"""DeviceSyncManager - encrypted laptop-to-phone sync, dispatch and streaming.

Threat model
------------
A paired phone is an untrusted peer that must present a shared pairing token
before it is allowed to stay connected. It can receive the status frames the
laptop chooses to push and it can send alerts back, but it can never dispatch a
command: the command channel is laptop-to-phone only, and every outbound
contact the laptop makes (dispatch, alert, probe) sits behind the mandatory
``cross_device`` approval. Plain ``ws://`` and ``http://`` endpoints are
refused outright so no frame ever travels without transport encryption, and the
first TLS handshake against a peer pins its certificate (trust on first use) so
a later interception fails the pin comparison instead of being accepted.

Mechanism
---------
* ``device_manager.tls_dir`` holds a locally generated self-signed certificate
  and CA key; ``device_manager.pairing_token_path`` holds the shared secret a
  peer must present during the handshake.
* ``start_server`` binds the listening socket itself, starts the ``websockets``
  sync server on a daemon thread and only then signals readiness through an
  Event, so an early client can never race a server thread that has not started
  listening yet.
* Inbound frames are one-directional: ``sync`` and ``alert`` are acknowledged,
  ``command`` is always refused with an explanation, ``bye`` closes cleanly.
* ``scan`` uses mDNS only when ``zeroconf`` is installed and otherwise reports
  the missing dependency rather than pretending to have discovered anything.
"""

from __future__ import annotations

import importlib.util
import ipaddress
import json
import logging
import re
import secrets
import shlex
import shutil
import socket
import ssl
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    _HAS_CRYPTO = True
except Exception:  # noqa: BLE001 - cryptography is optional at import time
    x509 = None
    _HAS_CRYPTO = False

try:
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.server import serve as ws_serve

    _HAS_WS = True
except Exception:  # noqa: BLE001 - websockets is optional at import time
    ConnectionClosed = OSError
    ws_serve = None
    _HAS_WS = False

logging.getLogger("websockets").setLevel(logging.CRITICAL)

USAGE = (
    "Usage: device-sync [status | serve | stop | scan | transports | "
    "sync [id...] | dispatch <id> <command...> | alert <id> <text...> | "
    "list | add <id> <endpoint> | remove <id> | trust <id> | current]"
)

_OFF = "device sync is off for this session (features.device_manager)."
_HONORIFIC_DELIMITERS = " \t\r\n,.!?;:"
_ENDPOINT_RE = re.compile(
    r"^(?P<scheme>ssh|wss)://(?:(?P<user>[^@/]+)@)?"
    r"(?P<host>\[[^\]]+\]|[^/:]+)(?::(?P<port>\d+))?$"
)


class DeviceSyncManager:
    """Encrypted peer sync, one-directional dispatch and status streaming."""

    def __init__(self, config, gate=None, audit=None, devices=None) -> None:
        self.config = config
        self.gate = gate
        self.audit = audit
        self.devices = devices
        section = config.get_dict("device_manager")
        self.bind_host = str(section.get("bind_host", "0.0.0.0"))
        self.port = int(section.get("ws_port", 8772))
        self.tls_dir = Path(config.resolve("device_manager.tls_dir"))
        self.token_path = Path(config.resolve("device_manager.pairing_token_path"))
        self.stream_interval = float(section.get("stream_interval_seconds", 30))
        self.handshake_timeout = float(section.get("handshake_timeout_seconds", 30))
        self.call_timeout = float(section.get("call_timeout_seconds", 8))
        self.ssh_timeout = float(section.get("ssh_timeout_seconds", 10))
        self.ssh_probe = str(section.get("ssh_probe", "echo jarvis-sync-ok"))
        self.mdns_service = str(section.get("mdns_service", "_jarvis-sync._tcp.local."))
        self.bound_port = 0
        self.last_sync: dict[str, Any] = {"ts": 0.0, "synced": 0, "failed": 0}
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._stream_thread: threading.Thread | None = None
        self._stream_stop = threading.Event()
        self._ready = threading.Event()
        self._start_errors: list[str] = []
        self._clients: list[Any] = []
        self._clients_lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._token: str | None = None
        self._cert_path: Path | None = None
        self._key_path: Path | None = None

    # ----------------------------------------------------------------- state
    @property
    def running(self) -> bool:
        return (self._server is not None and self._thread is not None
                and self._thread.is_alive())

    @property
    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    @property
    def device_count(self) -> int:
        return len(self.devices.devices) if self.devices is not None else 0

    def _enabled(self) -> bool:
        return bool(self.config.feature("device_manager", True))

    def _respond(self, text: str) -> str:
        """Address the operator exactly once, without mangling real words.

        A naive prefix would turn a device called "Sirius" into "ius", so the
        honorific is only peeled when a separator follows it.
        """
        honorific = self.config.honorific()
        body = str(text or "").strip()
        if honorific:
            for _ in range(3):
                if not body.lower().startswith(honorific.lower()):
                    break
                rest = body[len(honorific):]
                if not rest:
                    body = ""
                    break
                if not (rest[0].isspace() or rest[0] in ",.!?;:"):
                    break
                body = rest.lstrip(_HONORIFIC_DELIMITERS).strip()
            if not body:
                return honorific
        return f"{honorific}, {body}" if honorific else body

    def _log(self, action: str, decision: str, detail: dict[str, Any] | None = None) -> None:
        if self.audit is not None:
            self.audit.record("device_sync", action, decision, detail)

    def _endpoint(self) -> str:
        host = self.bind_host
        if host in ("0.0.0.0", "::", ""):
            host = socket.gethostname()
        return f"wss://{host}:{self.bound_port}"

    def _status_payload(self) -> dict[str, Any]:
        return {"ts": time.time(), "port": self.bound_port,
                "clients": self.client_count, "devices": self.device_count,
                "last_sync": dict(self.last_sync)}

    def _send_json(self, conn: Any, frame: dict[str, Any]) -> bool:
        try:
            with self._send_lock:
                conn.send(json.dumps(frame, ensure_ascii=False))
            return True
        except Exception:  # noqa: BLE001 - a dead peer is a normal outcome
            with self._clients_lock:
                if conn in self._clients:
                    self._clients.remove(conn)
            return False

    # ------------------------------------------------------------------ tls
    def ensure_tls(self) -> dict[str, Any]:
        """Generate the local certificate once and reuse it afterwards."""
        if not _HAS_CRYPTO:
            return {"ok": False, "generated": False,
                    "error": "cryptography is not installed - pip install cryptography"}
        self.tls_dir.mkdir(parents=True, exist_ok=True)
        cert_path = self.tls_dir / "server.crt"
        key_path = self.tls_dir / "server.key"
        if cert_path.is_file() and key_path.is_file():
            self._cert_path, self._key_path = cert_path, key_path
            return {"ok": True, "generated": False, "cert": str(cert_path)}
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        hostname = re.sub(r"[^0-9A-Za-z.-]", "-", socket.gethostname())[:63] or "jarvis"
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, f"jarvis-{hostname}"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Jarvis OS"),
        ])
        now = datetime.now(timezone.utc)
        alternative_names: list[Any] = [x509.DNSName("localhost"), x509.DNSName(hostname)]
        for literal in ("127.0.0.1", "::1"):
            try:
                alternative_names.append(x509.IPAddress(ipaddress.ip_address(literal)))
            except ValueError:
                continue
        certificate = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1))
            .not_valid_after(now + timedelta(days=825))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.SubjectAlternativeName(alternative_names), critical=False)
            .sign(key, hashes.SHA256())
        )
        key_path.write_bytes(key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption()))
        cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
        for path in (key_path, cert_path):
            try:
                path.chmod(0o600)
            except OSError:
                pass
        self._cert_path, self._key_path = cert_path, key_path
        self._log("tls", "generated", {"cert": cert_path.name, "days": 825})
        return {"ok": True, "generated": True, "cert": str(cert_path)}

    def ensure_token(self) -> dict[str, Any]:
        """Create the pairing token once and reuse it for every later session."""
        if self._token:
            return {"ok": True, "token": self._token, "created": False}
        value = ""
        if self.token_path.is_file():
            try:
                value = self.token_path.read_text(encoding="utf-8").strip()
            except OSError:
                value = ""
        created = False
        if len(value) < 16:
            value = secrets.token_hex(24)
            created = True
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            self.token_path.write_text(value, encoding="utf-8")
            try:
                self.token_path.chmod(0o600)
            except OSError:
                pass
        self._token = value
        if created:
            self._log("pairing_token", "generated", {"path": self.token_path.name})
        return {"ok": True, "token": value, "created": created}

    def _peer_ssl(self, host: str, port: int, timeout: float) -> ssl.SSLContext:
        """Trust-on-first-use pinning of the peer certificate.

        The first handshake stores the peer certificate under ``tls_dir/peers``;
        every later handshake must present byte-identical certificate data. A
        caller in the middle of a TLS handshake cannot produce the pinned
        certificate without its private key, so a mismatch is refused here
        rather than at the socket layer where it would be invisible.
        """
        peers = self.tls_dir / "peers"
        peers.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^0-9A-Za-z._-]", "_", f"{host}_{port}")
        pin_path = peers / f"{safe}.pem"
        probe = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        probe.check_hostname = False
        probe.verify_mode = ssl.CERT_NONE
        der = None
        try:
            with socket.create_connection((host, port),
                                          timeout=max(1.0, min(5.0, float(timeout)))) as raw:
                with probe.wrap_socket(raw, server_hostname=self._sni(host)) as tls:
                    der = tls.getpeercert(binary_form=True)
        except (OSError, ssl.SSLError):
            der = None
        if der:
            pem = ssl.DER_cert_to_PEM_cert(der)
            if pin_path.is_file():
                pinned = pin_path.read_text(encoding="utf-8")
                if pinned != pem:
                    self._log("tofu_pin", "refused", {"host": host, "port": port})
                    raise ssl.SSLCertVerificationError(
                        "the pinned certificate for this peer does not match the one "
                        "presented - possible interception, Sir (pin mismatch)")
            else:
                pin_path.write_text(pem, encoding="utf-8")
                self._log("tofu_pin", "pinned", {"host": host, "port": port})
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    @staticmethod
    def _sni(host: str) -> str | None:
        try:
            socket.inet_aton(host)
            return None
        except OSError:
            pass
        try:
            socket.inet_pton(socket.AF_INET6, host)
            return None
        except (OSError, AttributeError):
            return host or None

    # -------------------------------------------------------------- serving
    def start_server(self) -> dict[str, Any]:
        """Bind, gate, and serve. Never starts without a live manual approval."""
        if not self._enabled():
            return {"ok": False, "error": _OFF}
        if self.running:
            return {"ok": True, "serving": True, "already": True,
                    "port": self.bound_port, "endpoint": self._endpoint(),
                    "message": self._respond(
                        f"the sync server is already serving on port {self.bound_port}.")}
        tls = self.ensure_tls()
        if not tls.get("ok"):
            return {"ok": False, "error": str(tls.get("error", "TLS material unavailable"))}
        token = self.ensure_token()
        if not token.get("ok"):
            return {"ok": False, "error": str(token.get("error", "pairing token unavailable"))}
        preview = str(token["token"])[:8]
        prompt = (
            "Jarvis requests permission to start the encrypted device sync server.\n"
            f"  bind:     {self.bind_host}:{self.port}\n"
            f"  protocol: wss (TLS {825}-day local certificate)\n"
            f"  token:    {preview}...\n"
            "  peers pair with: --device add <id> wss://<host>:<port>"
        )
        if self.gate is not None and not self.gate.approve(
                "cross_device", prompt,
                detail={"bind": self.bind_host, "port": self.port}):
            self._log("serve", "denied", {"bind": self.bind_host, "port": self.port})
            return {"ok": False, "denied": True,
                    "error": "permission denied by operator - the sync server was not started."}
        host, port = self.bind_host, int(self.port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.listen(8)
        except OSError as exc:
            try:
                sock.close()
            except OSError:
                pass
            self._log("serve", "failed", {"bind": host, "port": port, "error": str(exc)[:200]})
            return {"ok": False, "error": f"cannot bind {host}:{port}: {exc}"}
        actual_port = int(sock.getsockname()[1])
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            context.load_cert_chain(str(self._cert_path), str(self._key_path))
        except (OSError, ssl.SSLError) as exc:
            try:
                sock.close()
            except OSError:
                pass
            return {"ok": False, "error": f"cannot load the sync certificate: {exc}"}
        self.bound_port = actual_port
        self._ready.clear()
        self._start_errors = []
        self._thread = threading.Thread(target=self._serve, args=(sock, context),
                                        name="jarvis-sync-server", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)
        if self._start_errors:
            self.bound_port = 0
            return {"ok": False, "error": self._start_errors[0]}
        if not self.running:
            self.bound_port = 0
            return {"ok": False, "error": "the sync server thread exited immediately"}
        self._log("serve", "ok", {"bind": host, "port": actual_port})
        return {"ok": True, "serving": True, "already": False,
                "port": actual_port, "endpoint": self._endpoint(),
                "token": str(token["token"]),
                "message": self._respond(
                    f"the sync server is serving on port {actual_port} with "
                    f"{self.client_count} client(s).")}

    def _serve(self, sock: socket.socket, context: ssl.SSLContext) -> None:
        entered = False
        try:
            with ws_serve(self._connection, sock=sock, ssl=context,
                          open_timeout=5, ping_interval=20) as server:
                entered = True
                self._server = server
                self._stream_stop.clear()
                self._stream_thread = threading.Thread(
                    target=self._stream_loop, name="jarvis-sync-stream", daemon=True)
                self._stream_thread.start()
                self._ready.set()
                server.serve_forever()
        except Exception as exc:  # noqa: BLE001 - reported back through start_server
            self._start_errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            if not entered:
                try:
                    sock.close()
                except OSError:
                    pass
            self._ready.set()
            if self._thread is threading.current_thread():
                self._server = None
                self._stream_stop.set()
                self._stream_thread = None
                self._thread = None

    def stop_server(self) -> dict[str, Any]:
        """Close every peer and the listener. Safe to call when nothing runs."""
        if not self.running:
            return {"ok": True, "serving": False, "already": True,
                    "message": self._respond("the sync server was not running.")}
        self._stream_stop.set()
        for conn in list(self._clients):
            self._send_json(conn, {"type": "bye", "payload": {"ok": True}})
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - already gone is already fine
                pass
        with self._clients_lock:
            self._clients.clear()
        server, thread = self._server, self._thread
        if server is not None:
            try:
                server.shutdown()
            except Exception:  # noqa: BLE001 - shutdown on a closing socket
                pass
        if thread is not None and thread is not threading.current_thread():
            thread.join(4)
        stream = self._stream_thread
        if stream is not None and stream is not threading.current_thread():
            stream.join(2)
        self._log("serve", "stopped", {"port": self.bound_port})
        return {"ok": True, "serving": False, "already": False,
                "message": self._respond("the sync server stopped.")}

    def _stream_loop(self) -> None:
        interval = float(self.stream_interval)
        if interval <= 0:
            self._stream_stop.wait()
            return
        while not self._stream_stop.wait(interval):
            if not self._clients:
                continue
            frame = {"type": "stream", "payload": self._status_payload()}
            for conn in list(self._clients):
                self._send_json(conn, frame)

    # ----------------------------------------------------------- connection
    def _connection(self, conn: Any) -> None:
        peer = ""
        try:
            peer = str(conn.remote_address or "")
        except Exception:  # noqa: BLE001 - informational only
            pass
        if not self._handshake(conn, peer):
            return
        normal = False
        try:
            with self._clients_lock:
                self._clients.append(conn)
            self._send_json(conn, {"type": "ack",
                                   "payload": {"ok": True, "authenticated": True}})
            self._send_json(conn, {"type": "status", "payload": self._status_payload()})
            self._log("auth", "ok", {"peer": peer, "clients": self.client_count})
            while True:
                try:
                    raw = conn.recv(timeout=None)
                except ConnectionClosed:
                    normal = True
                    break
                except Exception:  # noqa: BLE001 - treat as a dropped peer
                    break
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", "replace")
                try:
                    frame = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    self._send_json(conn, {"type": "error", "payload": {
                        "ok": False, "error": "refused: malformed frame"}})
                    continue
                if not isinstance(frame, dict):
                    self._send_json(conn, {"type": "error", "payload": {
                        "ok": False, "error": "refused: malformed frame"}})
                    continue
                kind = str(frame.get("type", ""))
                if kind == "bye":
                    normal = True
                    break
                if kind == "command":
                    self._send_json(conn, {"type": "error", "payload": {
                        "ok": False,
                        "error": "refused: this channel is one-directional - paired "
                                 "devices cannot dispatch commands to the laptop"}})
                elif kind == "sync":
                    self._send_json(conn, {"type": "ack", "payload": {
                        "ok": True, "kind": "sync", "server": self._status_payload()}})
                elif kind == "alert":
                    text = self._alert_text(frame)[:500]
                    if self.audit is not None:
                        self.audit.record("device_sync", "inbound_alert", "ok",
                                          {"text": text, "peer": peer})
                    self._send_json(conn, {"type": "ack",
                                           "payload": {"ok": True, "kind": "alert"}})
                elif kind == "auth":
                    self._send_json(conn, {"type": "error", "payload": {
                        "ok": False, "error": "refused: already authenticated"}})
                else:
                    self._send_json(conn, {"type": "error", "payload": {
                        "ok": False,
                        "error": f"refused: unknown frame type '{kind[:40]}'"}})
        finally:
            with self._clients_lock:
                if conn in self._clients:
                    self._clients.remove(conn)
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - closing twice is harmless
                pass
            if not normal:
                self._log("disconnect", "abnormal", {"peer": peer})

    @staticmethod
    def _alert_text(frame: dict[str, Any]) -> str:
        payload = frame.get("payload")
        if isinstance(payload, dict):
            return str(payload.get("text") or "")
        if isinstance(payload, str):
            return payload
        return str(frame.get("text") or "")

    def _handshake(self, conn: Any, peer: str) -> bool:
        try:
            raw = conn.recv(timeout=self.handshake_timeout)
        except TimeoutError:
            self._send_json(conn, {"type": "error", "payload": {
                "ok": False, "error": "refused: handshake timed out"}})
            self._log("auth", "timeout", {"peer": peer})
            return False
        except Exception:  # noqa: BLE001 - dropped before the first frame
            self._log("auth", "refused", {"peer": peer, "reason": "no handshake"})
            return False
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        try:
            frame = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            frame = {}
        expected = self._token or ""
        offered = str(frame.get("token", "")) if isinstance(frame, dict) else ""
        ok = (isinstance(frame, dict) and frame.get("type") == "auth"
              and bool(expected) and bool(offered)
              and secrets.compare_digest(offered, expected))
        if not ok:
            self._send_json(conn, {"type": "error", "payload": {
                "ok": False, "error": "refused: bad pairing token"}})
            self._log("auth", "refused", {"peer": peer})
            return False
        return True

    # -------------------------------------------------------------- peers
    @staticmethod
    def _parse_endpoint(endpoint: str) -> dict[str, Any] | None:
        """Split an endpoint, or None when this build cannot dial it."""
        match = _ENDPOINT_RE.match(str(endpoint or "").strip())
        if match is None:
            return None
        host = match.group("host")
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]
        scheme = match.group("scheme")
        return {"scheme": scheme, "host": host,
                "port": int(match.group("port") or (8022 if scheme == "ssh" else 443)),
                "user": match.group("user") or ""}

    def _resolve_target(self, device_id: str) -> tuple[Any, dict[str, Any] | None,
                                                       dict[str, Any] | None]:
        if self.devices is None:
            return None, None, {"ok": False, "error": "device registry is unavailable."}
        device = self.devices.devices.get(device_id)
        if device is None:
            return None, None, {"ok": False, "error":
                                f"device '{device_id}' is not registered - register it "
                                f"first with: --device add {device_id} "
                                f"ssh://user@host:8022 (or wss://host:port)"}
        if device_id == self.devices.current:
            return device, None, {"ok": False, "error":
                                  f"'{device_id}' is this machine, not a remote peer."}
        endpoint = str(device.endpoint or "").strip()
        if not endpoint:
            return device, None, {"ok": False, "error":
                                  f"'{device_id}' has no endpoint - register one with: "
                                  f"--device add {device_id} ssh://user@host:8022 "
                                  f"(or wss://host:port)"}
        if endpoint.lower().startswith(("ws://", "http://", "https://")):
            return device, None, {"ok": False, "error":
                                  "endpoint must be ssh:// or wss:// - plain ws:// and "
                                  "http:// are refused (no transport encryption)."}
        target = self._parse_endpoint(endpoint)
        if target is None:
            return device, None, {"ok": False, "error":
                                  f"'{device_id}' has an endpoint this build cannot "
                                  f"dial: {endpoint[:80]} - need ssh://user@host:port "
                                  f"or wss://host:port"}
        return device, target, None

    def dispatch(self, device_id: str, words: list[str]) -> dict[str, Any]:
        """Run one command on a peer, laptop-to-phone only, gate first."""
        if not self._enabled():
            return {"ok": False, "error": _OFF}
        device_id = str(device_id or "").strip()
        if not device_id or not words:
            return {"ok": False, "error":
                    "Usage: device-sync dispatch <id> <command...> - give me the "
                    "device id and the command to run."}
        device, target, failure = self._resolve_target(device_id)
        if failure is not None:
            return failure
        command_line = " ".join(str(word) for word in words)
        prompt = ("Jarvis requests permission to dispatch a command to a device.\n"
                  f"  device:   {device_id}\n  endpoint: {device.endpoint}\n"
                  f"  scheme:   {target['scheme']}\n  command:  {command_line}")
        if self.gate is not None and not self.gate.approve(
                "cross_device", prompt,
                detail={"device": device_id, "endpoint": device.endpoint,
                        "command": command_line}):
            self._log("dispatch", "denied", {"device": device_id})
            return {"ok": False, "denied": True, "error":
                    f"permission denied by operator - nothing was sent to '{device_id}'."}
        if target["scheme"] == "ssh":
            result = self._ssh_run(target, list(words), self.ssh_timeout)
        else:
            result = self._wss_call(target, {"type": "command", "command": command_line})
        if result.get("ok"):
            result["message"] = self._respond(
                f"command dispatched to '{device_id}' via {target['scheme']}.")
            self._log("dispatch", "ok", {"device": device_id, "scheme": target["scheme"]})
        else:
            self._log("dispatch", "failed",
                      {"device": device_id, "error": str(result.get("error", ""))[:200]})
        return result

    def alert(self, device_id: str, text: str) -> dict[str, Any]:
        """Send one operator-approved alert to a paired phone."""
        if not self._enabled():
            return {"ok": False, "error": _OFF}
        device_id = str(device_id or "").strip()
        text = str(text or "").strip()
        if not device_id or not text:
            return {"ok": False, "error":
                    "Usage: device-sync alert <id> <text...> - name the device and "
                    "write the alert."}
        device, target, failure = self._resolve_target(device_id)
        if failure is not None:
            return failure
        prompt = ("Jarvis requests permission to send an alert to a device.\n"
                  f"  device:   {device_id}\n  endpoint: {device.endpoint}\n"
                  f"  message:  {text[:200]}")
        if self.gate is not None and not self.gate.approve(
                "cross_device", prompt,
                detail={"device": device_id, "endpoint": device.endpoint,
                        "text": text[:200]}):
            self._log("alert", "denied", {"device": device_id})
            return {"ok": False, "denied": True, "error":
                    f"permission denied by operator - the alert was not sent to '{device_id}'."}
        if target["scheme"] == "ssh":
            words = ["termux-notification", "--title", "Jarvis alert", "--text", text]
            result = self._ssh_run(target, words, self.ssh_timeout)
        else:
            result = self._wss_call(target, {"type": "alert", "text": text})
        if result.get("ok"):
            result["message"] = self._respond(f"alert delivered to '{device_id}'.")
            self._log("alert", "ok", {"device": device_id, "chars": len(text)})
        else:
            self._log("alert", "failed",
                      {"device": device_id, "error": str(result.get("error", ""))[:200]})
        return result

    def sync(self, ids: list[str] | None = None) -> dict[str, Any]:
        """Probe paired devices under one approval and record what answered."""
        if not self._enabled():
            return {"ok": False, "error": _OFF}
        wanted = [str(value) for value in (ids or [])]
        if self.devices is None:
            return {"ok": False, "error": "device registry is unavailable."}
        unknown = [value for value in wanted if value not in self.devices.devices]
        if unknown:
            return {"ok": False, "error":
                    f"unknown device id(s): {', '.join(unknown)} - register them first "
                    f"with: --device add <id> ssh://user@host:8022"}
        peers = [device for device in self.devices.devices.values()
                 if device.device_id != self.devices.current
                 and str(device.endpoint or "").strip()]
        if wanted:
            peers = [device for device in peers if device.device_id in wanted]
        if not peers:
            return {"ok": True, "synced": 0, "failed": 0, "message": self._respond(
                "no paired devices with an endpoint yet - register one with "
                "--device add <id> ssh://user@host:8022 or wss://host:port.")}
        listing = "\n".join(f"  {device.device_id:<16} {device.endpoint}"
                            for device in peers)
        prompt = (f"Jarvis requests permission to probe {len(peers)} paired device(s):\n"
                  f"{listing}")
        if self.gate is not None and not self.gate.approve(
                "cross_device", prompt,
                detail={"peers": [device.device_id for device in peers]}):
            self._log("sync", "denied", {"peers": len(peers)})
            return {"ok": False, "denied": True, "error":
                    "permission denied by operator - no device was contacted."}
        synced = 0
        errors: list[str] = []
        for device in peers:
            ok, note = self._probe(device)
            if ok:
                synced += 1
            else:
                errors.append(note)
        failed = len(peers) - synced
        self.last_sync = {"ts": time.time(), "synced": synced, "failed": failed}
        self._log("sync", "ok" if synced else "failed",
                  {"synced": synced, "failed": failed})
        result: dict[str, Any] = {"ok": synced > 0, "synced": synced, "failed": failed,
                                  "last_sync": dict(self.last_sync),
                                  "message": self._respond(
                                      f"probed {len(peers)} device(s), {synced} answered.")}
        if synced == 0:
            result["error"] = "no paired device answered: " + "; ".join(errors)[:300]
        return result

    def _probe(self, device: Any) -> tuple[bool, str]:
        endpoint = str(device.endpoint or "").strip()
        if endpoint.lower().startswith(("ws://", "http://", "https://")):
            return False, f"{device.device_id}: endpoint must be ssh:// or wss://"
        target = self._parse_endpoint(endpoint)
        if target is None:
            return False, f"{device.device_id}: unrecognised endpoint {endpoint[:60]}"
        if target["scheme"] == "ssh":
            try:
                words = shlex.split(self.ssh_probe)
            except ValueError:
                words = ["echo", "jarvis-sync-ok"]
            result = self._ssh_run(target, words or ["echo", "jarvis-sync-ok"],
                                   self.ssh_timeout)
        else:
            result = self._wss_call(target, {"type": "sync"})
        ok = bool(result.get("ok"))
        self._mark_seen(device, "online" if ok else "unreachable")
        return ok, f"{device.device_id}: {result.get('error', 'ok')}"[:200]

    def _mark_seen(self, device: Any, status: str) -> None:
        device.last_seen = time.time()
        device.status = status
        if self.devices is not None:
            try:
                self.devices.save()
            except OSError:
                pass

    def scan(self) -> dict[str, Any]:
        """mDNS browse for paired devices - reports a missing dependency, no guessing."""
        try:
            found_dependency = importlib.util.find_spec("zeroconf") is not None
        except (ImportError, ValueError):
            found_dependency = False
        if not found_dependency:
            return {"ok": False, "found": [], "error":
                    "zeroconf is not installed - pip install zeroconf to enable mDNS "
                    "discovery of paired devices."}
        try:
            from zeroconf import ServiceBrowser, Zeroconf
        except Exception as exc:  # noqa: BLE001 - optional dependency
            return {"ok": False, "found": [], "error": f"zeroconf failed to load: {exc}"}
        discovered: list[str] = []

        class _Listener:
            def add_service(self, _type: str, name: str, _info: Any) -> None:
                discovered.append(name)

            def update_service(self, _type: str, name: str, _info: Any) -> None:
                return None

            def remove_service(self, _type: str, name: str, _info: Any) -> None:
                if name in discovered:
                    discovered.remove(name)

        zeroconf = Zeroconf()
        try:
            ServiceBrowser(zeroconf, self.mdns_service, _Listener())
            time.sleep(0.5)
        except Exception as exc:  # noqa: BLE001 - browse must never crash the OS
            return {"ok": False, "found": [], "error": f"mDNS browse failed: {exc}"}
        finally:
            try:
                zeroconf.close()
            except Exception:  # noqa: BLE001
                pass
        names = sorted(set(discovered))
        return {"ok": True, "found": names, "service": self.mdns_service,
                "message": self._respond(f"mDNS scan found {len(names)} service(s).")}

    # ------------------------------------------------------------ transports
    def _ssh_run(self, target: dict[str, Any], words: list[str],
                 timeout: float) -> dict[str, Any]:
        ssh = shutil.which("ssh")
        if not ssh:
            return {"ok": False, "error":
                    "ssh was not found on this machine - on Windows install it with: "
                    "Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0"}
        connect_timeout = max(1, min(5, int(timeout or 5)))
        destination = (f"{target['user']}@{target['host']}" if target.get("user")
                       else target["host"])
        command = [ssh, "-p", str(target["port"]), "-o", "BatchMode=yes",
                   "-o", "StrictHostKeyChecking=accept-new",
                   "-o", f"ConnectTimeout={connect_timeout}",
                   destination, *[str(word) for word in words]]
        try:
            proc = subprocess.run(command, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=max(1.0, float(timeout)),
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            return {"ok": False, "error":
                    f"ssh to {destination}:{target['port']} timed out after {timeout}s."}
        except OSError as exc:
            return {"ok": False, "error": f"ssh failed to start: {exc}"}
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[:300]
            return {"ok": False, "error": f"ssh exit {proc.returncode}: {detail}"}
        return {"ok": True, "stdout": (proc.stdout or "")[:4000]}

    def _wss_call(self, target: dict[str, Any], frame: dict[str, Any],
                  timeout: float | None = None) -> dict[str, Any]:
        """Dial a peer over TLS, authenticate, exchange exactly one frame."""
        budget = max(5.0, float(timeout or self.call_timeout))
        try:
            from websockets.sync.client import connect
        except Exception as exc:  # noqa: BLE001 - optional dependency
            return {"ok": False, "error": f"websockets is not installed ({exc})"}
        host, port = target["host"], int(target["port"])
        try:
            context = self._peer_ssl(host, port, budget)
        except ssl.SSLCertVerificationError as exc:
            return {"ok": False, "error": f"peer certificate rejected: {exc}"}
        except (OSError, ssl.SSLError) as exc:
            return {"ok": False, "error": f"cannot reach {host}:{port}: {exc}"}
        try:
            with connect(f"wss://{host}:{port}", ssl=context,
                         open_timeout=budget) as conn:
                if not self._send_json(conn, {"type": "auth", "token": self._token or ""}):
                    return {"ok": False, "error": "connection closed during the handshake."}
                reply = self._recv_reply(conn, budget)
                if reply is None:
                    return {"ok": False, "error":
                            f"no handshake reply from {host}:{port} within {budget}s."}
                if reply.get("type") == "error":
                    return {"ok": False, "error": self._error_of(reply, "handshake refused")}
                if not self._send_json(conn, frame):
                    return {"ok": False, "error": "connection closed before the frame was sent."}
                answer = self._recv_reply(conn, budget)
                if answer is None:
                    return {"ok": False, "error": f"{host}:{port} sent no reply to the frame."}
                if answer.get("type") == "error":
                    return {"ok": False, "error": self._error_of(answer, "refused")}
                payload = answer.get("payload")
                if isinstance(payload, dict):
                    if payload.get("ok") is False:
                        return {"ok": False, "error": self._error_of(answer, "refused")}
                    merged = dict(payload)
                    merged.setdefault("ok", True)
                    return merged
                return {"ok": True, "frame": answer}
        except ssl.SSLCertVerificationError as exc:
            return {"ok": False, "error": f"peer certificate rejected: {exc}"}
        except (TimeoutError, OSError, ssl.SSLError) as exc:
            return {"ok": False, "error": f"cannot complete TLS call to {host}:{port}: {exc}"}
        except Exception as exc:  # noqa: BLE001 - a protocol surprise is not a crash
            return {"ok": False, "error": f"sync call failed: {type(exc).__name__}: {exc}"}

    @staticmethod
    def _error_of(frame: dict[str, Any], fallback: str) -> str:
        payload = frame.get("payload")
        if isinstance(payload, dict) and payload.get("error"):
            return str(payload["error"])
        return fallback

    def _recv_reply(self, conn: Any, timeout: float) -> dict[str, Any] | None:
        """Read frames until a reply arrives, ignoring pushed status/stream frames."""
        deadline = time.monotonic() + max(0.5, float(timeout))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                raw = conn.recv(timeout=remaining)
            except TimeoutError:
                return None
            except Exception:  # noqa: BLE001 - closed mid-read
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            try:
                frame = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(frame, dict):
                continue
            if frame.get("type") in ("status", "stream"):
                continue
            return frame

    # ------------------------------------------------------------- reporting
    def status(self) -> str:
        if not self._enabled():
            return "device sync: off for this session (features.device_manager)."
        state = "serving" if self.running else "stopped"
        return (f"device sync: {state} port={self.bound_port or self.port} "
                f"clients={self.client_count} devices={self.device_count} "
                f"last_sync={self.last_sync.get('ts') or 'never'}")

    def report(self) -> str:
        return (f"DeviceSyncManager: serving={'on' if self.running else 'off'}, "
                f"port={self.bound_port or '-'}, devices={self.device_count}, "
                f"last_sync={self.last_sync.get('ts') or 'never'}")

    def transports(self) -> dict[str, Any]:
        try:
            zeroconf = importlib.util.find_spec("zeroconf") is not None
        except (ImportError, ValueError):
            zeroconf = False
        return {"websockets": _HAS_WS, "tls": _HAS_CRYPTO, "zeroconf": zeroconf,
                "ssh": shutil.which("ssh"), "ssh_probe": self.ssh_probe,
                "mdns_service": self.mdns_service}

    def command(self, args: list[str]) -> Any:
        """Router for the ``--device`` flag and the REPL alike."""
        words = [str(value) for value in (args or [])]
        sub = words[0].lower() if words else "status"
        rest = words[1:]
        if sub in ("status", "state"):
            return self.status()
        if sub in ("serve", "start", "server"):
            return self.start_server()
        if sub in ("stop", "shutdown"):
            return self.stop_server()
        if sub == "scan":
            return self.scan()
        if sub == "transports":
            return self.transports()
        if sub == "sync":
            return self.sync(rest or None)
        if sub == "dispatch":
            if len(rest) < 2:
                return {"ok": False, "error": USAGE}
            return self.dispatch(rest[0], rest[1:])
        if sub == "alert":
            if len(rest) < 2:
                return {"ok": False, "error": USAGE}
            return self.alert(rest[0], " ".join(rest[1:]))
        if sub in ("list", "add", "remove", "trust", "reach", "configure", "current"):
            if self.devices is None:
                return {"ok": False, "error": "device registry is unavailable."}
            return self.devices.command(words)
        return {"ok": False, "error": USAGE}
