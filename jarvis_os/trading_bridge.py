"""TradingBridge - offline market analysis and strictly gated execution.

A read-mostly analysis layer: deterministic synthetic market snapshot, SMA /
RSI / volatility indicators, a signal rule, Monk and Gorilla risk presets with
per-class position caps, and a Whale Radar that flags large notional moves.
Nothing here touches a live exchange and no model is talked to; the data is
generated once at import from a seeded PRNG so every run reproduces the same
numbers and the selftest stays deterministic.

Execution is symmetrical with the rest of Jarvis OS: a real shell routes every
``market_order`` through the ``trading_exec`` permission category, so Sir must
approve live. Unlike the BusinessAgent (where a missing gate degrades to
allowed), a bridge with *no* gate attached refuses to execute anything but a
dry-run decision - there is no capital-adjacent action taken without an
operator present. The ledger, the alerts and the Whale Radar history all live
under ``data/`` as append-only JSONL.
"""

from __future__ import annotations

import datetime as _dt
import random
from pathlib import Path
from typing import Any

from . import jsonl_store

RISK_PRESETS = {
    "monk": {
        "title": "Monk - capital preservation, index/gold/stablecoin only",
        "max_position_pct": 5,
        "max_leverage": 1,
        "max_drawdown_pct": 5,
        "daily_loss_stop_pct": 2,
        "allowed_classes": ("index", "gold", "stablecoin"),
    },
    "gorilla": {
        "title": "Gorilla - aggressive growth, all asset classes",
        "max_position_pct": 40,
        "max_leverage": 3,
        "max_drawdown_pct": 25,
        "daily_loss_stop_pct": 15,
        "allowed_classes": ("crypto", "index", "gold", "stablecoin"),
    },
}

_ASSET_DEFS = (
    ("BTC", "Bitcoin", "crypto", 64000.0),
    ("ETH", "Ether", "crypto", 3200.0),
    ("SOL", "Solana", "crypto", 150.0),
    ("SPX", "S&P 500 Index", "index", 5400.0),
    ("GLD", "Gold", "gold", 210.0),
)

_WHALE_EVENTS = (
    ("BTC", "buy", 3_200_000, "OTC accumulation across three venues"),
    ("SOL", "sell", 2_100_000, "large holder rotating into gold"),
    ("GLD", "buy", 1_750_000, "institutional bid near the close"),
    ("ETH", "buy", 250_000, "small algos front-running the open"),
    ("SPX", "sell", 900_000, "rebalancing flow, index-wide"),
)

_CLOSES = 60
_GATE_CATEGORY = "trading_exec"
_GATE_ACTION = "market_order"
_LOT = 1e-9  # smallest tradable quantity (satoshis / 1e-9 of an index unit)


def _build_closes(price: float) -> list[float]:
    """Deterministic 60-close series ending exactly at ``price``."""
    rng = random.Random(int(price) * 7919 + 104729)
    series = [price * (0.90 + 0.02 * rng.random())]
    for _ in range(_CLOSES - 1):
        series.append(series[-1] * (1.0 + rng.uniform(-0.012, 0.012)))
    scale = price / series[-1]
    return [round(value * scale, 6) for value in series]


def _market_snapshot() -> dict[str, dict[str, Any]]:
    snapshot = {}
    for symbol, name, asset_class, price in _ASSET_DEFS:
        snapshot[symbol] = {
            "symbol": symbol,
            "name": name,
            "class": asset_class,
            "price": price,
            "closes": _build_closes(price),
        }
    return snapshot


MARKET_SNAPSHOT = _market_snapshot()


def _sma(closes: list[float], window: int) -> float:
    tail = closes[-window:]
    return round(sum(tail) / len(tail), 4)


def _rsi(closes: list[float], window: int = 14) -> float:
    if len(closes) < window + 1:
        return 50.0
    gains = losses = 0.0
    for prev, cur in zip(closes[-window - 1:], closes[-window:]):
        delta = cur - prev
        gains += max(delta, 0.0)
        losses += max(-delta, 0.0)
    if losses == 0:
        return 100.0
    rs = (gains / window) / (losses / window)
    return round(100.0 - 100.0 / (1.0 + rs), 2)


def _volatility(closes: list[float], window: int = 20) -> float:
    tail = closes[-window:]
    mean = sum(tail) / len(tail)
    var = sum((value - mean) ** 2 for value in tail) / len(tail)
    return round(var ** 0.5, 4)


class TradingBridge:
    """Offline analysis + gated simulation of orders. Never trades live."""

    def __init__(self, kernel: Any, config: Any | None = None) -> None:
        self.kernel = kernel
        self.config = config if config is not None else kernel.config
        self.ledger_path = Path(
            str(self.config.get("trading.ledger_path", "data/trading_ledger.jsonl")))
        self.alerts_path = Path(
            str(self.config.get("trading.alerts_path", "data/trading_alerts.jsonl")))
        self.threshold_usd = float(self.config.get("trading.whale_threshold_usd", 1_000_000))
        self.reference_capital_usd = float(
            self.config.get("trading.reference_capital_usd", 100_000.0))
        self._seen_whales = {
            self._event_key(str(a.get("symbol", "")), str(a.get("side", "")),
                            str(a.get("ts", "")))
            for a in jsonl_store.replay_records(self.alerts_path).values()
            if a.get("kind") == "alert"
        }

    # ------------------------------------------------------------- accessors
    def risk_profile(self) -> dict[str, Any]:
        name = str(self.config.get("trading.risk_profile", "monk"))
        return {"name": name, "rules": RISK_PRESETS.get(name, RISK_PRESETS["monk"])}

    def dry_run_enabled(self) -> bool:
        return bool(self.config.get("trading.dry_run", True))

    def _cap_usd(self) -> float:
        rules = self.risk_profile()["rules"]
        return round(self.reference_capital_usd * rules["max_position_pct"] / 100.0, 2)

    def asset(self, symbol: str) -> dict[str, Any] | None:
        symbol = (symbol or "").upper().strip()
        return MARKET_SNAPSHOT.get(symbol)

    def _asset(self, symbol: str) -> dict[str, Any] | None:
        return self.asset(symbol)

    def symbol(self, symbol: str) -> dict[str, Any] | None:
        return self.asset(symbol)

    def ledger(self) -> list[dict[str, Any]]:
        out = []
        for record in jsonl_store.replay_records(self.ledger_path).values():
            if record.get("kind") == "order":
                out.append(record)
        return sorted(out, key=lambda r: str(r.get("ts", "")))

    def alerts(self) -> list[dict[str, Any]]:
        out = []
        for record in jsonl_store.replay_records(self.alerts_path).values():
            if record.get("kind") == "alert":
                out.append(record)
        return sorted(out, key=lambda r: str(r.get("ts", "")))

    # ------------------------------------------------------------ indicators
    @staticmethod
    def sma(closes: list[float], window: int) -> float:
        return _sma(closes, window)

    @staticmethod
    def rsi(closes: list[float], window: int = 14) -> float:
        return _rsi(closes, window)

    @staticmethod
    def volatility(closes: list[float], window: int = 20) -> float:
        return _volatility(closes, window)

    def analyze(self, symbol: str) -> dict[str, Any]:
        asset = self.asset(symbol)
        if asset is None:
            return {"error": f"unknown symbol {symbol!r}"}
        closes = asset["closes"]
        price = closes[-1]
        sma20 = _sma(closes, 20)
        sma50 = _sma(closes, 50)
        rsi = _rsi(closes)
        if rsi >= 70:
            signal = "bear"
        elif rsi <= 30:
            signal = "bull"
        elif price > sma20 > sma50:
            signal = "bull"
        elif price < sma20 < sma50:
            signal = "bear"
        else:
            signal = "neutral"
        return {
            "symbol": asset["symbol"],
            "class": asset["class"],
            "price": price,
            "sma20": sma20,
            "sma50": sma50,
            "rsi": rsi,
            "volatility": _volatility(closes),
            "signal": signal,
        }

    def market(self) -> list[dict[str, Any]]:
        return [self.analyze(symbol) for symbol in sorted(MARKET_SNAPSHOT)]

    # ------------------------------------------------------------- whale radar
    @staticmethod
    def _event_key(symbol: str, side: str, ts: str) -> str:
        return f"{symbol.upper()}:{side}:{ts}"

    def scan_whales(self) -> list[dict[str, Any]]:
        now = jsonl_store.now_iso()
        events = []
        for index, (symbol, side, usd, note) in enumerate(_WHALE_EVENTS):
            if usd < self.threshold_usd:
                continue
            ts = self._ts_at(index)
            record = {
                "id": jsonl_store.new_id("alert"),
                "ts": ts,
                "scanned_at": now,
                "kind": "alert",
                "symbol": symbol,
                "side": side,
                "usd": usd,
                "note": note,
                "status": "open",
            }
            key = self._event_key(symbol, side, ts)
            if key not in self._seen_whales:
                self._seen_whales.add(key)
                jsonl_store.append_record(self.alerts_path, record)
                events.append(record)
        return events

    def _ts_at(self, index: int) -> str:
        minutes = max(1, 3 - index)
        return (_dt.datetime.now(_dt.timezone.utc)
                - _dt.timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def resolve_alert(self, alert_id: str, decision: str) -> dict[str, Any]:
        records = jsonl_store.replay_records(self.alerts_path)
        record = records.get(alert_id)
        if record is None or record.get("kind") != "alert":
            return {"error": f"no alert record {alert_id}"}
        decision = decision.lower().strip()
        if decision not in ("ignore", "escalate"):
            return {"error": "decision must be ignore or escalate"}
        record["status"] = decision
        record["resolved_at"] = jsonl_store.now_iso()
        jsonl_store.append_record(self.alerts_path, record)
        self._record("resolve_alert", "ok", {"id": alert_id, "decision": decision})
        return record

    # ------------------------------------------------------------ compliance
    def _exposure_usd(self, symbol: str) -> float:
        asset = self.asset(symbol)
        if asset is None:
            return 0.0
        total = 0.0
        for order in self.ledger():
            if order.get("symbol") != symbol or order.get("result") != "approved":
                continue
            signed = float(order.get("qty", 0.0))
            if order.get("side") == "sell":
                signed = -signed
            total += signed * float(order.get("price", asset["price"]))
        return max(total, 0.0)

    def compliance_check(self, symbol: str, side: str, qty: float) -> tuple[bool, str]:
        rules = self.risk_profile()["rules"]
        asset = self.asset(symbol)
        if asset is None:
            return False, f"unknown symbol {symbol!r}"
        side = (side or "").lower()
        if side not in ("buy", "sell"):
            return False, f"side must be buy or sell, not {side!r}"
        try:
            qty = float(qty)
        except (TypeError, ValueError):
            return False, "quantity must be a number"
        if qty <= _LOT:
            return False, "quantity must be positive"
        if asset["class"] not in rules["allowed_classes"]:
            return False, (f"{asset['symbol']} is {asset['class']}, disallowed by "
                           f"{self.risk_profile()['name']} preset "
                           f"{rules['allowed_classes']}")
        notional = qty * asset["price"]
        exposure = self._exposure_usd(symbol)
        cap_usd = self._cap_usd()
        if notional + exposure > cap_usd + 0.01:
            return False, (f"notional ${notional:,.0f} + exposure ${exposure:,.0f} "
                           f"exceeds ${cap_usd:,.0f} cap")
        return True, "compliant"

    # ------------------------------------------------------------------ orders
    def _record(self, action: str, result: str, detail: dict[str, Any]) -> None:
        jsonl_store.append_record(self.ledger_path, {
            "id": jsonl_store.new_id("action"),
            "ts": jsonl_store.now_iso(),
            "kind": "action",
            "actor": "TradingBridge",
            "action": action,
            "result": result,
            "detail": detail,
        })
        audit = getattr(self.kernel, "audit", None)
        if audit is not None:
            audit.record(action, "trading",
                         "ok" if result in ("dry_run", "approved") else "denied", detail)

    def _authorize(self) -> tuple[bool, str]:
        gate = getattr(self.kernel, "gate", None)
        if gate is None:
            return False, "refused - no live gate attached"
        try:
            granted = gate.approve(_GATE_CATEGORY,
                                   f"execute simulated market order: {_GATE_ACTION}",
                                   {"category": _GATE_CATEGORY, "action": _GATE_ACTION})
        except Exception:  # noqa: BLE001 - a failing gate must fail closed
            return False, "gate error - denied"
        if not granted:
            return False, "denied by operator"
        return True, "approved live"

    def execute(self, symbol: str, side: str, qty: float,
                live: bool = False) -> dict[str, Any]:
        asset = self.asset(symbol)
        if asset is None:
            return {"error": f"unknown symbol {symbol!r}"}
        ok, reason = self.compliance_check(symbol, side, qty)
        if not ok:
            return {"decision": "refused", "reason": reason, "symbol": symbol}
        dry = self.dry_run_enabled() and not live
        if dry:
            result = "dry_run"
            mode = "dry_run"
        else:
            allowed, how = self._authorize()
            if not allowed:
                result = "refused" if how.startswith("refused") else "denied"
                mode = result
            else:
                result = "approved"
                mode = "approved_simulated"
        order = {
            "id": jsonl_store.new_id("order"),
            "ts": jsonl_store.now_iso(),
            "kind": "order",
            "symbol": asset["symbol"],
            "class": asset["class"],
            "side": side.lower(),
            "qty": float(qty),
            "price": asset["price"],
            "notional_usd": round(float(qty) * asset["price"], 2),
            "mode": mode,
            "result": result,
            "gate": "dry_run" if dry else "trading_exec",
        }
        jsonl_store.append_record(self.ledger_path, order)
        self._record("market_order", result, {"id": order["id"]})
        return order

    def pnl(self) -> dict[str, Any]:
        positions: dict[str, dict[str, Any]] = {}
        for order in self.ledger():
            if order.get("result") != "approved":
                continue
            position = positions.setdefault(order["symbol"], {"qty": 0.0, "cost": 0.0})
            qty = float(order["qty"])
            if order.get("side") == "sell":
                qty = -qty
            position["qty"] += qty
            position["cost"] += qty * float(order["price"])
        unrealized = 0.0
        lines = []
        for symbol in sorted(positions):
            position = positions[symbol]
            total = position["cost"]
            position["avg_price"] = round(abs(total / position["qty"]), 4) if position["qty"] else 0.0
            mark = float(asset["price"]) if (asset := self.asset(symbol)) else 0.0
            pnl = (mark - position["avg_price"]) * position["qty"]
            unrealized += pnl
            lines.append({"symbol": symbol, "qty": round(position["qty"], 6),
                          "avg_price": position["avg_price"], "mark": mark,
                          "unrealized": round(pnl, 2)})
        return {"positions": lines, "unrealized": round(unrealized, 2)}

    def status(self) -> str:
        profile = self.risk_profile()
        rules = profile["rules"]
        dry = "DRY-RUN" if self.dry_run_enabled() else "LIVE"
        symbols = ", ".join(f"{s}={self.asset(s)['price']:,.0f}" for s in sorted(MARKET_SNAPSHOT))
        open_alerts = sum(1 for a in self.alerts() if a.get("status") == "open")
        approved = [o for o in self.ledger() if o.get("result") == "approved"]
        return (f"trading: profile={profile['name']} ({rules['title']}) | cap=±${self._cap_usd():,.0f} "
                f"| mode={dry} | market {symbols} | open_alerts={open_alerts} "
                f"| orders={len(approved)}")