"""OS UI control via pyautogui - click, type, hotkeys, scroll, screenshots, OCR.

The tool is registered by the ToolRegistry under the name ``ui_control``
with permission category ``ui_control`` (gated: manual approval required),
because moving the mouse and typing into other applications is a system
modification like any other.

``pyautogui`` is imported **inside** :func:`run`, never at module import,
so kernel boots on machines without it keep working and the registry can
list the tool without pulling in a display stack.

Safety: pyautogui's FailSafe stays enabled - throwing the mouse into the
top-left corner aborts any running pyautogui call. The daemon and this
tool never disable it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

ACTIONS = ("position", "move", "click", "doubleclick", "rightclick",
           "type", "hotkey", "press", "scroll", "screenshot",
           "ocr", "click_text")


def _tesseract():
    """Import pytesseract lazily; returns (module, error_dict) - one is None."""
    try:
        import pytesseract
    except Exception as exc:  # noqa: BLE001 - package absent on lean installs
        return None, {"ok": False, "error":
                      f"pytesseract unavailable ({type(exc).__name__}: {exc}) - run "
                      "setup-windows-daemon.ps1 -Action install, Sir."}
    return pytesseract, None


def _keys(kwargs: dict[str, Any]) -> list[str]:
    raw = kwargs.get("keys", [])
    if isinstance(raw, str):
        raw = raw.split(",")
    return [str(k).strip().lower() for k in raw if str(k).strip()]


def run(action: str = "", **kwargs: Any) -> dict[str, Any]:
    """Drive the desktop with pyautogui.

    Actions: position, move(x, y), click(x, y, button, clicks),
    doubleclick(x, y), rightclick(x, y), type(text, interval),
    hotkey(keys=[...]), press(key), scroll(amount), screenshot(),
    ocr() - screen text via pytesseract, click_text(text) - find a phrase
    on screen and click its centre.
    Gated as permission category 'ui_control'; pyautogui FailSafe is kept on.
    """
    action = str(action or "").strip().lower()
    if action not in ACTIONS:
        return {"ok": False, "error":
                f"unknown action {action!r}; use one of: {', '.join(ACTIONS)}"}
    try:
        import pyautogui
    except Exception as exc:  # noqa: BLE001 - display-less sessions fail here
        return {"ok": False, "error":
                f"pyautogui unavailable ({type(exc).__name__}: {exc}) - run "
                "setup-windows-daemon.ps1 -Action install, Sir."}
    if action in ("ocr", "click_text"):
        pytesseract, t_err = _tesseract()
        if t_err is not None:
            return {**t_err, "action": action}
        image = pyautogui.screenshot()
        try:
            data = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT)
        except Exception as exc:  # noqa: BLE001 - engine missing, display headless
            return {"ok": False, "action": action, "error":
                    f"tesseract engine unavailable ({type(exc).__name__}: {exc}) - "
                    "install the Tesseract OCR engine, Sir."}
        texts = [str(t) for t in data.get("text", [])]
        if action == "ocr":
            words = [t.strip() for t in texts if t.strip()]
            return {"ok": True, "action": "ocr", "text": " ".join(words),
                    "words": len(words)}
        # action == "click_text"
        needle = str(kwargs.get("text", "")).strip().lower()
        if not needle:
            return {"ok": False, "action": action,
                    "error": "click_text needs text=<phrase on screen>"}
        match = next((i for i, t in enumerate(texts) if t.strip().lower() == needle),
                     None)
        if match is None:
            match = next((i for i, t in enumerate(texts)
                          if needle in t.strip().lower()), None)
        if match is None:
            return {"ok": False, "action": action,
                    "error": f"text {needle!r} not found on screen"}
        try:
            cx = int(data["left"][match]) + int(data["width"][match]) // 2
            cy = int(data["top"][match]) + int(data["height"][match]) // 2
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return {"ok": False, "action": action,
                    "error": f"ocr box data unusable: {exc}"}
        pyautogui.click(cx, cy)
        return {"ok": True, "action": action, "x": cx, "y": cy,
                "matched": texts[match].strip()}
    try:
        if action == "position":
            point = pyautogui.position()
            return {"ok": True, "action": action, "x": int(point.x), "y": int(point.y)}
        if action == "move":
            pyautogui.moveTo(int(kwargs.get("x", 0)), int(kwargs.get("y", 0)),
                             duration=min(max(float(kwargs.get("duration", 0.25)), 0.0), 5.0))
            return {"ok": True, "action": action,
                    "x": int(kwargs.get("x", 0)), "y": int(kwargs.get("y", 0))}
        if action in ("click", "doubleclick", "rightclick"):
            button = str(kwargs.get("button", "right" if action == "rightclick" else "left"))
            clicks = 2 if action == "doubleclick" else int(kwargs.get("clicks", 1))
            kwargs_click: dict[str, Any] = {"clicks": clicks, "button": button}
            if kwargs.get("x") is not None:
                kwargs_click["x"] = int(kwargs["x"])
            if kwargs.get("y") is not None:
                kwargs_click["y"] = int(kwargs["y"])
            pyautogui.click(**kwargs_click)
            return {"ok": True, "action": action, **kwargs_click}
        if action == "type":
            text = str(kwargs.get("text", ""))
            pyautogui.write(text, interval=min(max(float(kwargs.get("interval", 0.02)), 0.0), 1.0))
            return {"ok": True, "action": action, "chars": len(text)}
        if action == "hotkey":
            keys = _keys(kwargs)
            if not keys:
                return {"ok": False, "error": "hotkey needs keys=[...] (e.g. ctrl,shift,t)"}
            pyautogui.hotkey(*keys)
            return {"ok": True, "action": action, "keys": keys}
        if action == "press":
            key = str(kwargs.get("key", "enter")).strip().lower()
            pyautogui.press(key)
            return {"ok": True, "action": action, "key": key}
        if action == "scroll":
            amount = int(kwargs.get("amount", 3))
            pyautogui.scroll(amount)
            return {"ok": True, "action": action, "amount": amount}
        # action == "screenshot"
        target = Path("data") / f"ui_{int(time.time())}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        path = pyautogui.screenshot(str(target))
        return {"ok": True, "action": action, "path": str(target),
                "size": list(path.size) if path is not None else None}
    except Exception as exc:  # noqa: BLE001 - a tool must not kill the OS
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "action": action}
