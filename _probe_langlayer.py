import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

from jarvis_os.config import Config
from jarvis_os.language import UniversalLanguageLayer
from jarvis_os.language_layer import LanguageLayer

cfg = Config()
lang = UniversalLanguageLayer(cfg, llm=None, audit=None)
layer = LanguageLayer(cfg, lang)


def show(label, r):
    d = dict(r)
    if d.get("message"):
        d["message"] = d["message"][:70]
    if d.get("error"):
        d["error"] = d["error"][:95]
    print(label, "->", json.dumps(d, ensure_ascii=False, sort_keys=True))


r1 = layer.respond("ready.")
r2 = layer.respond(r1)
print("respond:", repr(r1), "idempotent:", r1 == r2)

show("empty", layer.translate(""))
show("unsupported", layer.translate("hello", "fr"))
show("already", layer.translate("All systems are nominal.", "en"))
show("default-target-ta", layer.translate("vanakkam"))
show("degraded-passthrough", layer.translate("hello"))
show("degraded-long", layer.translate("the quick brown fox jumps over the lazy dog"))

cfg.set_feature("language_layer", False, persist=False)
show("off", layer.translate("hello"))
cfg.set_feature("language_layer", True, persist=False)
show("back-on", layer.translate("hello"))

for t in ['translate "good morning"', "remember the milk", "hello there",
          "this is fine", "மொழிபெயர் இதை", "xyzzy nonsense",
          "Stop right now", "what is the time"]:
    r = layer.parse_intent(t)
    print("intent", repr(t), "->", r["intent"], r["confidence"],
          "q=", r["entities"]["quoted"], "msg0=", r["message"][:14])

print("status:", json.dumps(layer.status(), sort_keys=True))
print("all msgs honorific:",
      all(layer.parse_intent(t)["message"].startswith("Sir,")
          for t in ["hello", "stop", "zzz"]))
print("translate msg honorific:",
      layer.translate("All systems are nominal.", "en")["message"].startswith("Sir,"))
