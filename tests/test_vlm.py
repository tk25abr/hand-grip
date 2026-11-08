"""Vision-model plumbing: defensive JSON parsing, range validation, the source chain and the cache.
Model answers below are test inputs for the parser only; the demo path never uses canned answers."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import vlm  # noqa: E402

M = dict(found=True, height_m=0.044, width_m=0.044, length_m=0.058, shape="round, domed (sphere/ellipsoid)",
         cut_off=False, bbox=(280, 380, 60, 70), n_px=2000)
RGB = (np.random.default_rng(0).integers(0, 255, (480, 640, 3))).astype(np.uint8)


def test_extract_json_from_messy_text():
    t = 'Sure! Here it is:\n```json\n{"object": "egg {white}", "mass_g": [50, 70],}\n```\nHope that helps {ok}'
    assert vlm.extract_json(t) == {"object": "egg {white}", "mass_g": [50, 70]}
    with pytest.raises(ValueError):
        vlm.extract_json("I cannot see an object.")


def test_validate_clamps_and_fills():
    prior = vlm.depth_prior(M)
    raw = {"object": "hen egg", "material": "shell", "mass_g": 58, "fragile": "true",
           "max_safe_squeeze_n": [40, -5], "surface": "Dry, smooth", "friction_guess": "0.4-0.7",
           "graspable_by_parallel_jaw_14cm": "yes"}                 # notes missing
    g = vlm.validate(raw, prior)
    assert g["mass_g"] == [46.4, 69.6]                              # scalar -> +-20 %
    assert g["fragile"] is True and g["graspable_by_parallel_jaw_14cm"] is True
    assert g["max_safe_squeeze_n"] == [0.5, 40.0]                   # sorted + clamped
    assert g["surface"] == "dry" and g["friction_guess"] == [0.4, 0.7]
    assert g["invalid_fields"] == ["notes"] and "filled from depth prior" in g["notes"]


def test_depth_only_without_any_model(monkeypatch, tmp_path):
    for k in ("GEMINI_API_KEY", "GEMMA_URL", "OLLAMA_HOST"):
        monkeypatch.delenv(k, raising=False)
    g = vlm.first_guess(RGB, M, cache=vlm.Cache(str(tmp_path / "c.json")))
    assert g["source"] == "depth-only" and "no GEMINI_API_KEY" in g["fallback"]
    assert g["graspable_by_parallel_jaw_14cm"] is True


def test_too_wide_is_not_graspable_by_depth():
    wide = dict(M, width_m=0.18, shape="box (flat faces)")
    assert vlm.depth_prior(wide)["graspable_by_parallel_jaw_14cm"] is False


class _Ollama(BaseHTTPRequestHandler):
    """Local Ollama-protocol server for testing the HTTP wiring of the fallback (not the demo path)."""
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Ollama.seen.append(body)
        out = {"object": "egg", "material": "shell", "mass_g": [50, 66], "fragile": True,
               "max_safe_squeeze_n": [10, 40], "surface": "dry", "friction_guess": [0.5, 0.8],
               "graspable_by_parallel_jaw_14cm": True, "notes": "test"}
        data = json.dumps({"message": {"role": "assistant", "content": json.dumps(out)}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_gemini_failure_falls_back_to_gemma_and_caches(monkeypatch, tmp_path):
    _Ollama.seen = []
    srv = HTTPServer(("127.0.0.1", 0), _Ollama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET-KEY-123")
    monkeypatch.setattr(vlm, "ask_gemini", lambda *a: (_ for _ in ()).throw(RuntimeError("quota")))
    monkeypatch.setenv("GEMMA_URL", f"127.0.0.1:{srv.server_port}")
    cache = vlm.Cache(str(tmp_path / "c.json"))
    g = vlm.first_guess(RGB, M, cache=cache)
    assert g["source"] == "local gemma" and not g["cached"] and "quota" in g["fallback"]
    req = _Ollama.seen[-1]
    assert req["messages"][0]["images"] and "@H" not in req["messages"][0]["content"]
    assert "height 44 mm" in req["messages"][0]["content"]
    g2 = vlm.first_guess(RGB, M, cache=cache)                       # same picture -> no second call
    assert g2["cached"] and len(_Ollama.seen) == 1
    srv.shutdown()
    assert "SECRET-KEY-123" not in open(tmp_path / "c.json").read()            # keys never written to disk
