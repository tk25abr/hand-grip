"""What the robot remembers about objects it has already touched.

Keyed by what an object LOOKS like (the same small colour thumbnail the vision cache uses), so a repeat
visit of the same-looking object starts from experience instead of from a guess. Stores only things the
robot itself sensed - never ground truth:

  held_n    squeeze that held it (no yield seen)         -> proven safe up to about this force
  crush_n   squeeze at which it gave way (fingers closed -> never go above ~0.7 x this again
            suddenly further, sensed from the encoders)
  weight_g, mu, stiffness, needs_n   measured by the wrist and pads -> lets a repeat visit decide
                                                        before touching
  visits    how many times it has been seen

Two objects count as the same when their thumbnails differ by less than THUMB_DIFF (mean |pixel
difference| on a 0-255 scale), the same rule the vision cache uses.
"""
import json
import os

import numpy as np

THUMB_DIFF = 6.0


class ObjectMemory:
    def __init__(self, path=None):
        self.path = path
        self.items = []          # each: dict(thumb=np.ndarray, **facts)
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    for e in json.load(f):
                        e["thumb"] = np.array(e["thumb"], dtype=np.int16)
                        self.items.append(e)
            except (OSError, ValueError, KeyError):
                self.items = []

    def _find(self, thumb):
        best, best_d = None, THUMB_DIFF
        for e in self.items:
            d = float(np.mean(np.abs(e["thumb"].astype(np.int16) - thumb.astype(np.int16))))
            if d < best_d:
                best, best_d = e, d
        return best

    def recall(self, thumb):
        """The remembered facts for an object that looks like this one, or None."""
        e = self._find(thumb)
        return None if e is None else {k: v for k, v in e.items() if k != "thumb"}

    def record(self, thumb, **facts):
        """Add or update facts. A crush limit only ever moves down; a held force only ever moves up."""
        e = self._find(thumb)
        if e is None:
            e = dict(thumb=thumb.astype(np.int16), visits=0)
            self.items.append(e)
        for k, v in facts.items():
            if v is None:
                continue
            if k == "crush_n" and e.get("crush_n") is not None:
                v = min(v, e["crush_n"])
            if k == "held_n" and e.get("held_n") is not None:
                v = max(v, e["held_n"])
            e[k] = float(v) if isinstance(v, (int, float, np.floating)) else v
        return e

    def seen(self, thumb):
        e = self._find(thumb)
        if e is not None:
            e["visits"] = e.get("visits", 0) + 1

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump([dict(e, thumb=e["thumb"].tolist()) for e in self.items], f)

    def __len__(self):
        return len(self.items)
