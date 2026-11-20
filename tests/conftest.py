"""Make the official FEVER scorer importable in tests without relying on PYTHONPATH.

The notebook clones sheffieldnlp/fever-scorer to /kaggle/working/fever-scorer; a kernel restart
clears PYTHONPATH, which used to make the official-scorer test skip. Set FEVER_SCORER_DIR to use
another clone location.
"""
import glob
import os
import sys

for root in filter(None, [os.environ.get("FEVER_SCORER_DIR"), "/kaggle/working/fever-scorer"]):
    hits = glob.glob(os.path.join(root, "**", "fever", "scorer.py"), recursive=True)
    if hits:
        sys.path.insert(0, os.path.dirname(os.path.dirname(hits[0])))
        break
