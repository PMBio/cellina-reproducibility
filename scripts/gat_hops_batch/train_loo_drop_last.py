"""Run scripts/train_loo.py unchanged, but with drop_last=True on the training loader.

Used for the one MERFISH fold (C57BL6J-2.041 / oligodendrocyte) whose training set
size is 1 mod 256, so the last minibatch has a single seed cell and BatchNorm raises
"Expected more than 1 value per channel". train_loo.py replaces datasplitter_kwargs
wholesale, so the flag cannot be injected from the config; instead CellinaGCN.train
is wrapped here to add it. Only the trailing size-1 batch is dropped; everything
else is identical to the arm's normal run.

    python scripts/gat_hops_batch/train_loo_drop_last.py <train_loo.py args...>
"""
import os
import runpy
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(REPO, "scripts")
TRAIN_LOO = os.path.join(SCRIPTS, "train_loo.py")

import cellina  # noqa: E402

_orig_train = cellina.CellinaGCN.train


def _train_drop_last(self, *args, **kwargs):
    ds = dict(kwargs.get("datasplitter_kwargs") or {})
    ds["drop_last"] = True
    kwargs["datasplitter_kwargs"] = ds
    print("[train_loo_drop_last] CellinaGCN.train patched: datasplitter_kwargs['drop_last']=True",
          file=sys.stderr, flush=True)
    return _orig_train(self, *args, **kwargs)


cellina.CellinaGCN.train = _train_drop_last

sys.path.insert(0, SCRIPTS)
sys.argv = [TRAIN_LOO] + sys.argv[1:]
runpy.run_path(TRAIN_LOO, run_name="__main__")
