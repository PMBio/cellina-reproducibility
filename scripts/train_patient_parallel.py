#!/usr/bin/env python3
"""
Run `scripts/train_patient_loo.py` for multiple holdout patients/slides in parallel
with a global concurrency limit.

Behavior:
 - Unlike train_parallel.py (which loops over multiple per-slide adata files x
   holdout cell types), this launcher uses a single adata file spanning multiple
   patients/slides (--adata-path / ADATA_PATH below) and loops over HOLDOUT_SIDS
   x HOLDOUT_CELLTYPES x MODELS. Each (holdout_sid, holdout_celltype) pair gets
   its own train/test split (train_patient_loo.py's split_indices holds out that
   celltype in the holdout domain across ALL patients, in addition to holdout_sid
   entirely), so it requires its own trained model.
 - Each train_patient_loo.py invocation internally loops over all cell types
   (--holdout_celltypes, defaults to the same list) for evaluation once its
   model is trained/loaded, so no separate per-celltype process is spawned for
   evaluation — only for training/splitting.
 - Up to `--concurrency` processes run at a time.

Logs are written to `scripts/parallel_logs/patient_loo/<holdout_sid>/<holdout_celltype>/<model_class>.log`.
Per-job result CSVs land in `results/loo_patient/` and can be merged afterward, e.g.:
    import glob, pandas as pd
    df = pd.concat([pd.read_csv(f) for f in glob.glob("results/loo_patient/*.csv")])

Example:
 python scripts/train_patient_parallel.py --concurrency 3
"""
import os
import sys
import argparse
import subprocess
import shlex
import time
from pathlib import Path

DATA_ROOT = '/data2/a330d' #os.environ.get("DATA_ROOT", ".")

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_patient_loo.py"
LOG_ROOT = SCRIPT_DIR / "parallel_logs" / "patient_loo"
PY = sys.executable

# Define these here (populate manually before running).
DATASET_NAME = "crc"  # or "merfish"

ADATA_PATH = os.path.join(DATA_ROOT, "datasets/crc/processed/crc_patient_loo.h5ad")

# batch_key values (patients/slides) to hold out entirely, one job per entry
HOLDOUT_SIDS = [231, 232, 242]

CRC_HOLDOUT_CELLTYPES = [
    "Endothelial",
    "Epithelial",
    "Fibroblast",
    #"Myeloid",
    #"T_cell",
]

MERFISH_HOLDOUT_CELLTYPES = [
    'glutamatergic neuron',
    'oligodendrocyte',
    'astrocyte',
    'GABAergic neuron',
    'endothelial cell',
]

# cell type held out (in the holdout domain, across ALL patients) for the
# train/test split of each job, in addition to holdout_sid
HOLDOUT_CELLTYPES = CRC_HOLDOUT_CELLTYPES if DATASET_NAME == "crc" else MERFISH_HOLDOUT_CELLTYPES

# batch_key values to drop from adata before splitting (e.g. QC-failed slides)
EXCLUDE_SIDS = "" if DATASET_NAME == "crc" else ""

MODELS = [
    # Use a list of dicts so each model_class can have an associated model_name.
    # Populate these entries directly. Example:
    #{"class": "cellina", "name": "cellina"},
    #{"class": "cellina", "name": "cellina", "extra_args": "--inference_only"},
    #{"class": "cpa", "name": "cpa", "extra_args": "--inference_only"},
    # baseline has no train/load step (average domain shift computed at eval time), so
    # it doesn't need --inference_only and runs with the default python interpreter.
    {"class": "baseline", "name": "baseline"},
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=3,
                   help="Maximum number of concurrent training processes (default 3)")
    p.add_argument("--model-name-template", default="{model_class}_{holdout_sid}_{holdout_celltype}",
                   help="Template for --model_name passed to train_patient_loo.py. Available keys: model_class, holdout_sid, holdout_celltype")
    p.add_argument("--extra-args", default="",
                   help="Extra CLI args to append to each train_patient_loo invocation (quoted string)")
    p.add_argument("--dry-run", action='store_true', help="Print planned commands without executing them")
    return p.parse_args()


def make_cmd(adata_path, dataset_name, holdout_sid, holdout_celltype, model_class, model_name, exclude_sids, extra_args, model_extra_args=None):
    cmd = [PY, str(TRAIN_SCRIPT),
           "--dataset_name", str(dataset_name),
           "--adata_path", str(adata_path),
           "--holdout_sid", str(holdout_sid),
           "--holdout_celltype", str(holdout_celltype),
           "--model_class", str(model_class),
           "--model_name", str(model_name),
           ]
    if exclude_sids:
        cmd += ["--exclude_sids", str(exclude_sids)]
    if extra_args:
        cmd += shlex.split(extra_args)

    if model_extra_args:
        if isinstance(model_extra_args, str):
            cmd += shlex.split(model_extra_args)
        elif isinstance(model_extra_args, (list, tuple)):
            cmd += [str(x) for x in model_extra_args]
        elif isinstance(model_extra_args, dict):
            for k, v in model_extra_args.items():
                flag = f"--{k.replace('_', '-')}"
                if isinstance(v, bool):
                    if v:
                        cmd.append(flag)
                else:
                    cmd += [flag, str(v)]

    return cmd


def start_process(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, 'ab')
    fh.write((f"# START CMD: {' '.join(shlex.quote(c) for c in cmd)}\n").encode())
    fh.flush()
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return proc, fh


def run_batch(cmds, concurrency):
    """Run a list of (cmd, log_path) tuples, ensuring at most `concurrency` processes run concurrently.
    Waits until all in cmds complete before returning.
    """
    procs = []
    pending = list(cmds)
    try:
        while pending or procs:
            while pending and len(procs) < concurrency:
                cmd, log_path = pending.pop(0)
                proc, fh = start_process(cmd, log_path)
                procs.append((proc, fh, cmd, log_path))
                time.sleep(0.1)

            still_running = []
            for proc, fh, cmd, log_path in procs:
                ret = proc.poll()
                if ret is None:
                    still_running.append((proc, fh, cmd, log_path))
                else:
                    fh.write((f"# FINISHED exit_code={ret}\n").encode())
                    fh.close()
                    print(f"Finished: {' '.join(shlex.quote(c) for c in cmd)} -> exit {ret}; log: {log_path}")
            procs = still_running

            if procs:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received: terminating running processes...")
        for proc, fh, cmd, log_path in procs:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                fh.close()
            except Exception:
                pass
        raise


def main():
    args = parse_args()

    if not ADATA_PATH:
        print("ADATA_PATH is empty. Please populate ADATA_PATH at the top of this script before running.")
        sys.exit(1)
    if not HOLDOUT_SIDS:
        print("HOLDOUT_SIDS is empty. Please populate the HOLDOUT_SIDS list at the top of this script before running.")
        sys.exit(1)
    if not HOLDOUT_CELLTYPES:
        print("HOLDOUT_CELLTYPES is empty. Please populate the HOLDOUT_CELLTYPES list at the top of this script before running.")
        sys.exit(1)
    if not MODELS:
        print("MODELS is empty. Please populate the MODELS list at the top of this script before running.")
        sys.exit(1)

    holdout_sids = list(HOLDOUT_SIDS)
    holdout_celltypes = list(HOLDOUT_CELLTYPES)
    models = list(MODELS)
    concurrency = max(1, int(args.concurrency))

    print(f"Will run holdout_sids={holdout_sids} holdout_celltypes={holdout_celltypes} "
          f"models={[m.get('class', m) for m in models]} with concurrency={concurrency}")

    all_cmds = []
    for holdout_sid in holdout_sids:
        for holdout_celltype in holdout_celltypes:
            for model_entry in models:
                if isinstance(model_entry, dict):
                    model_class = model_entry.get('class')
                    model_name = model_entry.get('name') or args.model_name_template.format(model_class=model_class, holdout_sid=holdout_sid, holdout_celltype=holdout_celltype)
                    model_extra = model_entry.get('extra_args', None)
                else:
                    model_class = str(model_entry)
                    model_name = args.model_name_template.format(model_class=model_class, holdout_sid=holdout_sid, holdout_celltype=holdout_celltype)
                    model_extra = None

                cmd = make_cmd(ADATA_PATH, DATASET_NAME, holdout_sid, holdout_celltype, model_class, model_name, EXCLUDE_SIDS, args.extra_args, model_extra)
                if model_class == 'cpa':
                    cpa_python = "/data/a330d/miniforge3/envs/cpa_cuda/bin/python" #os.environ.get("CPA_PYTHON", sys.executable)
                    if len(cmd) > 0:
                        cmd[0] = cpa_python

                ct_slug = holdout_celltype.replace(' ', '_')
                log_path = LOG_ROOT / str(holdout_sid) / ct_slug / f"{model_class}.log"
                all_cmds.append((cmd, log_path))

    for cmd, log_path in all_cmds:
        print('  ', ' '.join(shlex.quote(c) for c in cmd), '->', log_path)

    if args.dry_run:
        print("DRY-RUN: not launching any processes")
        return

    print(f"Starting {len(all_cmds)} jobs with concurrency={concurrency}")
    run_batch(all_cmds, concurrency)
    print("Completed all jobs")


if __name__ == '__main__':
    main()
