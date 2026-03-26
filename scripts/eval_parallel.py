#!/usr/bin/env python3
"""
Sequential evaluator that runs `scripts/eval_loo.py` for a set of datasets, holdout cell types and model entries.

Behavior:
 - Uses module-level lists (populate these at the top of the file) for PATHS, HOLDOUTS and MODELS.
 - MODELS is a list of dicts with keys 'class' and 'name' (name may be None to fall back to template).
 - For each combination the script invokes `scripts/eval_loo.py --adata_path ... --holdout_celltype ... --model_class ... --model_name ...`
 - Runs sequentially, writes a small per-run log to `scripts/parallel_logs/<sid>/<holdout>/<model_class>.eval.log`.
 - Supports --dry-run and --extra-args for convenience.

Edit the lists below to point to the adata files and models you want to evaluate.
"""
import sys
import argparse
import subprocess
import shlex
import glob
from pathlib import Path
import os
import time

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = SCRIPT_DIR / "eval_loo.py"
LOG_ROOT = SCRIPT_DIR / "parallel_logs"
PY = sys.executable

# Populate these lists manually
PATHS = [
    # Example: #"/data2/a330d/datasets/crc/raw_zenodo/crc_222.h5ad",
    #"/data2/a330d/datasets/crc/raw_zenodo/crc_210.h5ad",
    #"/data2/a330d/datasets/crc/raw_zenodo/crc_221.h5ad",
    #"/data2/a330d/datasets/crc/raw_zenodo/crc_231.h5ad",
    #"/data2/a330d/datasets/crc/raw_zenodo/crc_232.h5ad",
    "/data2/a330d/datasets/crc/raw_zenodo/crc_242.h5ad",
    #"/data2/a330d/datasets/crc/raw_zenodo/crc_120.h5ad",
]

HOLDOUTS = [
    # Example: "Epithelial",
    "Endothelial",
    "Epithelial",
    "Fibroblast",
    "Myeloid",
    "T_cell",
    #"B_cell"
]
# TODO: Run eval for everything again because of correlation norm change
MODELS = [
    # Example entries: {"class": "cellina", "name": "cellina"}
    #{"class": "baseline", "name": "baseline", "extra_args": "--use_cf"},
    #{"class": "cellina", "name": "cellina"},
    {"class": "cellina", "name": "cellina", "extra_args": "--use_cf"},
    #{"class": "cellina", "name": "cellina", "extra_args": ["--use_recon","--use_cf"]},
    #{"class": "cellina_graph", "name": "cellina-graph", "extra_args": "--use_cf"},
    #{"class": "concert", "name": "concert"},
    #{"class": "cpa", "name": "cpa", "extra_args": "--use_cf"},
    #{"class": "cellina", "name": "cellina-ablated", "extra_args": "--use_cf"},
    #{"class": "cellina", "name": "cellina-mmd", "extra_args": "--use_cf"},
    #{"class": "scgen", "name": "scgen", "extra_args": "--use_cf"},
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--extra-args", default="",
                   help="Extra CLI args to append to each eval_loo invocation (quoted string)")
    p.add_argument("--concurrency", type=int, default=3,
                   help="Maximum number of concurrent evaluation processes (default 3)")
    p.add_argument("--dry-run", action='store_true', help="Print planned commands without executing them")
    return p.parse_args()


def expand_paths(path_patterns):
    paths = []
    for p in path_patterns:
        matched = glob.glob(p)
        if matched:
            paths.extend(sorted(matched))
        else:
            paths.append(p)
    return paths


def make_cmd(adata_path, holdout, model_class, model_name, extra_args, model_extra_args=None):
    cmd = [PY, str(EVAL_SCRIPT),
           "--adata_path", str(adata_path),
           "--holdout_celltype", str(holdout),
           "--model_class", str(model_class),
           "--model_name", str(model_name),
           ]
    # global extra args (string)
    if extra_args:
        cmd += shlex.split(extra_args)

    # per-model extra args: accept string or list
    if model_extra_args:
        if isinstance(model_extra_args, str):
            cmd += shlex.split(model_extra_args)
        elif isinstance(model_extra_args, (list, tuple)):
            cmd += [str(x) for x in model_extra_args]
        else:
            # if user accidentally passes a dict, try to convert to flags: {"use_recon": True} -> "--use_recon"
            if isinstance(model_extra_args, dict):
                for k, v in model_extra_args.items():
                    flag = f"--{k.replace('_','-')}"
                    if isinstance(v, bool):
                        if v:
                            cmd.append(flag)
                    else:
                        cmd += [flag, str(v)]
    return cmd


def run_eval(cmd, log_path):
    """Legacy synchronous runner (kept for compatibility)."""
    # keep for backward compatibility but not used in parallel mode
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'ab') as fh:
        fh.write((f"# START CMD: {' '.join(shlex.quote(c) for c in cmd)}\n").encode())
        fh.flush()
        res = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        fh.write((f"# FINISHED exit_code={res.returncode}\n").encode())
    return res.returncode


def start_process(cmd, log_path):
    """Start a process and stream its stdout/stderr into the given log file. Returns (proc, fh, cmd, log_path)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, 'ab')
    fh.write((f"# START CMD: {' '.join(shlex.quote(c) for c in cmd)}\n").encode())
    fh.flush()
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return proc, fh, cmd, log_path


def run_batch(cmds, concurrency):
    """Run a list of (cmd, log_path) tuples with at most `concurrency` parallel processes.
    Returns a list of failure dicts for non-zero exit codes.
    """
    pending = list(cmds)
    procs = []  # list of (proc, fh, cmd, log_path)
    failures = []
    try:
        while pending or procs:
            # start new procs if under concurrency
            while pending and len(procs) < concurrency:
                cmd, log_path = pending.pop(0)
                proc_tuple = start_process(cmd, log_path)
                procs.append(proc_tuple)
                time.sleep(0.05)

            # poll processes and handle finished
            still_running = []
            for proc, fh, cmd, log_path in procs:
                ret = proc.poll()
                if ret is None:
                    still_running.append((proc, fh, cmd, log_path))
                else:
                    fh.write((f"# FINISHED exit_code={ret}\n").encode())
                    fh.close()
                    if ret != 0:
                        failures.append({'cmd': ' '.join(shlex.quote(c) for c in cmd), 'log': str(log_path), 'returncode': int(ret)})
                    print(f"Finished: {' '.join(shlex.quote(c) for c in cmd)} -> exit {ret}; log: {log_path}")
            procs = still_running

            if procs:
                time.sleep(0.5)
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

    return failures


def main():
    args = parse_args()

    if not PATHS:
        print("PATHS list is empty; populate PATHS at top of script")
        sys.exit(1)
    if not HOLDOUTS:
        print("HOLDOUTS list is empty; populate HOLDOUTS at top of script")
        sys.exit(1)
    if not MODELS:
        print("MODELS list is empty; populate MODELS at top of script")
        sys.exit(1)

    paths = expand_paths(PATHS)
    if not paths:
        print("No input paths found after glob expansion")
        sys.exit(1)

    # build all commands
    all_cmds = []  # list of (cmd, log_path)
    for p in paths:
        sid = Path(p).stem
        for holdout in HOLDOUTS:
            for model_entry in MODELS:
                model_class = model_entry.get('class')
                model_name = model_entry.get('name') or f"{model_class}_{sid}_{holdout}"
                model_extra = model_entry.get('extra_args', None)

                cmd = make_cmd(p, holdout, model_class, model_name, args.extra_args, model_extra)
                log_path = LOG_ROOT / sid / holdout / f"{model_class}.eval.log"
                all_cmds.append((cmd, log_path))

    # show planned commands
    for cmd, log_path in all_cmds:
        print('  ', ' '.join(shlex.quote(c) for c in cmd), '->', log_path)

    if args.dry_run:
        print("Dry-run: not launching processes.")
        return

    # run in parallel batches
    concurrency = max(1, int(args.concurrency))
    print(f"Starting {len(all_cmds)} jobs with concurrency={concurrency}")
    failures = run_batch(all_cmds, concurrency)

    if failures:
        print('\nSome runs failed:')
        for f in failures:
            print(f" - cmd={f['cmd']} log={f['log']} rc={f['returncode']}")
        sys.exit(2)

    print('\nAll evaluations completed successfully')


if __name__ == '__main__':
    main()
