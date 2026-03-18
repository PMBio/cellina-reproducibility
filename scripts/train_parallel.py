#!/usr/bin/env python3
"""
Run `scripts/train_loo.py` for multiple datasets / holdout cell types in parallel with a global concurrency limit.

Behavior:
 - For each adata path (sid) and each holdout cell type, launches one `train_loo.py` process per requested model class.
 - For a given (sid, holdout) group the requested model-class processes are started and run concurrently up to `--concurrency` processes at a time.
 - The launcher waits for all processes for that holdout to complete before moving on to the next holdout cell type.

Logs are written to `scripts/parallel_logs/<sid>/<holdout>/<model_class>.log`.

Example:
 python scripts/train_parallel.py --concurrency 3

"""
import sys
import argparse
import subprocess
import shlex
import time
import glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_loo.py"
LOG_ROOT = SCRIPT_DIR / "parallel_logs"
PY = sys.executable

# Define lists here (populate manually)
# The user will edit these lists directly in the script before running.
PATHS = [
    # Example: "/data2/a330d/datasets/cosmx/*.h5ad",
    #"/data2/a330d/datasets/crc/raw_zenodo/crc_110.h5ad",
    #"/data2/a330d/datasets/crc/raw_zenodo/crc_210.h5ad",
    #"/data2/a330d/datasets/crc/raw_zenodo/crc_221.h5ad",
    #"/data2/a330d/datasets/crc/raw_zenodo/crc_222.h5ad",
    #"/data2/a330d/datasets/crc/raw_zenodo/crc_231.h5ad",
    "/data2/a330d/datasets/crc/raw_zenodo/crc_232.h5ad",
    #"/data2/a330d/datasets/crc/raw_zenodo/crc_242.h5ad",
    #"/data2/a330d/datasets/crc/raw_zenodo/crc_120.h5ad",
]
HOLDOUTS = [
    # Example: "Epithelial",
    "Fibroblast",
    #"Endothelial",
    #"Myeloid",
    #"T_cell",
    #"Epithelial",
    #"B_cell"
]
MODELS = [
    # Use a list of dicts so each model_class can have an associated model_name.
    # Populate these entries directly. Example:
    #{"class": "cellina", "name": "cellina-ablated"},
    #{"class": "cellina", "name": "cellina"},
    #{"class": "cpa", "name": "cpa", "extra_args": "--inference_only"},
    #{"class": "cellina_graph", "name": "cellina-graph"},
    #{"class": "concert", "name": "concert"},
    {"class": "scgen", "name": "scgen"},
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=3,
                   help="Maximum number of concurrent training processes (default 3)")
    p.add_argument("--model-name-template", default="{model_class}_{sid}_{holdout}",
                   help="Template for --model_name passed to train_loo.py. Available keys: model_class, sid, holdout")
    p.add_argument("--extra-args", default="",
                   help="Extra CLI args to append to each train_loo invocation (quoted string)")
    p.add_argument("--dry-run", action='store_true', help="Print planned commands without executing them")
    return p.parse_args()


def expand_paths(path_patterns):
    paths = []
    for p in path_patterns:
        matched = glob.glob(p)
        if matched:
            paths.extend(sorted(matched))
        else:
            # accept literal path even if not matched
            paths.append(p)
    return paths


def make_cmd(adata_path, holdout, model_class, model_name, extra_args, model_extra_args=None):
    cmd = [PY, str(TRAIN_SCRIPT),
           "--adata_path", str(adata_path),
           "--holdout_celltype", str(holdout),
           "--model_class", str(model_class),
           "--model_name", str(model_name),
           ]
    if extra_args:
        # simply split extra args the shell way
        cmd += shlex.split(extra_args)

    # support per-model extra args (string, list/tuple or dict -> flags)
    if model_extra_args:
        if isinstance(model_extra_args, str):
            cmd += shlex.split(model_extra_args)
        elif isinstance(model_extra_args, (list, tuple)):
            cmd += [str(x) for x in model_extra_args]
        elif isinstance(model_extra_args, dict):
            for k, v in model_extra_args.items():
                flag = f"--{k.replace('_', '-') }"
                if isinstance(v, bool):
                    if v:
                        cmd.append(flag)
                else:
                    cmd += [flag, str(v)]

    return cmd


def start_process(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, 'ab')
    # write header
    fh.write((f"# START CMD: {' '.join(shlex.quote(c) for c in cmd)}\n").encode())
    fh.flush()
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return proc, fh


def run_batch(cmds, concurrency):
    """Run a list of (cmd, log_path) tuples, ensuring at most `concurrency` processes run concurrently.
    Wait until all in cmds complete before returning.
    """
    procs = []  # list of (proc, fh)
    pending = list(cmds)
    try:
        while pending or procs:
            # start new procs if under concurrency
            while pending and len(procs) < concurrency:
                cmd, log_path = pending.pop(0)
                proc, fh = start_process(cmd, log_path)
                procs.append((proc, fh, cmd, log_path))
                time.sleep(0.1)

            # poll processes and remove finished
            still_running = []
            for proc, fh, cmd, log_path in procs:
                ret = proc.poll()
                if ret is None:
                    still_running.append((proc, fh, cmd, log_path))
                else:
                    # process finished
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
    # Use the lists defined in the script (the user should populate them)
    if not PATHS:
        print("PATHS is empty. Please populate the PATHS list at the top of this script before running.")
        sys.exit(1)
    if not HOLDOUTS:
        print("HOLDOUTS is empty. Please populate the HOLDOUTS list at the top of this script before running.")
        sys.exit(1)
    if not MODELS:
        print("MODELS is empty. Please populate the MODELS list at the top of this script before running.")
        sys.exit(1)

    paths = expand_paths(PATHS)
    holdouts = list(HOLDOUTS)
    models = list(MODELS)
    concurrency = max(1, int(args.concurrency))

    if not paths:
        print("No input paths found after glob expansion.")
        sys.exit(1)

    print(f"Found {len(paths)} adata paths; will run holdouts={holdouts} models={[m.get('class', m) for m in models]} with concurrency={concurrency}")

    # Build all commands across datasets / holdouts / models and run them with a global concurrency limit.
    all_cmds = []
    for p in paths:
        sid = Path(p).stem
        for holdout in holdouts:
            for model_entry in models:
                if isinstance(model_entry, dict):
                    model_class = model_entry.get('class')
                    model_name = model_entry.get('name') or args.model_name_template.format(model_class=model_class, sid=sid, holdout=holdout)
                    model_extra = model_entry.get('extra_args', None)
                else:
                    model_class = str(model_entry)
                    model_name = args.model_name_template.format(model_class=model_class, sid=sid, holdout=holdout)
                    model_extra = None

                cmd = make_cmd(p, holdout, model_class, model_name, args.extra_args, model_extra)
                # If the model class is 'cpa', run the command with the CPA env python directly
                if model_class == 'cpa':
                    cpa_python = "/data/a330d/miniforge3/envs/cpa_cuda/bin/python"
                    if len(cmd) > 0:
                        cmd[0] = cpa_python
                if model_class == 'concert':
                    concert_python = "/data/a330d/miniforge3/envs/concert/bin/python"
                    if len(cmd) > 0:
                        cmd[0] = concert_python
                if model_class == 'scgen':
                    scgen_python = "/data/a330d/miniforge3/envs/cellina-base/bin/python"
                    if len(cmd) > 0:
                        cmd[0] = scgen_python

                log_path = LOG_ROOT / sid / holdout / f"{model_class}.log"
                all_cmds.append((cmd, log_path))

    # show planned commands
    for cmd, log_path in all_cmds:
        print('  ', ' '.join(shlex.quote(c) for c in cmd), '->', log_path)

    # dry-run: just print commands and exit
    if args.dry_run:
        print("DRY-RUN: not launching any processes")
        return

    print(f"Starting {len(all_cmds)} jobs with concurrency={concurrency}")
    run_batch(all_cmds, concurrency)
    print("Completed all jobs")

    print("All jobs finished")


if __name__ == '__main__':
    main()
