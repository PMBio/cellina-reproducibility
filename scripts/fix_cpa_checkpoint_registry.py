"""
One-time repair for a bug in the installed `cpa` package (cpa/_model.py::setup_anndata):

    keys = categorical_covariate_keys
    if batch_key is not None:
        keys.append(batch_key)

This mutates the caller's `categorical_covariate_keys` list in place instead of
copying it. Since scvi's `_get_setup_method_args` snapshots locals by reference,
the corrupted (batch_key-appended) list gets serialized into the saved model's
`registry_['setup_args']['categorical_covariate_keys']`.

On `CPA.load()`, scvi replays `setup_anndata(adata, **saved_setup_args)`, which
prepends `batch_key` again (it's already baked into categorical_covariate_keys),
so the composite `CPA_cat` category string ends up with batch_key duplicated
(e.g. "231_Endothelial_231_CRC") and no longer matches the registry recorded at
training time (e.g. "231_Endothelial_CRC") -> ValueError on load.

This script strips `batch_key` back out of the saved `categorical_covariate_keys`
for each `cpa/model.pt` checkpoint under a given root, so `CPA.load()` reproduces
the original (correct) setup. The actual field registries (e.g. CPA_cat's
categorical_mapping) are untouched -- they were already correct at save time,
since the mutation happens after they're computed.

Usage:
  python scripts/fix_cpa_checkpoint_registry.py --root /data2/a330d/data/ood/trained/loo_patients [--dry_run] [--no_backup]

Run with the cpa_cuda conda env (needs torch):
  /data/a330d/miniforge3/envs/cpa_cuda/bin/python scripts/fix_cpa_checkpoint_registry.py --root ...
"""
import argparse
import shutil
from pathlib import Path

import torch


def fix_checkpoint(model_pt_path: Path, dry_run: bool, backup: bool):
    d = torch.load(model_pt_path, map_location="cpu", weights_only=False)
    setup_args = d["attr_dict"]["registry_"]["setup_args"]
    batch_key = setup_args.get("batch_key")
    cck = setup_args.get("categorical_covariate_keys")

    if not batch_key or not cck or batch_key not in cck:
        print(f"  SKIP (nothing to fix): {model_pt_path}")
        return False

    fixed = [k for k in cck if k != batch_key]
    print(f"  {model_pt_path}: categorical_covariate_keys {cck} -> {fixed}")

    if dry_run:
        return True

    if backup:
        backup_path = model_pt_path.with_suffix(model_pt_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(model_pt_path, backup_path)

    setup_args["categorical_covariate_keys"] = fixed
    torch.save(d, model_pt_path)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Directory to search for cpa/model.pt checkpoints")
    p.add_argument("--dry_run", action="store_true", help="Only print what would change")
    p.add_argument("--no_backup", action="store_true", help="Skip writing .bak copies before overwriting")
    args = p.parse_args()

    root = Path(args.root)
    model_paths = sorted(root.glob("**/cpa/model.pt"))
    print(f"Found {len(model_paths)} cpa model.pt checkpoints under {root}")

    n_fixed = 0
    for path in model_paths:
        if fix_checkpoint(path, dry_run=args.dry_run, backup=not args.no_backup):
            n_fixed += 1

    print(f"{'Would fix' if args.dry_run else 'Fixed'} {n_fixed}/{len(model_paths)} checkpoints")


if __name__ == "__main__":
    main()
