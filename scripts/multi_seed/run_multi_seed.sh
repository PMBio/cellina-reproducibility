#!/usr/bin/env bash
# Run multi-seed LOO training for all SEEDS, then evaluation for all SEEDS.
# Models, datasets and holdouts are configured in train_parallel.py / eval_parallel.py;
# the seed is passed to them through the SEED environment variable.
#
# Usage (from any directory, inside the env that has cellina installed):
#   bash scripts/multi_seed/run_multi_seed.sh
#   PYTHON=/path/to/python CONCURRENCY=2 bash scripts/multi_seed/run_multi_seed.sh

SEEDS=(1 2)

PYTHON="${PYTHON:-python}"
CONCURRENCY="${CONCURRENCY:-3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# configs are also resolved via ./scripts, so run from the repo root
cd "${REPO_ROOT}" || exit 1

failed=()

echo "===== Training seeds: ${SEEDS[*]} ====="
for seed in "${SEEDS[@]}"; do
    echo "----- [train] seed ${seed} ($(date)) -----"
    if ! SEED="${seed}" "${PYTHON}" "${SCRIPT_DIR}/train_parallel.py" --concurrency "${CONCURRENCY}"; then
        failed+=("train seed ${seed}")
    fi
done

echo "===== Evaluating seeds: ${SEEDS[*]} ====="
for seed in "${SEEDS[@]}"; do
    echo "----- [eval] seed ${seed} ($(date)) -----"
    if ! SEED="${seed}" "${PYTHON}" "${SCRIPT_DIR}/eval_parallel.py" --concurrency "${CONCURRENCY}"; then
        failed+=("eval seed ${seed}")
    fi
done

if [ ${#failed[@]} -gt 0 ]; then
    echo "Finished with failures:"
    printf ' - %s\n' "${failed[@]}"
    exit 1
fi
echo "All seeds finished ($(date))"
