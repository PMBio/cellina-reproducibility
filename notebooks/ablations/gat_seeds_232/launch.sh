#!/bin/bash
# 9 runs (seeds 0,1,2 x Epithelial/Fibroblast/Myeloid), 4 workers (2 per GPU). Skips runs with an existing result.
cd "$(dirname "$0")"
PY=/data/ddimitrov/software/miniforge3/envs/cellina_edge/bin/python
> logs/queue.txt
for seed in 0 1 2; do for ct in Fibroblast Epithelial Myeloid; do echo "$seed $ct" >> logs/queue.txt; done; done
split -n r/4 -d logs/queue.txt logs/q_
w=0
for q in logs/q_00 logs/q_01 logs/q_02 logs/q_03; do
  gpu=$((w % 2))
  nohup bash -c "while read seed ct; do
      name=\${ct}_seed\${seed}
      [ -f results/\$name.json ] && { echo skip \$name; continue; }
      echo \"\$(date +%T) start \$name\"
      $PY run_seed.py --seed \$seed --ct \$ct --gpu $gpu > logs/\$name.log 2>&1 \
        && echo \"\$(date +%T) done \$name: \$(tail -1 logs/\$name.log)\" || echo \"\$(date +%T) FAIL \$name (see logs/\$name.log)\"
    done < $q" > logs/worker_$w.log 2>&1 &
  echo "worker $w -> GPU $gpu pid $! ($(wc -l < $q) jobs)"
  w=$((w+1))
done
