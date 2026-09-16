#!/bin/bash
# Frozen-encoder arm only (no fine-tune), all remaining CRC slides, two GPUs.
# crc_232 is already scored (frozen + lora-ep5 + lora-ep1) and is not re-run.
# Slides are ordered small->large per GPU and each one's tokenizer cache is dropped when it is
# done: /data cannot hold five caches at once.  Caches rebuild from the raw h5ad on demand.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
L=scripts/terra/queue/logs; mkdir -p $L
export FROZEN_ONLY=1 PRUNE_TOK=1
nohup scripts/terra/queue/worker.sh 0 crc_231 crc_242 crc_120 > $L/worker0.log 2>&1 &
echo "GPU 0 -> crc_231 crc_242 crc_120  (pid $!)"
nohup scripts/terra/queue/worker.sh 1 crc_221 crc_210 > $L/worker1.log 2>&1 &
echo "GPU 1 -> crc_221 crc_210  (pid $!)"
df -h /data | tail -1
