#!/bin/bash
# One-off: drop crc_232 artefacts no arm depends on any more (~21 GB).
# KEEPS: lora_run checkpoints, lora_ep1 + lora_ep5 bundles (the scored arms),
#        epoch_selection.json / ft_config.json, emb_frozen.npz, every correlations/*.json,
#        and the terra_tok / terra_tok_terra2k caches.
set -eu
C=${DATA_ROOT:-/data/ddimitrov/data}/datasets/crc
S=$C/crc_232
[ -d "$S/terra/lora_ep5/lora_bundle" ] || { echo "refusing: $S/terra/lora_ep5/lora_bundle missing"; exit 1; }
[ -d "$S/terra/lora_ep1/lora_bundle" ] || { echo "refusing: $S/terra/lora_ep1/lora_bundle missing"; exit 1; }

before=$(df --output=avail -BG /data | tail -1)
rm -rf $S/*/terra-swap-*                      # decoder-swap 2x2 diagnostic
rm -rf $S/*/terra-lora_* $S/*/terra-lora-null*  # pre-epoch arm naming, superseded by lora-ep{1,5}
rm -rf $S/*/*hoFibroblast* $S/*/*112M*        # retired -ho tag and the 112M model
rm -rf $S/*/lora_1ep $S/terra/lora_1ep $S/terra/collapsed_lr1e-3  # lr 1e-3 collapse + 1-epoch trials
rm -f  $S/terra/emb_lora.npz $S/terra/lora_bundle                 # stale slide-level names
rm -rf $S/terra/lora_ep2 $S/terra/lora_ep3 $S/terra/lora_ep4      # unscored epochs, re-repackable
rm -rf $C/crc_232_max3000 $C/crc_232_terra2k                      # smoke-test work dirs
echo "avail before=$before after=$(df --output=avail -BG /data | tail -1)"
