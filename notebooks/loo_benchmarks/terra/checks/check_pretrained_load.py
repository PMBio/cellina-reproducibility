"""Pre-flight check for FINETUNE_PLAN.md section 4.

Builds the TERRA encoder the way each entry point builds it, and loads the real
pretrained checkpoint into each. The PATCHED arm is the one the plan requires to pass.
Usage: python check_pretrained_load.py [112M|96M]   (default 112M)
"""
import sys, glob, pickle, yaml, torch, logging, warnings
warnings.filterwarnings("ignore"); logging.disable(logging.INFO)
from terra.utils.helper import init_model, parse_arch_kwargs, parse_protein_init_kwargs

MODEL = sys.argv[1] if len(sys.argv) > 1 else "112M"
d = glob.glob(f"/home/d573v/.cache/huggingface/hub/models--lotfollahi-lab--TERRA-{MODEL}/snapshots/*/")[0]
print(f"=== TERRA-{MODEL} ===")
cfg  = yaml.safe_load(open(d + "model_config.yaml"))
tokd = pickle.load(open(d + "token_dictionary.pkl", "rb"))
ck   = torch.load(d + "model_checkpoint.pt", map_location="cpu")["target_encoder"]

n_spv_counted = sum(1 for k in tokd if "spv" in k)
print(f"n_special_values: counted-from-token_dict={n_spv_counted}  config={cfg['data'].get('n_special_values')}")
print(f"nz_spc in config: {cfg['data'].get('nz_spc')}   (finetune paths never pass it -> init_model default False)\n")

n_special_tokens = len(cfg['meta']['special_tokens'])
seq_len = cfg['data']['seq_len_cell'] + cfg['data']['seq_len_neighborhood'] + n_special_tokens
common = dict(
    gt_type=cfg['meta']['gt_type'], count_encoding=cfg['meta']['count_encoding'],
    n_value_bins=cfg['meta']['n_value_bins'], cell_pos_enc=cfg['meta']['cell_pos_enc'],
    device="cpu", vocab_size=len(tokd), seq_len=seq_len,
    n_special_tokens=n_special_tokens, n_segments=cfg['data']['n_segments'],
    enc_emb_dim=cfg['meta']['enc_emb_dim'], enc_depth=cfg['meta']['enc_depth'],
    pred_emb_dim=cfg['meta']['pred_emb_dim'], pred_depth=cfg['meta']['pred_depth'],
    num_heads=cfg['meta']['num_heads'], mlp_ratio=cfg['meta']['mlp_ratio'],
    use_flash_attention=cfg['meta']['use_flash_attention'],
    api_version=cfg['meta']['api_version'],
    sep_gene_tokens_neb=cfg['data']['sep_gene_tokens_neb'],
    predict_gene=cfg['meta']['predict_gene'], pos_learnable=cfg['meta']['pos_learnable'],
)

def check(name, extra):
    enc, _ = init_model(**common, **extra)
    try:
        enc.load_state_dict(ck); print(f"  {name:38s} -> ALL KEYS MATCHED")
    except RuntimeError as e:
        bad = [l.strip() for l in str(e).split("\n") if "size mismatch" in l or "Unexpected key" in l or "Missing key" in l]
        print(f"  {name:38s} -> FAILS ({len(bad)} mismatches)")
        for l in bad[:3]: print(f"  {'':38s}    {l[:105]}")

check("embed.py (inference, known-good)", dict(
    n_special_values=cfg['data'].get('n_special_values', 0),
    nz_spc=cfg['data'].get('nz_spc', False), mlp_bias=cfg['meta'].get('mlp_bias', True),
    protein_init_kwargs=parse_protein_init_kwargs(cfg, tokd), **parse_arch_kwargs(cfg)))

check("finetune_self_supervised (xenium path)", dict(
    n_special_values=n_spv_counted,
    mlp_bias=cfg['meta'].get('mlp_bias', True)))     # line 284: mlp_bias only, no nz_spc

check("finetune.py (supervised path)", dict(
    n_special_values=n_spv_counted, protein_init_kwargs=None))

# --- the arm FINETUNE_PLAN.md section 4 requires to pass -----------------------
# finetune.py's own call, with the five fields embed.py passes injected back in.
check("finetune.py + PATCH (plan section 4)", dict(
    n_special_values=cfg['data'].get('n_special_values', 0),
    nz_spc=cfg['data'].get('nz_spc', False),
    mlp_bias=cfg['meta'].get('mlp_bias', True),
    protein_init_kwargs=parse_protein_init_kwargs(cfg, tokd),
    **parse_arch_kwargs(cfg)))
