# Disentanglement Benchmark

Benchmarks the latent disentanglement of Cellina against SCANVI and scVIVA on CRC CosMx data, using scib metrics evaluated at both the cell-type and niche level.

## Files

| File | Description |
|------|-------------|
| `get_cosmx_crc.ipynb` | Download and preprocess CosMx CRC data for this benchmark |
| `run_crc.ipynb` | Train SCANVI and scVIVA on CRC holdout splits; save latent representations |
| `eval_crc.ipynb` | Compute scib disentanglement scores; write `results/*_scib_*.csv` |
