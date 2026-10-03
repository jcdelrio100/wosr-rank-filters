# Train with Ramps, Deploy with Ranks

**Tiny, robust rank-order filters that compete with neural networks.**

<!-- After the first Zenodo release, replace XXXXXXX with the concept DOI number -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Rank-order filters (median, erosion, dilation, weighted medians) need no multiplications, tolerate a known number of
corrupted samples and are 1-Lipschitz, but they are hard to train because selecting "the k-th largest value" has zero
gradient almost everywhere. **WOS-R** trains cascades of weighted order statistic (WOS) filters by spreading each sample
as a small *ramp* between its sorted neighbours during training, and deploys the *exact* rank filter.
This repository contains the preprint, the full benchmark code and every result file used in it.

This work continues the author's 1993 telecommunications engineering thesis (UPC, supervised by P. Salembier) on adaptive
cascades of order filters.

| Test | WOS-R (parameters) | Best network (parameters) | Verdict |
|---|---|---|---|
| 1D impulsive noise, NMSE (dB, lower is better) | −21.1 (137) | −22.3 (22 081) | 161× smaller, 1.2 dB worse |
| Same, impulses twice as large (unseen) | −17.8 | −17.4 | WOS-R better; Transformer fails |
| Images, 50 % salt & pepper, PSNR (dB) | 26.22 (52) | 26.74 (74 593) | 1 434× smaller, 0.5 dB worse |
| Images, random-valued impulses (unseen) | 24.46 (52) | 17.74 | WOS-R +6.7 dB |
| Radar CFAR, 8 interferers, SNR for Pd = 0.5 (dB, lower is better) | 17.09 (34) | 20.35 (6 337) | WOS-R +3.3 dB |
| Federated learning, worst-case accuracy | 0.753 (rank cascade) | 0.100 (learned aggregator) | learned works only after a rank cascade |

Where networks remain better: Gaussian noise at the training level, contamination above the filter's breakdown point,
and operators that need subtraction or squaring.

## Contents

```
paper/
  wosr_zenodo.pdf / .tex       accessible preprint (main document)
  wosr_technical.pdf / .tex    detailed technical version (propositions, full protocol)
  refs.bib, tikz/, figs/, tables/
  make_tables.py               regenerates every table from code/bench/*.json
  make_bench_figs.py           benchmark dashboards (accuracy, residual error, convergence, latency, parameters)
  make_qual.py, make_ramp_fig.py, make_props.py
code/
  wosr / ramp / rank-filter library modules (ramp.py, rankfilters.py, wos.py, robust.py, amo.py, cfar*.py, fl_bench.py, ...)
  experiments of the earlier technical notes (*.py) and their results (*.json)
  bench/
    wosnet.py                  WOS-R layers (chain, bank, ramp continuation) and all baselines
    b1_signal1d.py             B1  1D restoration
    b2_coverage.py             B2  operator coverage (system identification)
    b3_image.py, b3x_switch.py, b3fix_dncnn17.py, b3bank.py   B3  image impulse denoising (Set12, BSD68)
    b4_cfar.py                 B4  CFAR detection in K-distributed clutter
    b5_fl.py                   B5  Byzantine/fault-robust federated aggregation
    b6_convergence.py          B6  training curves and time to converge
    b7_structure.py            B7  learning depth, window and bank size (pruning)
    b8_restarts.py             B8  sensitivity to initialisation, best-of-N restarts
    latency.py, ablation_bank.py
    *_results.json             raw results used in the paper
  data/get_data.py             downloads Set12, BSD68, Train400 and Fashion-MNIST
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python code/data/get_data.py              # ~50 MB of public datasets

cd code/bench   # scripts are resumable: they skip runs already in *_results.json (delete those files to re-run from scratch)
python b1_signal1d.py                      # ~1 h on 2 CPU cores
python b2_coverage.py                      # ~1.5 h
python b3_image.py && python b3x_switch.py && python b3fix_dncnn17.py && python b3bank.py   # ~4 h
python b4_cfar.py                          # ~1.5 h
python b5_fl.py                            # ~30 min
python b6_convergence.py && python b7_structure.py && python b8_restarts.py && python latency.py

cd ../../paper
python make_tables.py b1 b2 b3 b4 b5 cost fit
python make_bench_figs.py 1d img acc struct
latexmk -pdf wosr_zenodo.tex wosr_technical.tex
```

All experiments in the paper ran on a 2-core CPU with PyTorch 2.x; no GPU is needed. The tables can be regenerated from
the included JSON files without re-running any experiment.

## Minimal use

```python
import sys; sys.path += ["code", "code/bench"]
import torch
from wosnet import WOSChain, WOSBank, set_deploy

model = WOSChain(stages=2, dim=1, k=7)        # 2 WOS stages, 7-sample windows, 16 parameters
# ... train with any optimiser: in train() mode each stage uses the ramp (beta = 1)
set_deploy(model)                              # eval mode: exact weighted order statistics (beta = 0)
y = model(torch.randn(1, 1024))

bank = WOSBank(C=4, stages=2, dim=1, k=7, anneal=2100)   # banks: ramp annealed to 0 over the first 2100 steps
```

## Citation

See `CITATION.cff`, or:

> J. C. del Río, *Train with Ramps, Deploy with Ranks: Tiny, robust rank-order filters that compete with neural
> networks*, preprint, Zenodo, 2026. doi:10.5281/zenodo.XXXXXXX

## Licence

Code: MIT (`LICENSE`). Paper, figures and tables in `paper/`: CC BY 4.0 (`paper/LICENSE`).
Datasets are not redistributed; see `code/data/get_data.py` for their sources and terms.
