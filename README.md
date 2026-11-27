# The First 100 Hours: model code

Team DESTINY, Tata Elxsi. Asian Paints Alchemy 2026, Case Study 4, Phase I.

Forecasting exterior paint durability from ATR-FTIR spectra read in the first 100 hours of the ASTM G154 UV test.

## What is here

| Path | What it does |
|---|---|
| `model/sim.py` | Simulation of 120 + 80 + 80 recipes: oxidation kinetics, stabiliser induction, infrared spectra with noise, features (5 band rates + carbonyl spread), elastic net with 5×5 repeated CV, split-conformal 90% range. Writes `out/sim_results.json`. Seed 20261115. |
| `model/open_model.py` | Real data: re-analysis of the Kyoto University open ATR-FTIR spectra; NASA POWER India climate layer; decision analysis on 80 held-out simulated recipes. Writes `out/open_model_results.json`. |
| `model/robustness.py` | Re-runs the simulation with each kinetic setting halved or doubled. Writes `out/robustness.json`. |
| `model/make_workbook.py` | Builds the supporting Excel workbook (number trace, business-case calculator). |
| `model/figs*.py`, `visuals2.py`, `exhibits.py` | Charts and equation images for the deck. |
| `model/failmodes.py`, `render.py`, `product_render.py`, `trim_renders.py` | Illustrations used in the deck. |
| `model/open_data/` | Reduced Kyoto spectra (`.npz`) and NASA POWER climatology (`.json`). |
| `model/out/` | Model results the deck reads. |

## How to run

```
cd model
pip install -r requirements.txt
python3 sim.py out
python3 open_model.py out open_data ../figures
python3 robustness.py out ../figures
python3 make_workbook.py out open_data ../figures/workbook.xlsx
```

## Key results (from `model/out/`)

- Real spectra (acrylic): acid/ester ratio rises 45% in the first 100 h, 8–11× the repeat scatter; UV dose read back with R² 0.72–0.88 from six spectral features vs 0.43–0.78 from the carbonyl index. These numbers are for UV dose, not failure time.
- Simulation: median error of hours to failure 11.1% vs 25.2% for the carbonyl index (120 recipes, 5×5 CV); lower in 10 of 10 robustness settings.
- Simulation, target life 2,000 h: 47 of 80 held-out recipes get a verdict at 100 h, with 0 wrong verdicts among the 47.

The failure-time forecast is simulation-based until a pilot pairs early spectra with real failure times.

## Data sources and licences

- Teramoto, Y. (Kyoto University). Data and code for GAWNSPLS analysis of accelerated weathering spectra of wood coatings. Zenodo, [doi:10.5281/zenodo.22971354](https://doi.org/10.5281/zenodo.22971354). CC BY 4.0. A reduced copy is redistributed here with attribution.
- NASA Langley Research Center, POWER project, Climatology API v2.10: [power.larc.nasa.gov](https://power.larc.nasa.gov/).

## Methods cited

- Zou, H., Hastie, T. Regularization and variable selection via the elastic net. J. R. Stat. Soc. B 67, 301–320 (2005). [doi](https://doi.org/10.1111/j.1467-9868.2005.00503.x)
- Lei, J. et al. Distribution-free predictive inference for regression. JASA 113, 1094–1111 (2018). [doi](https://doi.org/10.1080/01621459.2017.1307116)
