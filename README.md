# Double-RIS Wireless-Powered Communication — Monte Carlo Simulation

Professional research-grade implementation of the Monte Carlo simulation used
in the study "Double-RIS Enabled Physical Layer Security for WPC Systems Over
Rayleigh Fading Channels" (Kunrui Cao et al., IEEE, 2025). The code reproduces
figures and metrics reported in the paper, including COP, SOP and EST for four
Double-RIS (DRIS) schemes.

## Contents
- `simulation.py` — main simulation script (power sweep, N sweep, plotting)
- `DRIS_power_sweep.png` — generated power-sweep figure
- `DRIS_N_sweep.png` — generated N-sweep figure

## Requirements
- Python 3.8 or later
- NumPy
- Matplotlib

Install the required packages with pip:

```bash
pip install numpy matplotlib
```

## Quick start
Run the simulation from the repository root:

```bash
python simulation.py
```

This will perform the transmit-power sweep and the RIS-element (`N`) sweep,
save the figures `DRIS_power_sweep.png` and `DRIS_N_sweep.png` and print a
tabulated results summary to stdout.

## Key configurable parameters
The top of `simulation.py` contains the main simulation configuration. Important
parameters you may want to edit:

- `N` — number of RIS reflecting elements (default: 32)
- `eta` — RF-to-DC energy conversion efficiency (default: 0.8)
- `alpha_t` — energy-harvesting time ratio (T_EH / T_total, default: 0.5)
- `R_target` / `R_s` — rate thresholds used for COP/SOP (bits/s/Hz)
- `M_sim` — Monte Carlo iterations (default: 1e5). Reduce for faster runs.
- `sigma2` — noise power (Watts)
- `path_exp` — path-loss exponent
- `d` — dictionary of distances (metres) used to compute mean channel gains
- RNG seed (`rng = np.random.default_rng(42)`) for reproducible results

Adjusting `M_sim` is the quickest way to trade runtime for statistical
accuracy during development.

## Outputs
- `DRIS_power_sweep.png` — three-panel figure (COP, SOP, EST vs transmit power)
- `DRIS_N_sweep.png` — two-panel figure (COP and SOP vs number of RIS elements)
- Console summary table with selected numerical values for quick inspection

## Notes and recommendations
- The script uses Matplotlib's `Agg` backend so it can run on headless servers.
- For exploratory work set `M_sim` to a smaller value (e.g. 1e4) and increase it
  for final results or reproducible experiments.
- The implementation assumes Rayleigh fading and follows the model and system
  parameters described in the referenced paper. Use caution if adapting to
  different channel models or parameter regimes.

## Citation
If you use this implementation in academic work, please cite:

Kunrui Cao et al., "Double-RIS Enabled Physical Layer Security for WPC Systems
Over Rayleigh Fading Channels", IEEE, 2025.

## License & contact
This repository is provided for academic research and experimentation. If you
need a license statement or wish to collaborate, open an issue or contact the
maintainer.
