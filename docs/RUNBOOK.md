# MIAA versus SNR: synthetic OCT wedge simulator

This project synthesizes a noise-contaminated OCT fringe from two equal reflectors whose separation decreases across a B-scan. It then compares:

1. **Given-band DFT**: conventional reconstruction from only the measured central spectral band.
2. **IAA**: iterative adaptive depth estimation on the measured band.
3. **MIAA**: IAA followed by MMSE prediction of the missing left and right spectral samples.
4. **Oracle full-band DFT**: reconstruction from the true synthetic target spectrum. This is a reference, not an available measurement.

The primary experiment measures two-reflector resolution versus input SNR. The MIAA-specific secondary metric is missing-spectrum normalized mean squared error (NMSE).

## 1. Setup in VS Code on Windows

Open this folder in VS Code, then open a terminal and run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
```

Confirm the installation:

```powershell
python -c "import miaa_oct; print('miaa_oct imported successfully')"
```

## 2. Run the quick experiment first

```powershell
.\run_quick.ps1
```

This uses small matrices and only three noise seeds. It checks the complete pipeline but is not the final experiment.

You may also run the stages manually:

```powershell
python scripts\00_plot_simulation_setup.py --config config_quick.yaml
python scripts\01_validate_single_reflector.py --config config_quick.yaml --snr 50
python scripts\02_validate_two_reflectors.py --config config_quick.yaml --snr 50
python scripts\03_generate_wedge_bscan.py --config config_quick.yaml --snrs 10 30 50
python scripts\04_run_snr_sweep.py --config config_quick.yaml --separation-step-um 2
python scripts\05_make_report_figures.py
python -m pytest -q
```

Outputs are written to:

```text
results/figures/
results/tables/
results/wedge_<SNR>dB.npz
```

## 3. Run the final experiment

After inspecting the quick outputs:

```powershell
.\run_final.ps1
```

The final sweep uses 11 SNR levels, separations from 0 to 29 micrometers in 0.5-micrometer steps, and 30 independent noise seeds. Brute-force IAA is intentionally used for transparency, so this run can take substantial time on a CPU.

## 4. File-by-file map

### Configuration

- `config_quick.yaml`: small test configuration.
- `config_final.yaml`: report configuration.
- `config.yaml`: copy of the final defaults.

### Core source code

- `src/miaa_oct/config.py`
  - Loads and validates YAML parameters.
- `src/miaa_oct/grids.py`
  - Builds the centered wavenumber grid `q = k - k_c`.
  - Splits the target spectrum into measured and missing indices.
  - Builds Fourier dictionaries for the given and missing samples.
- `src/miaa_oct/fringe.py`
  - Synthesizes a complex analytic OCT fringe from discrete reflectors.
  - Builds the two approaching reflector trajectories for the wedge B-scan.
- `src/miaa_oct/noise.py`
  - Adds circular complex Gaussian noise at an exact requested power SNR.
- `src/miaa_oct/iaa.py`
  - Implements brute-force IAA.
  - Uses a zero-padded-DFT-equivalent initialization.
  - Builds the covariance from the current reflectivity powers.
  - Uses Cholesky linear solves; it never explicitly computes a matrix inverse.
- `src/miaa_oct/miaa.py`
  - Implements the MIAA MMSE missing-data equation.
  - Copies measured samples unchanged and predicts only missing samples.
- `src/miaa_oct/reconstruction.py`
  - Performs centered zero padding, inverse FFT reconstruction, and spectral tapering.
- `src/miaa_oct/metrics.py`
  - Calculates missing-spectrum NMSE.
  - Applies the 3 dB valley criterion and localization checks to two reflectors.
- `src/miaa_oct/pipeline.py`
  - Runs one complete noisy A-line through DFT, IAA, MIAA, and oracle reconstruction.
- `src/miaa_oct/plotting.py`
  - Produces A-line and B-scan figures using a consistent dB scale.

### Experiment scripts

- `scripts/00_plot_simulation_setup.py`
  - Plots wedge geometry, measured/missing spectral regions, and example noisy fringes.
- `scripts/01_validate_single_reflector.py`
  - Checks depth accuracy and missing-spectrum recovery for one reflector.
- `scripts/02_validate_two_reflectors.py`
  - Checks separated and close reflector pairs before the large experiment.
- `scripts/03_generate_wedge_bscan.py`
  - Generates the approaching-reflector B-scans at selected SNRs.
- `scripts/04_run_snr_sweep.py`
  - Runs the Monte Carlo experiment over SNR, separation, and random seed.
- `scripts/05_make_report_figures.py`
  - Converts the raw CSV into resolution-versus-SNR, NMSE, and probability heatmap figures.

## 5. The model used by the code

For reflectors at depths `z_j`, the normalized analytic fringe is

```text
y(q_n) = sum_j a_j exp[-i 2 (k_c + q_n) z_j].
```

The constant carrier phase `exp(-i 2 k_c z_j)` is absorbed into each complex reflector amplitude. This keeps the model physically phase-consistent while allowing a centered `q` grid.

The B-scan uses

```text
z_1(x) = z_c - Delta_z(x)/2
z_2(x) = z_c + Delta_z(x)/2
```

with `Delta_z(x)` decreasing linearly from 29 micrometers to 0.

## 6. SNR definition

The code uses complex-sample power SNR:

```text
SNR_dB = 10 log10(mean(|y_clean|^2) / mean(|noise|^2)).
```

The noise realization is rescaled so every trial has the requested SNR exactly. This isolates reconstruction variability from accidental SNR variation.

## 7. IAA implementation

IAA initializes

```text
a^(0) = F_g^H y_g / N_g.
```

At each iteration it builds

```text
R_g = F_g diag(|a|^2) F_g^H + sigma^2 I,
```

then updates all candidate depths using

```text
a_l = (f_l^H R_g^-1 y_g) / (f_l^H R_g^-1 f_l).
```

The code solves linear systems with Cholesky factorization instead of forming `R_g^-1` explicitly.

For this controlled synthetic experiment, the known injected noise variance is supplied to IAA. That makes the experiment specifically about IAA/MIAA performance versus SNR, rather than mixing in a separate noise-estimation error.

## 8. MIAA implementation

After IAA converges on the measured band:

```text
v = R_g^-1 y_g
c_l = |a_l|^2 (f_g,l^H v)
y_m = F_m c
```

This is the MMSE missing-data expression. The final target spectrum contains the actual noisy measured samples in the central band and MIAA predictions only at missing indices.

## 9. Resolution metric

A reflector pair is counted as resolved only when:

1. Two peaks are detected near the two known depths.
2. Both peaks are within the configured localization tolerance.
3. The valley between them satisfies

```text
valley <= 0.5 * min(peak_1, peak_2).
```

The report script determines the 90% resolution threshold while sweeping from large to small separation and stopping at the first loss of reliable resolution. This is more stable than taking an isolated successful point because coherent two-reflector fringes can show nonmonotonic beating.

## 10. How to interpret the main outputs

- `05_resolution_vs_snr.png`
  - DFT should be approximately SNR-independent.
  - IAA/MIAA should improve as SNR increases, until limited by grid/model choices.
- `05_missing_nmse_vs_snr.png`
  - More-negative NMSE is better.
  - This is the direct test of MIAA spectral extrapolation.
- `05_resolution_probability_heatmap.png`
  - Shows where MIAA resolves the pair reliably across noise seeds.
- `03_wedge_*.png`
  - Visual comparison of the approaching reflector geometry.

## 11. Experimental boundaries

- This simulator studies axial spectral estimation only. It does not implement ISAM or CAO.
- Brute-force IAA is used for clarity. FIAA/RFIAA are speed optimizations and can be added after scientific validation.
- The oracle full band is synthetic ground truth, not data available to MIAA.
- MIAA performance depends on SNR and on the assumed sum-of-complex-exponentials model.
- The exact numerical resolution will depend on the target spectral extension, taper, grid density, SNR definition, carrier phase, and peak criterion. Do not force the result to match a paper value.

## 12. Report

Use `report/report_template.md`. Replace placeholders only after running `config_final.yaml`. Include the exact configuration and the generated CSV files as supplementary material.
