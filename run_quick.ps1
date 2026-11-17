$ErrorActionPreference = "Stop"
python scripts\00_plot_simulation_setup.py --config config_quick.yaml
python scripts\01_validate_single_reflector.py --config config_quick.yaml --snr 50
python scripts\02_validate_two_reflectors.py --config config_quick.yaml --snr 50
python scripts\03_generate_wedge_bscan.py --config config_quick.yaml --snrs 10 30 50
python scripts\04_run_snr_sweep.py --config config_quick.yaml --separation-step-um 2
python scripts\05_make_report_figures.py
python -m pytest -q
