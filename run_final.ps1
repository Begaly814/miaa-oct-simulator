$ErrorActionPreference = "Stop"
python scripts\00_plot_simulation_setup.py --config config_final.yaml
python scripts\01_validate_single_reflector.py --config config_final.yaml --snr 60
python scripts\02_validate_two_reflectors.py --config config_final.yaml --snr 50
python scripts\03_generate_wedge_bscan.py --config config_final.yaml --snrs 10 30 50
python scripts\04_run_snr_sweep.py --config config_final.yaml --separation-step-um 0.5
python scripts\05_make_report_figures.py
python -m pytest -q
