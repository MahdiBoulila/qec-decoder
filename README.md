# QEC Observatory

A local workspace for inspecting quantum error correction circuits, playing back
syndrome measurements, and comparing logical memory experiments.

## Run locally

Use Python 3.12 and run these commands from the repository root.

```powershell
python -m pip install -r requirements.txt
python -B compare_hardware_shots.py --serve --port 8765
```

Open http://127.0.0.1:8765/. The viewer starts with six seeded Stim distance-3
memory replays when no local run archive is present. Shot replay and the
simulated QEC views work with no IBM account.

For Sycamore comparisons, download the ZIP from
[Google Quantum AI's public dataset](https://zenodo.org/records/6804040)
into `runs/sycamore-memory/google_qec3v5_experiment_data.zip`.
Refresh the page after downloading. The archive contains published decoder
predictions and is distributed by its authors.

## Command line

```powershell
python -B pipeline.py --help
python -B hardware_trial.py --help
python -B sweep.py --help
python -B -m unittest discover
```

The hardware workflow translates a Stim circuit into Qiskit, runs a Sampler
job, and scores the returned shots offline. Use your own IBM Quantum Runtime
account. Hardware records and credentials are stored locally. Tests that
require optional local experiment archives are skipped on a fresh checkout.

The viewer binds to loopback for local use.
