GYARADOS structured IICR/SMC pipeline
=====================================

This repository runs fastsimcoal2 island/stepping-stone simulations, calculates
IICR summaries, and launches PSMC/SMC++ inference jobs on the Genobioinfo
cluster. The code is old, beloved, and occasionally sharp-edged, so the current
entry points keep the original defaults while adding explicit run modes.

Main entry points
-----------------

- `gyarados_call.sh`: SLURM-facing pipeline launcher.
- `run_island_models.py`: small island-model grid, now with optional mode and
  PSMC parameters.
- `run_psmc_fim_vector.py`: island-model driver with a configurable PSMC
  time vector (`psmc -p`).

Run Modes
---------

Use `--mode` with `gyarados_call.sh` or the Python drivers.

Modes are atomic steps. Combine them with commas:

- `iicr`: calculate the simulated IICR curve.
- `simulate`: run fastsimcoal2 for repetitions.
- `stats`: calculate sequence summary statistics from existing `.gen.gz`.
- `psmc`: run PSMC from existing repetitions.
- `smcpp`: run SMC++ from existing repetitions.
- `none`: create/copy model files but submit no repetition jobs.

For IICRsim + summary statistics + PSMC:

```bash
--mode iicr,simulate,stats,psmc
```

For the same plus SMC++:

```bash
--mode iicr,simulate,stats,psmc,smcpp
```

PSMC settings
-------------

Two PSMC knobs are now exposed:

- `--psmc-patterns`: comma-separated PSMC `-p` time vectors.
- `--psmc-s`: fq2psmcfa `-s` bin size.

Default PSMC settings:

```bash
--psmc-patterns '4+25*2+4+6' --psmc-s 100
```

The model/repetition folder tree stays the same: simulated IICR, sequence
summaries, PSMC, and SMC++ remain under the same demographic-model folder. PSMC
outputs get a `_p...` suffix such as `_p4_25_2_4_6` or `_p27_2_4_6`; the model
folder name still carries the simulated demographic parameters. Each PSMC
summary contains that vector's mean PSMC-inferred IICR and the same model/deme
IICRsim column for comparison. SMC++ summaries are written separately with the
same IICRsim column.

The simplest workflow is one PSMC vector per run/output directory. The code can
accept comma-separated vectors, but separate folders are easier to audit.

Examples
--------

Original-style island run:

```bash
sbatch gyarados_call.sh -n 50 -N 2000 -m 2.5510E-05 -b 100000000 -c 5 \
  -t 1 -p 1 -i 1 -d 1e-8 -s 2 -M 5 -P 100000
```

Same kind of run, but with PSMC and SMC++ enabled and multiple PSMC vectors:

```bash
sbatch gyarados_call.sh -n 50 -N 2000 -m 2.5510E-05 -b 100000000 -c 5 \
  -t 1 -p 1 -i 1 -d 1e-8 -s 20 -M 5 -P 100000 \
  --mode iicr,simulate,stats,psmc,smcpp \
  --psmc-s 100 \
  --psmc-patterns '4+25*2+4+6,27*2+4+6,2*2+2*2+25*2+1*4+1*6'
```

Submit the built-in PSMC-vector driver:

```bash
python run_psmc_fim_vector.py --mode iicr,simulate,stats,psmc --samples 20 \
  --psmc-pattern '27*2+4+6'
```

Preview commands without submitting:

```bash
python run_psmc_fim_vector.py --dry-run --mode iicr,simulate,stats,psmc,smcpp \
  --psmc-pattern '4+25*2+4+6' \
  --psmc-pattern '27*2+4+6'
```

Rerun only PSMC on an existing model/repetition set:

```bash
python run_psmc_fim_vector.py --mode psmc --samples 20 \
  --psmc-pattern '16*1+19*2+1*4+1*6'
```
