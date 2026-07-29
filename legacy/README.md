# Gyarados: structured IICR/SMC pipeline

Gyarados simulates structured demographic models with fastsimcoal2, calculates
the corresponding IICR, and can run summary statistics, PSMC, and SMC++.

> **Important:** the pipeline is currently designed for the Genotoul
> Genobioinfo SLURM cluster. The scripts submit jobs with `sbatch` and load
> cluster-specific environment modules. Having the programs installed locally
> is not sufficient unless the SLURM scripts are adapted. The main code and its
> cluster configuration are intentionally unchanged in this repository.

## Software required before starting

The directly runnable example below needs every item whose “When it is
required” entry says either “Always” or “runnable example.” The versions in the
last column are the versions named by the existing SLURM scripts and are
therefore the safest tested choices.

| Software | When it is required | Version/module used by the scripts |
|---|---|---|
| Linux | Always | Bash, GNU `getopt`, and standard tools such as `awk`, `cat`, `cp`, `fold`, `gzip`, `tail`, and `tar` |
| SLURM (`sbatch`) | Always | A cluster on which users may submit jobs |
| Environment Modules (`module`) | Always | Must be available inside SLURM jobs |
| Python | Always | `devel/python/Python-3.7.9` |
| NumPy and pandas | Always | Install the versions in `requirements.txt` |
| fastsimcoal2, exposed as `fsc` | IICR and sequence simulation | `bioinfo/fastsimcoal2/2709` |
| PSMC (`psmc` and `fq2psmcfa`) | PSMC mode and the runnable example | `bioinfo/psmc/0.6.5` |
| seqtk | PSMC mode and the runnable example | `bioinfo/Seqtk/1.3` |
| bcftools (`bgzip` and `tabix`) | SMC++ mode only | `bioinfo/Bcftools/1.9` |
| SMC++ (`smc++`) | SMC++ mode only | `bioinfo/SMC++/1.15.5` |
| SciPy | Ditto mode only; recommended | Version in `requirements.txt` |

The exact module names above must exist on the cluster because they are loaded
by `gyarados_call.sh`, the worker scripts, `call_psmc_snif_gy.sh`, and
`call_smcpp_gy.sh`.

## Install

Log in to the Genotoul Genobioinfo cluster, then run:

```bash
git clone <REPOSITORY_URL> structured_IICR_SMC
cd structured_IICR_SMC
module load devel/python/Python-3.7.9
python -m pip install --user -r requirements.txt
```

Replace `<REPOSITORY_URL>` with this repository's clone URL. If NumPy, pandas,
and SciPy are already provided for the Python 3.7.9 module, the last command may
be skipped.

Confirm the two commands needed to launch the example:

```bash
command -v python
command -v sbatch
```

Both commands must print a path. You can also confirm that the cluster modules
are visible:

```bash
module avail devel/python/Python-3.7.9
module avail bioinfo/fastsimcoal2/2709
module avail bioinfo/psmc/0.6.5
module avail bioinfo/Seqtk/1.3
```

If any required module is absent, ask the cluster administrator for the
equivalent module. Changing module names in the SLURM scripts is outside the
installation described here.

## Run the minimal complete example

First preview the job without submitting anything:

```bash
./run_example.sh --dry-run
```

The last line should begin with:

```text
sbatch gyarados_call.sh
```

Then submit the example:

```bash
./run_example.sh
```

This command can be launched from any directory: the example changes to the
repository directory before running. It submits one reduced-size
structured-island model with:

- five demes;
- 1,400 haploid individuals per deme;
- \(M=5\);
- two sampled haploid genomes;
- one 10 Mb chromosome;
- one repetition; and
- the `iicr,simulate,stats,psmc` steps.

The launcher returns after submitting jobs; the analysis continues in SLURM.
Check it with:

```bash
squeue -u "$USER"
```

SLURM `.out` and `.err` logs are written in the repository directory. Model
files and final copied outputs are written below `RESULTS/`, with working files
in a model-named directory. The example creates data, so running it repeatedly
may create or reuse model directories.

For a full-size analysis, edit a copy of `run_example.sh` and increase
`--samples`, `--sizes`, `--chromosomes`, or `--iterations`. The minimal values
are intended to verify installation and workflow, not to produce a
publication-ready PSMC estimate.

## Run modes

Use `--mode` with `gyarados_call.sh` or the Python drivers. Modes are atomic
steps and may be combined with commas:

| Mode | Action |
|---|---|
| `iicr` | Calculate the simulated IICR curve |
| `simulate` | Run fastsimcoal2 for each repetition |
| `stats` | Calculate sequence summaries from existing `.gen.gz` files |
| `psmc` | Run PSMC from existing repetitions |
| `smcpp` | Run SMC++ from existing repetitions |
| `transition_matrix` | Produce/copy sequence simulation output for transition-matrix inspection |
| `none` | Create/copy model files without submitting repetition jobs |

For the normal IICR, simulation, statistics, and PSMC workflow:

```bash
--mode iicr,simulate,stats,psmc
```

Add SMC++ with:

```bash
--mode iicr,simulate,stats,psmc,smcpp
```

`transition_matrix` alone stops before statistics and inference. If combined
with `stats`, `psmc`, or `smcpp`, those steps also run.

## Main entry points

- `run_example.sh`: minimal installation-checking example; start here.
- `run_psmc_fim_vector.py`: submit one or more island-model parameter sets.
- `gyarados_call.sh`: SLURM-facing pipeline launcher.
- `ditto.py`: convert a simulated IICR to a panmictic fastsimcoal2 `.par` file
  and optionally launch the mimic through Gyarados.
- `run_island_models.py`: submit a small island-model grid.

## PSMC settings

Two settings are exposed:

- `--psmc-pattern`: a PSMC `-p` time vector; repeat the option for multiple
  vectors when using `run_psmc_fim_vector.py`.
- `--psmc-s`: the `fq2psmcfa -s` bin size.

The defaults are:

```bash
--psmc-pattern '4+25*2+4+6' --psmc-s 100
```

PSMC outputs receive a suffix describing the vector, for example
`_p4_25_2_4_6`. One vector per run/output directory is easiest to audit.

To preview two vectors without submitting:

```bash
python run_psmc_fim_vector.py \
  --dry-run \
  --mode iicr,simulate,stats,psmc \
  --psmc-pattern '4+25*2+4+6' \
  --psmc-pattern '27*2+4+6'
```

To rerun only PSMC on an existing compatible model/repetition set:

```bash
python run_psmc_fim_vector.py \
  --mode psmc \
  --samples 20 \
  --psmc-pattern '16*1+19*2+1*4+1*6'
```

The `psmc`-only command does not create missing simulations; the expected
model/repetition files must already exist.

## Ditto panmictic mimic

Pass `--ditto` to `run_psmc_fim_vector.py`, or `--ditto true` to
`gyarados_call.sh`, to build a panmictic fastsimcoal2 model from the simulated
IICR. Ditto uses the same requested modes and PSMC settings as the structured
model.

The original model tree remains unchanged. Ditto creates a separate
`<original_model>_deme_1_ditto` directory, with its own `RESULTS/...` outputs.

Example:

```bash
python run_psmc_fim_vector.py \
  --mode iicr,simulate,stats,psmc \
  --samples 20 \
  --psmc-pattern '4+25*2+4+6' \
  --ditto
```

## Existing larger examples

The repository also contains parameter-specific scripts:

- `run_example_fim_M5_d50_N1400_psmc.sh`
- `run_example_fim_M5_d50_N1400_psmc_ditto.sh`
- `run_example_fim_M5_d50_N1400_psmc_only.sh`
- `run_example_fim_M1_d50_N1400_smcpp_ditto.sh`

These are analysis-oriented examples and may request substantially more
compute, memory, and storage than `run_example.sh`. Use the minimal example
first.
