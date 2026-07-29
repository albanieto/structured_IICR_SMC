# Gyarados: portable structured-IICR pipeline

Gyarados simulates symmetric island models with fastsimcoal2, calculates the
simulated IICR, and can run sequence statistics, PSMC, and SMC++ locally.

It runs from any directory and does not require SLURM, cluster modules, or a
particular folder structure. All generated files are written below the
directory selected with `--output-dir`.

The old cluster-specific repository is preserved unchanged in
[`legacy/`](legacy/). New users should follow this README and use the files in
the repository root.

## Getting started

### Software required

- Every Gyarados run requires fastsimcoal2.
- The `psmc` mode additionally requires PSMC.
- The `smcpp` mode additionally requires SMC++ in its own environment.
- Git and Conda are always required.

The supplied `gyarados` Conda environment installs Python, NumPy, pandas,
SciPy, PyYAML, seqtk, bgzip, and tabix. It does not install fastsimcoal2,
PSMC, or SMC++.

Linux is recommended. Windows users should use WSL with Conda for Linux.

### 1. Download Gyarados

```bash
git clone <REPOSITORY_URL> structured_IICR_SMC
cd structured_IICR_SMC
```

Replace `<REPOSITORY_URL>` with this repository's real URL.

### 2. Install the external programs

#### fastsimcoal2: required for every run

1. Open the
   [official fastsimcoal2 version 2.8 page](https://cmpg.unibe.ch/software/fastsimcoal28/).
2. Download the archive for Linux or macOS.
3. Extract it in a permanent location, for example
   `/home/your_name/software/fastsimcoal2/`.
4. Make the `fsc28` executable runnable:

```bash
chmod +x /absolute/path/to/fsc28
```

Keep its exact absolute path for `tools.yml`.

#### PSMC: required only for `psmc` mode

Install a C compiler, `make`, and the zlib development headers. On Ubuntu or
Debian:

```bash
sudo apt update
sudo apt install git build-essential zlib1g-dev
```

On macOS:

```bash
xcode-select --install
```

Then clone [Heng Li's official PSMC repository](https://github.com/lh3/psmc)
and compile PSMC and its utilities:

```bash
cd /where/you/keep/software
git clone https://github.com/lh3/psmc.git
cd psmc
make
(cd utils && make)
```

This creates the two commands needed by Gyarados:

```text
/where/you/keep/software/psmc/psmc
/where/you/keep/software/psmc/utils/fq2psmcfa
```

Keep the absolute path to `psmc`. Gyarados obtains the `fq2psmcfa` path
automatically from the `utils` directory of the same PSMC installation.

#### SMC++: required only for `smcpp` mode

Follow the installation instructions in the
[official SMC++ repository maintained by the Terhorst group](https://github.com/popgenmethods/smcpp).
Install it in a separate Conda environment. For example, if that environment
is named `smcpp`, confirm the completed installation with:

```bash
conda activate smcpp
smc++ --help
conda deactivate
```

Keep only the Conda environment name for `tools.yml`.

### 3. Edit `tools.yml`

Every user must edit [`tools.yml`](tools.yml) before running Gyarados. Enter
the three program calls and, when needed, the SMC++ Conda environment name:

```yaml
fastsimcoal2: /home/user/software/fastsimcoal2/fsc28
psmc: /home/user/software/psmc/psmc
smcpp: smc++
smcpp_environment: smcpp
```

Use absolute installation paths without spaces. Gyarados executes exactly the
values written here and does not guess executable names.

If PSMC or SMC++ will not be used, write `none` for that program. If SMC++
does not require a separate environment, write `none` for
`smcpp_environment`:

```yaml
fastsimcoal2: /home/user/software/fastsimcoal2/fsc28
psmc: none
smcpp: none
smcpp_environment: none
```

`fq2psmcfa` is taken from `utils/fq2psmcfa` beside the configured PSMC
installation. `seqtk`, `bgzip`, and `tabix` are provided by the `gyarados`
Conda environment and therefore do not appear in `tools.yml`.

When `smcpp_environment` is not `none`, Gyarados runs only the SMC++ commands
with `conda run -n ENVIRONMENT_NAME smc++ ...`. The main process remains in
the `gyarados` environment throughout the analysis.

### 4. Create the supplied `gyarados` environment

From the repository root, run:

```bash
conda env create --file environment.yml
conda activate gyarados
```

There is no additional `pip install` step for Gyarados. In every new terminal,
activate this environment before running the pipeline:

```bash
conda activate gyarados
```

To update an existing environment:

```bash
conda env update --name gyarados --file environment.yml --prune
```

### 5. Check the configuration

The command can be launched from any directory. Use the absolute path to the
repository:

```bash
conda activate gyarados

python /absolute/path/to/structured_IICR_SMC/gyarados_cli.py \
  --config /absolute/path/to/structured_IICR_SMC/tools.yml \
  doctor
```

The three program calls and the SMC++ environment setting are displayed.
Programs set to `none` remain unavailable and are ignored unless their mode is
requested.

`doctor` reports the values read from `tools.yml`. It does not search for other
program names or replace the user's choices.

## Run the included example

The included example uses `psmc`, so both fastsimcoal2 and PSMC must be
configured in `tools.yml`.

First preview it without creating files or executing scientific programs:

```bash
/absolute/path/to/structured_IICR_SMC/examples/run_minimal.sh --dry-run
```

Then run it:

```bash
/absolute/path/to/structured_IICR_SMC/examples/run_minimal.sh
```

The example script finds the repository from its own location, so it works
regardless of the current directory.

By default, it creates `gyarados_example_output` in the current directory. To
choose another output directory:

```bash
GYARADOS_OUTPUT=/data/my_test \
  /absolute/path/to/structured_IICR_SMC/examples/run_minimal.sh
```

The example uses:

- five symmetric demes;
- 1,400 haploid individuals per deme;
- scaled migration \(M=5\);
- two sampled haploid genomes from deme 1;
- one 1 Mb chromosome;
- one repetition;
- 10,000 fastsimcoal2 loci for the demonstration IICR; and
- `iicr,simulate,stats,psmc`.

These deliberately small values test the installation and file flow. They are
not suitable for a final biological analysis.

## Run a model

```bash
conda activate gyarados

python /absolute/path/to/structured_IICR_SMC/gyarados_cli.py \
  --config /absolute/path/to/structured_IICR_SMC/tools.yml \
  run \
  --output-dir /data/project/run_01 \
  --demes 50 \
  --population-size 1400 \
  --M 5 \
  --samples 20 \
  --size 100000000 \
  --size 100000000 \
  --iterations 1 \
  --population 1 \
  --mu 1e-8 \
  --rho 1e-8 \
  --mode iicr,simulate,stats,psmc \
  --iicr-replicates 10000000 \
  --psmc-s 100 \
  --psmc-pattern '4+25*2+4+6'
```

Repeat `--size` once per chromosome. `--samples` is a haploid count and must be
even for PSMC.

The migration rate is:

```text
m = M / (2 × N × (d - 1))
```

`--migration` can be supplied instead of `--M`.

Always preview a large run first:

```bash
python /absolute/path/to/gyarados_cli.py \
  --config /absolute/path/to/tools.yml \
  run ... --dry-run
```

## Run modes

Modes are comma-separated:

| Mode | Action |
|---|---|
| `iicr` | Simulate pairwise coalescence times with fastsimcoal2 and calculate the IICR |
| `simulate` | Simulate sequence data with fastsimcoal2 |
| `stats` | Calculate summary statistics from the simulated `.gen.gz` file |
| `psmc` | Build diploid consensus sequences and run PSMC |
| `smcpp` | Build/index VCF data and run the separately installed SMC++ |
| `transition_matrix` | Produce output for transition-matrix inspection |
| `none` | Create model files without external analysis |

Examples:

```bash
--mode iicr
--mode iicr,simulate,stats,psmc
--mode iicr,simulate,stats,psmc,smcpp
```

`psmc`, `smcpp`, and `stats` require a compatible existing simulation when
`simulate` is omitted.

## Outputs

Everything stays below `--output-dir`:

```text
output-dir/
├── RESULTS/
├── StSI_.../
├── gyarados_models.log
└── gyarados_time_check.log
```

## Repository layout

```text
.
├── README.md
├── environment.yml          supplied Conda environment named gyarados
├── tools.yml                external calls and SMC++ environment
├── header.txt               VCF header
├── gyarados_cli.py          command-line interface
├── gyarados.py              simulation and inference implementation
├── ditto.py
├── examples/
│   └── run_minimal.sh
└── legacy/                  unchanged historical repository
```
