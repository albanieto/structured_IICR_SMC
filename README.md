# Gyarados

Gyarados simulates demographic models with fastsimcoal2, calculates the IICR,
and can run sequence statistics, PSMC and SMC++. It can be launched from any
directory. All generated files are written below `--output-dir`.

## Requirements

| Run | Required software |
|---|---|
| All modes | Git, Conda and fastsimcoal2 |
| `psmc` | PSMC |
| `smcpp` | SMC++ installed in a separate Conda environment |

The supplied `gyarados` environment installs Python, NumPy, pandas, SciPy,
PyYAML, seqtk, bgzip and tabix. fastsimcoal2, PSMC and SMC++ must be installed
separately.

## Installation

### 1. Download Gyarados

```bash
git clone <REPOSITORY_URL> structured_IICR_SMC
cd structured_IICR_SMC
```

### 2. Install fastsimcoal2

Download fastsimcoal2 2.8 from its
[official page](https://cmpg.unibe.ch/software/fastsimcoal28/), extract it and
make the executable runnable:

```bash
chmod +x /absolute/path/to/fsc28
```

### 3. Install PSMC

Follow the instructions in the
[official PSMC repository](https://github.com/lh3/psmc):

```bash
git clone https://github.com/lh3/psmc.git
cd psmc
make
(cd utils && make)
```

PSMC requires a C compiler, `make` and zlib development headers. Gyarados uses
the compiled `psmc` executable and obtains `utils/fq2psmcfa` from the same
installation.

### 4. Install SMC++

Create a separate Conda environment and follow the installation instructions
in the
[official SMC++ repository](https://github.com/popgenmethods/smcpp) while that
environment is active:

```bash
conda create --name smcpp python=3
conda activate smcpp
```

After installing SMC++, verify it and return to the base environment:

```bash
smc++ --help
conda deactivate
```

### 5. Edit `tools.yml`

Enter the exact fastsimcoal2 and PSMC executable paths. `smcpp_environment` is
the Conda environment name, not a path:

```yaml
fastsimcoal2: /home/user/software/fastsimcoal2/fsc28
psmc: /home/user/software/psmc/psmc
smcpp: smc++
smcpp_environment: smcpp
```

Gyarados runs only SMC++ commands inside the `smcpp` environment. The rest of
the pipeline remains in the `gyarados` environment. Each `vcf2smc`, `estimate`
and `plot` command is executed as:

```text
conda run --no-capture-output -n smcpp smc++ ...
```

The SMC++ environment ends when that command finishes, so the next pipeline
step continues in `gyarados`. No manual activation or deactivation is needed
inside the scripts.

### 6. Create the Gyarados environment

From the repository root:

```bash
conda env create --file environment.yml
conda activate gyarados
```

No additional installation command is required.

### 7. Check the configuration

```bash
python /absolute/path/to/structured_IICR_SMC/gyarados_cli.py \
  --config /absolute/path/to/structured_IICR_SMC/tools.yml \
  doctor
```

## Run the examples

Preview an example without creating files or running external programs:

```bash
/absolute/path/to/structured_IICR_SMC/examples/run_fim.sh --dry-run
```

Run it:

```bash
/absolute/path/to/structured_IICR_SMC/examples/run_fim.sh
```

The examples work from any directory. By default, results are created in the
current directory. Set another location with:

```bash
GYARADOS_OUTPUT=/data/my_run \
  /absolute/path/to/structured_IICR_SMC/examples/run_fim.sh
```

| Model | Example |
|---|---|
| Finite island model | `examples/run_fim.sh` |
| Two-dimensional stepping stone | `examples/run_2dsst.sh` |
| One-dimensional stepping stone | `examples/run_1dsst.sh` |
| Panmictic | `examples/run_panmictic.sh` |
| Free model | `examples/run_free.sh` |
| FIM and its panmictic Ditto clone, including PSMC and SMC++ | `examples/run_fim_ditto.sh` |

The first five examples run `iicr,simulate` and require fastsimcoal2. The
`examples/run_minimal.sh` example also runs statistics and PSMC. The complete
Ditto example requires fastsimcoal2, PSMC and SMC++:

```bash
/absolute/path/to/structured_IICR_SMC/examples/run_fim_ditto.sh
```

## Run a model directly

```bash
conda activate gyarados

python /absolute/path/to/structured_IICR_SMC/gyarados_cli.py \
  --config /absolute/path/to/structured_IICR_SMC/tools.yml \
  run \
  --model fim \
  --output-dir /data/project/run_01 \
  --demes 50 \
  --population-size 1400 \
  --M 5 \
  --samples 20 \
  --chromosomes 2 \
  --size 100000000 \
  --iterations 1 \
  --mu 1e-8 \
  --rho 1e-8 \
  --mode iicr,simulate,stats,psmc
```

`--chromosomes` sets the chromosome count. One `--size` applies to every
chromosome; repeat `--size` when chromosome lengths differ.

For 1D-SST and 2D-SST, repeat the option to sample several populations:

```bash
--sampled-population 1 \
--sampled-population 2
```

`--samples` is the number of sampled haploid genomes per selected population
and must be even for PSMC.

For a free model, population samples, chromosome numbers and chromosome
lengths are read from its `.par` file:

```bash
--model free --par-file /absolute/path/to/model.par
```

The FIM examples use population 1 for inference. Its simulated `.gen.gz`
contains populations 1 and 2 so that FST can be calculated automatically.

The IICR always uses \(10^7\) simulated T2 values.

To calculate the original model, generate its panmictic Ditto clone and run
all analyses on both:

```bash
--mode iicr,simulate,stats,psmc,smcpp,ditto
```

`ditto` requires `iicr`. One panmictic clone is generated for each sampled
population.

## Modes

Modes are comma-separated:

| Mode | Action |
|---|---|
| `iicr` | Calculate the IICR with fastsimcoal2 |
| `simulate` | Simulate sequence data with fastsimcoal2 |
| `stats` | Calculate sequence statistics and FST |
| `psmc` | Run PSMC |
| `smcpp` | Run SMC++ |
| `ditto` | Generate a panmictic clone and run the selected analyses on it |
| `none` | Create model files only |

For example:

```bash
--mode iicr,simulate,stats,psmc,smcpp
```

## Outputs

Everything is written below `--output-dir`:

```text
output-dir/
├── RESULTS/
├── <model-name>/
├── gyarados_models.log
└── gyarados_time_check.log
```

Population-specific filenames contain `_deme_N`. Shared `.par`, `.json` and
`.gen.gz` files do not.
