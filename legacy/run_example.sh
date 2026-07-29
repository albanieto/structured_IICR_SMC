#!/usr/bin/env bash
#
# Minimal end-to-end installation example for the Genotoul Genobioinfo cluster.
# It submits IICR + simulation + statistics + PSMC for one 10 Mb chromosome.
#
# Usage:
#   ./run_example.sh --dry-run   # print the sbatch command only
#   ./run_example.sh             # submit the example

set -euo pipefail

REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPOSITORY_DIR}"

if command -v python3 >/dev/null 2>&1; then
  GYARADOS_PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  GYARADOS_PYTHON=python
else
  echo "Error: Python is not available. Follow the README prerequisites." >&2
  exit 1
fi

if ! "${GYARADOS_PYTHON}" -c 'import numpy, pandas' >/dev/null 2>&1; then
  echo "Error: NumPy or pandas is missing for ${GYARADOS_PYTHON}." >&2
  echo "Install requirements.txt as described in the README." >&2
  exit 1
fi

GYARADOS_DRY_RUN=false
for GYARADOS_ARGUMENT in "$@"; do
  if [[ "${GYARADOS_ARGUMENT}" == "--dry-run" ]]; then
    GYARADOS_DRY_RUN=true
  fi
done

if [[ "${GYARADOS_DRY_RUN}" == false ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch is not available. This example must run on a SLURM login node." >&2
  exit 1
fi

echo "Running the minimal Gyarados example from ${REPOSITORY_DIR}"

"${GYARADOS_PYTHON}" run_psmc_fim_vector.py \
  --d-vector 5 \
  --n-vector 1400 \
  --m-vector 5 \
  --samples 2 \
  --sizes 10000000 \
  --chromosomes 1 \
  --iterations 1 \
  --population 1 \
  --mu 1e-8 \
  --mode iicr,simulate,stats,psmc \
  --psmc-s 100 \
  --psmc-pattern '4+25*2+4+6' \
  "$@"
