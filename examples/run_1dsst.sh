#!/usr/bin/env bash

set -euo pipefail

GYARADOS_REPOSITORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GYARADOS_OUTPUT="${GYARADOS_OUTPUT:-${PWD}/gyarados_1dsst_output}"

python3 "${GYARADOS_REPOSITORY}/gyarados_cli.py" \
  --config "${GYARADOS_REPOSITORY}/tools.yml" run \
  --model 1dsst \
  --output-dir "${GYARADOS_OUTPUT}" \
  --demes 5 \
  --population-size 1400 \
  --M 5 \
  --samples 2 \
  --chromosomes 1 \
  --size 1000000 \
  --iterations 1 \
  --sampled-population 1 \
  --sampled-population 3 \
  --mu 1e-8 \
  --rho 1e-8 \
  --mode iicr,simulate \
  "$@"
