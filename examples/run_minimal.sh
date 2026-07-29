#!/usr/bin/env bash
#
# Small end-to-end local example. Run from anywhere:
#   /path/to/structured_IICR_SMC/examples/run_minimal.sh
#
# Add --dry-run to inspect the configuration without running external tools.

set -euo pipefail

GYARADOS_REPOSITORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GYARADOS_OUTPUT="${GYARADOS_OUTPUT:-${PWD}/gyarados_example_output}"

python3 "${GYARADOS_REPOSITORY}/gyarados_cli.py" \
  --config "${GYARADOS_REPOSITORY}/tools.yml" run \
  --output-dir "${GYARADOS_OUTPUT}" \
  --demes 5 \
  --population-size 1400 \
  --M 5 \
  --samples 2 \
  --size 1000000 \
  --iterations 1 \
  --population 1 \
  --mu 1e-8 \
  --rho 1e-8 \
  --mode iicr,simulate,stats,psmc \
  --iicr-replicates 10000 \
  --psmc-s 100 \
  --psmc-pattern '4+25*2+4+6' \
  "$@"
