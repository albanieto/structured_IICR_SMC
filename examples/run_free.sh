#!/usr/bin/env bash

set -euo pipefail

GYARADOS_REPOSITORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GYARADOS_OUTPUT="${GYARADOS_OUTPUT:-${PWD}/gyarados_free_output}"

python3 "${GYARADOS_REPOSITORY}/gyarados_cli.py" \
  --config "${GYARADOS_REPOSITORY}/tools.yml" run \
  --model free \
  --par-file "${GYARADOS_REPOSITORY}/examples/free_model.par" \
  --output-dir "${GYARADOS_OUTPUT}" \
  --iterations 1 \
  --mode iicr,simulate \
  "$@"
