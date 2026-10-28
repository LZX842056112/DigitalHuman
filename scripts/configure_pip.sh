#!/usr/bin/env bash
set -euo pipefail
ROOT=$(realpath "$1")
# These two environments belong only to this project. No user/global config is changed.
for env in avatar speech; do
  mkdir -p "/opt/digitalhuman/$env"
  printf '[global]\nconstraint = %s\n' "$ROOT/configs/$env-constraints.txt" > "/opt/digitalhuman/$env/pip.conf"
done
