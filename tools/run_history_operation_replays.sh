#!/usr/bin/env bash
# Replay accepted-history operations one after another, each into a fresh folder.
# Usage: tools/run_history_operation_replays.sh <snapshot> [--profile|--native] op1 op2 ...
set -u
cd "$(dirname "$0")/.."
M=validation/completed_base_20261012/memory
SNAP=$1; shift
FLAG=""
if [ "${1:-}" = "--profile" ] || [ "${1:-}" = "--native" ]; then FLAG=$1; shift; fi
for op in "$@"; do
  out=$M/runs/op_replay/$(date +%Y%m%d-%H%M%S)_${op}${FLAG//-/_}
  python tools/replay_history_operation.py --engine development/cd68_full_red_green/rigid_analysis/A1/tracking/engine \
    --snapshot "$SNAP" --config development/cd68_full_red_green/rigid_batch/.j/A1/inputs/motion_config.json \
    --output "$out" --run pymicroglia_95_full_A1 --stem 95_full_A1 --operation "$op" $FLAG > "$out.log" 2>&1
  echo "$op exit=$? $(head -c 600 "$out.log")"
done
