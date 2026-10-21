#!/usr/bin/env bash
# Replay one accepted-history operation with its working folders in local temp
# (not Dropbox); keep only the report and log under memory/runs/op_replay.
# Usage: tools/run_history_operation_local.sh <snapshot> <operation> [--profile|--native]
set -u
cd "$(dirname "$0")/.."
M=validation/completed_base_20261012/memory
SNAP=$1; OP=$2; FLAG=${3:-}
NAME=$(date +%Y%m%d-%H%M%S)_${OP}${FLAG//-/_}
WORK="${LOCALAPPDATA:-/tmp}/pymicroglia_op_replay/$NAME"
mkdir -p "$(dirname "$WORK")" "$M/runs/op_replay/$NAME"
python tools/replay_history_operation.py --engine development/cd68_full_red_green/rigid_analysis/A1/tracking/engine \
  --snapshot "$SNAP" --config development/cd68_full_red_green/rigid_batch/.j/A1/inputs/motion_config.json \
  --output "$WORK" --run pymicroglia_95_full_A1 --stem 95_full_A1 --operation "$OP" $FLAG > "$M/runs/op_replay/$NAME.log" 2>&1
code=$?
cp "$WORK/report.json" "$M/runs/op_replay/$NAME/" 2>/dev/null
[ -f "$WORK/profile.pstats" ] && cp "$WORK/profile.pstats" "$M/runs/op_replay/$NAME/"
rm -rf "$WORK"
python -c "import json,sys;r=json.load(open(sys.argv[1]));print(sys.argv[2],r['status'],round(r['seconds'],1),'labels',r['labels_equal'],'ledger',r['unclaimed_equal'],'files',all(r['checks'].values()) if r['checks'] else '-','peakGiB',round(r['peak_resident_bytes']/2**30,2))" "$M/runs/op_replay/$NAME/report.json" "$OP" 2>/dev/null || { echo "$OP exit=$code"; tail -c 800 "$M/runs/op_replay/$NAME.log"; }
