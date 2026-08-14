#!/usr/bin/env bash
# rendezvous_gpfs.sh -- GPFS file-rendezvous for TRUE multi-node torchrun on the qz cluster.
#
# WHY THIS EXISTS
#   qz CreateJob has NO native distributed mode and does NOT inject MASTER_ADDR / NODE_RANK
#   (confirmed: 0 of 80 succeeded jobs are multi-node; the reference ic=2 job defaulted
#   MASTER_ADDR=127.0.0.1 and died in ~7s). Every pod shares the cluster-wide GPFS mount,
#   so the pods self-coordinate through a shared directory: each pod drops a UNIQUE file
#   carrying its IP, waits until all NNODES files appear, then EVERY pod sorts the IPs the
#   SAME way -> identical MASTER_ADDR for all + a distinct 0-based NODE_RANK per pod.
#   We then exec the EXACT proven STATIC torchrun form (NOT --rdzv_*), which is the only
#   piece that was ever missing: supply MASTER_ADDR + NODE_RANK and the proven launch works.
#
# CONTRACT (env in)
#   RDZV_DIR     (required)   shared GPFS dir for rendezvous files
#   RDZV_RUN_ID  (required)   validated run identity that scopes peer markers
#   NNODES       (default 2)  number of pods/nodes expected
#   NGPUS        (default 8)  procs per node (only used on the real exec path)
#   MOCK_IP      (optional)   if set, use it as MYIP and SKIP `hostname -I` (drives the local mock test)
#   DRY_RUN      (optional)   if "1", print the RESOLVED line and exit 0 WITHOUT exec'ing torchrun
#   RDZV_TIMEOUT (default 600) seconds to wait for all peers before giving up (exit 2)
#   MASTER_PORT  (default 29500)
# ARGS in: the trainer argv handed straight to torchrun (e.g. train/train_finetune_dit.py --config-name ...)
# EXIT: 0 (dry-run resolved) | torchrun's rc (real) | 2 (timeout, duplicate-IP collapse, or unresolved MYIP)

set -uo pipefail

RDZV_DIR="${RDZV_DIR:?RDZV_DIR must be set (shared GPFS rendezvous dir)}"
RDZV_RUN_ID="${RDZV_RUN_ID:?RDZV_RUN_ID must be set (run-scoped marker identity)}"
NNODES="${NNODES:-2}"
NGPUS="${NGPUS:-8}"
RDZV_TIMEOUT="${RDZV_TIMEOUT:-600}"
MASTER_PORT="${MASTER_PORT:-29500}"

case "$RDZV_RUN_ID" in
  ""|.|..|*[!A-Za-z0-9._-]*)
    echo "[rdzv] FATAL: RDZV_RUN_ID must be one non-empty path-safe component"
    exit 2
    ;;
esac

MARKER_PREFIX="host_${RDZV_RUN_ID}_"

# 1) Resolve MY ip. Honor MOCK_IP (skip hostname -I) for the local mock test.
echo "[rdzv] hostname=$(hostname)"
if [ -n "${MOCK_IP:-}" ]; then
  MYIP="$MOCK_IP"
  echo "[rdzv] MOCK_IP set -> MYIP=$MYIP (skipping hostname -I)"
else
  HOSTI="$(hostname -I)"
  echo "[rdzv] hostname -I: $HOSTI"
  MYIP="$(printf '%s\n' "$HOSTI" | awk '{print $1}')"
fi
echo "[rdzv] MYIP=$MYIP NNODES=$NNODES RDZV_TIMEOUT=$RDZV_TIMEOUT"

if [ -z "$MYIP" ]; then
  echo "[rdzv] FATAL: could not determine MYIP (hostname -I empty and MOCK_IP unset)"
  exit 2
fi

# 2) Ensure the shared rendezvous dir exists.
mkdir -p "$RDZV_DIR"

# 3) Drop a UNIQUE per-pod file so concurrent pods never clobber one another.
printf '%s\n' "$MYIP" > "$RDZV_DIR/${MARKER_PREFIX}${MYIP}.txt"
printf '%s\n' "$(hostname)" > "$RDZV_DIR/name_${RDZV_RUN_ID}_${MYIP}.txt"
echo "[rdzv] wrote run-scoped peer marker"

# 4) WAIT for all peers, with a HARD timeout (never hang forever).
START="$(date +%s)"
while [ "$(ls "$RDZV_DIR"/"${MARKER_PREFIX}"*.txt 2>/dev/null | wc -l)" -lt "$NNODES" ]; do
  sleep 2
  NOW="$(date +%s)"
  ELAPSED="$(( NOW - START ))"
  if [ "$ELAPSED" -gt "$RDZV_TIMEOUT" ]; then
    CUR="$(ls "$RDZV_DIR"/"${MARKER_PREFIX}"*.txt 2>/dev/null | wc -l)"
    IPLIST="$(ls "$RDZV_DIR"/"${MARKER_PREFIX}"*.txt 2>/dev/null | xargs -n1 cat 2>/dev/null | sort -u | tr '\n' ' ')"
    echo "[rdzv] RDZV TIMEOUT: only $CUR/$NNODES peers arrived after ${ELAPSED}s: $IPLIST"
    exit 2
  fi
done

# 5) Collect the agreed, sorted, de-duplicated peer set. Every pod reads the SAME files and
#    sorts them identically -> identical ordering -> identical MASTER_ADDR across all pods.
mapfile -t ALLIPS < <(ls "$RDZV_DIR"/"${MARKER_PREFIX}"*.txt | xargs -n1 cat 2>/dev/null | sort -u)
if [ "${#ALLIPS[@]}" -ne "$NNODES" ]; then
  echo "[rdzv] FATAL: resolved ${#ALLIPS[@]} unique IPs but NNODES=$NNODES (duplicate IP collapsed pods?): ${ALLIPS[*]}"
  exit 2
fi

# 6) MASTER_ADDR = first IP in the sorted set; NODE_RANK = 0-based index of MYIP in that set.
MASTER_ADDR="${ALLIPS[0]}"
NODE_RANK=-1
for i in "${!ALLIPS[@]}"; do
  if [ "${ALLIPS[$i]}" = "$MYIP" ]; then
    NODE_RANK="$i"
    break
  fi
done
if [ "$NODE_RANK" -lt 0 ]; then
  echo "[rdzv] FATAL: my IP $MYIP not present in resolved set: ${ALLIPS[*]}"
  exit 2
fi
echo "[rdzv] MASTER_ADDR=$MASTER_ADDR NODE_RANK=$NODE_RANK (of $NNODES)"

# 7) MASTER_PORT already defaulted at the top (29500 unless overridden).

# 8) Dry-run: print the resolution and exit 0 WITHOUT exec'ing torchrun (this drives the mock test).
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "RESOLVED master=$MASTER_ADDR node_rank=$NODE_RANK nnodes=$NNODES myip=$MYIP allips=${ALLIPS[*]}"
  exit 0
fi

# 9) Real path: exec the EXACT proven STATIC torchrun form. The trainer argv is "$@".
echo "[rdzv] exec torchrun --nnodes=$NNODES --nproc_per_node=$NGPUS --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $*"
exec torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node="$NGPUS" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$@"
