#!/usr/bin/env bash
# launch_capability_eval.sh -- per-pod entrypoint for the DiffRWKV capability
# eval (MMLU / ARC / HellaSwag multichoice, GSM8K / MATH-500 math, HumanEval /
# MBPP code) on qz.
#
# One pod = one multi-GPU node. The pod loads the RELAY checkpoint once per
# GPU-process and scores a round-robin slice of the benchmark (NUM_SHARDS total,
# NGPUS in parallel here). Each (task, condition, shard) triple writes a
# per-shard JSON to OUTDIR on GPFS; merge_eval folds them offline.
#
# This is a DIAGNOSTIC eval (registry execution_class: replicated_shards), not a
# training run. Telemetry is still collected and reported honestly; the
# utilization gate is the human's call at approval time, not auto-enforced.
#
# Env (set at submit time via the CreateJob command):
#   CKPT_DIR   (required) step_*/ dir holding model.pt  (e.g. iter-1 step_00100000)
#   TASK       (required) multichoice | math | code
#   TASK_DIR   (multichoice) dir of pre-tokenized .npz examples
#   TASKS_FILE (math|code)  vendored JSONL task file
#   SUITE      (code)       humaneval | mbpp        (default humaneval)
#   CONDITIONS (multichoice) comma-sep raw/ddpm<N>/ddim<N>/flow<N> (default raw,ddpm100)
#   NUM_SHARDS (default NGPUS) total shards across the whole run
#   SAMPLE_STEPS (default 100) diffusion steps for injected conditions
#   MAX_NEW_TOKENS (default 512) gen budget for math/code
#   TEMPERATURE (default 0.2) / TOP_P (0.95) / TOP_K (50) / REP_PENALTY (1.1)
#   OUTDIR     (required) GPFS dir for per-shard + merged JSONs
#   PROBE_ONLY (default 0) 1 => run only the 4-example smoke probe, then exit
#   NGPUS      (default 4) GPUs on this pod
#
# Repo rule: local work is CPU-only; this script requires CUDA and runs ONLY
# inside an approved qz job.

set -uo pipefail

SCALE_DIR="${DAN_SCALE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --- Environment: mirror the proven training launcher (offline HF, data path). ---
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Defaults (overridable at submit time). ---
NGPUS="${NGPUS:-4}"
NUM_SHARDS="${NUM_SHARDS:-$NGPUS}"
SUITE="${SUITE:-humaneval}"
CONDITIONS="${CONDITIONS:-raw,ddpm100}"
SAMPLE_STEPS="${SAMPLE_STEPS:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-50}"
REP_PENALTY="${REP_PENALTY:-1.1}"
SEED="${SEED:-42}"
PROBE_ONLY="${PROBE_ONLY:-0}"
PROBE_N="${PROBE_N:-4}"
MODEL_KIND="${MODEL_KIND:-relay}"

: "${CKPT_DIR:?CKPT_DIR is required (step_*/ dir with model.pt)}"
: "${TASK:?TASK is required (multichoice|math|code)}"
: "${OUTDIR:?OUTDIR is required (GPFS dir for shard + merged JSONs)}"
if [[ "$TASK" == "multichoice" ]]; then : "${TASK_DIR:?TASK_DIR is required for multichoice}"; fi
if [[ "$TASK" == "math" || "$TASK" == "code" ]]; then : "${TASKS_FILE:?TASKS_FILE is required for math/code}"; fi

mkdir -p "$OUTDIR"
HOST="$(hostname)"
BOOTLOG="$OUTDIR/${HOST}.${TASK}.boot.txt"
RUNLOG="$OUTDIR/${HOST}.${TASK}.run.log"

# --- Self-written boot log (qz GetJobLog is unreliable; GPFS file is the source of truth). ---
{
  echo "==== CAP-EVAL BOOT $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  echo "hostname=$HOST"
  echo "TASK=$TASK  CKPT_DIR=$CKPT_DIR  OUTDIR=$OUTDIR"
  echo "NGPUS=$NGPUS  NUM_SHARDS=$NUM_SHARDS  PROBE_ONLY=$PROBE_ONLY"
  case "$TASK" in
    multichoice) echo "TASK_DIR=$TASK_DIR  CONDITIONS=$CONDITIONS  SAMPLE_STEPS=$SAMPLE_STEPS";;
    math)        echo "TASKS_FILE=$TASKS_FILE  MAX_NEW_TOKENS=$MAX_NEW_TOKENS";;
    code)        echo "TASKS_FILE=$TASKS_FILE  SUITE=$SUITE  MAX_NEW_TOKENS=$MAX_NEW_TOKENS  T=$TEMPERATURE";;
  esac
  echo "---- nvidia-smi -L ----"
  nvidia-smi -L 2>&1 || echo "<nvidia-smi -L failed>"
  echo "---- deps ----"
  python -c "import torch,transformers,omegaconf;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'n_gpu',torch.cuda.device_count())" 2>&1 || echo "<dep import failed>"
  echo "---- ckpt ----"
  ls -la "$CKPT_DIR/model.pt" 2>&1 || echo "<model.pt missing>"
  echo "==== END BOOT ===="
} > "$BOOTLOG" 2>&1
echo "[cap-eval] wrote boot log: $BOOTLOG"

cd "$SCALE_DIR"

# --- Telemetry sampler (per-host files; mirrors launch_dan_n4.sh). ---
TL="${OUTDIR}/telemetry-${HOST}.${TASK}"
python "$SCALE_DIR/qz/gpu_telemetry.py" collect --output-dir "$TL" &
telemetry_pid=$!
cleanup() {
  status=$?
  kill "$telemetry_pid" >/dev/null 2>&1 || true
  wait "$telemetry_pid" >/dev/null 2>&1 || true
  python "$SCALE_DIR/qz/gpu_telemetry.py" summarize \
    --telemetry-csv "$TL/gpu-telemetry.csv" \
    --processes-csv "$TL/gpu-processes.csv" \
    --output "$TL/utilization-summary.json" >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM

# Common run_eval argv tail (provenance stubs come from env / are filled by the
# orchestrator; "unpinned" is the honest default for a first run).
COMMON_ARGS=(
  --ckpt_dir "$CKPT_DIR"
  --task "$TASK"
  --model_kind "$MODEL_KIND"
  --shard 0 --num_shards 1
  --sample_steps "$SAMPLE_STEPS"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE" --top_p "$TOP_P" --top_k "$TOP_K"
  --repetition_penalty "$REP_PENALTY" --seed "$SEED"
  --progress_every 50
)
if [[ -n "${MODEL_DIR:-}" ]]; then
  COMMON_ARGS+=(--model_dir "$MODEL_DIR")
fi

# --- PREFLIGHT: validate every flag this script passes against run_eval's own
# argparse, on CPU, before a single GPU second is spent. A flag typo
# (underscore vs hyphen) otherwise costs a full pod scheduling round-trip to
# discover. Parses the argparse tree statically; never loads the model.
echo "[cap-eval] PREFLIGHT: checking argv against run_eval argparse"
preflight_err="$(python - "$SCALE_DIR" <<'PYEOF' 2>&1
import ast, pathlib, re, sys
scale = pathlib.Path(sys.argv[1])
launcher = scale / "qz" / "launch_capability_eval.sh"
runner = scale / "eval" / "capability" / "run_eval.py"
accepted = set()
for node in ast.walk(ast.parse(runner.read_text())):
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"):
        for a in node.args:
            if isinstance(a, ast.Constant) and str(a.value).startswith("-"):
                accepted.add(a.value)
# Scan only executable lines: comments discuss flag names (including wrong ones)
# and would otherwise self-trip this check.
code_lines = [ln for ln in launcher.read_text().splitlines()
              if not ln.lstrip().startswith("#")]
used = {m for m in re.findall(r"(?<![\w-])--[A-Za-z0-9][\w-]*", "\n".join(code_lines))}
merge_only = {"--shards-glob", "--output-dir", "--bootstrap-samples",
              "--telemetry-csv", "--processes-csv", "--help"}
bad = sorted(f for f in used - merge_only if f not in accepted)
if bad:
    print("unknown run_eval flags in launcher: " + " ".join(bad))
PYEOF
)"
if [[ -n "$preflight_err" ]]; then
  echo "[cap-eval] PREFLIGHT FAILED: $preflight_err" | tee -a "$RUNLOG"
  exit 3
fi
echo "[cap-eval] PREFLIGHT OK." | tee -a "$RUNLOG"

# --- PROBE: tiny smoke that fail-closes if load_relay or scoring breaks. ---
# Scores PROBE_N examples under the first condition (raw for multichoice) to
# validate the full load -> sample -> score -> write path before the sweep.
probe_out="$OUTDIR/${HOST}.${TASK}.probe.json"
echo "[cap-eval] PROBE ($PROBE_N samples) -> $probe_out"
case "$TASK" in
  multichoice)
    first_cond="${CONDITIONS%%,*}"
    python -m eval.capability.run_eval "${COMMON_ARGS[@]}" \
      --task_dir "$TASK_DIR" --conditions "$first_cond" \
      --max_samples "$PROBE_N" --output "$probe_out" 2>&1 | tee -a "$RUNLOG"
    ;;
  math)
    python -m eval.capability.run_eval "${COMMON_ARGS[@]}" \
      --tasks_file "$TASKS_FILE" --arm "math_probe" \
      --max_samples "$PROBE_N" --output "$probe_out" 2>&1 | tee -a "$RUNLOG"
    ;;
  code)
    python -m eval.capability.run_eval "${COMMON_ARGS[@]}" \
      --tasks_file "$TASKS_FILE" --suite "$SUITE" --arm "code_probe" \
      --max_samples "$PROBE_N" --output "$probe_out" 2>&1 | tee -a "$RUNLOG"
    ;;
esac
probe_rc="${PIPESTATUS[0]}"
if [[ "$probe_rc" -ne 0 ]]; then
  echo "[cap-eval] PROBE FAILED rc=$probe_rc — aborting sweep (fail-closed)." | tee -a "$RUNLOG"
  exit "$probe_rc"
fi
echo "[cap-eval] PROBE OK." | tee -a "$RUNLOG"

if [[ "$PROBE_ONLY" == "1" ]]; then
  echo "[cap-eval] PROBE_ONLY=1 — stopping after probe." | tee -a "$RUNLOG"
  exit 0
fi

# --- FULL SWEEP: distribute NUM_SHARDS across NNODES*NGPUS GPUs in parallel. ---
# Multinode: qz injects no rank env, but pod hostnames end in the worker index
# ("...-worker-0-<N>"), so NODE_RANK is derived from $HOST. Global worker w =
# NODE_RANK*NGPUS + g handles shards {w, w+W, w+2W, ...} < NUM_SHARDS with
# W = NNODES*NGPUS. NNODES=1 (the default) reproduces the single-pod behavior
# exactly. Each shard is one run_eval invocation writing its own JSON.
# Processes share nothing (one model load each); CUDA_VISIBLE_DEVICES pins
# each to its GPU.
#
# NOTE: the assignment MUST be `export` — a bare `CUDA_VISIBLE_DEVICES=$g` sets
# only a shell variable, which python never sees, so every shard would land on
# GPU 0 (OOM / serialized). The probe cannot catch this: it is single-process.
NNODES="${NNODES:-1}"
if [[ "$NNODES" -gt 1 ]]; then
  NODE_RANK="${NODE_RANK:-$(echo "$HOST" | grep -oE '[0-9]+$' || echo 0)}"
else
  NODE_RANK=0
fi
WORLD_GPUS=$((NNODES * NGPUS))
echo "[cap-eval] sweep topology: NNODES=$NNODES NODE_RANK=$NODE_RANK NGPUS=$NGPUS (world $WORLD_GPUS, $NUM_SHARDS shards)" | tee -a "$RUNLOG"
declare -a pids=()
for g in $(seq 0 $((NGPUS - 1))); do
  ( export CUDA_VISIBLE_DEVICES=$g
    w=$((NODE_RANK * NGPUS + g))
    # Per-GPU log: NGPUS concurrent `tee -a` into one file interleaves lines and
    # makes a failing shard unattributable. One file per GPU keeps them readable.
    SHARDLOG="$OUTDIR/${HOST}.${TASK}.gpu${g}.log"
    for (( s=w; s<NUM_SHARDS; s+=WORLD_GPUS )); do
      case "$TASK" in
        multichoice)
          out="$OUTDIR/${TASK}.${CONDITIONS//,/-}.shard${s}of${NUM_SHARDS}.json"
          python -m eval.capability.run_eval "${COMMON_ARGS[@]}" \
            --task_dir "$TASK_DIR" --conditions "$CONDITIONS" \
            --shard "$s" --num_shards "$NUM_SHARDS" \
            --output "$out" 2>&1 | tee -a "$SHARDLOG"
          ;;
        math)
          out="$OUTDIR/math.${CONDITIONS//,/-}.shard${s}of${NUM_SHARDS}.json"
          python -m eval.capability.run_eval "${COMMON_ARGS[@]}" \
            --tasks_file "$TASKS_FILE" --conditions "$CONDITIONS" \
            --shard "$s" --num_shards "$NUM_SHARDS" \
            --output "$out" 2>&1 | tee -a "$SHARDLOG"
          ;;
        code)
          out="$OUTDIR/code-${SUITE}.${CONDITIONS//,/-}.shard${s}of${NUM_SHARDS}.json"
          python -m eval.capability.run_eval "${COMMON_ARGS[@]}" \
            --tasks_file "$TASKS_FILE" --suite "$SUITE" --conditions "$CONDITIONS" \
            --shard "$s" --num_shards "$NUM_SHARDS" \
            --output "$out" 2>&1 | tee -a "$SHARDLOG"
          ;;
      esac
    done
  ) &
  pids+=($!)
done

# Wait for all GPU workers; rc = first non-zero (fail-closed on any shard).
rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=$?
done
echo "[cap-eval] sweep exited rc=$rc (run log: $RUNLOG)" | tee -a "$RUNLOG"

# --- MERGE: fold this pod's shard JSONs into a summary (CPU-only; safe on the pod). ---
if [[ "$rc" -eq 0 ]]; then
  # NOTE the glob is ${TASK}* (not ${TASK}.*): code shards are named
  # "code-${SUITE}.…", which a literal "code." prefix silently never matched --
  # the merge then "succeeded" having merged nothing.
  #
  # Multinode: every pod reaches this line, but only the one that can see ALL
  # NUM_SHARDS shard files merges -- a pod that finishes first would otherwise
  # publish a partial summary. GPFS is shared, so the last-finishing pod always
  # sees the complete set.
  shard_count="$(ls "$OUTDIR"/${TASK}*.shard*of${NUM_SHARDS}.json 2>/dev/null | wc -l)"
  expected=$((NUM_SHARDS))
  case "$TASK" in multichoice) : ;; *) : ;; esac
  if [[ "$shard_count" -ge "$expected" ]]; then
    python -m eval.capability.merge_eval \
      --shards-glob "$OUTDIR/${TASK}*.shard*of${NUM_SHARDS}.json" \
      --output-dir "$OUTDIR/merged" --bootstrap-samples 1000 2>&1 | tee -a "$RUNLOG" || true
  else
    echo "[cap-eval] merge skipped on $HOST: $shard_count/$expected shard files present" | tee -a "$RUNLOG"
  fi
fi

exit "$rc"
