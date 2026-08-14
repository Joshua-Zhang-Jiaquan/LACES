#!/usr/bin/env bash
# launch_birwkv_diffusion.sh -- token-level BiRWKV masked-diffusion training/parity.
#
# Modes (MODE env):
#   parity  - run scale/train/parity_birwkv_warmstart.py (1 GPU, exits 0/1)
#   smoke   - 20-step single-node training smoke (checkpoint + resume check)
#   train   - full run (NNODES>1 uses GPFS rendezvous, else torchrun --standalone)
#
# Env:
#   MODE (parity|smoke|train)  MODEL_DIR  TOKEN_DIR  SAVE_ROOT  RUN_NAME
#   NNODES (default 1)  NGPUS (default 8)  MICROBATCH  GRAD_ACCUM  STEPS
#   EXTRA_ARGS (e.g. "--force-forward" for the causal control arm)
#   TOKEN (rdzv token, multi-node only)  LOGDIR

set -uo pipefail

SCALE_DIR="${DAN_SCALE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RDZV_SCRIPT="$SCALE_DIR/qz/rendezvous_gpfs.sh"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODE="${MODE:-smoke}"
NNODES="${NNODES:-1}"
NGPUS="${NGPUS:-8}"
MODEL_DIR="${MODEL_DIR:-/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY/base_models/rwkv7-0.4B-world}"
TOKEN_DIR="${TOKEN_DIR:-/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/fineweb_4096_packed_full}"
SAVE_ROOT="${SAVE_ROOT:-/inspire/hdd/global_user/zhangjiaquan-253108540222/outputs_birwkv_diffusion}"
RUN_NAME="${RUN_NAME:-birwkv-diff-dev}"
MICROBATCH="${MICROBATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
STEPS="${STEPS:-4000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOGDIR="${LOGDIR:-$SAVE_ROOT/logs}"
HOST="$(hostname)"
mkdir -p "$LOGDIR" "$SAVE_ROOT"
BOOTLOG="$LOGDIR/${HOST}.${MODE}.boot.txt"
RUNLOG="$LOGDIR/${HOST}.${MODE}.run.log"

{
  echo "==== BOOT(birwkv-diffusion:$MODE) $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  echo "host=$HOST NNODES=$NNODES NGPUS=$NGPUS MODEL_DIR=$MODEL_DIR"
  echo "TOKEN_DIR=$TOKEN_DIR SAVE_ROOT=$SAVE_ROOT RUN_NAME=$RUN_NAME"
  echo "MICROBATCH=$MICROBATCH GRAD_ACCUM=$GRAD_ACCUM STEPS=$STEPS EXTRA_ARGS=$EXTRA_ARGS"
  nvidia-smi -L 2>&1 || echo "<nvidia-smi failed>"
  echo "---- PREFLIGHT: fla kernel import ----"
  python -c "import torch; from fla.ops.rwkv7 import chunk_rwkv7; from fla.layers.rwkv7 import RWKV7Attention; print('PREFLIGHT OK: torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1
  echo "---- PREFLIGHT: model + trainer import ----"
  python -c "
import sys; sys.path.insert(0, '$SCALE_DIR')
from models.birwkv7_diffusion import BiRWKV7ForMaskedDiffusion, MASK_TOKEN_ID
print('PREFLIGHT OK: birwkv7_diffusion importable, mask_id', MASK_TOKEN_ID)" 2>&1
  FIRST_TOKEN_DIR="${TOKEN_DIR%%,*}"
  ls "$FIRST_TOKEN_DIR"/*.pkl >/dev/null 2>&1 && echo "PREFLIGHT OK: pkl shards present" || echo "PREFLIGHT FAIL: no pkl shards in $FIRST_TOKEN_DIR"
  echo "==== END BOOT ===="
} > "$BOOTLOG" 2>&1
cat "$BOOTLOG"
grep -q "PREFLIGHT FAIL" "$BOOTLOG" && { echo "[launch] FATAL: preflight failed"; exit 2; }

cd "$SCALE_DIR"

case "$MODE" in
  parity)
    python train/parity_birwkv_warmstart.py --model-dir "$MODEL_DIR" 2>&1 | tee "$RUNLOG"
    exit "${PIPESTATUS[0]}"
    ;;
  smoke)
    torchrun --standalone --nproc_per_node="$NGPUS" train/train_birwkv_diffusion.py \
      --model-dir "$MODEL_DIR" --token-dir "$TOKEN_DIR" \
      --save-root "$SAVE_ROOT" --run-name "${RUN_NAME}-smoke" \
      --steps 20 --microbatch "$MICROBATCH" --grad-accum 1 \
      --save-every 10 --log-every 5 --val-every 20 --sampler-every 20 \
      --val-samples 32 --max-samples 2000 $EXTRA_ARGS 2>&1 | tee "$RUNLOG"
    exit "${PIPESTATUS[0]}"
    ;;
  train)
    TRAIN_ARGS=(train/train_birwkv_diffusion.py
      --model-dir "$MODEL_DIR" --token-dir "$TOKEN_DIR"
      --save-root "$SAVE_ROOT" --run-name "$RUN_NAME"
      --steps "$STEPS" --microbatch "$MICROBATCH" --grad-accum "$GRAD_ACCUM")
    if [ "$NNODES" -gt 1 ]; then
      TOKEN="${TOKEN:?TOKEN required for multi-node}"
      RDZV_BASE="${RDZV_BASE:-/inspire/hdd/global_user/zhangjiaquan-253108540222/rdzv}"
      export RDZV_DIR="$RDZV_BASE/rdzv_$TOKEN" RDZV_RUN_ID="$TOKEN" NNODES NGPUS
      mkdir -p "$RDZV_DIR"
      find "$RDZV_DIR" -maxdepth 1 \( -name "host_*" -o -name "name_*" \) -mmin +20 -delete 2>/dev/null || true
      bash "$RDZV_SCRIPT" "${TRAIN_ARGS[@]}" $EXTRA_ARGS 2>&1 | tee "$RUNLOG"
    else
      torchrun --standalone --nproc_per_node="$NGPUS" "${TRAIN_ARGS[@]}" $EXTRA_ARGS 2>&1 | tee "$RUNLOG"
    fi
    exit "${PIPESTATUS[0]}"
    ;;
  *)
    echo "[launch] FATAL: unknown MODE=$MODE"; exit 2 ;;
esac
