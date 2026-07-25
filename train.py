"""Train State-Hijacking RELAY + Denoiser latent diffusion.

Combined loss:   L = CE(text_logits, text_shifted) + λ_diff * MSE(ε̂, ε)

Single-GPU:
    python projects/LACES/train.py --config-name ... logging.run_name=...

Multi-GPU (DDP via torchrun):
    torchrun --nproc_per_node=8 projects/LACES/train.py --config-name ...
"""
# pyright: reportAny=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportConstantRedefinition=false, reportDeprecated=false, reportImplicitStringConcatenation=false, reportMissingParameterType=false
# pyright: reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportOperatorIssue=false, reportOptionalCall=false, reportOptionalIterable=false, reportOptionalMemberAccess=false, reportPossiblyUnboundVariable=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnnecessaryComparison=false, reportUnusedCallResult=false, reportUnusedImport=false, reportUnusedVariable=false
import datetime
import logging
import os
import sys
import time
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["WANDB_MODE"] = "disabled"

_project_dir = os.path.abspath(os.path.dirname(__file__))
_repo_root = os.path.abspath(os.path.join(_project_dir, os.pardir, os.pardir))
sys.path.insert(0, _repo_root)
sys.path.insert(0, _project_dir)

from model import StateHijackingRELAY
from data_simple import get_simple_dataloaders
from utils import get_lr, parse_dtype


def _init_distributed():
    env_has_ddp = all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if not env_has_ddp:
        return 0, 0, 1, True, False

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=60),
        init_method="env://",
        device_id=torch.device("cuda", local_rank),
    )
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main_process = global_rank == 0
    is_distributed = dist.is_available() and dist.is_initialized()
    return local_rank, global_rank, world_size, is_main_process, is_distributed


def _compatible_trainable_state(state, model=None):
    state = dict(state)
    for key, value in list(state.items()):
        if key.startswith("alpha_predictors."):
            state.setdefault("alpha_heads." + key[len("alpha_predictors."):], value)
    if model is not None:
        current = model.state_dict()
        incompatible_roots = {
            key.split(".", 1)[0]
            for key, value in state.items()
            if key in current and current[key].shape != value.shape
        }
        state = {
            key: value
            for key, value in state.items()
            if key.split(".", 1)[0] not in incompatible_roots
            and (key not in current or current[key].shape == value.shape)
        }
    return state


def _report_load_mismatch(missing, unexpected, prefix=""):
    missing_real = [key for key in missing if not key.startswith("rwkv_model.")]
    unexpected_real = list(unexpected)
    if not missing_real and not unexpected_real:
        return
    print(f"{prefix}load_state_dict: missing={len(missing_real)} unexpected={len(unexpected_real)}")
    alpha_missing = [key for key in missing_real if key.startswith("alpha_heads.")]
    alpha_unexpected = [key for key in unexpected_real if key.startswith("alpha_predictors.")]
    if alpha_missing or alpha_unexpected:
        print(
            f"{prefix}alpha key mismatch: missing_alpha_heads={len(alpha_missing)} "
            f"unexpected_alpha_predictors={len(alpha_unexpected)}"
        )


def _resolve_num_train_steps(config, world_size):
    requested_steps = int(config.training.num_train_steps)
    mode = str(config.training.get("step_scale_mode", "none")).strip().lower()
    if bool(config.training.get("scale_train_steps_by_global_batch", False)) and mode in ("", "none", "off", "false", "0"):
        mode = "global_batch"
    if mode in ("", "none", "off", "false", "0"):
        return requested_steps, None

    aliases = {
        "gpu": "world_size",
        "gpus": "world_size",
        "world": "world_size",
        "sample": "global_batch",
        "samples": "global_batch",
        "token": "global_batch",
        "tokens": "global_batch",
        "batch": "global_batch",
    }
    mode = aliases.get(mode, mode)
    if mode not in ("world_size", "global_batch"):
        raise ValueError(
            "training.step_scale_mode must be one of none, world_size, or global_batch; "
            f"got {mode!r}."
        )

    import math

    reference_world_size = int(config.training.get("step_scale_reference_world_size", config.training.get("world_size", 1)))
    reference_batch_size = int(config.training.get("step_scale_reference_train_batch_size", config.training.train_batch_size))
    actual_batch_size = int(config.training.train_batch_size)
    if reference_world_size <= 0 or reference_batch_size <= 0 or actual_batch_size <= 0 or world_size <= 0:
        raise ValueError(
            "Step scaling requires positive reference/actual world sizes and batch sizes: "
            f"reference_world_size={reference_world_size}, reference_batch_size={reference_batch_size}, "
            f"world_size={world_size}, actual_batch_size={actual_batch_size}."
        )

    if mode == "world_size":
        reference_units = reference_world_size
        actual_units = world_size
    else:
        reference_units = reference_world_size * reference_batch_size
        actual_units = world_size * actual_batch_size

    scaled_steps = max(1, int(math.ceil(requested_steps * reference_units / actual_units)))
    return scaled_steps, {
        "mode": mode,
        "requested_steps": requested_steps,
        "scaled_steps": scaled_steps,
        "reference_world_size": reference_world_size,
        "reference_batch_size": reference_batch_size,
        "reference_units": reference_units,
        "actual_world_size": world_size,
        "actual_batch_size": actual_batch_size,
        "actual_units": actual_units,
    }


# This is only the direct-entry default. Launch scripts pass --config-name
# explicitly; e.g. run_plan_b_13.3B.sh overrides this with the 13.3B config.
@hydra.main(config_path="configs",
            config_name="rwkv_relay_0.4B_state_relay_vae32",
            version_base="1.1")
def main(config):
    local_rank, global_rank, world_size, is_main_process, is_distributed = \
        _init_distributed()

    seed = config.training.seed + global_rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")

    dtype = parse_dtype(config.training.dtype)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if is_main_process:
        print(f"DDP: rank={global_rank}, world_size={world_size}")
        print(f"Using device={device} dtype={dtype}")

    rwkv_path = config.model.rwkv_local_path
    if is_main_process:
        print(f"Loading frozen RWKV-7 from {rwkv_path} ...")
    rwkv = AutoModelForCausalLM.from_pretrained(
        rwkv_path, torch_dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        rwkv_path, trust_remote_code=True, local_files_only=True
    )

    model = StateHijackingRELAY(
        config=config.model,
        rwkv_model=rwkv,
        vocab_size=len(tokenizer),
        latent_dim=int(config.model.latent_dim),
        n_basis=int(config.model.get("n_basis", 16)),
        denoiser_hidden=int(config.model.get("denoiser_hidden", config.model.get("dit_hidden", 256))),
        denoiser_depth=int(config.model.get("denoiser_depth", config.model.get("dit_depth", 4))),
        denoiser_num_heads=int(config.model.get("denoiser_num_heads", config.model.get("dit_num_heads", 4))),
        denoiser_num_tokens=int(config.model.get("denoiser_num_tokens", config.model.get("dit_num_tokens", 4))),
        encoder_type=str(config.model.get("encoder_type", "mlp")),
        alpha_type=str(config.model.get("alpha_type", "linear")),
        alpha_hidden=int(config.model.get("alpha_hidden", 256)),
        latent_stats_path=config.model.get("latent_stats_path", None),
    ).to(device)

    for p in model.parameters():
        if p.requires_grad and p.dtype != dtype and p.dtype.is_floating_point:
            if p.numel() == 1:
                continue
            p.data = p.data.to(dtype)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    if is_main_process:
        print(f"\n{'='*60}")
        print(f"State-Hijack + Denoiser RELAY initialized")
        print(f"  num_layers     : {model.num_layers}")
        print(f"  hidden_size    : {model.hidden_size}")
        print(f"  n_heads / head : {model.num_heads} / {model.head_dim}")
        print(f"  latent_dim     : {model.latent_dim}")
        print(f"  n_basis        : {model.n_basis}")
        print(f"  Denoiser hidden/dep : {model.latent_denoiser.hidden_size} / "
              f"{len(model.latent_denoiser.blocks)}")
        print(f"  trainable      : {n_trainable/1e6:.1f}M / {n_total/1e9:.2f}B "
              f"({100.0*n_trainable/n_total:.2f}%)")
        print(f"  world_size     : {world_size}")
        print(f"{'='*60}\n")

    if is_main_process:
        print("Loading data ...")
    train_loader, _ = get_simple_dataloaders(config, tokenizer)

    num_steps, step_scale_info = _resolve_num_train_steps(config, world_size)
    save_dir = Path(config.logging.save_dir) / config.logging.run_name
    if is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train.log"
    logger = logging.getLogger("train_statediffrwkv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    if is_main_process:
        fh = logging.FileHandler(str(log_path))
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
        logger.addHandler(fh)
        logger.info(f"Training: {config.logging.run_name}")
        logger.info(f"Config label: {config.get('config_name', 'unknown_config')}")
        logger.info(f"Resolved RWKV path: {config.model.rwkv_local_path}")
        logger.info(
            f"Resolved backbone: layers={model.num_layers} hidden={model.hidden_size} "
            f"heads={model.num_heads} head_dim={model.head_dim}"
        )
        logger.info(f"Resolved latent RWKV: {model.latent_rwkv_evidence}")
        logger.info(
            f"Resolved training: stage={config.training.get('stage', 'unknown')} "
            f"gen_type={config.training.get('gen_type', 'unknown')} "
            f"batch={config.training.train_batch_size} lr={config.optimizer.lr}"
        )
        logger.info(f"Save dir: {save_dir}")
        logger.info(f"DDP: world_size={world_size}")
        if step_scale_info is not None:
            logger.info(
                "Step scaling: "
                f"mode={step_scale_info['mode']} requested_steps={step_scale_info['requested_steps']} "
                f"scaled_steps={step_scale_info['scaled_steps']} "
                f"reference_global_batch={step_scale_info['reference_units']} "
                f"actual_global_batch={step_scale_info['actual_units']} "
                f"reference_world_size={step_scale_info['reference_world_size']} "
                f"actual_world_size={step_scale_info['actual_world_size']}"
            )
    save_every = int(config.training.save_every_n_steps)
    log_freq = int(config.logging.log_freq)
    diff_w = float(config.loss.get("diff_loss_weight", 1.0))
    text_w = float(config.loss.get("text_loss_weight", 1.0))
    traj_delta_w = float(config.loss.get("trajectory_delta_loss_weight", 0.0))
    s2_align_w = float(config.loss.get("s2_align_loss_weight", 0.0))
    s2_coadapt_ce_w = float(config.loss.get("s2_coadapt_ce_loss_weight", 0.0))
    s2_coadapt_ce_checkpoint = bool(config.training.get("s2_coadapt_ce_checkpoint", True))
    s1_ldlm_mse_w = float(config.loss.get("s1_ldlm_mse_loss_weight", 0.0))
    s1_ldlm_max_sigma = float(config.training.get("s1_ldlm_max_sigma", 1.0))
    simcot_step_w = float(config.loss.get("simcot_step_loss_weight", 0.0))
    s2_unfreeze_s1 = bool(config.training.get("s2_unfreeze_s1", False))
    freeze_s2 = bool(config.training.get("freeze_s2", False))
    s1_noise_cond = bool(config.training.get("s1_noise_cond", False))
    s1_global_anchor = bool(config.training.get("s1_global_anchor", False))
    s1_xchunk = bool(config.training.get("s1_xchunk", False))
    s1_noise_cond_max_sigma = float(config.training.get("s1_noise_cond_max_sigma", 1.0))
    state_anchor_w = float(config.loss.get("state_anchor_loss_weight", 0.0))
    s2_align_start = int(config.loss.get("s2_align_start_steps", 0))
    s2_align_warmup = int(config.loss.get("s2_align_warmup_steps", 0))
    s2_align_max_chunks = int(config.loss.get("s2_align_max_chunks", 2))
    s2_align_temperature = float(config.loss.get("s2_align_temperature", 1.0))
    tea_warmup = int(config.loss.get("teacher_force_warmup", 20000))
    train_stage = int(config.training.get("stage", 0))
    model._training_stage = train_stage
    gen_type = str(config.training.get("gen_type", "ddpm"))
    model._gen_type = gen_type
    prefix_suffix_s1 = bool(config.training.get("prefix_suffix_s1", False))
    prefix_suffix_s2 = bool(config.training.get("prefix_suffix_s2", False))
    prefix_suffix_trajectory_s1 = bool(config.training.get("prefix_suffix_trajectory_s1", False))
    prefix_suffix_trajectory_s2 = bool(config.training.get("prefix_suffix_trajectory_s2", False))
    min_prefix_len = int(config.training.get("prefix_suffix_min_prefix", 32))
    min_suffix_len = int(config.training.get("prefix_suffix_min_suffix", 32))
    cfg_drop_prob = float(config.training.get("cfg_drop_prob", 0.0))
    sft_response_only = bool(config.training.get("sft_response_only", False))
    s2_single_z_sft = bool(config.training.get("s2_single_z_sft", False))
    s2_trajectory_sft = bool(config.training.get("s2_trajectory_sft", False))
    trajectory_mode = bool(config.model.get("trajectory_mode", False))
    trajectory_condition_first = bool(config.training.get("trajectory_condition_first", False))
    trajectory_s1_mode = str(config.model.get("trajectory_s1_mode", "independent"))
    s1_self_forcing = bool(config.training.get("s1_self_forcing", False))
    self_forcing_prob = float(config.training.get("self_forcing_prob", 0.5))
    self_forcing_prob_warmup = int(config.training.get("self_forcing_prob_warmup", 1000))
    self_forcing_bptt_chunks = max(1, int(config.training.get("self_forcing_bptt_chunks", 1)))
    self_forcing_ban_eos = bool(config.training.get("self_forcing_ban_eos", False))
    self_forcing_eos_id = int(getattr(tokenizer, "eos_token_id", None) or 0)
    if s1_self_forcing and not (train_stage == 1 and (trajectory_mode or prefix_suffix_trajectory_s1)):
        raise ValueError(
            "training.s1_self_forcing=true is only supported for stage-1 trajectory S1 rollouts "
            "(trajectory_mode or +training.prefix_suffix_trajectory_s1=true)."
        )
    if s1_self_forcing and trajectory_s1_mode not in ("transformer", "rwkv", "birwkv"):
        raise ValueError("training.s1_self_forcing=true requires model.trajectory_s1_mode=transformer|rwkv|birwkv.")
    if s2_trajectory_sft:
        if train_stage != 2 or not prefix_suffix_trajectory_s2:
            raise ValueError(
                "training.s2_trajectory_sft=true expects training.stage=2 with "
                "+training.prefix_suffix_trajectory_s2=true."
            )
    if s2_single_z_sft:
        if train_stage != 2 or not prefix_suffix_s2:
            raise ValueError(
                "training.s2_single_z_sft=true expects training.stage=2 with "
                "+training.prefix_suffix_s2=true."
            )
    if sft_response_only:
        incompatible = (
            train_stage in (2, 3)
            or prefix_suffix_s1
            or prefix_suffix_s2
            or prefix_suffix_trajectory_s1
            or prefix_suffix_trajectory_s2
            or trajectory_mode
        )
        if incompatible:
            raise ValueError(
                "training.sft_response_only=true expects plain stage 0/1 state-injection CE "
                "with response_mask or prompt_lengths; do not combine it with S2/S3, "
                "trajectory_mode, or prefix/suffix training flags."
            )
    if train_stage == 1:
        for p in model.parameters():
            p.requires_grad = False
        if trajectory_mode and trajectory_s1_mode in ("transformer", "rwkv", "birwkv"):
            for p in model.trajectory_state_decoder.parameters():
                p.requires_grad = True
        else:
            for p in model.alpha_heads.parameters():
                p.requires_grad = True
            if model.alpha_trunk is not None:
                for p in model.alpha_trunk.parameters():
                    p.requires_grad = True
        model.state_basis.requires_grad = True
        model.state_scale.requires_grad = True
        if getattr(model, "use_learnable_blend", False) and getattr(model, "blend_gate_logit", None) is not None:
            model.blend_gate_logit.requires_grad = True
    elif train_stage == 2:
        denoiser_module = model.trajectory_denoiser if (trajectory_mode or prefix_suffix_trajectory_s2) else model.latent_denoiser
        denoiser_ids = {id(p) for p in denoiser_module.parameters()}
        s1_modules = []
        if s2_unfreeze_s1:
            for _mn in ("encoder_trunk", "mu_head", "logvar_head", "alpha_heads", "alpha_trunk"):
                _m = getattr(model, _mn, None)
                if _m is not None:
                    s1_modules.append(_m)
        s1_ids = {id(p) for m in s1_modules for p in m.parameters()}
        for p in model.parameters():
            if id(p) not in denoiser_ids and id(p) not in s1_ids and p.requires_grad:
                p.requires_grad = False
        if not freeze_s2:
            for p in denoiser_module.parameters():
                p.requires_grad = True
        else:
            for p in denoiser_module.parameters():
                p.requires_grad = False
        if s2_unfreeze_s1:
            for m in s1_modules:
                for p in m.parameters():
                    p.requires_grad = True
            model.state_basis.requires_grad = True
            model.state_scale.requires_grad = True
    elif train_stage == 3:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.encoder_trunk.parameters():
            p.requires_grad = True
        for p in model.mu_head.parameters():
            p.requires_grad = True
        for p in model.logvar_head.parameters():
            p.requires_grad = True
        for p in model.aux_decoder.parameters():
            p.requires_grad = True

    if sft_response_only:
        for p in model.latent_denoiser.parameters():
            p.requires_grad = False
        if getattr(model, "trajectory_denoiser", None) is not None:
            for p in model.trajectory_denoiser.parameters():
                p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.optimizer.lr),
        weight_decay=float(config.optimizer.weight_decay),
        betas=(float(config.optimizer.beta1), float(config.optimizer.beta2)),
        eps=float(config.optimizer.eps),
    )

    # ── Resume from previous stage checkpoint ──
    resume_path = config.training.get("resume", None)
    if resume_path:
        ckpt_path = Path(resume_path) / "model.pt"
        if not ckpt_path.exists():
            ckpt_path = Path(resume_path)
        if is_main_process:
            print(f"Resuming from {ckpt_path} ...")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(
            _compatible_trainable_state(ckpt["trainable_state"], model), strict=False
        )
        if is_main_process:
            print(f"  Loaded checkpoint at step {ckpt.get('step', '?')} "
                  f"({len(ckpt['trainable_state'])} keys)")
            _report_load_mismatch(missing, unexpected, prefix="  ")

    if is_distributed:
        ddp_find_unused = config.training.get("ddp_find_unused_parameters", None)
        if ddp_find_unused is None:
            ddp_find_unused = train_stage == 2
        else:
            ddp_find_unused = bool(ddp_find_unused)
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_find_unused,
            static_graph=not ddp_find_unused,
            gradient_as_bucket_view=True,
        )
    else:
        ddp_model = model

    if is_main_process:
        if step_scale_info is None:
            print(f"Training: {num_steps} steps  "
                  f"(text_w={text_w}, diff_w={diff_w})")
        else:
            print(
                f"Training: {num_steps} scaled steps from requested "
                f"{step_scale_info['requested_steps']} "
                f"(mode={step_scale_info['mode']}, "
                f"reference_global_batch={step_scale_info['reference_units']}, "
                f"actual_global_batch={step_scale_info['actual_units']})  "
                f"(text_w={text_w}, diff_w={diff_w})"
            )
        print(f"Teacher-force warmup: {tea_warmup} steps")
        if (train_stage == 1 and (prefix_suffix_s1 or prefix_suffix_trajectory_s1)) or (train_stage == 2 and (prefix_suffix_s2 or prefix_suffix_trajectory_s2)):
            print(
                f"Prefix/suffix S{train_stage} enabled: "
                f"min_prefix={min_prefix_len}, min_suffix={min_suffix_len}"
            )
        if train_stage == 2 and (prefix_suffix_s2 or prefix_suffix_trajectory_s2):
            print(f"CFG condition dropout: {cfg_drop_prob}")
        if sft_response_only:
            print("Response-only SFT enabled: CE loss is masked to response tokens")
        if s2_trajectory_sft:
            print("S2 trajectory SFT enabled: prompt_lengths provide the supervised prefix/response split")
        if s2_single_z_sft:
            print("S2 single-z SFT enabled: prompt_lengths provide the supervised prefix/response split")
        print(
            "S1 self-forcing: "
            f"enabled={s1_self_forcing} target_prob={self_forcing_prob} "
            f"warmup={self_forcing_prob_warmup} bptt_chunks={self_forcing_bptt_chunks} "
            f"ban_eos={self_forcing_ban_eos} eos_id={self_forcing_eos_id}"
        )
        print(f"DDP model ready" if is_distributed else "Single-GPU mode")

    # ── Sanity check: vanilla RWKV CE ──
    if is_main_process:
        print("\nSanity check (vanilla frozen RWKV, no injection)...")
        sanity_batch = next(iter(train_loader))
        sanity_tokens = sanity_batch["input_ids"].to(device)
        sanity_mask = sanity_batch.get("attention_mask")
        if sanity_mask is not None:
            sanity_mask = sanity_mask.to(device).bool()
            actual_lens = sanity_mask.sum(dim=1)
            max_len = actual_lens.max().item()
            if max_len < sanity_tokens.shape[1]:
                sanity_tokens = sanity_tokens[:, :max_len]
                sanity_mask = sanity_mask[:, :max_len]
        with torch.no_grad():
            vanilla = rwkv(input_ids=sanity_tokens, attention_mask=sanity_mask,
                           return_dict=True).logits
            vl = vanilla[:, :-1, :].float()
            vt = sanity_tokens[:, 1:]
            if sanity_mask is not None:
                vm = sanity_mask[:, 1:].bool()
                vc = F.cross_entropy(vl.reshape(-1, vl.shape[-1]), vt.reshape(-1),
                                     reduction="none")
                vanilla_ce = (vc * vm.reshape(-1)).sum() / vm.sum().clamp(min=1)
            else:
                vanilla_ce = F.cross_entropy(vl.reshape(-1, vl.shape[-1]),
                                             vt.reshape(-1))
        print(f"  vanilla frozen RWKV CE    : {vanilla_ce.item():.4f}\n")

    if is_distributed:
        dist.barrier()

    step = 0
    start_time = time.time()
    log_window_start = start_time
    log_buf = []
    train_iter = iter(train_loader)
    ddp_model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def save_checkpoint(step_value):
        if not is_main_process:
            return
        try:
            ck_dir = save_dir / f"step_{step_value:08d}"
            ck_dir.mkdir(parents=True, exist_ok=True)
            raw_to_save = ddp_model.module if is_distributed else ddp_model
            trainable_state = {
                name: p.detach().cpu()
                for name, p in raw_to_save.state_dict().items()
                if not name.startswith("rwkv_model.")
            }
            payload = {
                "trainable_state": trainable_state,
                "state_basis": raw_to_save.state_basis.data.cpu(),
                "state_scale": raw_to_save.state_scale.data.cpu(),
                "step": step_value,
                "config": OmegaConf.to_yaml(config),
            }
            torch.save(payload, ck_dir / "model.pt")
            msg = f"  saved ckpt: {ck_dir}/model.pt"
            print(msg, flush=True)
            logger.info(msg)
        except Exception as ck_err:
            msg = f"  [warn] checkpoint save failed at step {step_value}: {type(ck_err).__name__}: {ck_err}"
            print(msg, flush=True)
            logger.warning(msg)

    while step < num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        text_tokens = batch["input_ids"].to(device)
        attn_mask = batch.get("attention_mask")
        response_mask = batch.get("response_mask")
        prompt_lengths = batch.get("prompt_lengths")
        z_0_external = None
        if bool(config.data.get("use_external_latents", False)):
            _lat = batch.get("latent")
            if _lat is not None:
                z_0_external = _lat.to(device)
        if prompt_lengths is not None:
            prompt_lengths = prompt_lengths.to(device)
        if response_mask is not None:
            response_mask = response_mask.to(device).bool()
        if attn_mask is not None:
            attn_mask = attn_mask.to(device).bool()
            actual_lens = attn_mask.sum(dim=1)
            max_len = actual_lens.max().item()
            if max_len < text_tokens.shape[1]:
                text_tokens = text_tokens[:, :max_len]
                attn_mask = attn_mask[:, :max_len]
                if response_mask is not None:
                    response_mask = response_mask[:, :max_len]

        lr = get_lr(config, float(config.optimizer.lr), step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        raw_model = ddp_model.module if is_distributed else ddp_model
        raw_model._prefix_suffix_s1 = train_stage == 1 and prefix_suffix_s1
        raw_model._prefix_suffix_s2 = train_stage == 2 and prefix_suffix_s2
        raw_model._prefix_suffix_trajectory_s1 = train_stage == 1 and prefix_suffix_trajectory_s1
        raw_model._prefix_suffix_trajectory_s2 = train_stage == 2 and prefix_suffix_trajectory_s2
        raw_model._cfg_drop_prob = cfg_drop_prob
        raw_model._trajectory_mode = trajectory_mode and not (raw_model._prefix_suffix_trajectory_s1 or raw_model._prefix_suffix_trajectory_s2)
        raw_model._trajectory_condition_first = trajectory_condition_first
        if s1_self_forcing:
            if self_forcing_prob_warmup > 0:
                self_forcing_p_effective = self_forcing_prob * min(1.0, step / self_forcing_prob_warmup)
            else:
                self_forcing_p_effective = self_forcing_prob
        else:
            self_forcing_p_effective = 0.0
        raw_model._s1_self_forcing = bool(s1_self_forcing and train_stage == 1)
        raw_model._s1_self_forcing_prob_effective = float(self_forcing_p_effective)
        raw_model._s1_self_forcing_bptt_chunks = int(self_forcing_bptt_chunks)
        raw_model._s1_self_forcing_ban_eos = bool(self_forcing_ban_eos)
        raw_model._s1_self_forcing_eos_id = int(self_forcing_eos_id)
        if s2_align_w > 0.0 and train_stage == 2 and prefix_suffix_trajectory_s2:
            if step < s2_align_start:
                s2_align_active_w = 0.0
            elif s2_align_warmup > 0:
                s2_align_active_w = s2_align_w * min(1.0, (step - s2_align_start + 1) / s2_align_warmup)
            else:
                s2_align_active_w = s2_align_w
        else:
            s2_align_active_w = 0.0
        raw_model._s2_align_enabled = s2_align_active_w > 0.0
        raw_model._s2_align_max_chunks = s2_align_max_chunks
        raw_model._s2_align_temperature = s2_align_temperature
        raw_model._s2_coadapt_ce_enabled = bool(s2_coadapt_ce_w > 0.0 and train_stage == 2 and (prefix_suffix_trajectory_s2 or prefix_suffix_s2))
        raw_model._s2_coadapt_ce_checkpoint = s2_coadapt_ce_checkpoint
        raw_model._s1_ldlm_mse_enabled = bool(s1_ldlm_mse_w > 0.0 and train_stage == 2 and prefix_suffix_trajectory_s2)
        raw_model._s1_ldlm_max_sigma = s1_ldlm_max_sigma
        raw_model._simcot_enabled = bool(simcot_step_w > 0.0 and train_stage == 2 and prefix_suffix_trajectory_s2)
        raw_model._s1_noise_cond_enabled = bool(s1_noise_cond and train_stage == 1)
        raw_model._s1_global_anchor_enabled = bool(s1_global_anchor)
        raw_model._s1_xchunk_enabled = bool(s1_xchunk)
        raw_model._s1_noise_cond_max_sigma = s1_noise_cond_max_sigma
        split_idx = None
        if raw_model._prefix_suffix_s1 or raw_model._prefix_suffix_s2 or raw_model._prefix_suffix_trajectory_s1 or raw_model._prefix_suffix_trajectory_s2:
            seq_len = text_tokens.shape[1]
            if s2_trajectory_sft or s2_single_z_sft:
                if prompt_lengths is None:
                    raise ValueError(
                        "S2 SFT requires prompt_lengths in each sample. "
                        "Use scripts/preprocess/preprocess_response_sft.py or an equivalent format."
                    )
                prompt_lengths_flat = prompt_lengths.reshape(-1).long()
                if prompt_lengths_flat.numel() != text_tokens.shape[0]:
                    raise ValueError(
                        "prompt_lengths must contain exactly one split index per batch sample."
                    )
                if attn_mask is not None:
                    real_lengths = attn_mask.long().sum(dim=1)
                else:
                    real_lengths = torch.full_like(prompt_lengths_flat, seq_len)
                min_suffix_tokens = int(config.model.get("trajectory_chunk_size", 32)) if s2_trajectory_sft else 1
                max_prompt_lengths = real_lengths - min_suffix_tokens
                invalid_prompt_lengths = (
                    (prompt_lengths_flat < 1)
                    | (max_prompt_lengths < 1)
                    | (prompt_lengths_flat > max_prompt_lengths)
                )
                if invalid_prompt_lengths.any().item():
                    bad_idx = int(invalid_prompt_lengths.nonzero(as_tuple=False)[0].item())
                    raise ValueError(
                        "S2 SFT requires each prompt_lengths value "
                        f"to leave at least {min_suffix_tokens} response token(s); "
                        f"sample {bad_idx} has prompt_lengths={int(prompt_lengths_flat[bad_idx].item())}, "
                        f"real_length={int(real_lengths[bad_idx].item())}, "
                        f"min_suffix_tokens={min_suffix_tokens}."
                    )
                split_idx = prompt_lengths_flat.to(device=text_tokens.device)
            else:
                if attn_mask is not None:
                    min_real_len = int(attn_mask.sum(dim=1).min().item())
                else:
                    min_real_len = seq_len
                usable_len = max(3, min(seq_len, min_real_len))
                lo = min(max(1, min_prefix_len), max(1, usable_len - 1))
                hi = max(lo, usable_len - max(1, min_suffix_len))
                if hi <= lo:
                    split_idx = max(1, usable_len // 2)
                else:
                    split_idx = random.randint(lo, hi)
            raw_model._prefix_suffix_split_idx = split_idx

        if train_stage == 3:
            raw_model._teacher_force_ratio = 0.0
            text_logits, eps_pred, eps_target, extras = ddp_model(
                text_tokens, attention_mask=attn_mask
            )
            recon_loss = extras.pop("recon_loss", torch.tensor(0.0, device=device))
            ce_loss = torch.tensor(0.0, device=device)
            diff_loss = torch.tensor(0.0, device=device)
        else:
            if sft_response_only:
                ratio = 1.0
            elif train_stage == 0 and tea_warmup > 0:
                import math
                ratio = max(0.0, math.cos(0.5 * math.pi * min(step / tea_warmup, 1.0)))
            else:
                ratio = 1.0
            raw_model._teacher_force_ratio = ratio
            text_logits, eps_pred, eps_target, extras = ddp_model(
                text_tokens, attention_mask=attn_mask, z_0_external=z_0_external
            )

        s2_align_loss = extras.pop(
            "_s2_align_loss_tensor",
            torch.tensor(0.0, device=device),
        )
        s2_coadapt_ce_loss = extras.pop(
            "_s2_coadapt_ce_loss_tensor",
            torch.tensor(0.0, device=device),
        )
        s1_ldlm_mse_loss = extras.pop(
            "_s1_ldlm_mse_loss_tensor",
            torch.tensor(0.0, device=device),
        )
        simcot_step_loss = extras.pop(
            "_simcot_step_loss_tensor",
            torch.tensor(0.0, device=device),
        )
        state_anchor_loss = extras.pop(
            "_state_anchor_loss_tensor",
            torch.tensor(0.0, device=device),
        )
        state_anchor_active_w = state_anchor_w if train_stage == 1 else 0.0
        trajectory_loss_mask = extras.pop("_trajectory_loss_mask", None)

        # CE on text (shift-1) — skip for S2 (only trains Denoiser with MSE)
        if train_stage not in (2, 3):
            if raw_model._prefix_suffix_trajectory_s1:
                chunk_size = int(config.model.get("trajectory_chunk_size", 32))
                horizon = int(config.model.get("trajectory_horizon", 16))
                suffix_tokens = text_tokens[:, split_idx:]
                usable = min(suffix_tokens.shape[1], chunk_size * horizon)
                usable = (usable // chunk_size) * chunk_size
                target_chunks = suffix_tokens[:, :usable].reshape(-1, chunk_size)
                loss_logits = text_logits[:, :-1, :].float()
                loss_target = target_chunks[:, 1:]
                if attn_mask is not None:
                    suffix_mask = attn_mask[:, split_idx:]
                    loss_mask = suffix_mask[:, :usable].reshape(-1, chunk_size)[:, 1:].bool()
                else:
                    loss_mask = torch.ones_like(loss_target, dtype=torch.bool)
            elif trajectory_mode:
                chunk_size = int(config.model.get("trajectory_chunk_size", 32))
                horizon = int(config.model.get("trajectory_horizon", 16))
                usable = min(text_tokens.shape[1], chunk_size * horizon)
                usable = (usable // chunk_size) * chunk_size
                target_chunks = text_tokens[:, :usable].reshape(-1, chunk_size)
                loss_logits = text_logits[:, :-1, :].float()
                loss_target = target_chunks[:, 1:]
                if attn_mask is not None:
                    loss_mask = attn_mask[:, :usable].reshape(-1, chunk_size)[:, 1:].bool()
                else:
                    loss_mask = torch.ones_like(loss_target, dtype=torch.bool)
            elif raw_model._prefix_suffix_s1:
                loss_logits = text_logits[:, :-1, :].float()
                loss_target = text_tokens[:, split_idx:]
                if attn_mask is not None:
                    loss_mask = attn_mask[:, split_idx:].bool()
                else:
                    loss_mask = torch.ones_like(loss_target, dtype=torch.bool)
            else:
                loss_logits = text_logits[:, :-1, :].float()
                loss_target = text_tokens[:, 1:]
                loss_mask = attn_mask[:, 1:].bool() if attn_mask is not None else None
            if sft_response_only:
                if response_mask is None:
                    raise ValueError(
                        "training.sft_response_only=true requires each sample to provide "
                        "response_mask or prompt_lengths."
                    )
                response_loss_mask = response_mask[:, 1:].bool()
                if loss_mask is None:
                    loss_mask = response_loss_mask
                else:
                    loss_mask = loss_mask & response_loss_mask
                if loss_mask.sum().item() == 0:
                    raise ValueError(
                        "training.sft_response_only=true produced an empty response-token "
                        "loss mask for this batch. Check response_mask or prompt_lengths."
                    )
                extras["response_tokens"] = float(loss_mask.sum().detach().item())
            if loss_mask is not None:
                ce = F.cross_entropy(
                    loss_logits.reshape(-1, loss_logits.shape[-1]),
                    loss_target.reshape(-1),
                    reduction="none",
                )
                ce_loss = (ce * loss_mask.reshape(-1)).sum() / loss_mask.sum().clamp(min=1)
            else:
                ce_loss = F.cross_entropy(
                    loss_logits.reshape(-1, loss_logits.shape[-1]),
                    loss_target.reshape(-1),
                )

        # Denoiser diffusion MSE
        if sft_response_only:
            diff_loss = torch.tensor(0.0, device=device)
        elif trajectory_loss_mask is not None and train_stage not in (1, 3):
            mask = trajectory_loss_mask.to(device=eps_pred.device, dtype=eps_pred.dtype)
            diff_sq = (eps_pred.float() - eps_target.float()).pow(2)
            diff_loss = (diff_sq * mask.float()).sum() / mask.expand_as(diff_sq).sum().clamp(min=1)
        else:
            diff_loss = F.mse_loss(eps_pred.float(), eps_target.float()) if train_stage not in (1, 3) else torch.tensor(0.0, device=device)

        traj_delta_loss = torch.tensor(0.0, device=device)
        if (
            train_stage == 2
            and traj_delta_w > 0.0
            and trajectory_mode
            and gen_type in ("flow", "rf")
            and eps_pred.ndim == 3
            and eps_pred.shape[1] > 1
        ):
            pred_z = eps_pred.float()
            target_z = eps_target.float()
            if trajectory_loss_mask is not None:
                delta_mask = (trajectory_loss_mask[:, 1:] * trajectory_loss_mask[:, :-1]).float()
                pred_delta = pred_z[:, 1:] - pred_z[:, :-1]
                target_delta = target_z[:, 1:] - target_z[:, :-1]
                delta_sq = (pred_delta - target_delta).pow(2)
                traj_delta_loss = (delta_sq * delta_mask).sum() / delta_mask.expand_as(delta_sq).sum().clamp(min=1)
            else:
                traj_delta_loss = F.mse_loss(
                    pred_z[:, 1:] - pred_z[:, :-1],
                    target_z[:, 1:] - target_z[:, :-1],
                )
            if pred_z.shape[1] > 2:
                pred_dd = pred_z[:, 2:] - 2.0 * pred_z[:, 1:-1] + pred_z[:, :-2]
                target_dd = target_z[:, 2:] - 2.0 * target_z[:, 1:-1] + target_z[:, :-2]
                if trajectory_loss_mask is not None:
                    dd_mask = (trajectory_loss_mask[:, 2:] * trajectory_loss_mask[:, 1:-1] * trajectory_loss_mask[:, :-2]).float()
                    dd_sq = (pred_dd - target_dd).pow(2)
                    traj_delta_loss = traj_delta_loss + (dd_sq * dd_mask).sum() / dd_mask.expand_as(dd_sq).sum().clamp(min=1)
                else:
                    traj_delta_loss = traj_delta_loss + F.mse_loss(pred_dd, target_dd)

        # KL (variational only)
        kl_per_dim = extras.pop("kl_per_dim", None)
        kl_value = 0.0
        eff_kl_w = 0.0
        if kl_per_dim is not None and train_stage in (0, 3) and not sft_response_only:
            kl_weight = float(config.loss.get("kl_weight", 0.0))
            free_bits = float(config.loss.get("free_bits", 0.0))
            kl_warmup = int(config.loss.get("kl_warmup_steps", 0))
            warmup_factor = min(1.0, step / kl_warmup) if kl_warmup > 0 else 1.0
            eff_kl_w = kl_weight * warmup_factor
            kl_floored = kl_per_dim.clamp(min=free_bits) if free_bits > 0 else kl_per_dim
            kl_term = kl_floored.sum(-1).mean()
            kl_value = float(kl_term.detach().item())
            if train_stage == 3:
                loss = recon_loss + eff_kl_w * kl_term
            elif train_stage == 2:
                loss = diff_w * diff_loss + traj_delta_w * traj_delta_loss + s2_align_active_w * s2_align_loss + s2_coadapt_ce_w * s2_coadapt_ce_loss + s1_ldlm_mse_w * s1_ldlm_mse_loss + simcot_step_w * simcot_step_loss + eff_kl_w * kl_term
            else:
                loss = text_w * ce_loss + diff_w * diff_loss + state_anchor_active_w * state_anchor_loss + eff_kl_w * kl_term
        else:
            if train_stage == 3:
                loss = recon_loss
            elif train_stage == 2:
                loss = diff_w * diff_loss + traj_delta_w * traj_delta_loss + s2_align_active_w * s2_align_loss + s2_coadapt_ce_w * s2_coadapt_ce_loss + s1_ldlm_mse_w * s1_ldlm_mse_loss + simcot_step_w * simcot_step_loss
            else:
                loss = text_w * ce_loss + diff_w * diff_loss + state_anchor_active_w * state_anchor_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        # Average loss across ranks for logging
        if is_distributed and loss.dim() == 0:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        log_buf.append({
            "loss": float(loss.detach().item()),
            "ce": float(ce_loss.detach().item()) if train_stage not in (2, 3) else 0.0,
            "diff": float(diff_loss.detach().item()),
            "traj_delta": float(traj_delta_loss.detach().item()),
            "s2_align": float(s2_align_loss.detach().item()),
            "s2_align_w": float(s2_align_active_w),
            "s2_coadapt_ce": float(s2_coadapt_ce_loss.detach().item()),
                "s1_ldlm_mse": float(s1_ldlm_mse_loss.detach().item()),
                "simcot_step": float(simcot_step_loss.detach().item()),
            "state_anchor": float(state_anchor_loss.detach().item()),
            "state_anchor_w": float(state_anchor_active_w),
            "kl": kl_value,
            "kl_w": eff_kl_w,
            "lr": lr,
            "split_idx": float(split_idx.float().mean().detach().item()) if isinstance(split_idx, torch.Tensor) else float(split_idx or 0),
            **extras,
        })

        if (step + 1) % log_freq == 0 and is_main_process:
            def avg(k, default=0.0):
                vals = [d.get(k, default) for d in log_buf]
                return sum(vals) / len(vals)
            now = time.time()
            elapsed = int(now - start_time)
            window_elapsed = max(now - log_window_start, 1e-9)
            window_steps = max(len(log_buf), 1)
            samples_per_step = int(config.training.train_batch_size) * world_size
            step_per_sec = window_steps / window_elapsed
            samples_per_sec = samples_per_step * step_per_sec
            if device.type == "cuda":
                mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
            else:
                mem_gb = 0.0
            perf_str = (
                f"step/s={step_per_sec:.2f} samples/s={samples_per_sec:.2f} "
                f"mem={mem_gb:.1f}GB"
            )
            kl_str = ""
            if kl_per_dim is not None:
                kl_str = f" kl={avg('kl'):.3f}*{avg('kl_w'):.4f}"
            if train_stage == 3:
                log_line = (
                    f"[step {step+1}] "
                    f"loss={avg('loss'):.4f} recon={avg('recon_val'):.4f} "
                    f"cosine={avg('cosine'):.4f}{kl_str}  | "
                    f"z_norm={avg('z_norm'):.3f} | "
                    f"lr={lr:.2e} {perf_str} elapsed={elapsed//60}m{elapsed%60}s"
                )
            elif train_stage == 2:
                align_str = ""
                if avg('s2_align_w') > 0.0:
                    align_str = f" align={avg('s2_align'):.4f}*{avg('s2_align_w'):.4f}"
                ldlm_str = ""
                if avg('s1_ldlm_mse') > 0.0:
                    ldlm_str = f" ldlm={avg('s1_ldlm_mse'):.4f}"
                log_line = (
                    f"[step {step+1}] "
                    f"loss={avg('loss'):.4f} diff={avg('diff'):.4f} "
                    f"traj_delta={avg('traj_delta'):.4f}{align_str}{ldlm_str}  | "
                    f"z_norm={avg('z_norm'):.3f} "
                    f"eps_pred_norm={avg('eps_pred_norm'):.3f} "
                    f"t_mean={avg('t_mean'):.2f} | "
                    f"lr={lr:.2e} {perf_str} elapsed={elapsed//60}m{elapsed%60}s"
                )
            else:
                anchor_str = ""
                if avg('state_anchor_w') > 0.0:
                    anchor_str = f" anchor={avg('state_anchor'):.4f}*{avg('state_anchor_w'):.4f}"
                self_forcing_str = ""
                if avg('s1_self_forcing') > 0.0:
                    self_forcing_str = (
                        f" self_forcing=on p={avg('s1_self_forcing_p_effective'):.3f}"
                        f" tokens={avg('s1_self_forcing_token_steps'):.0f}"
                    )
                log_line = (
                    f"[step {step+1}] "
                    f"loss={avg('loss'):.4f} ce={avg('ce'):.4f} "
                    f"diff={avg('diff'):.4f}{anchor_str}{kl_str}{self_forcing_str}  | "
                    f"z_norm={avg('z_norm'):.3f} state_norm={avg('state_norm'):.3f} "
                    f"state_scale={avg('state_scale'):.4f} "
                    f"eps_pred_norm={avg('eps_pred_norm'):.3f} "
                    f"t_mean={avg('t_mean'):.2f} | "
                    f"lr={lr:.2e} {perf_str} elapsed={elapsed//60}m{elapsed%60}s"
                )
            print(log_line, flush=True)
            logger.info(log_line)
            log_buf = []
            log_window_start = now
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

        step += 1

        if step % save_every == 0:
            save_checkpoint(step)

    if step > 0 and step % save_every != 0:
        save_checkpoint(step)

    if is_main_process:
        print("Training done.")
        logger.info("Training done.")
    if is_distributed:
        dist.destroy_process_group()

def sample_from_ckpt(
    ckpt_dir,
    prompt="",
    max_len=256,
    temperature=0.7,
    top_k=50,
    seed=42,
    use_encoder_z=False,
    top_p=0.9,
    repetition_penalty=1.1,
    sample_steps=1000,
    trajectory_s1_mode=None,
    trajectory_state_blend=None,
    trajectory_sampler=None,
):
    """Rebuild model from checkpoint and generate text.
    
    Args:
        prompt: Optional debug/seed text. Empty prompt means pure unconditional
                sampling from token id 0, with no semantic prefix.
        use_encoder_z: If True, use encoder's z_0 from prompt instead of Denoiser sampling.
                        Useful for debugging: if this works but Denoiser sampling doesn't,
                        the issue is Denoiser quality, not sampling logic.
    """
    import torch, sys, os
    _proj = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(_proj, os.pardir, os.pardir)))
    sys.path.insert(0, _proj)
    from omegaconf import OmegaConf
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model import StateHijackingRELAY

    def apply_repetition_penalty(logits, generated_ids, penalty):
        if penalty == 1.0 or not generated_ids:
            return logits
        token_ids = torch.tensor(list(set(generated_ids)), device=logits.device, dtype=torch.long)
        token_scores = logits[token_ids]
        logits = logits.clone()
        logits[token_ids] = torch.where(token_scores < 0, token_scores * penalty, token_scores / penalty)
        return logits

    def apply_top_p(probs, p):
        if p <= 0.0 or p >= 1.0:
            return probs
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        filtered = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
        return filtered / filtered.sum().clamp(min=1e-12)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    ckpt = torch.load(f"{ckpt_dir}/model.pt", map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    if trajectory_s1_mode is not None:
        cfg.model.trajectory_s1_mode = trajectory_s1_mode
    if trajectory_state_blend is not None:
        cfg.model.trajectory_state_blend = float(trajectory_state_blend)

    rwkv_path = cfg.model.rwkv_local_path
    print(f"Loading backbone from {rwkv_path}")
    rwkv = AutoModelForCausalLM.from_pretrained(rwkv_path, trust_remote_code=True, torch_dtype=dtype, local_files_only=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(rwkv_path, trust_remote_code=True, local_files_only=True)
    model = StateHijackingRELAY(
        config=cfg.model,
        rwkv_model=rwkv,
        vocab_size=len(tokenizer),
        latent_dim=int(cfg.model.latent_dim),
        n_basis=int(cfg.model.get("n_basis", 16)),
        denoiser_hidden=int(cfg.model.get("denoiser_hidden", cfg.model.get("dit_hidden", 256))),
        denoiser_depth=int(cfg.model.get("denoiser_depth", cfg.model.get("dit_depth", 4))),
        denoiser_num_heads=int(cfg.model.get("denoiser_num_heads", cfg.model.get("dit_num_heads", 4))),
        denoiser_num_tokens=int(cfg.model.get("denoiser_num_tokens", cfg.model.get("dit_num_tokens", 4))),
        encoder_type=str(cfg.model.get("encoder_type", "mlp")),
        alpha_type=str(cfg.model.get("alpha_type", "linear")),
        alpha_hidden=int(cfg.model.get("alpha_hidden", 256)),
        latent_stats_path=cfg.model.get("latent_stats_path", None),
    ).to(device)
    missing, unexpected = model.load_state_dict(
        _compatible_trainable_state(ckpt["trainable_state"], model), strict=False
    )
    _report_load_mismatch(missing, unexpected)
    for p in model.parameters():
        if p.requires_grad and p.dtype != dtype and p.dtype.is_floating_point:
            p.data = p.data.to(dtype)
    model.eval()
    
    # Set generation type from checkpoint config
    gen_type = str(cfg.training.get("gen_type", "ddpm"))
    model._gen_type = gen_type
    print(f"Loaded step {ckpt['step']}, gen_type={gen_type}, ready for generation")

    if prompt:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        print(f"Using explicit seed prompt: {prompt!r}")
    else:
        input_ids = torch.tensor([[0]], device=device, dtype=torch.long)
        print("Using pure unconditional start token: 0")
    
    torch.manual_seed(seed)
    trajectory_mode = bool(cfg.model.get("trajectory_mode", False))
    with torch.no_grad():
        if use_encoder_z:
            print("Using encoder z_0 from prompt (debug mode)")
            out_pool = model.rwkv_model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            h_last = out_pool.hidden_states[-1]
            # Use masked mean pooling (same as training)
            attention_mask = torch.ones_like(input_ids, dtype=h_last.dtype)
            m = attention_mask.unsqueeze(-1)
            pooled = (h_last * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            if model.encoder_type == "variational":
                pooled = pooled.to(next(model.encoder_trunk.parameters()).dtype)
                h = model.encoder_trunk(pooled)
                z = model.mu_head(h)
            elif model.encoder_type == "mlp":
                pooled = pooled.to(next(model.encoder.parameters()).dtype)
                z = model.encoder(pooled)
            elif model.encoder_type == "identity":
                pooled = pooled.to(model.latent_mu.dtype)
                z = (pooled - model.latent_mu) / model.latent_sigma
                z = z.to(next(model.alpha_heads.parameters()).dtype)
            print(f"  z_0 norm: {z.norm().item():.3f}")
            states = model.predict_states(z)
        elif trajectory_mode:
            sampler_name = trajectory_sampler or "default"
            print(f"Using trajectory sampling ({sample_steps} steps, sampler={sampler_name})")
            z_traj = model.trajectory_sample(1, num_steps=sample_steps, sampler=trajectory_sampler)
            print(f"  z_traj shape: {tuple(z_traj.shape)}")
            print(f"  z_traj has NaN: {z_traj.isnan().any().item()}")
            print(f"  z_traj norm: {z_traj.norm(dim=-1).mean().item():.3f}")
            z = None
            states = None
        else:
            print(f"Using DDPM sampling ({sample_steps} steps)")
            z = model.ddpm_sample(1, num_steps=sample_steps)
            print(f"  z has NaN: {z.isnan().any().item()}")
            print(f"  z has Inf: {z.isinf().any().item()}")
            print(f"  z norm: {z.norm().item():.3f}")
            print(f"  z min/max: {z.min().item():.3f} / {z.max().item():.3f}")
            states = model.predict_states(z)
            print(f"  states[0] norm: {states[0].norm().item():.3f}")
            print(f"  states[0] has NaN: {states[0].isnan().any().item()}")

    model.rwkv_model.eval()
    with torch.no_grad():
        out = model.rwkv_model(input_ids=input_ids, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        if trajectory_mode and not use_encoder_z:
            generated = list(input_ids[0].tolist())
            logits = out.logits[0, -1]
            chunk_size = int(cfg.model.get("trajectory_chunk_size", 32))
            max_chunks = min(z_traj.shape[1], max(1, (max_len + chunk_size - 1) // chunk_size))
            trajectory_s1_mode = str(cfg.model.get("trajectory_s1_mode", "independent"))
            if trajectory_s1_mode in ("transformer", "rwkv", "birwkv"):
                layer_states = model.predict_trajectory_states(z_traj)
                trajectory_state_blend = float(cfg.model.get("trajectory_state_blend", getattr(model, "trajectory_state_blend", 1.0)))
            else:
                layer_states = None
                trajectory_state_blend = 1.0
            for h in range(max_chunks):
                if trajectory_s1_mode in ("transformer", "rwkv", "birwkv"):
                    states_h = [layer_state[:, h] for layer_state in layer_states]
                    past_kv = model.blend_into_cache(past_kv, states_h, trajectory_state_blend)
                else:
                    states_h = model.predict_states(z_traj[:, h])
                    past_kv = model.inject_into_cache(past_kv, states_h)
                for _ in range(chunk_size):
                    if len(generated) - input_ids.shape[1] >= max_len:
                        break
                    logits = apply_repetition_penalty(logits.float(), generated, repetition_penalty)
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    if top_k > 0:
                        topk_vals, topk_idx = torch.topk(probs, top_k)
                        probs = torch.zeros_like(probs).scatter(-1, topk_idx, topk_vals)
                        probs /= probs.sum().clamp(min=1e-12)
                    probs = apply_top_p(probs, top_p)
                    next_id = torch.multinomial(probs, 1).item()
                    generated.append(next_id)
                    out = model.rwkv_model(
                        input_ids=torch.tensor([[next_id]], device=device),
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                    past_kv = out.past_key_values
                    logits = out.logits[0, -1]
            return tokenizer.decode(generated)

        past_kv = model.inject_into_cache(past_kv, states)

        out = model.rwkv_model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values

        if input_ids.shape[1] > 1:
            ce_logits = out.logits[:, :-1, :].float()
            ce_targets = input_ids[:, 1:]
            ce = F.cross_entropy(ce_logits.reshape(-1, ce_logits.shape[-1]), ce_targets.reshape(-1))
            print(f"  CE (injected state): {ce.item():.4f}")
        else:
            print("  CE (injected state): skipped for single-token unconditional start")

        generated = list(input_ids[0].tolist())
        logits = out.logits[0, -1]
        for _ in range(max_len):
            logits = apply_repetition_penalty(logits.float(), generated, repetition_penalty)
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs).scatter(-1, topk_idx, topk_vals)
                probs /= probs.sum()
            probs = apply_top_p(probs, top_p)
            next_id = torch.multinomial(probs, 1).item()
            generated.append(next_id)
            out = model.rwkv_model(input_ids=torch.tensor([[next_id]], device=device), past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[0, -1]
    return tokenizer.decode(generated)


if __name__ == "__main__":
    import argparse, sys
    if "--sample" in sys.argv:
        sample_argv = [arg for arg in sys.argv[1:] if arg != "--sample"]
        p = argparse.ArgumentParser(allow_abbrev=False)
        p.add_argument("--ckpt_dir", required=True)
        p.add_argument("--prompt", default="", help="Optional debug/seed prompt. Default is pure unconditional start token 0.")
        p.add_argument("--max_len", type=int, default=256)
        p.add_argument("--temperature", type=float, default=0.7)
        p.add_argument("--top_k", type=int, default=50)
        p.add_argument("--top_p", type=float, default=0.9)
        p.add_argument("--repetition_penalty", type=float, default=1.1)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--use-encoder-z", action="store_true", help="Use encoder z_0 from prompt instead of Denoiser sampling (debug)")
        p.add_argument("--sample_steps", type=int, default=1000)
        p.add_argument("--trajectory_s1_mode", choices=["independent", "transformer", "rwkv", "birwkv"], default=None)
        p.add_argument("--trajectory_state_blend", type=float, default=None)
        p.add_argument("--trajectory_sampler", choices=["rf_heun"], default=None)
        args, _ = p.parse_known_args(sample_argv)
        text = sample_from_ckpt(
            args.ckpt_dir,
            args.prompt,
            args.max_len,
            args.temperature,
            args.top_k,
            args.seed,
            args.use_encoder_z,
            args.top_p,
            args.repetition_penalty,
            args.sample_steps,
            args.trajectory_s1_mode,
            args.trajectory_state_blend,
            args.trajectory_sampler,
        )
        print(text)
        sys.exit(0)
    main()
