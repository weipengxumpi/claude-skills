#!/usr/bin/env python3
"""Generate a livedealer_infer.py evaluation snippet for a training run.

Reads a training `*_leo_<N>.sh` script and emits the inference command that
evaluates it on the eyes-only test set, mirroring the canonical evals at the
bottom of live_dealer/infer/infer_card_class.sh:

  HF_HUB_OFFLINE=1 <WAN_* flags from training> python live_dealer/infer/livedealer_infer.py \
    --lora_path outputs/<run>/step-<STEP>.safetensors \
    --pose_video   <test_set>/infer_list_shape_body_only_eye.txt \  # if s2v_pose_video trained
    --object_video <test_set>/infer_list_shape_card_only.txt     \  # if s2v_object_video trained
    --input_image  <test_set>/infer_list_input_image.txt \
    --audio_path   <test_set>/infer_list_audio.txt \
    --gt_path      <test_set>/infer_list_gt.txt \
    --save_path output/<run>-<STEP> \
    --infer_frames 12 --num_clips 1 [--width W --height H] \
    --no_motion_video --use_block_attn

What it derives from the training script:
  * run name        <- basename of --output_path
  * WAN_* env flags  <- the tokens preceding `accelerate launch`
  * --width/--height <- training --width/--height
  * which inputs     <- training --extra_inputs (pose/object gating)
  * checkpoint step  <- latest step-*.safetensors (or --step)
"""
import argparse
import getpass
import glob
import os
import re
import shlex
import subprocess
import sys

TEST_SET = "data/project21_snapshot_12032025_packed/test_set"
SLURM_LOGS_DIR = "/leonardo_scratch/large/userexternal/{user}/slurm_logs"
DEFAULT_EXTRA = "input_image,input_audio,s2v_pose_video,s2v_object_video"

# Canonical denoising schedule for distillation-model evals, matching
# direct_distill_loss() in diffsynth/pipelines/wan_video_new.py.
DISTILL_TIMESTEPS = "1000,768,358"

# Defaults for --launch, mirroring the interactive launch wrapper at the top of
# live_dealer/infer/infer_card_class.sh (srun + singularity exec --nv).
WORK_DEFAULT = "/leonardo_work/AIFAC_F02_378"
SIF_REL = "shared/singularity/diffsynth-a100/diffsynth-bind-a100.sif"
LAUNCH = dict(account="aifac_f02_378", partition="boost_usr_prod",
              qos="boost_qos_dbg", time="00:30:00", cpus="8", mem="64G")
LISTS = {
    "pose_video": f"{TEST_SET}/infer_list_shape_body_only_eye.txt",
    "object_video": f"{TEST_SET}/infer_list_shape_card_only.txt",
    "card_detection": f"{TEST_SET}/infer_list_card_detection.txt",
    "input_image": f"{TEST_SET}/infer_list_input_image.txt",
    "audio_path": f"{TEST_SET}/infer_list_audio.txt",
    "gt_path": f"{TEST_SET}/infer_list_gt.txt",
}


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def arg_value(text, flag):
    # Match the value on the SAME line as the flag and skip comment lines, so a
    # comment that merely mentions "--extra_inputs" at end-of-line doesn't make
    # us grab the first token of the next line.
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = re.search(rf"--{re.escape(flag)}[ \t]+(\S+)", line)
        if m:
            return m.group(1)
    return None


def resolve_base_from_jobid(job_id, logs_dir):
    """Map a SLURM job id back to the training *_leo_<N>.sh it ran.

    1. `scontrol show job <id>` -> `Command=<script>` (works while the job is
       running or still in the controller's memory).
    2. Fall back to the job's stdout log `<logs_dir>/*_<id>.out`, which echoes
       `The original script is located at: <script>` (works after the job is
       gone from the controller, as long as the log survives).
    """
    try:
        out = subprocess.run(
            ["scontrol", "show", "job", str(job_id)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            universal_newlines=True, timeout=15,
        )
        m = re.search(r"Command=(\S+)", out.stdout)
        if m and os.path.isfile(m.group(1)):
            return m.group(1)
    except (OSError, subprocess.SubprocessError):
        pass

    user = os.environ.get("USER") or getpass.getuser()
    logdir = logs_dir.format(user=user)
    for log in sorted(glob.glob(os.path.join(logdir, f"*_{job_id}.out"))):
        try:
            with open(log) as f:
                for line in f:
                    m = re.search(r"original script is located at:\s*(\S+)", line)
                    if m and os.path.isfile(m.group(1)):
                        return m.group(1)
        except OSError:
            continue

    fail(f"could not resolve job {job_id} to a training script via scontrol or "
         f"{logdir}/*_{job_id}.out; pass --base instead")


def host_outputs(text, override):
    if override:
        return override
    m = re.search(r"^SCRATCH_PROJECT_DIR=(\S+)", text, re.MULTILINE)
    if not m:
        fail("could not find SCRATCH_PROJECT_DIR; pass --outputs-dir")
    return os.path.join(m.group(1).split("#")[0].strip(), "outputs")


def pick_step(ckpt_dir, want):
    if not os.path.isdir(ckpt_dir):
        fail(f"checkpoint dir does not exist: {ckpt_dir}")
    steps = []
    for f in os.listdir(ckpt_dir):
        m = re.fullmatch(r"step-(\d+)\.safetensors", f)
        if m:
            steps.append(int(m.group(1)))
    steps.sort()
    if not steps:
        fail(f"no step-*.safetensors in {ckpt_dir}")
    if want is None:
        return steps[-1], steps
    if want not in steps:
        fail(f"step-{want} not in {ckpt_dir} (have: {steps})")
    return want, steps


def wan_flags(text):
    """WAN_* env assignments that prefix the `accelerate launch` line."""
    m = re.search(r"^(.*)\baccelerate launch\b", text, re.MULTILINE)
    if not m:
        return []
    return re.findall(r"\bWAN_[A-Z0-9_]+=\S+", m.group(1))


def launch(snippet, run, step, args, tag=""):
    """Run the snippet on a GPU node via srun + singularity exec --nv.

    Mirrors the interactive launch wrapper at the top of infer_card_class.sh,
    converted to a non-interactive single-command srun. Output is teed to
    eval_logs/eval_<run>-<step>.log under the workspace. Blocks until the srun
    finishes; the caller (agent) typically runs this in the background.
    """
    workspace = args.workspace or os.getcwd()
    work = os.environ.get("WORK", WORK_DEFAULT)
    shared = os.path.join(work, "shared")
    sif = args.sif or os.path.join(work, SIF_REL)
    if not os.path.isfile(sif):
        fail(f"SIF not found: {sif} (pass --sif)")

    log_dir = os.path.join(workspace, "eval_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"eval_{run}-{step}{tag}.log")

    cmd = [
        "srun", "-A", args.account, "-p", args.partition, "--qos", args.qos,
        "--gres=gpu:1", "--ntasks=1", f"--cpus-per-task={args.cpus}",
        f"--mem={args.mem}", f"--time={args.time}",
        "singularity", "exec", "--nv",
        "-B", f"{workspace}:/workspace",
        "-B", f"{shared}:{shared}",
        "-B", f"{shared}/wan_models:/workspace/models",
        "-B", f"{workspace}/outputs:/workspace/outputs",
        "--pwd", "/workspace",
        sif,
        "/bin/bash", "-c", snippet,
    ]
    print(f"# launching on slurm: {args.qos} {args.time} 1xGPU; "
          f"log -> {log_path}", file=sys.stderr)
    print("# " + " ".join(shlex.quote(c) for c in cmd), file=sys.stderr)
    with open(log_path, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"# eval finished rc={rc}; full log -> {log_path}", file=sys.stderr)
    sys.exit(rc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--base", help="path to training *_leo_<N>.sh")
    src.add_argument("--job-id", help="SLURM job id; resolves the training script "
                                      "via scontrol, then the job's slurm log")
    ap.add_argument("--checkpoint", "--lora", dest="checkpoint", default=None,
                    metavar="PATH",
                    help="evaluate a checkpoint path directly (e.g. "
                         "outputs/<run>/step-<N>.safetensors). The run name and step are "
                         "derived from the path, so the checkpoint dir need not match any "
                         "training script's --output_path. Combine with --base/--job-id to "
                         "pull the training config (WAN_* flags, resolution, extra_inputs), "
                         "or pass --wan-env/--extra-inputs/--width/--height to set it explicitly.")
    ap.add_argument("--wan-env", action="append", default=None, metavar="WAN_X=Y",
                    help="WAN_* env assignment(s) for inference; repeatable, or one "
                         "space/comma-separated string. Overrides those derived from --base.")
    ap.add_argument("--extra-inputs", dest="extra_inputs", default=None,
                    help="override the trained --extra_inputs gating "
                         "(e.g. 'input_image,input_audio,s2v_pose_video,card_detection')")
    ap.add_argument("--width", default=None, help="override inference width")
    ap.add_argument("--height", default=None, help="override inference height")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest); "
                    "ignored when --checkpoint is given (step comes from the path)")
    ap.add_argument("--outputs-dir", default=None)
    ap.add_argument("--slurm-logs-dir", default=SLURM_LOGS_DIR,
                    help="slurm log dir template ({user} expands to $USER) for --job-id")
    ap.add_argument("--append", action="store_true",
                    help="append the snippet to live_dealer/infer/infer_card_class.sh")
    ap.add_argument("--distill", nargs="?", const=DISTILL_TIMESTEPS, default=None,
                    metavar="T0,T1,...",
                    help="evaluate as a distillation model: denoise in a few fixed timesteps "
                         f"instead of the default schedule. Bare --distill uses {DISTILL_TIMESTEPS} "
                         "(the direct_distill_loss schedule); pass a comma-separated list to override. "
                         "Disables tea-cache and tags the save_path with -distill.")
    ap.add_argument("--launch", action="store_true",
                    help="run the eval on a GPU node via srun + singularity exec --nv "
                         "(non-interactive), teeing output to eval_logs/")
    ap.add_argument("--workspace", default=None,
                    help="repo root to bind as /workspace (default: cwd)")
    ap.add_argument("--sif", default=None, help="override singularity image path")
    ap.add_argument("--account", default=LAUNCH["account"])
    ap.add_argument("--partition", default=LAUNCH["partition"])
    ap.add_argument("--qos", default=LAUNCH["qos"])
    ap.add_argument("--time", default=LAUNCH["time"], help="srun --time (HH:MM:SS)")
    ap.add_argument("--cpus", default=LAUNCH["cpus"])
    ap.add_argument("--mem", default=LAUNCH["mem"])
    args = ap.parse_args()

    if not (args.base or args.job_id or args.checkpoint):
        ap.error("provide --checkpoint, --base, or --job-id")

    # Config source (training script): optional when --checkpoint is given and the
    # config is supplied via --wan-env / --extra-inputs / --width / --height.
    base = args.base
    if args.job_id:
        base = resolve_base_from_jobid(args.job_id, args.slurm_logs_dir)
        print(f"# job {args.job_id} -> {base}", file=sys.stderr)
    text = None
    if base:
        if not os.path.isfile(base):
            fail(f"training script not found: {base}")
        text = open(base).read()

    # Run name + step + lora_path.
    all_steps = None
    if args.checkpoint:
        ckpt = args.checkpoint
        m = re.search(r"step-(\d+)\.safetensors$", os.path.basename(ckpt))
        if not m:
            fail(f"--checkpoint must point at a step-<N>.safetensors file: {ckpt}")
        step = int(m.group(1))
        run = os.path.basename(os.path.dirname(ckpt.rstrip("/")))
        if not run:
            fail(f"could not derive a run name from --checkpoint parent dir: {ckpt}")
        check = ckpt if os.path.isabs(ckpt) else os.path.join(args.workspace or os.getcwd(), ckpt)
        if not os.path.isfile(check):
            fail(f"checkpoint not found: {check}")
        lora_path = ckpt
    else:
        out_path = arg_value(text, "output_path")
        if not out_path:
            fail("no --output_path in training script")
        run = os.path.basename(out_path.rstrip("/"))
        step, all_steps = pick_step(
            os.path.join(host_outputs(text, args.outputs_dir), run), args.step)
        lora_path = f"outputs/{run}/step-{step}.safetensors"

    # Config: explicit overrides win, then the training script, then defaults.
    if args.extra_inputs is not None:
        extra = args.extra_inputs
    elif text is not None:
        extra = arg_value(text, "extra_inputs") or DEFAULT_EXTRA
    else:
        extra = DEFAULT_EXTRA
        print("# WARN: no --base/--extra-inputs; defaulting --extra_inputs gating to "
              f"'{DEFAULT_EXTRA}'", file=sys.stderr)
    width = args.width if args.width is not None else (arg_value(text, "width") if text else None)
    height = args.height if args.height is not None else (arg_value(text, "height") if text else None)
    if args.wan_env:
        flags = [f for item in args.wan_env for f in re.split(r"[,\s]+", item.strip()) if f]
    elif text is not None:
        flags = wan_flags(text)
    else:
        flags = []
        print("# WARN: no --base/--wan-env; emitting no WAN_* flags", file=sys.stderr)

    # Distillation runs (`--task direct_distill`) freeze extra modules like
    # card_encoder and DON'T save them in the checkpoint (only the LoRA). At
    # inference card_encoder is created via .to_empty() and, left unloaded, NaNs
    # the forward pass -> all-black video. Load those frozen modules from the base
    # checkpoint the run resumed from (its --lora_checkpoint), converting the
    # container path (/output/<base>/...) to the host/workspace-relative form
    # (outputs/<base>/...) that inference uses.
    is_distill = bool(args.distill) or (text is not None and "direct_distill" in (arg_value(text, "task") or ""))
    lora_ckpt = arg_value(text, "lora_checkpoint") if text else None
    extra_module_ckpt = None
    if is_distill and lora_ckpt:
        extra_module_ckpt = re.sub(r"^/?output/", "outputs/", lora_ckpt)

    env = " ".join(["HF_HUB_OFFLINE=1", *flags])
    lines = [f"{env} python live_dealer/infer/livedealer_infer.py \\",
             f"  --lora_path {lora_path} \\"]
    if extra_module_ckpt:
        lines.append(f"  --extra_module_ckpt_path {extra_module_ckpt} \\")
    if "s2v_pose_video" in extra:
        lines.append(f"  --pose_video {LISTS['pose_video']} \\")
    if "s2v_object_video" in extra:
        lines.append(f"  --object_video {LISTS['object_video']} \\")
    # WAN_CARD_CLASS_EMBED runs feed raw (x, y, class) detections instead of a
    # rendered object video; the WAN_CARD_CLASS_EMBED=true flag is already carried
    # over by wan_flags(), and livedealer_infer.py ignores object_video in this mode.
    if "card_detection" in extra:
        lines.append(f"  --card_detection {LISTS['card_detection']} \\")
    lines.append(f"  --input_image {LISTS['input_image']} \\")
    lines.append(f"  --audio_path {LISTS['audio_path']} \\")
    lines.append(f"  --gt_path {LISTS['gt_path']} \\")
    save_suffix = "-distill" if args.distill else ""
    lines.append(f"  --save_path output/{run}-{step}{save_suffix} \\")
    lines.append("  --infer_frames 12 \\")
    lines.append("  --num_clips 1 \\")
    if width:
        lines.append(f"  --width {width} \\")
    if height:
        lines.append(f"  --height {height} \\")
    # Distillation models denoise in a few fixed steps; feed them explicitly and
    # disable tea-cache (a many-step skip heuristic that's meaningless here).
    if args.distill:
        lines.append(f"  --custom_timesteps {args.distill} \\")
        lines.append("  --no_tea_cache \\")
    lines.append("  --no_motion_video \\")
    lines.append("  --use_block_attn")
    snippet = "\n".join(lines)

    if all_steps:
        print(f"# checkpoints in {run}: {', '.join(f'step-{s}' for s in all_steps)}", file=sys.stderr)
    print(f"# evaluating {run} step-{step}\n", file=sys.stderr)
    print(snippet)

    if args.append:
        tgt = "live_dealer/infer/infer_card_class.sh"
        distill_note = f" (distill {args.distill})" if args.distill else ""
        with open(tgt, "a") as f:
            f.write(f"\n\n# eval {run} @ step-{step}{distill_note}\n  {snippet}\n")
        print(f"\n# appended to {tgt}", file=sys.stderr)

    if args.launch:
        launch(snippet, run, step, args, tag=save_suffix)


if __name__ == "__main__":
    main()
