---
name: eval-training
description: Generate the livedealer_infer.py evaluation command for a Wan2.2-S2V training run (e.g. leo_114.sh, or a SLURM job id). Reads the training script and emits the inference snippet on the eyes-only test set, mirroring the canonical evals in live_dealer/infer/infer_card_class.sh — correct WAN_* env flags, lora_path/step, width/height, and pose/object inputs. Use when the user says "evaluate this training job", "make the infer command for run X", "eval job 43983676", or "generate the eval snippet".
---

# Generate an evaluation snippet for a training run

Produce the `livedealer_infer.py` command that evaluates a training run on the
eyes-only test set, following the pattern at the bottom of
`live_dealer/infer/infer_card_class.sh`.

## What it derives from the training script

| inference field | source in the training `*_leo_<N>.sh` |
|---|---|
| `WAN_*` env flags | tokens preceding `accelerate launch` (e.g. `WAN_CARD_SEQUENTIAL=true`) |
| `--lora_path outputs/<run>/step-<STEP>` | `<run>` = basename of `--output_path`; STEP = latest checkpoint (or `--step`) |
| `--width` / `--height` | training `--width` / `--height` |
| `--pose_video` / `--object_video` | included only if `s2v_pose_video` / `s2v_object_video` is in `--extra_inputs` |
| `--save_path output/<run>-<STEP>` | derived |

Fixed (eyes-only test set): `--input_image`, `--audio_path`, `--gt_path`, the pose/object
list files, and constants `--infer_frames 12 --num_clips 1 --no_motion_video --use_block_attn`,
prefixed with `HF_HUB_OFFLINE=1`.

## How to run it

```bash
python3 .claude/skills/eval-training/eval_snippet.py \
  ( --base <path/to/training_leo_N.sh> | --job-id <SLURM_JOB_ID> ) \
  [--step <STEP>]   # default: latest checkpoint \
  [--distill[=T0,T1,...]]  # distillation model: denoise in a few fixed steps \
  [--append]        # also append the snippet to infer_card_class.sh \
  [--launch]        # also run the eval on a GPU node (srun + singularity)
```

## Distillation models (`--distill`)

For a distillation-trained model, the eval should denoise in the model's few fixed
timesteps rather than the default ~25-step schedule. Pass `--distill` to emit
`--custom_timesteps 1000,768,358` (the schedule from `direct_distill_loss()` in
`diffsynth/pipelines/wan_video_new.py`). Override the steps with
`--distill=1000,500,250`. In distill mode the helper also adds `--no_tea_cache`
(the many-step skip heuristic is meaningless at 3 steps) and tags the save path /
log with `-distill` so it doesn't clobber a regular eval of the same checkpoint.

`livedealer_infer.py` accepts `--custom_timesteps T0,T1,...` directly and forwards
it to the pipeline's `custom_timesteps` (see `__call__`); `--distill` is just the
convenience wrapper that fills in the canonical schedule.

Pass **either** `--base` (the training script directly) **or** `--job-id` (a SLURM
job id). With `--job-id` the helper resolves the training script automatically:
1. `scontrol show job <id>` → `Command=` (while the job is running / still in the
   controller's memory);
2. falling back to the job's stdout log
   `/leonardo_scratch/large/userexternal/$USER/slurm_logs/*_<id>.out`, which echoes
   `The original script is located at: <script>` (for finished jobs whose logs survive).
Override the log location with `--slurm-logs-dir '<template with {user}>'` if needed.
The resolved `job <id> -> <script>` mapping is printed to stderr.

By default it prints the snippet to stdout (checkpoint list goes to stderr). Show the
snippet to the user; only pass `--append` when they want it added to
`live_dealer/infer/infer_card_class.sh`.

## Auto-launch (`--launch`)

When the user asks to **run / launch / execute** the eval (not just generate it),
pass `--launch`. The helper wraps the generated snippet in a non-interactive
`srun` + `singularity exec --nv` invocation — the same launch wrapper documented
at the top of `infer_card_class.sh`, but turned into a single batchable command
(no `--pty bash` shell). It:

- requests `1×GPU` on `boost_qos_dbg` for `00:30:00` (override with `--qos`,
  `--time`, `--account`, `--partition`, `--cpus`, `--mem`),
- binds `/workspace`, `/workspace/models` (`wan_models`), and `/workspace/outputs`
  the same way the interactive wrapper does (override the repo root with
  `--workspace`, default cwd; image with `--sif`, default `$WORK/<SIF_REL>`),
- tees all output to `eval_logs/eval_<run>-<step>.log` under the workspace, and
- blocks until the srun finishes, exiting with the srun's return code.

Because it blocks (model load + inference can take minutes), **the agent should
invoke the helper with `--launch` as a background Bash command** (`run_in_background:
true`) and report the Slurm job id + log path, then surface the result on
completion. Example:

```bash
python3 .claude/skills/eval-training/eval_snippet.py --job-id 44063494 --launch
```

## Procedure for the agent

1. Identify the training run. If the user gives a SLURM job id (e.g. "eval job
   43983676"), pass `--job-id`. Otherwise identify the training script (the most
   recently opened `*_leo_<N>.sh` is a strong default; or match the run name to a
   script's `--output_path`) and pass `--base`.
2. Run the helper and present the generated snippet (and which step it used).
3. If the user wants it saved, re-run with `--append` or add it where they prefer.
4. If the user wants it **run**, re-run with `--launch` (in the background) and
   report the job id + `eval_logs/` path; verify the checkpoint and test-set list
   files exist first (the helper fails fast if the checkpoint dir is missing).

## Notes / gotchas

- WAN_* flags are mirrored verbatim from training to inference — that's the convention in
  infer_card_class.sh (e.g. `WAN_CARD_COMPRESSION`, `WAN_CARD_ENCODER`, `WAN_CARD_SEQUENTIAL`).
- The eyes-only list files are hard-coded to match recent evals. If a run needs a
  different test set, edit the printed snippet.
- Checkpoint paths use `outputs/<run>/...` (host/workspace-relative), as the inference
  runs from /workspace; this differs from the `/output/<run>` container path in training.
