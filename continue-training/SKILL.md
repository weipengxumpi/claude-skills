---
name: continue-training
description: Continue/resume a Wan2.2-S2V LoRA training run from its latest (or a chosen) checkpoint. Generates a new `..._leo_<N+1>.sh` SLURM script that resumes via --lora_checkpoint with a configurable --skip_frames, following the project's `_cont<STEP>_SF<SF>` convention. Use when the user says "continue training X", "resume the run with skip frames N", or "pick up from the latest checkpoint".
---

# Continue a training run

Generate a continuation SLURM script that resumes an existing run from a checkpoint,
following the project convention used across `live_dealer/train/*_leo_<N>.sh`.

## What a continuation script changes (vs. the base script)

Everything is preserved except the launch args:
- `--output_path` gets a `_cont<STEP>_SF<SF>` suffix (a **new** run dir; the source run's
  checkpoints are never touched). A pre-existing `_cont…_SF…` suffix is stripped first so
  chained continuations don't grow unbounded.
- `--lora_checkpoint <run>/step-<STEP>.safetensors` is added/replaced.
- `--skip_frames <SF>` is added/replaced.
- `#SBATCH --job-name` becomes a concise `<run>_sf<SF>_cont<STEP>`.

`<STEP>` defaults to the **latest** `step-*.safetensors` in the run's output dir.

## How to run it

Use the helper (don't hand-edit unless the base script is unusual):

```bash
python3 .claude/skills/continue-training/continue_training.py \
  --base <path/to/base_leo_N.sh> \
  --skip-frames <N> \
  [--step <STEP>]        # default: latest checkpoint \
  [--out <path>]         # default: next *_leo_<N+1>.sh \
  [--submit]             # default: write only, print the sbatch command
```

Default behavior writes the script and prints the suggested `sbatch` command **without
submitting** — let the user review first (they often want to resume from a newer
checkpoint than the one currently on disk). Only pass `--submit` when the user has
clearly asked to launch it now.

## Procedure for the agent

1. Identify the base script. If the user names a run rather than a file, find the
   `*_leo_<N>.sh` whose `--output_path` matches; the most recently opened/edited script
   is a strong default.
2. Run the helper (without `--submit`) to generate the script. It prints the available
   checkpoints and which one it picked.
3. Show the user the generated path, the resume checkpoint, the new run name, and the
   `sbatch` command. Confirm before submitting.
4. On confirmation, `sbatch` it (or re-run with `--submit`) and report the job id +
   `squeue` status.

## Checkpoint-path resolution

The container sees `--output_path /output/<run>`; the host dir is
`$SCRATCH_PROJECT_DIR/outputs/<run>` (the `--bind ${OUTPUT_DIR}:/output` mount). The
helper derives this from the base script automatically; override with `--outputs-dir`
if a script uses a non-standard layout.

## Notes / gotchas

- Submitting an N-node job is outward-facing and hard to reverse — confirm the
  checkpoint step with the user, since an active run may write a newer one any minute.
- The helper requires the base name to match `*_leo_<N>.sh`; pass `--out` for other names.
- It anchors the inserted args after the `--lora_rank` line. If a base script has no
  `--lora_rank`, the helper errors — handle that case manually.
