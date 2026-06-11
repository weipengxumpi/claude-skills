#!/usr/bin/env python3
"""Generate a continuation SLURM script from an existing training run.

Given a base training script, this:
  1. reads the run's --output_path (the run that produced the checkpoints),
  2. resolves the host checkpoint dir and picks the latest step-*.safetensors
     (or a specific step via --step),
  3. writes a new `..._leo_<N+1>.sh` that resumes from that checkpoint with
     --lora_checkpoint + --skip_frames, an updated --output_path suffix
     (_cont<STEP>_SF<SF>) and a concise #SBATCH --job-name.

It only edits the launch args; everything else in the base script is preserved.
Use --submit to sbatch the result; by default it just writes the file and
prints the suggested sbatch command so you can review first.
"""
import argparse
import os
import re
import subprocess
import sys


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def parse_arg_value(text, flag):
    """Return the value passed to a long flag like `--output_path /foo \\`."""
    m = re.search(rf"--{re.escape(flag)}\s+(\S+)", text)
    return m.group(1) if m else None


def resolve_host_outputs(text, override):
    if override:
        return override
    # Convention in these templates: OUTPUT_DIR defaults to SCRATCH_PROJECT_DIR/outputs,
    # and the container binds OUTPUT_DIR -> /output.
    m = re.search(r"^SCRATCH_PROJECT_DIR=(\S+)", text, re.MULTILINE)
    if not m:
        fail("could not find SCRATCH_PROJECT_DIR in base script; pass --outputs-dir explicitly")
    scratch = m.group(1).split("#")[0].strip()
    return os.path.join(scratch, "outputs")


def latest_step(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        fail(f"checkpoint dir does not exist: {ckpt_dir}")
    steps = []
    for f in os.listdir(ckpt_dir):
        m = re.fullmatch(r"step-(\d+)\.safetensors", f)
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        fail(f"no step-*.safetensors checkpoints found in {ckpt_dir}")
    return max(steps), sorted(steps)


def next_script_path(base_path):
    d, name = os.path.dirname(base_path), os.path.basename(base_path)
    m = re.match(r"(.*_leo_)(\d+)(\.sh)$", name)
    if not m:
        fail(f"base script name does not match *_leo_<N>.sh: {name}")
    prefix, num, ext = m.group(1), int(m.group(2)), m.group(3)
    n = num + 1
    while os.path.exists(os.path.join(d, f"{prefix}{n}{ext}")):
        n += 1
    return os.path.join(d, f"{prefix}{n}{ext}")


def short_jobname(run_name, step, sf):
    # e.g. leo_new_eyes_only_16nodes_card_sequential_rope_compression -> "rope_compression"
    tail = re.sub(r"_cont\d+_SF\d+$", "", run_name)
    tail = "_".join(tail.split("_")[-2:]) or tail
    return f"{tail}_sf{sf}_cont{step}"[:64]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="path to base training .sh script")
    ap.add_argument("--skip-frames", type=int, required=True, dest="sf")
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step to resume from (default: latest)")
    ap.add_argument("--outputs-dir", default=None,
                    help="host dir holding run output folders (default: derived from SCRATCH_PROJECT_DIR/outputs)")
    ap.add_argument("--out", default=None, help="output script path (default: next *_leo_<N+1>.sh)")
    ap.add_argument("--submit", action="store_true", help="sbatch the generated script")
    args = ap.parse_args()

    if not os.path.isfile(args.base):
        fail(f"base script not found: {args.base}")
    text = open(args.base).read()

    base_run = parse_arg_value(text, "output_path")
    if not base_run:
        fail("no --output_path found in base script")
    run_name = os.path.basename(base_run.rstrip("/"))           # container path /output/<run>
    container_out_root = os.path.dirname(base_run.rstrip("/"))  # e.g. /output

    host_outputs = resolve_host_outputs(text, args.outputs_dir)
    ckpt_dir = os.path.join(host_outputs, run_name)

    if args.step is not None:
        step = args.step
        ckpt = os.path.join(ckpt_dir, f"step-{step}.safetensors")
        if not os.path.isfile(ckpt):
            fail(f"requested checkpoint does not exist: {ckpt}")
    else:
        step, all_steps = latest_step(ckpt_dir)
        print(f"checkpoints in {run_name}: {', '.join(f'step-{s}' for s in all_steps)}")
        print(f"-> resuming from latest: step-{step}")

    # New run name: strip any prior _cont/_SF suffix so chained continuations don't grow unbounded.
    stripped = re.sub(r"_cont\d+_SF\d+$", "", run_name)
    new_run = f"{stripped}_cont{step}_SF{args.sf}"
    new_output_path = f"{container_out_root}/{new_run}"
    ckpt_container = f"{container_out_root}/{run_name}/step-{step}.safetensors"

    new = text
    # 1) job-name
    new = re.sub(r"(#SBATCH --job-name=)\S+(.*)",
                 lambda m: f"{m.group(1)}{short_jobname(run_name, step, args.sf)}"
                           f"            # continue {run_name} @ step-{step}, skip_frames {args.sf}",
                 new, count=1)
    # 2) output_path
    new = re.sub(r"(--output_path\s+)\S+", rf"\g<1>{new_output_path}", new, count=1)
    # 3) drop any existing checkpoint/skip_frames lines so we don't duplicate
    new = re.sub(r"\n\s*--lora_checkpoint\s+\S+\s*\\", "", new)
    new = re.sub(r"\n\s*--skip_frames\s+\S+\s*\\", "", new)
    # 4) insert fresh checkpoint + skip_frames right after --lora_rank <n> \
    m = re.search(r"(\n)(\s*)--lora_rank\s+\S+\s*\\", new)
    if not m:
        fail("could not find a `--lora_rank` line to anchor the inserted args")
    indent = m.group(2)
    insert = (f"{m.group(0)}"
              f"\n{indent}--lora_checkpoint {ckpt_container} \\"
              f"\n{indent}--skip_frames {args.sf} \\")
    new = new[:m.start()] + insert + new[m.end():]

    out_path = args.out or next_script_path(args.base)
    with open(out_path, "w") as f:
        f.write(new)
    os.chmod(out_path, 0o755)

    print(f"\nwrote: {out_path}")
    print(f"  job-name   : {short_jobname(run_name, step, args.sf)}")
    print(f"  resume from: {ckpt_container}")
    print(f"  new run    : {new_output_path}")
    print(f"  skip_frames: {args.sf}")

    if args.submit:
        r = subprocess.run(["sbatch", out_path], capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        sys.exit(r.returncode)
    else:
        print(f"\nreview, then submit with:\n  sbatch {out_path}")


if __name__ == "__main__":
    main()
