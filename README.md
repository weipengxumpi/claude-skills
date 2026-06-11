# Claude Code Custom Skills

Custom [Claude Code](https://claude.com/claude-code) skills for working with
**Wan2.2-S2V LoRA training** and **SLURM** on an HPC cluster.

## Skills

| Skill | What it does |
|-------|--------------|
| [`continue-training`](continue-training/) | Resume a Wan2.2-S2V LoRA run from its latest (or chosen) checkpoint. Generates a new `..._leo_<N+1>.sh` SLURM script that resumes via `--lora_checkpoint` with a configurable `--skip_frames`, following the project's `_cont<STEP>_SF<SF>` convention. |
| [`eval-training`](eval-training/) | Generate the `livedealer_infer.py` evaluation command for a training run (script or SLURM job id). Emits the inference snippet on the eyes-only test set with the correct `WAN_*` env flags, `lora_path`/step, width/height, and pose/object inputs. |
| [`slurm-wait-analysis`](slurm-wait-analysis/) | Query `sacct` for your jobs, compute queue wait time and run time, and write a readable Markdown/HTML report with a per-job table, summary cards, and averages. Supports filtering by node count and minimum run time. |

## Installation

Each skill is a directory containing a `SKILL.md` (with name/description
frontmatter) plus its supporting scripts. To use them in a project, copy or
symlink the skill directories into that project's `.claude/skills/`:

```bash
git clone https://github.com/<your-username>/claude-skills.git
ln -s "$(pwd)/claude-skills/continue-training"   /path/to/project/.claude/skills/continue-training
ln -s "$(pwd)/claude-skills/eval-training"       /path/to/project/.claude/skills/eval-training
ln -s "$(pwd)/claude-skills/slurm-wait-analysis" /path/to/project/.claude/skills/slurm-wait-analysis
```

Or install them user-wide under `~/.claude/skills/`.

Claude Code discovers each skill from its `SKILL.md` and invokes it when your
request matches the skill's description.
