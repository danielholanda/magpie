# Installing the Magpie skill

The Magpie skill lets AI agents (Cursor, Claude Code, Codex, etc.) perform all Magpie tasks via CLI instructions when MCP is not available. The skill is stored in this repo under **`skills/magpie/`** (IDE-neutral). To use it in an editor, install it into that editor’s skills location using the script below or the manual steps.

## Install script (recommended)

From the Magpie repo root:

```bash
# Make the script executable once
chmod +x skills/install-skill.sh

# Install for Cursor (global: ~/.cursor/skills/magpie)
./skills/install-skill.sh cursor

# Install for Claude Code (global: ~/.claude/skills/magpie)
./skills/install-skill.sh claude

# Install for Codex (global: ~/.codex/skills/magpie)
./skills/install-skill.sh codex

# Install for all three
./skills/install-skill.sh all

# Project scope: install into current directory for Cursor
./skills/install-skill.sh cursor --project

# Project scope: install into a specific project directory
./skills/install-skill.sh claude --project /path/to/your/project
```

No restart is usually required; each editor discovers skills automatically.

## Manual install

Source folder: **`skills/magpie/`** in this repo (contains `SKILL.md`, `reference.md`, `examples.md`).

### Cursor

- **Global:** Copy the skill into your Cursor skills folder:
  ```bash
  cp -r /path/to/Magpie/skills/magpie ~/.cursor/skills/magpie
  ```
- **Project:** Copy into your project’s Cursor skills folder so only that project uses it:
  ```bash
  mkdir -p /path/to/your/project/.cursor/skills
  cp -r /path/to/Magpie/skills/magpie /path/to/your/project/.cursor/skills/magpie
  ```
  (Replace `/path/to/Magpie` and `/path/to/your/project` with the actual paths.)

### Claude Code

Same `SKILL.md` format and layout.

- **Global:**
  ```bash
  mkdir -p ~/.claude/skills
  cp -r /path/to/Magpie/skills/magpie ~/.claude/skills/magpie
  ```
- **Project:**
  ```bash
  mkdir -p /path/to/your/project/.claude/skills
  cp -r /path/to/Magpie/skills/magpie /path/to/your/project/.claude/skills/magpie
  ```

### Codex and other IDEs

- If the IDE has a **skills** or **custom instructions** directory (e.g. `~/.codex/skills/` or a project `.codex/skills/`), copy the **`magpie`** directory there:
  ```bash
  mkdir -p ~/.codex/skills
  cp -r /path/to/Magpie/skills/magpie ~/.codex/skills/magpie
  ```
  Use the path your IDE documents for custom skills.
- If the IDE only has a **single “custom instructions”** or “rules” field, paste the body of [skills/magpie/SKILL.md](../skills/magpie/SKILL.md) (the markdown after the YAML frontmatter) into that field, and add a note that these instructions apply when working with Magpie, GPU kernel analysis/compare, or vLLM/SGLang benchmarks.

## Verifying the skill

1. **Discovery:** In the target IDE, ask: “What can you do with Magpie?” or “I want to analyze a HIP kernel with Magpie.” The agent should use the Magpie skill (reference its instructions or run Magpie commands).
2. **Correctness:** Ask “Show GPU info using Magpie.” The agent should run `magpie --gpu-info` or `python -m Magpie --gpu-info` from the Magpie repo (or from a directory where Magpie is installed).
3. **CLI check:** From the Magpie repo root, run `magpie --gpu-info` (or `python -m Magpie --gpu-info`) to confirm the CLI and environment work.
