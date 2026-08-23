# Plan: Upgrade PySpice's bundled ngspice 34 → 39

**Status:** PLANNED — no execution yet. Awaiting user-supplied download.

## Goal

Fix VM-08 (ngspice-34 mis-evaluates the CPM2 behavioral switch equation at
high Vgs) and restore `ddt()` support (disabled Crss/Qgd charge source) by
swapping the ngspice shared library that PySpice loads, from the bundled
ngspice-34 to an official ngspice-39 build.

## What you (user) need to download

Place in: `C:\Users\GNKRISHNAN\Documents\GitHub\PieceSpice\download\`

- **ngspice-39_64.zip** — official Windows build from SourceForge:
  https://sourceforge.net/projects/ngspice/files/ng-spice-rework/39/ngspice-39_64.zip
  (~50 MB). It contains `bin\ngspice.dll` plus the console exe and share data.
- Alternative if you prefer the older layout: **ngspice-39_dll_64.zip**
  (DLL-only package, contains just the dll + msys deps).

Either is fine; we only need `ngspice.dll` from it. Do NOT run its installer
or add it to PATH — PieceSpice must keep using the project venv's copy.

## Current state to be preserved (non-destructive)

The active library lives inside the venv:

```
.venv/Lib/site-packages/PySpice/Spice/NgSpice/Spice64_dll/ngspice.dll
```

We will NOT delete or overwrite it in place. Instead:

1. Rename the existing dll to `ngspice.dll.bak34` (instant rollback path).
2. Copy the new ngspice-39 dll in as `ngspice.dll`.
3. Keep the original download zip untouched in `download/`.

## Execution steps (in order, with gates)

### Phase 0 — Preflight
- Confirm `download/` contains the zip; list its contents (unzip -l) and
  locate `bin/ngspice.dll`.
- Record current dll size/hash for rollback verification.
- Verify pytest passes BEFORE any change (baseline: 46 passed).

### Phase 1 — Swap
- Extract only `bin/ngspide.dll` → temp folder under `download/_extract/`.
- Rename old → `.bak34`; copy new → `ngspice.dll`.

### Phase 2 — Smoke gate (must pass before proceeding)
Run a minimal RC-circuit OP through PySpice (`NgSpiceShared`):
- If import/load fails here (ABI mismatch), STOP, roll back (rename back),
  report. No further changes.
- Check `version` output reports ngspice-39.

### Phase 3 — VM-08 verification probes (single batched script)
- Probe 1: `ddt()` inside a B-source expression — previously failed
  ("no such function ddt"); expect OK on 39.
- Probe 2: gmos-equation replica at Vgs = 13/15/20, compare against Python
  ground-truth evaluation (the numbers recorded in the dev log). Expect
  monotonic Id(Vgs) matching Python within a few %.
- If either probe fails: roll back and report; do not proceed to notebook.

### Phase 4 — Full regression
- pytest (expect 46 passed).
- Notebook 01 executed as-is with VENDOR=cpm2-1200:
  - Expect Rds(on) monotonicity restored at high Vgs,
  - Crss now physical (~15 pF class, not flat 1 pF),
  - Qgd/Qg rise toward datasheet values.

### Phase 5 — Undo blocked-out code (ONLY after everything above passes)
Non-destructive review of preprocessor changes made to work around ngspice-34:
- `parser/preprocessor.py`: the block that comments out B/G sources containing
  `ddt(`. On ngspice-39 ddt works, so this workaround becomes unnecessary.
- Change: make the ddt-filter conditional (e.g. honor a flag/version check),
  or remove it — decided after seeing probe results.
- Re-run notebook + pytest after this change to confirm nothing breaks.

## Rollback plan (any failure)

```
cp ngspice.dll.bak34 ngspice.dll   # inside Spice64_dll folder
```
Then re-run pytest to confirm restoration. No other artifacts are touched.

## Explicit non-goals / safety rules

- No edits to vendor files, yaml configs, benches, or notebooks during Phases 0–4.
- The `.spiceinit` (`set ngbehavior=ltpsa`) stays unless Phase 4 shows a
  compatibility-mode problem specific to 39; any change there gets reported first.
- Original ngspice-34 dll is never deleted — only renamed.
- Every step's result is reported before moving to the next phase.

## Acceptance criteria

1. `version` reports ngspice-39 through PySpice shared lib.
2. All 46 tests pass.
3. Notebook 01 runs end-to-end, 0 errors.
4. Output-characteristic family shows monotonic Id(Vgs) ordering at all drives.
5. Crss curve is physical (non-flat, pF-class rising toward low Vds).
