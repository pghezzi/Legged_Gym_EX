# Depth-WaQ TorchScript Exporter

Export single- or multi-checkpoint Depth-WaQ policies to TorchScript for deployment. Supports combining a base policy with FFT (foot-force-terrain?) specialist checkpoints that can be hot-swapped at inference time.

## Overview

The script (`legged_gym/scripts/depth_waq_exporter.py`) loads one or more `ActorCriticDreamWaQDepth` checkpoints and exports them as TorchScript modules:

- **`policy.pt`** — the combined policy (depth encoder + VAE + actor in one module)
- **`DepthCNN.pt`** — the depth encoder split out on its own
- **`FeaturesWaQ.pt`** — the VAE + actor split out on its own
- **`manifest.json`** — metadata describing the export (mode, checkpoint sources, input signatures, and switching indices)

## Modes

### `single`

Exports exactly one checkpoint as a standalone policy. The exported modules still expose a `swap(index)` method for interface compatibility, but it's a no-op.

### `multi`

Exports a base checkpoint plus one or more specialist checkpoints into a single switchable module. Use `swap(index)` on the exported modules to select which policy is active:

- `swap(-1)` → base policy
- `swap(0)` → first specialist
- `swap(1)` → second specialist
- ...and so on

## Usage

### Export a single policy

```bash
python legged_gym/scripts/depth_waq_exporter.py single \
    --checkpoint /path/to/model_10000.pt \
    --args-file /path/to/current_actor_args.pt
```

### Export a multi-policy (base + specialists)

```bash
python legged_gym/scripts/depth_waq_exporter.py multi \
    --checkpoint /path/to/base/model.pt \
    --checkpoint /path/to/gap/model.pt \
    --checkpoint /path/to/stairs/model.pt \
    --checkpoint /path/to/pit/model.pt \
    --args-file /path/to/current_actor_args.pt
```

## Arguments

| Flag | Short | Required | Repeatable | Description |
|---|---|---|---|---|
| `mode` | — | yes | no | `single` or `multi` |
| `--checkpoint` | `-c` | yes | yes | Path to a model checkpoint. Repeat once per policy in `multi` mode. |
| `--args-file` | `-a` | yes | yes | Path to `current_actor_args.pt`. Pass once to share across all checkpoints, or once per checkpoint. |
| `--output-dir` | `-o` | no | no | Directory to write exported artifacts. Defaults to `exported/<timestamp>_<mode>_fft` under the project root. |

**Notes:**
- `single` mode requires exactly one `--checkpoint`.
- `multi` mode requires at least two `--checkpoint` values (one base + one or more specialists).
- `--args-file` must be supplied either once (shared by all checkpoints) or once per checkpoint (matching order).

## Output

Given an output directory, the script writes:

```
<output-dir>/
├── policy.pt        # combined depth encoder + VAE + actor
├── DepthCNN.pt       # depth encoder only
├── FeaturesWaQ.pt     # VAE + actor only
└── manifest.json      # export metadata
```

### `manifest.json` contents

- `mode` — `"single"` or `"multi"`
- `policy_count` — number of checkpoints exported
- `specialist_count` — number of specialists (0 in single mode)
- `artifacts` — filenames for each exported module
- `inputs` — expected input tensor names for each module's `forward()`
- `switching` — base/specialist index mapping (`null` in single mode)
- `sources` — resolved paths to the original checkpoint and args files used

## Requirements

- Python 3
- PyTorch (`torch`)
- `rsl_rl` with `ActorCriticDreamWaQDepth` available on the Python path

## How it works

1. Each checkpoint is loaded on CPU using the corresponding `current_actor_args.pt` to reconstruct the `ActorCriticDreamWaQDepth` architecture, then its `state_dict` is loaded and the model is set to eval mode.
2. Depending on `mode`, the checkpoints are wrapped in either:
   - `SinglePolicyExporterDepthWaQ` (single mode), or
   - `PolicyExporterDepthWaQ` (multi mode, with `swap()` support across `ModuleList`s of actors, VAEs, and visual encoders)
3. The combined module is compiled with `torch.jit.script` and saved.
4. The module is split into a depth-encoder-only and actor-only submodule (via `split_cnn()`), each of which is also scripted and saved separately.
5. A `manifest.json` is written alongside the TorchScript files to document the export without embedding metadata inside the scripted modules themselves.
