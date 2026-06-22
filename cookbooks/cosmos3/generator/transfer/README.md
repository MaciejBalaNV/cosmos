# Cosmos3 Generator Transfer Examples

Cosmos3-Nano video **transfer** examples for the native PyTorch (Cosmos
Framework) path and the OpenAI-compatible vLLM-Omni server path. Sample assets
under [`assets/`](./assets) cover spatial control signals paired with
`prompt.json` files:

- **Edge (Canny)** — edge map control plus caption.
- **Blur** — blurred-reference control plus caption.
- **Depth** — depth map control plus caption.
- **Segmentation** — segmentation map control plus caption.
- **World scenario (WSM)** — world-scenario map control plus caption.

Environment setup is centralized in the shared
[Cosmos3 cookbooks environment setup](../../README.md) guide.

## Transfer Definition

Video transfer generates a target clip from a `prompt.json` caption and a
precomputed control video on a hint block (`control_path`). The Framework path
uses `model_mode` `video2video` in a local JSON spec. The vLLM-Omni path uses
`POST /v1/videos/sync` and passes the same hint key (`edge`, `blur`, `depth`,
`seg`, or `wsm`) inside `extra_params`.

There is no source RGB video at run time for the checked-in transfer examples.
Output frame count and geometry come from the control video; see the spec field
reference for how `fps` and `aspect_ratio` are resolved. All examples share
`assets/negative_prompt.json` for the negative caption.

| Control | Asset folder | Inference input | Generation duration |
| --- | --- | --- | --- |
| Edge (Canny) | `assets/edge/` | `control_edge.mp4` + `prompt.json` | 121 frames @ 30 FPS |
| Blur | `assets/blur/` | `control_blur.mp4` + `prompt.json` | 121 frames @ 30 FPS |
| Depth | `assets/depth/` | `control_depth.mp4` + `prompt.json` | 121 frames @ 30 FPS |
| Segmentation | `assets/seg/` | `control_seg.mp4` + `prompt.json` | 121 frames @ 30 FPS |
| World scenario (WSM) | `assets/wsm/` | `control_wsm.mp4` + `prompt.json` | 101 frames @ 10 FPS |

Transfer inference is selected automatically when any hint key is present in the
Framework spec or in vLLM-Omni `extra_params`.

## Run with Cosmos Framework

### Quickstart

Set up the environment: [Cosmos Framework setup](../../README.md#cosmos-framework).
Activate the framework venv, then run inference (checked-in `specs/*.json` use paths
relative to `specs/`). Transfer on Nano looks like:

```bash
cd cookbooks/cosmos3/generator/transfer

# edge
torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i specs/edge.json \
  -o ./output/ \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026

# blur
torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i specs/blur.json \
  -o ./output/ \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026

# depth
torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i specs/depth.json \
  -o ./output/ \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026

# seg
torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i specs/seg.json \
  -o ./output/ \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026

# wsm
torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i specs/wsm.json \
  -o ./output/ \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026
```

The input spec sets `prompt_path` and a hint block with `control_path` pointing at the
checked-in assets under [`assets/`](./assets) via paths relative to [`specs/`](./specs).

Outputs are written under the directory passed to `-o`, with one subdirectory per sample name,
for example `output/transfer_edge/vision.mp4`. Batch size must be 1 for transfer.

## Run with vLLM-Omni

### Quickstart

Set up the environment and start the server:
[vLLM-Omni setup](../../README.md#vllm-omni). Transfer controls require the
native vLLM-Omni install from `main` until the `vllm/vllm-omni:v0.23.0` Docker
image is released. If you use a released image with transfer, run the
Docker command from the `cosmos` repo root so the repo is mounted at
`/workspace` and the server runs from that directory inside the container:

```bash
export COSMOS3_WORKDIR="$(pwd)"
export COSMOS3_HOST_PORT=8000
```

The transfer examples send repo-local `control_path` strings to the server. For
Docker, those paths must be visible from the server working directory. With the
shared Docker setup, the checked-in depth control video is:

```text
cookbooks/cosmos3/generator/transfer/assets/depth/control_depth.mp4
```

If your server does not run from the repo root, start it from the repo root or
adjust `control_path` to a path the server process can read.

Transfer requests should also pass the spec `resolution` inside `extra_params`.
Cosmos3 transfer bucket selection reads `extra_params.resolution` and the
control/input aspect ratio; set the video API `size` field to the matching
`<width>x<height>` value for a consistent request.

Send a depth-transfer request:

```python
import json
from pathlib import Path

import requests

transfer_root = Path("cookbooks/cosmos3/generator/transfer")
prompt = json.dumps(json.load(open(transfer_root / "assets/depth/prompt.json")))
negative = json.dumps(json.load(open(transfer_root / "assets/negative_prompt.json")))
control_path = transfer_root / "assets/depth/control_depth.mp4"

response = requests.post(
    "http://localhost:8000/v1/videos/sync",
    data={
        "prompt": prompt,
        "negative_prompt": negative,
        "size": "1280x720",
        "num_frames": "121",
        "fps": "30",
        "num_inference_steps": "50",
        "guidance_scale": "3.0",
        "flow_shift": "10.0",
        "seed": "2026",
        "extra_params": json.dumps(
            {
                "use_resolution_template": False,
                "use_duration_template": False,
                "guardrails": True,
                "depth": {"control_path": control_path.as_posix()},
                "resolution": "720",
                "control_guidance": 1.5,
                "num_video_frames_per_chunk": 121,
                "max_frames": 121,
            }
        ),
    },
    headers={"Accept": "video/mp4"},
)
response.raise_for_status()
Path("/tmp/cosmos3_transfer_depth.mp4").write_bytes(response.content)
```

### Spec field reference

A representative spec (`specs/edge.json`):

```json
{
  "name": "transfer_edge",
  "model_mode": "video2video",
  "resolution": "720",
  "aspect_ratio": "16,9",
  "num_frames": 121,
  "fps": 30,
  "num_video_frames_per_chunk": 121,
  "num_conditional_frames": 1,
  "num_first_chunk_conditional_frames": 0,
  "share_vision_temporal_positions": true,
  "guidance": 3.0,
  "control_guidance": 1.5,
  "negative_prompt_file": "../assets/negative_prompt.json",
  "prompt_path": "../assets/edge/prompt.json",
  "edge": {
    "control_path": "../assets/edge/control_edge.mp4",
    "preset_edge_threshold": "medium"
  }
}
```

Key fields:

- **`resolution`** — target resolution (e.g. `720` for 720p).

- **`aspect_ratio`** — aspect ratio of the control video; together with `resolution` determines the spatial dimensions (e.g. `720` + `16,9` → 1280 × 720).

- **`fps`** — model conditioning signal and playback rate of the saved output video. Should match the native fps of the control video.

- **`num_frames`** — number of video frames.


### Cookbook entrypoints

- [`run_video_transfer_with_cosmos_framework.ipynb`](./run_video_transfer_with_cosmos_framework.ipynb) —
  full tutorial on a **GPU host**: environment setup, `nvidia-smi` check, then five inference blocks
  (edge, blur, depth, seg, wsm) with previews. See [Cosmos3 environment setup](../../README.md).
- [`run_video_transfer_with_vllm_omni.ipynb`](./run_video_transfer_with_vllm_omni.ipynb) —
  full tutorial against an already-running vLLM-Omni server: endpoint checks, repo-local
  control paths, five transfer requests, and compact previews.
- [`specs/`](./specs) — checked-in Framework input JSON per control (paths relative to `specs/`).

### Troubleshooting

If inference fails inside attention with a NATTEN/libnatten error, verify that the active Python
environment uses a matching Torch and NATTEN build. Avoid mixing a container-provided Torch/NATTEN
stack with packages from `~/.local` or another venv. In containerized environments,
`PYTHONNOUSERSITE=1` can help prevent user-site packages from shadowing the container stack.
