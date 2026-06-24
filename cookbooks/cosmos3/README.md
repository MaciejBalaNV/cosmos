# Cosmos3 Cookbooks: Environment Setup

Shared environment setup for every Cosmos3 cookbook (Reasoner and Generator).
Each cookbook README links back here for the backend(s) it supports — pick the
backend you want to run and follow that one section.

| Backend | Use it for | Used by |
| --- | --- | --- |
| [Cosmos Framework](#cosmos-framework) | Native PyTorch inference, launched with `torchrun` | Reasoner, Generator (Audiovisual, Action, **Transfer**) |
| [Diffusers](#diffusers) | Direct generation with `Cosmos3OmniPipeline` | Generator (Audiovisual) |
| [TensorRT-LLM](#tensorrt-llm) | OpenAI-compatible VisualGen server (image/video generation) | Generator (Audiovisual) |
| [Transformers](#transformers) | Hugging Face Transformers inference | Reasoner |
| [vLLM](#vllm) | OpenAI-compatible reasoning server (image/video understanding) | Reasoner |
| [vLLM-Omni](#vllm-omni) | OpenAI-compatible generation server (image/video/audio/action) | Generator (Audiovisual, Action) |
| [NIM](#nim) | Prebuilt OpenAI-compatible reasoning server (image/video understanding); no venv | Reasoner |

## Prerequisites

- Linux with NVIDIA GPU access.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/), `git`, and `git-lfs` installed.
- Hugging Face access to the gated Cosmos3 model repos. Generator also requires
  access to the gated
  [nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail)
  HF repository. Authenticate once before the first run:

  ```bash
  uvx hf@latest auth login
  # or:
  export HF_TOKEN=<your_token>
  ```

  To disable the guardrail, set `enable_safety_checker=False` (Diffusers),
  `TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1` or `use_guardrails: false` through
  `extra_params` (TensorRT-LLM), `guardrails: false` (vLLM-Omni
  `extra_params`/`extra_args`), or `--no-guardrails` (Cosmos Framework).
- For the Cosmos Framework backend: access to `git@github.com:NVIDIA/cosmos-framework.git`.
- For the NIM backend: an NGC API key (used as `NGC_API_KEY`), which you can generate on [build.nvidia.com](https://build.nvidia.com/nvidia/cosmos3-nano-reasoner) or [NGC](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/cosmos3-reasoner), plus a one-time `docker login nvcr.io` (username `$oauthtoken`, password = your key). The HF login above is not needed for NIM.
- Enough local disk for the venv/image, the uv cache, and the model cache. Nano
  downloads plus CUDA dependencies can take tens of GiB.

### CUDA driver and the `cuXXX` backend

Several backends pin a CUDA build of `torch`/`vllm` that **must match your NVIDIA
driver**. Pick the tag that matches the CUDA version your driver supports:

| Driver CUDA | Backend tag | Notes |
| --- | --- | --- |
| 13.x | `cu130` | Default in the notebooks. |
| 12.x | `cu128` | Use when a compatible wheel is available for the selected package version. |

vLLM does not publish a wheel for every CUDA minor version, so
`--torch-backend=auto` is not reliable here — choose the pair that matches your
driver.

## Cosmos Framework

Native PyTorch inference through the Cosmos Framework checkout. Used by the
`run_*_with_cosmos_framework.ipynb` notebooks and the Cosmos Framework
quickstarts.

From the `cosmos` repo root, clone (or reuse) the framework checkout:

```bash
mkdir -p packages
git clone https://github.com/NVIDIA/cosmos-framework.git packages/cosmos3
cd packages/cosmos3
```

Install the framework dependencies into its venv. The inference path currently
imports modules from the training extras, so use the `*-train` dependency group
that matches your driver (see [CUDA driver and the `cuXXX` backend](#cuda-driver-and-the-cuxxx-backend)):

```bash
# lerobot tracks test artifacts with git-LFS that this cookbook does not need;
# skipping smudge avoids failures from missing LFS blobs in uv's git mirror.
export GIT_LFS_SKIP_SMUDGE=1

# CUDA 13 driver (default):
uv sync --all-extras --group=cu130-train

# CUDA 12.x driver:
# uv sync --all-extras --group=cu128-train
```

The notebooks honor `COSMOS3_UV_GROUP` (default `cu130-train`); set
`export COSMOS3_UV_GROUP=cu128-train` before launching them on CUDA 12.x systems.

This produces a venv at `packages/cosmos3/.venv`. Run framework commands either
by activating it (`source .venv/bin/activate`) or via its absolute interpreter
(`.venv/bin/python`, `.venv/bin/torchrun`).

### Recommended base image (optional)

For CUDA 13, NVIDIA documents the [NGC PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
`nvcr.io/nvidia/pytorch:25.09-py3` as the recommended starting point; for CUDA 12 use
`nvcr.io/nvidia/pytorch:25.06-py3`. See the repo root
[Which base container should I use?](../../README.md#which-base-container-should-i-use)
and [Cosmos Framework setup](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/setup.md#recommended-base-image).

Inside that image (or any minimal GPU host), install the system packages below **before**
your first `torchrun` inference — `uv sync --all-extras` alone is not enough for
guardrails.

### System packages (required for Framework guardrails)

Framework inference enables **guardrails by default**. The video guardrail path imports
OpenCV (via RetinaFace), which needs graphics libraries that are often missing on
headless servers and minimal containers.

From `packages/cosmos3` (or the framework repo root), with `apt-get` available.
NGC and many training containers run as **root** — use `apt-get` directly (no `sudo`).
On a normal host where you are not root, prefix with `sudo`.

```bash
apt-get update
apt-get install -y --no-install-recommends \
  curl ffmpeg git-lfs libgl1 libglib2.0-0 libx11-dev libxcb1 tree wget
```

Verify OpenCV imports after `source .venv/bin/activate`:

```bash
python -c "import cv2; print(cv2.__version__)"
```

If you see `libxcb.so.1: cannot open shared object file`, the `libxcb1` / `libgl1`
packages above were not installed. The same fix is documented in the repo root
[troubleshooting guide](../../README.md#import-fails-with-libxcbso1-cannot-open-shared-object-file).

When using the **NGC PyTorch base image**, clear `LD_LIBRARY_PATH` after activating the
venv so the container’s bundled libtorch does not shadow the venv (see
[Cosmos Framework FAQ — PyTorch import inside NGC](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/faq.md)):

```bash
source .venv/bin/activate
export LD_LIBRARY_PATH=
```

Guardrails also require Hugging Face access to the gated safety models (accept the
license and set `HF_TOKEN` as in [Prerequisites](#prerequisites)). To disable guardrails
for a one-off run, pass `--no-guardrails` to `cosmos_framework.scripts.inference`.

## Diffusers

Direct generation with `Cosmos3OmniPipeline` (Generator · Audiovisual). Create a
venv and install the backend, choosing `--torch-backend` to match your driver
(see [CUDA driver and the `cuXXX` backend](#cuda-driver-and-the-cuxxx-backend)):

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate

uv pip install --torch-backend=cu130 \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate \
  av \
  cosmos_guardrail \
  huggingface_hub \
  imageio \
  imageio-ffmpeg \
  torch \
  torchvision \
  transformers
```

## TensorRT-LLM

OpenAI-compatible **VisualGen** server for Generator audiovisual text-to-image,
text-to-video, and image-to-video examples. Cosmos3 support was added in TensorRT-LLM PR
[#14824](https://github.com/NVIDIA/TensorRT-LLM/pull/14824); use a
TensorRT-LLM checkout or package that includes that change.

Install TensorRT-LLM following its upstream documentation.

To build TensorRT-LLM from source, follow NVIDIA's
[Build from Source](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source.html)
guide. This is the right path when you need a checkout that contains a recent
Cosmos3 VisualGen change before it is available in your installed package or
release image.

```bash
apt-get update && apt-get -y install git git-lfs
git lfs install

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull

# Pick a devel tag from the upstream build-from-source guide or NGC.
docker pull nvcr.io/nvidia/tensorrt-llm/devel:<tag>
docker run --rm -it \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all \
  --volume "$PWD":"$PWD" \
  --workdir "$PWD" \
  nvcr.io/nvidia/tensorrt-llm/devel:<tag>

# Inside the container:
python3 scripts/build_wheel.py --use_ccache --skip_building_wheel --linking_install_binary
pip install -e .
```

For Python-only changes, the upstream guide also documents
`TRTLLM_USE_PRECOMPILED=1 pip install -e .` to reuse precompiled binaries while
installing the checkout in editable mode.

Then install the Cosmos3 guardrail package in the same environment unless you
explicitly disable guardrails before starting the server:

```bash
pip install cosmos_guardrail==0.3.0
# If needed by your OpenCV stack:
# pip uninstall opencv-python
```

Set the TensorRT-LLM source root for the shared VisualGen config YAMLs:

```bash
export TRTLLM_ROOT="${TRTLLM_ROOT:-$PWD/TensorRT-LLM}"
export COSMOS3_TRTLLM_PORT="${COSMOS3_TRTLLM_PORT:-8000}"
```

**Cosmos3-Nano** (single GPU):

```bash
trtllm-serve nvidia/Cosmos3-Nano \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml" \
  --port "$COSMOS3_TRTLLM_PORT"
```

**Cosmos3-Super** (four GPUs; CFG parallelism with Ulysses, plus parallel VAE):

```bash
torchrun --nproc_per_node=4 -m tensorrt_llm.commands.serve \
  nvidia/Cosmos3-Super \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-super-4gpu.yaml" \
  --port "$COSMOS3_TRTLLM_PORT"
```

The server exposes `/health`, `/v1/videos/generations`, `/v1/videos`, and
`/v1/images/generations`. The audiovisual notebook uses the validated video
generation endpoint for text-to-image, text-to-video, and image-to-video. Cosmos3
text-to-image is sent as a one-frame video request, matching the TensorRT-LLM
Cosmos3 pipeline; the notebook sends it as `num_frames=1`, `seconds=1`, and
`fps=8` to satisfy the video request schema while preserving a single generated
frame. Requests send Cosmos3 controls through `extra_params`,
so use a TensorRT-LLM build that includes the Cosmos3 VisualGen API schema.
The notebook sets request-level `max_sequence_length=2048` for longer structured
JSON prompts.

## Transformers

Local Python inference for the Cosmos3 Reasoner. This backend uses the
Transformers Cosmos3 integration and loads only the Reasoner tower from the
unified Cosmos3 checkpoint.

Cosmos3 support first appears in the Transformers `v5.11.0` release tag. Create
a venv and install Transformers `5.11.0` or newer:

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate

uv pip install --torch-backend=auto \
  accelerate \
  av \
  pillow \
  "safetensors>=0.8.0" \
  torch \
  torchvision \
  "transformers>=5.11.0"
```

`--torch-backend=auto` lets uv detect the CUDA build of `torch`/`torchvision`
that matches your NVIDIA driver. Pin a backend such as `cu128` or `cu130` if
your environment needs an explicit CUDA wheel.

Use `Cosmos3OmniForConditionalGeneration` with `AutoProcessor` and either
`nvidia/Cosmos3-Nano` or `nvidia/Cosmos3-Super`. See the
[Reasoner Transformers quickstart](reasoner/README.md#run-with-transformers)
for a runnable image example and video input notes.

## vLLM

OpenAI-compatible **reasoning** server for the Reasoner cookbook (image/video
understanding). Native Cosmos3 Reasoner support first appears in the vLLM
`v0.23.0` stable release:

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate

# CUDA 13 driver:
uv pip install --torch-backend=cu130 "vllm>=0.23.0"

# CUDA 12.x driver:
# uv pip install --torch-backend=cu128 "vllm>=0.23.0"
```

The vLLM wheel and the torch backend must be compatible — see
[CUDA driver and the `cuXXX` backend](#cuda-driver-and-the-cuxxx-backend).

If your vLLM build reports that DeepGEMM is unavailable, disable it before
starting the server:

```bash
export VLLM_USE_DEEP_GEMM=0
```

> When launching with `.venv/bin/vllm` instead of activating the venv, make sure
> `.venv/bin` is on `PATH` (e.g. `source .venv/bin/activate`). FlashInfer's
> just-in-time kernel build shells out to `ninja`, which lives in the venv.

### Start the server

All Reasoner cookbooks talk to an OpenAI-compatible chat-completions API. After
[installing vLLM](#vllm), run the commands below from
`cookbooks/cosmos3/reasoner` (same working directory as
[`run_with_vllm.ipynb`](reasoner/run_with_vllm.ipynb)). That sets
`$(dirname "$(pwd)")` to `<cosmos>/cookbooks/cosmos3`, which matches the
notebook's `COSMOS3_MEDIA_ROOT`.

**Cosmos3-Nano** (single GPU, port 8000):

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve nvidia/Cosmos3-Nano \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --allowed-local-media-path "$(dirname "$(pwd)")" \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --port 8000
```

**Cosmos3-Super** (four GPUs; default in [`run_with_vllm.ipynb`](reasoner/run_with_vllm.ipynb), port 8001):

```bash
export COSMOS3_MEDIA_ROOT="$(dirname "$(pwd)")"
export VLLM_PORT="${VLLM_PORT:-8001}"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve nvidia/Cosmos3-Super \
  --tensor-parallel-size 4 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --allowed-local-media-path "$COSMOS3_MEDIA_ROOT" \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --port "$VLLM_PORT"
```

The Super notebook polls `/health` for up to 1800 seconds on first start while CUDA
graphs compile.

| Option | Use |
| --- | --- |
| `--tensor-parallel-size` | Number of GPUs for tensor-parallel inference |
| `--mm-encoder-tp-mode data` | Data parallelism for the visual encoder |
| `--media-io-kwargs '{"video": {"num_frames": -1}}'` | Lets the processor see all frames before downstream sampling |
| `--allowed-local-media-path` | Must cover local `file://` media paths; defaults to `<cosmos>/cookbooks/cosmos3` when run from `cookbooks/cosmos3/reasoner` |

## vLLM-Omni

OpenAI-compatible **generation** server (image/video/audio/action) for the
Generator cookbooks.

Cosmos3 checkpoints can exceed the default server init timeout — always pass
`--init-timeout 1800` on every `vllm serve` command below.

### Option 1: Docker (recommended)

The prebuilt image `vllm/vllm-omni:cosmos3` supports every Generator modality
(including action). Pull once:

```bash
docker pull vllm/vllm-omni:cosmos3
```

Set paths once; adjust for your checkout and cache location:

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export COSMOS3_WORKDIR="${COSMOS3_WORKDIR:-$(pwd)}"
export COSMOS3_HOST_PORT="${COSMOS3_HOST_PORT:-8000}"
```

The container listens on port 8000; `-p "${COSMOS3_HOST_PORT}:8000"` publishes it
on the host. Generator notebooks often use `COSMOS3_HOST_PORT=8001` so port 8000
stays free for a Reasoner server.

**Cosmos3-Nano** (single GPU):

```bash
docker run --runtime nvidia --gpus '"device=0"' \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${COSMOS3_WORKDIR}:/workspace" \
  -p "${COSMOS3_HOST_PORT}:8000" --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --port 8000 \
    --init-timeout 1800
```

**Cosmos3-Super** (all GPUs; add tensor parallelism and layerwise offload):

```bash
docker run --runtime nvidia --gpus all \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${COSMOS3_WORKDIR}:/workspace" \
  -p "${COSMOS3_HOST_PORT}:8000" --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Super \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --tensor-parallel-size 4 \
    --enable-layerwise-offload \
    --port 8000 \
    --init-timeout 1800
```

Mount any directory that holds local media or action JSON files referenced in
requests. Set `--allowed-local-media-path /` (as above) when the whole container
filesystem should be readable.

vLLM-Omni prints `Application startup complete.` when the API is ready.

### Option 2: Native venv (limited modalities)

To install from the upstreaming PR branch instead of Docker (text-to-image,
text-to-video, and image-to-video only — not action or sound yet), create a venv
and pick the CUDA build that matches your driver (see
[CUDA driver and the `cuXXX` backend](#cuda-driver-and-the-cuxxx-backend)):

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate

# CUDA 13 driver:
uv pip install --torch-backend=cu130 \
  "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@refs/pull/3454/head"

# CUDA 12.x driver:
# uv pip install --torch-backend=cu128 \
#   "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@refs/pull/3454/head"
```

Run the same `vllm serve` arguments as in the Docker commands above, directly on
the host (no `docker run` wrapper):

```bash
vllm serve nvidia/Cosmos3-Nano \
  --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --allowed-local-media-path / \
  --port 8000 \
  --init-timeout 1800
```

For Super, add `--tensor-parallel-size 4 --enable-layerwise-offload`.

Additional parallelism options (Docker or native):

| Option | Use |
| --- | --- |
| `--cfg-parallel-size 2` | Runs positive and negative CFG branches on two GPUs |
| `--ulysses-degree 2` | Ulysses sequence parallelism across GPUs |

Ensure the server has enough GPUs for the product of enabled degrees
(`tensor_parallel_size` × `cfg_parallel_size` × `ulysses_degree`).

## NIM

A prebuilt container that serves the Reasoner over an OpenAI-compatible API for
image and video understanding. Like vLLM-Omni this is a Docker image, so there is
no venv or `--torch-backend` to manage; unlike the other backends it
authenticates with an NGC API key instead of Hugging Face (see
[Prerequisites](#prerequisites)).

Start a Nano server (publishes the OpenAI-compatible API on port 8000; the first
run downloads the model into `~/.cache/nim`):

```bash
export NGC_API_KEY=<your_key>

docker run --runtime=nvidia --gpus all \
  --shm-size=32GB \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -e NIM_MODEL_SIZE=nano \
  -v ~/.cache/nim:/opt/nim/.cache \
  -u $(id -u) \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0
```

For **Cosmos3-Super-Reasoner** (the larger model), set `-e NIM_MODEL_SIZE=super`.
The container serves `nvidia/cosmos3-nano-reasoner` (or
`nvidia/cosmos3-super-reasoner`); pass that exact name as the request `model`, or
resolve it dynamically with `client.models.list()`.

## Verify the environment

For the Cosmos Framework / Diffusers / vLLM venvs, check that PyTorch sees the GPU:

```bash
.venv/bin/python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

For a vLLM / vLLM-Omni / NIM server, confirm it is serving the model (use the host
port you set with `COSMOS3_HOST_PORT` or `VLLM_PORT`):

```bash
curl http://localhost:8000/v1/models
```

A NIM server also exposes a readiness endpoint that returns `200` once the model
is loaded:

```bash
curl http://localhost:8000/v1/health/ready
```
