# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run a one-shot Cosmos3 DROID policy rollout through vLLM-Omni.

This is the command-line equivalent of ``run_policy_with_vllm_omni.ipynb``.
It composes the first frames from the checked-in DROID camera videos, submits
them to the asynchronous ``/v1/videos`` API, and writes both the generated
rollout and the predicted action chunk to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
import requests
from PIL import Image


DEFAULT_PROMPT = "Pick up the object and place it in the target container."
ACTION_VIDEO_RES_SIZE_INFO = {
    "480": {
        "1,1": (640, 640),
        "4,3": (736, 544),
        "3,4": (544, 736),
        "16,9": (832, 480),
        "9,16": (480, 832),
    }
}
CAMERA_VIDEO_RELATIVE_PATHS = {
    "observation/wrist_image_left": (
        "videos/observation.image.wrist_image_left/chunk-000/file-000.mp4"
    ),
    "observation/exterior_image_1_left": (
        "videos/observation.image.exterior_image_1_left/chunk-000/file-000.mp4"
    ),
    "observation/exterior_image_2_left": (
        "videos/observation.image.exterior_image_2_left/chunk-000/file-000.mp4"
    ),
}


def find_repo_root(start: Path) -> Path:
    """Find the cosmos cookbook repository root."""
    for path in (start, *start.parents):
        if (path / "README.md").exists() and (path / "cookbooks").exists():
            return path
    raise RuntimeError(f"Could not find the cosmos repository root from {start}")


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path.parent)
    action_root = repo_root / "cookbooks" / "cosmos3" / "generator" / "action"
    default_output_root = Path(
        os.environ.get(
            "COSMOS3_VLLM_OUTPUT_ROOT",
            repo_root / "outputs" / "cosmos3_action_vllm",
        )
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("COSMOS3_VLLM_BASE_URL", "http://localhost:8001"),
        help="vLLM-Omni server URL (default: %(default)s)",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=action_root / "assets" / "droid_lerobot_example",
        help="DROID LeRobot sample directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help="Root directory for generated inputs and outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        default=os.environ.get("COSMOS3_POLICY_PROMPT", DEFAULT_PROMPT),
        help="Robot instruction (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--server-timeout", type=float, default=600)
    parser.add_argument("--server-check-interval", type=float, default=10)
    parser.add_argument(
        "--job-timeout",
        type=float,
        default=1800,
        help="Maximum seconds to wait for the submitted video job",
    )
    parser.add_argument("--poll-interval", type=float, default=2)
    return parser.parse_args()


def extract_first_frame(video_path: Path, output_path: Path) -> Path:
    """Extract frame zero, refreshing an older cached output if necessary."""
    if not output_path.exists() or output_path.stat().st_mtime < video_path.stat().st_mtime:
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                str(output_path),
            ],
            check=True,
        )
    return output_path


def prepare_policy_image(asset_root: Path, input_dir: Path) -> Path:
    """Compose wrist and exterior camera frames in the policy layout."""
    video_paths = {
        key: asset_root / relative_path
        for key, relative_path in CAMERA_VIDEO_RELATIVE_PATHS.items()
    }
    missing = [f"{key}: {path}" for key, path in video_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing DROID camera video(s):\n" + "\n".join(missing))

    input_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = {
        key: extract_first_frame(
            video_path,
            input_dir / f"policy_{key.rsplit('/', 1)[-1]}.png",
        )
        for key, video_path in video_paths.items()
    }
    frames: dict[str, Image.Image] = {}
    try:
        frames = {key: Image.open(path).convert("RGB") for key, path in frame_paths.items()}
        wrist = frames["observation/wrist_image_left"]
        if any(frame.size != wrist.size for frame in frames.values()):
            sizes = {key: frame.size for key, frame in frames.items()}
            raise ValueError(f"DROID camera frames must have matching dimensions: {sizes}")

        bottom_height = wrist.height // 2
        half_width = wrist.width // 2
        left = frames["observation/exterior_image_1_left"].resize(
            (half_width, bottom_height), Image.Resampling.BILINEAR
        )
        right = frames["observation/exterior_image_2_left"].resize(
            (half_width, bottom_height), Image.Resampling.BILINEAR
        )
        try:
            policy_image = Image.new("RGB", (wrist.width, wrist.height + bottom_height))
            policy_image.paste(wrist, (0, 0))
            policy_image.paste(left, (0, wrist.height))
            policy_image.paste(right, (half_width, wrist.height))
            output_path = input_dir / "droid_policy_first_frame.png"
            policy_image.save(output_path)
        finally:
            left.close()
            right.close()
    finally:
        for frame in frames.values():
            frame.close()

    print(f"Saved conditioning image: {output_path}")
    return output_path


def wait_for_server(
    base_url: str,
    timeout_s: float,
    interval_s: float,
) -> str:
    """Wait for model metadata and return the first active model ID."""
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=10)
            response.raise_for_status()
            payload = response.json()
            model_ids = [
                entry.get("id")
                for entry in payload.get("data", [])
                if entry.get("id")
            ]
            if not model_ids:
                raise RuntimeError(f"Server returned no model IDs: {payload}")
            return str(model_ids[0])
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            print(f"Waiting for vLLM server at {base_url}: {exc}")
            time.sleep(interval_s)
    raise RuntimeError(
        f"vLLM server did not become ready at {base_url} within {timeout_s:g}s"
    ) from last_error


def closest_action_size(height: int, width: int) -> tuple[int, int]:
    input_ratio = height / width
    return min(
        ACTION_VIDEO_RES_SIZE_INFO["480"].values(),
        key=lambda size: abs(input_ratio - size[1] / size[0]),
    )


def make_edge_policy_prompt(
    instruction: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
) -> str:
    aspect_ratio = next(
        ratio
        for ratio, size in ACTION_VIDEO_RES_SIZE_INFO["480"].items()
        if size == (width, height)
    )
    duration_seconds = num_frames / fps
    prompt = {
        "cinematography": {
            "framing": (
                "This video contains concatenated views from multiple camera perspectives. "
                "The top row is the wrist camera and the bottom row contains two external cameras."
            )
        },
        "actions": [
            {
                "time": f"0:00-0:{round(duration_seconds):02d}",
                "description": instruction.rstrip(".!?") + ".",
            }
        ],
        "duration": f"{int(duration_seconds)}s",
        "fps": float(fps),
        "resolution": {"H": height, "W": width},
        "aspect_ratio": aspect_ratio,
    }
    return json.dumps(prompt, separators=(",", ":"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def submit_policy_video(
    *,
    base_url: str,
    active_model: str,
    policy_image_path: Path,
    prompt: str,
    run_dir: Path,
    seed: int,
    num_inference_steps: int,
    job_timeout_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    """Submit, poll, validate, and download one policy rollout."""
    run_dir.mkdir(parents=True, exist_ok=True)
    request_image_path = run_dir / "policy_input.png"
    with Image.open(policy_image_path) as policy_image:
        policy_image.save(request_image_path)
        input_width, input_height = policy_image.size

    target_width, target_height = closest_action_size(input_height, input_width)
    request_prompt = (
        make_edge_policy_prompt(prompt, target_width, target_height, 17, 15)
        if "Cosmos3-Edge-Policy-DROID" in active_model
        else prompt
    )
    extra_params = {
        "action_mode": "policy",
        "domain_name": "droid_lerobot",
        "raw_action_dim": 8,
        "action_chunk_size": 16,
        "image_size": 480,
        "guardrails": False,
    }
    form: dict[str, Any] = {
        "prompt": request_prompt,
        "num_frames": 17,
        "fps": 15,
        "size": f"{target_width}x{target_height}",
        "num_inference_steps": num_inference_steps,
        "guidance_scale": 1.0,
        "flow_shift": 5.0,
        "seed": seed,
        "extra_params": json.dumps(extra_params),
    }

    with request_image_path.open("rb") as image_file:
        response = requests.post(
            f"{base_url}/v1/videos",
            data={key: str(value) for key, value in form.items()},
            files={"input_reference": (request_image_path.name, image_file, "image/png")},
            timeout=120,
        )
    if not response.ok:
        (run_dir / "error_response.txt").write_text(response.text, encoding="utf-8")
        print(f"vLLM request failed ({response.status_code}); request form follows:")
        print(json.dumps(form, indent=2))
        response.raise_for_status()

    initial = response.json()
    write_json(run_dir / "response.json", initial)
    job_id = initial.get("id")
    if not job_id:
        raise RuntimeError(f"vLLM response did not include a job ID: {initial}")

    deadline = time.monotonic() + job_timeout_s
    while time.monotonic() < deadline:
        response = requests.get(f"{base_url}/v1/videos/{job_id}", timeout=30)
        response.raise_for_status()
        final = response.json()
        write_json(run_dir / "final.json", final)
        print(job_id, final.get("status"), f"{final.get('progress', 0)}%")
        if final.get("status") == "completed":
            break
        if final.get("status") in {"failed", "cancelled"}:
            raise RuntimeError(json.dumps(final, indent=2))
        time.sleep(poll_interval_s)
    else:
        raise TimeoutError(f"Video job {job_id} did not complete within {job_timeout_s:g}s")

    action = final.get("action")
    if not isinstance(action, dict) or "data" not in action:
        raise RuntimeError(
            "vLLM response did not include action data: " + json.dumps(final, indent=2)
        )
    action_array = np.asarray(action["data"], dtype=np.float32)
    if action_array.shape != (16, 8):
        raise RuntimeError(f"Expected a [16, 8] DROID action chunk, got {action_array.shape}")
    if not np.isfinite(action_array).all():
        raise RuntimeError("DROID action response contains non-finite values")

    action_path = run_dir / "action.json"
    write_json(action_path, action)
    write_json(
        run_dir / "sample_outputs.json",
        {"outputs": [{"content": {"action": action["data"]}}]},
    )

    content_response = requests.get(f"{base_url}/v1/videos/{job_id}/content", timeout=300)
    content_response.raise_for_status()
    video_path: Path | None = None
    if content_response.content:
        video_path = run_dir / "policy_rollout.mp4"
        video_path.write_bytes(content_response.content)

    return {
        "job_id": job_id,
        "action": action,
        "action_array": action_array,
        "action_path": action_path,
        "video_path": video_path,
        "run_dir": run_dir,
    }


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    asset_root = args.asset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    input_dir = output_root / "inputs"
    run_dir = output_root / "action_policy_droid" / "video_api"

    print(f"vLLM base URL: {base_url}")
    print(f"DROID assets: {asset_root}")
    print(f"Output directory: {run_dir}")
    print(f"Policy prompt: {args.prompt}")

    policy_image_path = prepare_policy_image(asset_root, input_dir)
    active_model = wait_for_server(
        base_url,
        timeout_s=args.server_timeout,
        interval_s=args.server_check_interval,
    )
    print(f"Active vLLM model: {active_model}")

    result = submit_policy_video(
        base_url=base_url,
        active_model=active_model,
        policy_image_path=policy_image_path,
        prompt=args.prompt,
        run_dir=run_dir,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        job_timeout_s=args.job_timeout,
        poll_interval_s=args.poll_interval,
    )
    action = result["action"]
    print(f"Saved action: {result['action_path']}")
    print(
        "Action metadata:",
        f"shape={action.get('shape')}",
        f"dtype={action.get('dtype')}",
        f"domain_id={action.get('domain_id')}",
    )
    print("First predicted action rows:")
    print(result["action_array"][:5])
    if result["video_path"] is not None:
        print(f"Saved rollout video: {result['video_path']}")
    else:
        print("Video content endpoint returned an empty body")


if __name__ == "__main__":
    main()
