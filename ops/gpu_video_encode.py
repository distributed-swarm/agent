# ops/gpu_video_encode.py
from __future__ import annotations

from typing import Any, Dict, List

from . import register_op
from . import artifacts

import os
import shutil
import subprocess
import tempfile


@register_op("gpu_video_encode")
def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    frame_refs = payload.get("frame_refs")
    if not isinstance(frame_refs, list) or not frame_refs:
        raise ValueError("frame_refs must be a non-empty list of artifact refs")

    fps = int(payload.get("fps", 30))
    codec = str(payload.get("codec", "h264")).lower()
    container = str(payload.get("container", "mp4")).lower()

    # Resolve frames
    frame_paths: List[str] = [artifacts.get_path(str(r)) for r in frame_refs]

    with tempfile.TemporaryDirectory() as td:
        # copy into sequential names
        for i, p in enumerate(frame_paths, start=1):
            outp = os.path.join(td, f"frame_{i:06d}.jpg")
            shutil.copyfile(p, outp)

        out_path = os.path.join(td, f"out.{container}")

        # ffmpeg -y -framerate 30 -i frame_%06d.jpg -c:v libx264 -pix_fmt yuv420p out.mp4
        if codec == "h264":
            vcodec = "libx264"
        elif codec == "hevc":
            vcodec = "libx265"
        else:
            vcodec = codec  # let ffmpeg decide

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(td, "frame_%06d.jpg"),
            "-c:v",
            vcodec,
            "-pix_fmt",
            "yuv420p",
            out_path,
        ]

        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found in PATH (install ffmpeg or remove gpu_video_encode)")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed: {e.stderr.decode('utf-8', errors='ignore')[:4000]}")

        with open(out_path, "rb") as f:
            data = f.read()

    art = artifacts.put_bytes(data, ext=container)
    return {"video_ref": art.ref, "fps": fps, "codec": codec, "frames": len(frame_refs)}

