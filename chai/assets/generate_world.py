"""
Generate a World Labs Marble world and download the GLB collider mesh.

Usage:
    python chai/assets/generate_world.py \
        --prompt "indoor office corridor" \
        --out    chai/assets/world.glb

    python chai/assets/generate_world.py \
        --image  path/to/room.jpg \
        --model  "Marble 0.1-mini" \
        --out    chai/assets/world.glb

Requires WLT_API_KEY in .env.local (or environment).
"""

import argparse
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv(".env.local")

BASE_URL = "https://api.worldlabs.ai/marble/v1"


def _headers() -> dict:
    key = os.environ.get("WLT_API_KEY")
    if not key:
        sys.exit("WLT_API_KEY not set. Add it to .env.local or export it.")
    return {"WLT-Api-Key": key, "Content-Type": "application/json"}


def _generate_from_prompt(prompt: str, model: str) -> str:
    """Submit a text-to-world generation job. Returns operation name."""
    resp = requests.post(
        f"{BASE_URL}/worlds:generate",
        headers=_headers(),
        json={"prompt": prompt, "model": model},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["name"]


def _generate_from_image(image_path: str, model: str) -> str:
    """Upload a local image then submit a generation job. Returns operation name."""
    headers = _headers()

    # Step 1: request a signed upload URL
    filename = os.path.basename(image_path)
    upload_resp = requests.post(
        f"{BASE_URL}/uploads",
        headers=headers,
        json={"filename": filename, "content_type": "image/jpeg"},
        timeout=30,
    )
    upload_resp.raise_for_status()
    upload = upload_resp.json()

    # Step 2: PUT the file
    with open(image_path, "rb") as f:
        put_resp = requests.put(
            upload["upload_url"], data=f,
            headers={"Content-Type": "image/jpeg"},
            timeout=120,
        )
    put_resp.raise_for_status()

    # Step 3: generate
    resp = requests.post(
        f"{BASE_URL}/worlds:generate",
        headers=headers,
        json={"uploaded_image_id": upload["id"], "model": model},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["name"]


def _poll(operation_name: str, timeout: int = 600) -> dict:
    """Poll until the operation is done. Returns the response payload."""
    url = f"{BASE_URL}/{operation_name}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        op = requests.get(url, headers=_headers(), timeout=30).json()
        state = op.get("metadata", {}).get("state", "running")
        if op.get("done"):
            if "error" in op:
                sys.exit(f"Generation failed: {op['error']}")
            return op["response"]
        print(f"  [{state}] waiting …")
        time.sleep(10)
    sys.exit("Timed out waiting for world generation.")


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Marble world and save GLB.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", help="Text description of the world")
    group.add_argument("--image", help="Local image path to use as reference")
    parser.add_argument("--model", default="Marble 0.1-plus",
                        choices=["Marble 0.1-plus", "Marble 0.1-mini"],
                        help="Generation model (default: Marble 0.1-plus)")
    parser.add_argument("--out", default="chai/assets/world.glb",
                        help="Output path for the GLB file")
    args = parser.parse_args()

    print(f"Submitting generation request (model={args.model}) …")
    if args.prompt:
        op_name = _generate_from_prompt(args.prompt, args.model)
    else:
        op_name = _generate_from_image(args.image, args.model)

    print(f"Operation: {op_name}")
    print("Polling for completion (this may take 2–5 minutes) …")
    result = _poll(op_name)

    glb_url = result.get("glb_url") or result.get("collider_url")
    if not glb_url:
        sys.exit(f"No GLB URL in response. Full response:\n{result}")

    print(f"Downloading GLB → {args.out}")
    _download(glb_url, args.out)
    print(f"Done. Saved to {args.out}")

    ply_url = result.get("ply_url") or result.get("splat_url")
    if ply_url:
        ply_out = args.out.replace(".glb", ".ply")
        print(f"Downloading PLY splat → {ply_out}")
        _download(ply_url, ply_out)


if __name__ == "__main__":
    main()
