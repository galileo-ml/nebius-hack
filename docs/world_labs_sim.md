# World Labs Marble API — Simulated Environments for the G1 Robot

## 1. Overview

[World Labs Marble](https://marble.worldlabs.ai) is a **3D world generation service**: you give it a text or image prompt and it returns a photorealistic 3D environment as a mesh. It is not a physics simulator — MuJoCo handles that.

**Why it's useful for CHAI:** training and testing the G1 navigation stack in varied, realistic indoor environments (hallways, offices, kitchens) without building each scene by hand.

**Architecture:**

```
Prompt / image
      │
      ▼
World Labs API  ──►  .GLB collider mesh  ──►  MuJoCo (physics + G1)
                 └►  .PLY Gaussian splat       (static world body)
```

---

## 2. Prerequisites

1. Get an API key at [platform.worldlabs.ai](https://platform.worldlabs.ai).
2. Add it to `.env.local` (never commit this file):
   ```
   WLT_API_KEY=wlt_...
   ```
3. `requests` is already in `requirements.txt`.

Load it in Python via:
```python
import os
from dotenv import load_dotenv
load_dotenv(".env.local")
api_key = os.environ["WLT_API_KEY"]
```

---

## 3. Generating a World

### 3a. Portal (interactive)

Visit [marble.worldlabs.ai](https://marble.worldlabs.ai), enter a text prompt or upload an image, click **Generate**, then export the mesh when done.

### 3b. API (automated)

**Endpoint:** `POST https://api.worldlabs.ai/marble/v1/worlds:generate`

**Headers:** `WLT-Api-Key: <your key>`

#### Text prompt

```python
import requests, os, time

API_KEY = os.environ["WLT_API_KEY"]
HEADERS = {"WLT-Api-Key": API_KEY, "Content-Type": "application/json"}

resp = requests.post(
    "https://api.worldlabs.ai/marble/v1/worlds:generate",
    headers=HEADERS,
    json={
        "prompt": "indoor office corridor, tiled floor, fluorescent lights",
        "model": "Marble 0.1-plus",   # or "Marble 0.1-mini" for speed
    },
)
resp.raise_for_status()
operation_id = resp.json()["name"]   # e.g. "operations/abc123"
```

#### Image prompt (URL)

```python
resp = requests.post(
    "https://api.worldlabs.ai/marble/v1/worlds:generate",
    headers=HEADERS,
    json={"image_url": "https://example.com/room.jpg", "model": "Marble 0.1-plus"},
)
```

#### Image prompt (local file — 2-step signed URL)

```python
# Step 1: get upload URL
upload = requests.post(
    "https://api.worldlabs.ai/marble/v1/uploads",
    headers=HEADERS,
    json={"filename": "room.jpg", "content_type": "image/jpeg"},
).json()

# Step 2: upload the file
with open("room.jpg", "rb") as f:
    requests.put(upload["upload_url"], data=f,
                 headers={"Content-Type": "image/jpeg"})

# Step 3: generate
resp = requests.post(
    "https://api.worldlabs.ai/marble/v1/worlds:generate",
    headers=HEADERS,
    json={"uploaded_image_id": upload["id"], "model": "Marble 0.1-plus"},
)
operation_id = resp.json()["name"]
```

#### Polling for completion

```python
def poll_operation(operation_id: str, timeout: int = 300) -> dict:
    url = f"https://api.worldlabs.ai/marble/v1/{operation_id}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        op = requests.get(url, headers=HEADERS).json()
        if op.get("done"):
            return op["response"]
        print(f"  status: {op.get('metadata', {}).get('state', 'running')} …")
        time.sleep(10)
    raise TimeoutError("World generation timed out")
```

The `response` object contains download URLs for the GLB and PLY files.

---

## 4. Exporting the Mesh

### From the portal

Click **Download** → choose **GLB** (physics collider) and optionally **PLY** (Gaussian splat for visuals).

### From the API

After polling completes:

```python
result = poll_operation(operation_id)
glb_url = result["glb_url"]   # key name may vary; check actual response schema
ply_url = result.get("ply_url")

with open("chai/assets/world.glb", "wb") as f:
    f.write(requests.get(glb_url).content)
```

Save the GLB to `chai/assets/world.glb` — this is what MuJoCo will load.

---

## 5. Importing the Mesh into MuJoCo

MuJoCo supports `.glb` / `.obj` mesh files natively as assets.

### XML snippet

```xml
<mujoco>
  <asset>
    <!-- Marble world mesh (scale down from ~50m to ~5m) -->
    <mesh name="world_mesh" file="chai/assets/world.glb" scale="0.1 0.1 0.1"/>
  </asset>

  <worldbody>
    <!-- Static environment — no joints, only collision geometry -->
    <body name="marble_world" pos="0 0 0">
      <geom name="world_geom" mesh="world_mesh" type="mesh"
            contype="1" conaffinity="1" rgba="0.8 0.8 0.8 1"/>
    </body>

    <!-- G1 robot spawns at origin, inside the mesh -->
    <include file="unitree_mujoco/unitree_robots/g1/g1.xml"/>
  </worldbody>
</mujoco>
```

### Extending `chai/sim/scene.py`

`patch_scene_xml` already injects bodies into the G1 scene XML. Add a helper to inject the Marble mesh:

```python
def inject_marble_mesh(xml_string: str, glb_path: str, scale: float = 0.1) -> str:
    """
    Inject the Marble world mesh asset and a static body into the G1 scene XML.

    Args:
        xml_string: Contents of the MuJoCo scene XML.
        glb_path:   Absolute or relative path to the .glb file.
        scale:      Uniform scale applied to the mesh (default 0.1 — 50m → 5m).

    Returns:
        Modified XML string with the mesh asset and body injected.
    """
    s = f"{scale} {scale} {scale}"
    asset_xml = f'  <mesh name="world_mesh" file="{glb_path}" scale="{s}"/>'
    body_xml = (
        '  <body name="marble_world" pos="0 0 0">\n'
        '    <geom mesh="world_mesh" type="mesh" contype="1" conaffinity="1"/>\n'
        '  </body>'
    )

    # Inject into <asset> block (or create one)
    if "<asset>" in xml_string:
        xml_string = xml_string.replace("<asset>", f"<asset>\n{asset_xml}")
    else:
        xml_string = xml_string.replace("<mujoco>", f"<mujoco>\n<asset>\n{asset_xml}\n</asset>")

    # Inject body into worldbody
    xml_string = xml_string.replace("</worldbody>", f"\n{body_xml}\n</worldbody>")
    return xml_string
```

---

## 6. End-to-End Script

`chai/assets/generate_world.py` handles the full flow: generate → poll → download GLB.

```
python chai/assets/generate_world.py \
    --prompt "indoor office corridor with tiled floor" \
    --model  "Marble 0.1-mini" \
    --out    chai/assets/world.glb
```

Then load it in the simulator:

```python
from chai.sim.scene import patch_scene_xml, inject_marble_mesh
import mujoco

xml_path = "unitree_mujoco/unitree_robots/g1/scene.xml"
with open(xml_path) as f:
    xml = f.read()

xml = inject_marble_mesh(xml, glb_path="chai/assets/world.glb")
model = mujoco.MjModel.from_xml_string(xml)
```

Or run the existing demo with the patched scene:

```
python chai/sim_demo.py
```

---

## 7. Tips & Troubleshooting

| Issue | Fix |
|-------|-----|
| Mesh too large / robot falls through floor | Scale down: `scale="0.05 0.05 0.05"` |
| No collision response | Ensure `contype` and `conaffinity` are set on the geom |
| GLB not loading | MuJoCo ≥ 3.x required; confirm with `python -c "import mujoco; print(mujoco.__version__)"` |
| PLY splat not visible in MuJoCo viewer | MuJoCo can't render Gaussian splats; use the GLB for physics and render the PLY externally (e.g. in a WebGL viewer) |
| Generation takes > 5 minutes | Switch to `"Marble 0.1-mini"` for faster (lower-quality) results |
| Want to edit the world before exporting | Use the **Chisel** tool in the portal to remove objects or fill gaps |

### Mesh scale reference

Marble worlds are typically 30–60 m across. The G1 is ~1.3 m tall.

| `scale` value | Effective world size |
|---------------|----------------------|
| `0.1` | ~4–6 m — good for single-room scenes |
| `0.05` | ~2–3 m — tight corridor |
| `0.2` | ~8–12 m — large open area |

Adjust until the robot spawns comfortably inside the environment without clipping through walls.
