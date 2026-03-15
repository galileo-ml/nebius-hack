"""
MuJoCo scene setup helper.

Adds a static chair obstacle and a person marker to the G1 scene for demo testing.

Usage — patch the scene XML before loading:
    from sim.scene import patch_scene_xml
    patched_path = patch_scene_xml("unitree_mujoco/unitree_robots/g1/scene.xml")
    model = mujoco.MjModel.from_xml_path(patched_path)

Or add the XML snippets below directly to the worldbody of scene.xml:

    <!-- chair obstacle ~1.5m ahead -->
    <body name="chair" pos="1.5 0 0.25">
      <geom type="box" size="0.25 0.25 0.25" rgba="0.6 0.4 0.2 1" mass="5"/>
    </body>

    <!-- person marker ~3m ahead -->
    <body name="person_marker" pos="3.0 0 0.9">
      <geom type="cylinder" size="0.2 0.9" rgba="0.2 0.5 0.9 0.5"/>
    </body>
"""

import os
import shutil
import tempfile


CHAIR_XML = """    <body name="chair" pos="3.0 0 0">
      <freejoint name="chair"/>
      <!-- seat -->
      <geom type="box" size="0.22 0.22 0.03" pos="0 0 0.46" rgba="0.55 0.35 0.15 1" mass="3"/>
      <!-- backrest -->
      <geom type="box" size="0.22 0.03 0.28" pos="0 -0.19 0.77" rgba="0.55 0.35 0.15 1" mass="1"/>
      <!-- front-left leg -->
      <geom type="cylinder" size="0.025 0.23" pos="-0.17 0.15 0.23" rgba="0.4 0.25 0.1 1" mass="0.5"/>
      <!-- front-right leg -->
      <geom type="cylinder" size="0.025 0.23" pos=" 0.17 0.15 0.23" rgba="0.4 0.25 0.1 1" mass="0.5"/>
      <!-- back-left leg -->
      <geom type="cylinder" size="0.025 0.23" pos="-0.17 -0.19 0.23" rgba="0.4 0.25 0.1 1" mass="0.5"/>
      <!-- back-right leg -->
      <geom type="cylinder" size="0.025 0.23" pos=" 0.17 -0.19 0.23" rgba="0.4 0.25 0.1 1" mass="0.5"/>
    </body>"""

CAMERA_XML = """    <camera name="front_camera" pos="0 -3 1.5" xyaxes="1 0 0 0 0.5 1"/>"""

PERSON_MARKER_XML = """    <body name="person_marker" pos="-2.0 0 0">
      <freejoint name="person_marker"/>
      <!-- head -->
      <geom type="sphere" size="0.11" pos="0 0 1.62" rgba="0.9 0.7 0.5 1" contype="0" conaffinity="0"/>
      <!-- torso -->
      <geom type="capsule" size="0.13 0.25" pos="0 0 1.15" rgba="0.3 0.4 0.7 1" contype="0" conaffinity="0"/>
      <!-- left upper arm -->
      <geom type="capsule" size="0.04 0.13" pos="0 0.18 1.15" euler="0 0 0" rgba="0.3 0.4 0.7 1" contype="0" conaffinity="0"/>
      <!-- left lower arm -->
      <geom type="capsule" size="0.035 0.12" pos="0 0.18 0.86" euler="0 0 0" rgba="0.9 0.7 0.5 1" contype="0" conaffinity="0"/>
      <!-- right upper arm -->
      <geom type="capsule" size="0.04 0.13" pos="0 -0.18 1.15" euler="0 0 0" rgba="0.3 0.4 0.7 1" contype="0" conaffinity="0"/>
      <!-- right lower arm (bent forward to hold cane) -->
      <geom type="capsule" size="0.035 0.12" pos="0.1 -0.18 0.98" euler="0 90 0" rgba="0.9 0.7 0.5 1" contype="0" conaffinity="0"/>
      <!-- pelvis -->
      <geom type="box" size="0.11 0.14 0.07" pos="0 0 0.78" rgba="0.2 0.2 0.6 1" contype="0" conaffinity="0"/>
      <!-- left thigh -->
      <geom type="capsule" size="0.05 0.2" pos="0 0.09 0.52" rgba="0.2 0.2 0.6 1" contype="0" conaffinity="0"/>
      <!-- right thigh -->
      <geom type="capsule" size="0.05 0.2" pos="0 -0.09 0.52" rgba="0.2 0.2 0.6 1" contype="0" conaffinity="0"/>
      <!-- left shin -->
      <geom type="capsule" size="0.04 0.18" pos="0 0.09 0.22" rgba="0.9 0.7 0.5 1" contype="0" conaffinity="0"/>
      <!-- right shin -->
      <geom type="capsule" size="0.04 0.18" pos="0 -0.09 0.22" rgba="0.9 0.7 0.5 1" contype="0" conaffinity="0"/>
      <!-- mass carrier (invisible) -->
      <geom type="sphere" size="0.01" pos="0 0 0.9" rgba="0 0 0 0" mass="60"/>
      <!-- sunglasses left lens -->
      <geom type="box" size="0.04 0.035 0.015" pos="0.105 0.055 1.635" rgba="0.05 0.05 0.05 0.9" contype="0" conaffinity="0"/>
      <!-- sunglasses right lens -->
      <geom type="box" size="0.04 0.035 0.015" pos="0.105 -0.055 1.635" rgba="0.05 0.05 0.05 0.9" contype="0" conaffinity="0"/>
      <!-- sunglasses bridge -->
      <geom type="box" size="0.005 0.015 0.008" pos="0.105 0.0 1.635" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>
      <!-- white cane (in right hand) - top half white, bottom half red -->
      <geom type="capsule" size="0.012 0.333" pos="0.325 -0.180 0.735" euler="0 137.437 0" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
      <geom type="capsule" size="0.012 0.333" pos="0.775 -0.180 0.245" euler="0 137.437 0" rgba="0.8 0.1 0.1 1" contype="0" conaffinity="0"/>
    </body>"""


WORLD_ENV_GLB = os.path.join(
    os.path.dirname(__file__), "../../world_envs/event_collider.glb"
)

HQ_MESH_GLB = os.path.join(
    os.path.dirname(__file__), "../../world_envs/high_quality_mesh.glb"
)

HQ_SPLAT_SPZ = os.path.join(
    os.path.dirname(__file__), "../../world_envs/high_quality_splat.spz"
)


def detect_glb_bounds(glb_path: str) -> dict:
    """Auto-detect floor offset and room extents from a GLB file."""
    import trimesh
    scene = trimesh.load(glb_path, force="scene")
    mesh = trimesh.util.concatenate(list(scene.geometry.values()))
    b = mesh.bounds  # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    bounds = {
        "xmin": float(b[0][0]), "xmax": float(b[1][0]),
        "ymin": float(b[0][1]), "ymax": float(b[1][1]),
        "zmin": float(b[0][2]), "zmax": float(b[1][2]),
        "floor_offset": float(-b[0][1]),
    }
    print(f"[SCENE] World bounds: X=[{bounds['xmin']:.3f}, {bounds['xmax']:.3f}] "
          f"Y=[{bounds['ymin']:.3f}, {bounds['ymax']:.3f}] "
          f"Z=[{bounds['zmin']:.3f}, {bounds['zmax']:.3f}]")
    print(f"[SCENE] Floor offset: {bounds['floor_offset']:.3f}m")
    return bounds


def generate_wall_colliders(bounds: dict) -> str:
    """Generate invisible box wall/ceiling colliders from AABB bounds.

    After euler="90 0 0": GLB X → MuJoCo X, GLB Y → MuJoCo Z, GLB Z → MuJoCo -Y.
    """
    half_x = (bounds["xmax"] - bounds["xmin"]) / 2
    half_z = (bounds["zmax"] - bounds["zmin"]) / 2
    H      = bounds["ymax"] - bounds["ymin"]
    T = 0.4

    walls = [
        f'<geom name="wall_n" type="box" size="{half_x:.3f} {T} {H/2:.3f}" '
        f'pos="0 {-half_z - T:.3f} {H/2:.3f}" contype="1" conaffinity="1" rgba="0 0 0 0"/>',
        f'<geom name="wall_s" type="box" size="{half_x:.3f} {T} {H/2:.3f}" '
        f'pos="0 {half_z + T:.3f} {H/2:.3f}" contype="1" conaffinity="1" rgba="0 0 0 0"/>',
        f'<geom name="wall_e" type="box" size="{T} {half_z:.3f} {H/2:.3f}" '
        f'pos="{half_x + T:.3f} 0 {H/2:.3f}" contype="1" conaffinity="1" rgba="0 0 0 0"/>',
        f'<geom name="wall_w" type="box" size="{T} {half_z:.3f} {H/2:.3f}" '
        f'pos="{-half_x - T:.3f} 0 {H/2:.3f}" contype="1" conaffinity="1" rgba="0 0 0 0"/>',
        f'<geom name="ceiling" type="box" size="{half_x:.3f} {half_z:.3f} {T}" '
        f'pos="0 0 {H + T:.3f}" contype="1" conaffinity="1" rgba="0 0 0 0"/>',
    ]
    return "\n  ".join(walls)


def decode_spz_to_obj(spz_path: str) -> str:
    """Decompress .spz (gzip'd binary) → extract Gaussian positions → OBJ point cloud."""
    import gzip
    import struct
    import numpy as np

    obj_path = spz_path[:-4] + "_splat.obj"
    if os.path.exists(obj_path):
        # Validate cached file isn't corrupt (empty or zero valid vertices)
        with open(obj_path) as f:
            if sum(1 for line in f if line.startswith("v ")) > 0:
                return obj_path
        os.remove(obj_path)  # corrupt cache — regenerate

    with gzip.open(spz_path, "rb") as f:
        data = f.read()

    # SPZ header (Niantic/World Labs format):
    # magic(4) + version(4) + numPoints(4) + shDegree(1) + antialiased(1) + reserved(2)
    magic, version, num_points, sh_degree = struct.unpack_from("<IIIB", data, 0)
    assert magic == 0x5053474e, f"Bad SPZ magic: {magic:#010x}"
    print(f"[SCENE] SPZ: {num_points:,} Gaussians, SH degree {sh_degree}")

    # Positions immediately follow 16-byte header as float32 triplets
    positions = np.frombuffer(data, dtype=np.float32,
                              count=num_points * 3, offset=16).reshape(-1, 3)

    # Filter garbage positions (quantization artifact: |v| > 1e6 means bad decode)
    mask = np.all(np.abs(positions) <= 1e6, axis=1)
    positions = positions[mask]
    print(f"[SCENE] SPZ: kept {len(positions):,} / {num_points:,} valid positions")
    if len(positions) == 0:
        raise ValueError(
            "SPZ positions are all out of range — format may use quantized encoding "
            "not raw float32. Skipping splat."
        )

    with open(obj_path, "w") as out:
        out.write("# CHAI Gaussian splat point cloud\n")
        for x, y, z in positions:
            out.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
    print(f"[SCENE] Splat OBJ → {obj_path}")
    return obj_path


def _glb_to_obj(glb_path: str) -> str:
    """Convert a GLB file to OBJ (cached next to the original). Returns OBJ path."""
    obj_path = glb_path[:-4] + ".obj"
    if not os.path.exists(obj_path):
        import trimesh
        scene = trimesh.load(glb_path, force="scene")
        combined = scene.to_geometry()
        combined.export(obj_path)
        print(f"[SCENE] Converted {glb_path} → {obj_path}")
    return obj_path


def inject_marble_mesh(xml_string: str, glb_path: str = WORLD_ENV_GLB, scale: float = 1.0,
                       add_walls: bool = False) -> str:
    """
    Inject the pre-generated World Labs GLB mesh as a static visual body.

    When add_walls=True, also injects invisible box colliders around the room perimeter.
    Floor offset is auto-detected from the GLB bounds so the mesh lands at Z=0.
    """
    glb_path = os.path.abspath(glb_path)
    mesh_path = _glb_to_obj(glb_path)
    s = f"{scale} {scale} {scale}"
    asset_xml = f'  <mesh name="world_mesh" file="{mesh_path}" scale="{s}"/>'

    bounds = detect_glb_bounds(glb_path)
    offset = bounds["floor_offset"]

    # euler="90 0 0": rotates GLB Y-up → MuJoCo Z-up (R_x(90°): y→z, z→-y)
    # pos offset lifts mesh so its floor lands at Z=0
    # contype/conaffinity=0: visual-only; robot walks on invisible flat collision floor
    cx_glb = (bounds["xmax"] + bounds["xmin"]) / 2
    cz_glb = (bounds["zmax"] + bounds["zmin"]) / 2
    body_xml = (
        f'  <body name="marble_world" pos="{-cx_glb:.4f} {cz_glb:.4f} {offset:.4f}" euler="90 0 0">\n'
        f'    <geom mesh="world_mesh" type="mesh" contype="0" conaffinity="0"/>\n'
        f'  </body>'
    )

    if add_walls:
        body_xml += "\n  " + generate_wall_colliders(bounds)

    # Hide the default checker floor visually (keep it as invisible collision plane)
    xml_string = xml_string.replace(
        'material="groundplane"',
        'rgba="0 0 0 0"'
    )

    if "<asset>" in xml_string:
        xml_string = xml_string.replace("<asset>", f"<asset>\n{asset_xml}")
    else:
        xml_string = xml_string.replace("<mujoco>", f"<mujoco>\n<asset>\n{asset_xml}\n</asset>")

    xml_string = xml_string.replace("</worldbody>", f"\n{body_xml}\n</worldbody>")
    return xml_string


def inject_splat_mesh(xml_string: str, spz_path: str = HQ_SPLAT_SPZ) -> str:
    """Inject a Gaussian splat point-cloud OBJ as a semi-transparent visual layer."""
    obj_path = decode_spz_to_obj(os.path.abspath(spz_path))
    asset_xml = f'  <mesh name="splat_mesh" file="{obj_path}"/>'
    body_xml = (
        '  <body name="splat_world" euler="90 0 0">\n'
        '    <geom mesh="splat_mesh" type="mesh" contype="0" conaffinity="0" rgba="0.85 0.82 0.78 0.5"/>\n'
        '  </body>'
    )
    xml_string = xml_string.replace("<asset>", f"<asset>\n{asset_xml}")
    xml_string = xml_string.replace("</worldbody>", f"\n{body_xml}\n</worldbody>")
    return xml_string


def patch_scene_xml(source_path: str) -> str:
    """
    Copy the scene XML to a temp file and inject the obstacle/person bodies
    into the worldbody. Returns the path to the patched file.
    """
    with open(source_path, "r") as f:
        xml = f.read()

    if "name=\"chair\"" in xml:
        # Already patched
        return source_path

    injection = f"\n{CHAIR_XML}\n{PERSON_MARKER_XML}\n"
    xml = xml.replace("</worldbody>", injection + "</worldbody>")

    tmp = tempfile.NamedTemporaryFile(
        suffix=".xml", delete=False,
        dir=os.path.dirname(source_path)
    )
    tmp.write(xml.encode())
    tmp.close()
    return tmp.name
