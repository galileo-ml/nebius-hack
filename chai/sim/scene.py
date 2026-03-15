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


CHAIR_XML = """    <body name="chair" pos="1.8 0 0">
      <freejoint/>
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

PERSON_MARKER_XML = """    <body name="person_marker" pos="-1.5 0 0">
      <freejoint/>
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
      <!-- white cane (in right hand) -->
      <geom type="capsule" size="0.012 0.58" pos="0.51 -0.18 0.48" euler="0 30 0" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
    </body>"""


WORLD_ENV_GLB = os.path.join(
    os.path.dirname(__file__), "../../world_envs/event_collider.glb"
)


def inject_marble_mesh(xml_string: str, glb_path: str = WORLD_ENV_GLB, scale: float = 0.1) -> str:
    """
    Inject the pre-generated World Labs GLB mesh as a static collision body.

    Set CHAI_WORLD_MESH=1 to enable; off by default.
    """
    glb_path = os.path.abspath(glb_path)
    s = f"{scale} {scale} {scale}"
    asset_xml = f'  <mesh name="world_mesh" file="{glb_path}" scale="{s}"/>'
    body_xml = (
        '  <body name="marble_world" pos="0 0 0">\n'
        '    <geom mesh="world_mesh" type="mesh" contype="1" conaffinity="1"/>\n'
        '  </body>'
    )

    if "<asset>" in xml_string:
        xml_string = xml_string.replace("<asset>", f"<asset>\n{asset_xml}")
    else:
        xml_string = xml_string.replace("<mujoco>", f"<mujoco>\n<asset>\n{asset_xml}\n</asset>")

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

    injection = f"\n{CHAIR_XML}\n{PERSON_MARKER_XML}\n{CAMERA_XML}\n"
    xml = xml.replace("</worldbody>", injection + "</worldbody>")

    tmp = tempfile.NamedTemporaryFile(
        suffix=".xml", delete=False,
        dir=os.path.dirname(source_path)
    )
    tmp.write(xml.encode())
    tmp.close()
    return tmp.name
