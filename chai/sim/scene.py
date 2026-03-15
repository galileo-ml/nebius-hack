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


CHAIR_XML = """    <body name="chair" pos="1.5 0 0.25">
      <freejoint/>
      <geom type="box" size="0.25 0.25 0.25" rgba="0.6 0.4 0.2 1" mass="5"/>
    </body>"""

CAMERA_XML = """    <camera name="front_camera" pos="0 -3 1.5" xyaxes="1 0 0 0 0.5 1"/>"""

PERSON_MARKER_XML = """    <body name="person_marker" pos="3.0 0 0.9">
      <freejoint/>
      <geom type="cylinder" size="0.2 0.9" rgba="0.2 0.5 0.9 0.5"/>
    </body>"""


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
