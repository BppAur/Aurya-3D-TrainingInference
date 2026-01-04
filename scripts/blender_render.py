#!/usr/bin/env python3
"""
Blender headless rendering script for multi-view mesh rendering.
Usage: blender --background --python blender_render.py -- --mesh <path> --output <dir>
"""
import bpy
import sys
import argparse
import math
from pathlib import Path


def clear_scene():
    """Remove all default objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def setup_camera(distance=2.5):
    """Create and setup camera."""
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.location = (0, -distance, 0)
    camera.rotation_euler = (math.radians(90), 0, 0)
    bpy.context.scene.camera = camera
    return camera


def setup_lighting():
    """Setup 3-point lighting."""
    # Key light
    bpy.ops.object.light_add(type='AREA', location=(3, -3, 3))
    key_light = bpy.context.active_object
    key_light.data.energy = 300

    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-2, -2, 2))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 150

    # Back light
    bpy.ops.object.light_add(type='AREA', location=(0, 2, 2))
    back_light = bpy.context.active_object
    back_light.data.energy = 100


def load_mesh(mesh_path):
    """Load mesh file."""
    if mesh_path.endswith('.obj'):
        bpy.ops.import_scene.obj(filepath=mesh_path)
    elif mesh_path.endswith('.glb') or mesh_path.endswith('.gltf'):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_path}")

    # Get imported object and center it
    if not bpy.context.selected_objects:
        raise RuntimeError(f"No objects were imported from {mesh_path}")

    imported_obj = bpy.context.selected_objects[0]
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    imported_obj.location = (0, 0, 0)

    return imported_obj


def render_view(mesh_obj, output_path, angle_deg=0):
    """Render single view at specified angle."""
    # Rotate mesh object
    mesh_obj.rotation_euler = (0, 0, math.radians(angle_deg))

    # Render
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def main():
    # Parse arguments after --
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Input mesh file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--views", type=int, default=4, help="Number of views")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution")
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup scene
    clear_scene()
    setup_camera()
    setup_lighting()

    # Configure render settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_x = args.resolution
    bpy.context.scene.render.resolution_y = args.resolution
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.cycles.samples = 128

    # Load mesh
    mesh_obj = load_mesh(args.mesh)

    # Render multiple views
    angles = [i * (360 / args.views) for i in range(args.views)]
    for i, angle in enumerate(angles):
        output_path = str(output_dir / f"view_{i}.png")
        render_view(mesh_obj, output_path, angle)
        print(f"Rendered view {i} at {angle}° -> {output_path}")


if __name__ == "__main__":
    main()
