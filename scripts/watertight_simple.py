import pymeshlab as ml
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

print(f"Loading: {input_file}")
ms = ml.MeshSet()
ms.load_new_mesh(input_file)
print(f"Loaded! Vertices: {ms.current_mesh().vertex_number()}")

print("Removing duplicates...")
ms.meshing_remove_duplicate_vertices()

print("Closing holes...")
ms.meshing_close_holes(maxholesize=30)

print("Saving...")
ms.save_current_mesh(output_file)
print(f"Done! Saved to {output_file}")
