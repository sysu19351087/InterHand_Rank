import trimesh
import os
import numpy as np
import open3d
from tqdm import tqdm

def fine_mesh(mesh_path):
    
    ori_mesh = trimesh.load(mesh_path)
    ori_verts = np.array(ori_mesh.vertices)
    ori_faces = np.array(ori_mesh.faces)
    nv = ori_verts.shape[0]
    nf = ori_faces.shape[0]
    if nv > 778*2:
        return
    
    new_verts = ori_verts[:-1, :] if nv == 779 else ori_verts
    new_faces = ori_faces[:nf//2, :] if nv == 779 else ori_faces
    
    mesh = open3d.geometry.TriangleMesh()

    mesh.triangles = open3d.utility.Vector3iVector(new_faces)
    mesh.vertices = open3d.utility.Vector3dVector(new_verts)
    
    fine_mesh = mesh.subdivide_loop(number_of_iterations=2)
    fine_verts = np.array(fine_mesh.vertices)
    fine_faces = np.array(fine_mesh.triangles)
    mesh2save = trimesh.Trimesh(fine_verts, fine_faces)

    mesh2save.export(mesh_path)


if __name__ == "__main__":
    
    mesh_dir = '../mesh/train'
    
    all_obj_file = []
    
    for c in os.listdir(mesh_dir):
        for a in os.listdir(os.path.join(mesh_dir, c)):
            for f in os.listdir(os.path.join(mesh_dir, c, a)):
                all_obj_file.append(os.path.join(mesh_dir, c, a, f))
                
    for mesh_path in tqdm(all_obj_file):
        fine_mesh(mesh_path)