# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os.path

from tqdm import tqdm
import numpy as np
import cv2
from tqdm import tqdm
import trimesh
import json
import PIL
from config import cfg
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import sys
current_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(current_path + '/../../common'))
sys.path.append(os.path.abspath(current_path + '/../../data/InterHand2.6M'))
from base import Tester
from utils.vis import vis_keypoints
import torch.backends.cudnn as cudnn
from utils.transforms import flip
from dj_dataset import Dj_Dataset
import open3d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', type=str, dest='set')
    parser.add_argument('--save_dir', type=str, dest='save_dir')
    args = parser.parse_args()

    return args

def main():

    args = parse_args()
    cudnn.benchmark = True
    
    testset_loader = Dj_Dataset(transforms.ToTensor(), args.set, False)
    # batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus * cfg.test_batch_size,
    #                             shuffle=False, num_workers=cfg.num_thread, pin_memory=True)
    jsonData = {
            'hand_type': [],   # [2,]
            'shape': [],           # [2, 10]
            'pose': [],             # [2, 48]
            'trans': [],           # [2, 3]
            'img_path': [],
        }
    for idx in tqdm(range(len(testset_loader))):
        item = testset_loader.__getitem__(idx)
        for k, v in item.items():
            item[k] = v.cpu().numpy()

        for k in jsonData.keys():
            if k == 'img_path':
                jsonData[k].append(testset_loader.datalist[idx]['file_name'])
            else:
                jsonData[k].append(item[k])
        img2save = item['img'].transpose(1, 2, 0)   # [H, W, 3]
        
        verts = item['verts']
        faces = item['faces']
        mesh = open3d.geometry.TriangleMesh()

        mesh.triangles = open3d.utility.Vector3iVector(faces)
        mesh.vertices = open3d.utility.Vector3dVector(verts)
        
        fine_mesh = mesh.subdivide_loop(number_of_iterations=2)
        fine_verts = np.array(fine_mesh.vertices)
        fine_faces = np.array(fine_mesh.triangles)
        mesh2save = trimesh.Trimesh(fine_verts, fine_faces)
        
        img_path = jsonData['img_path'][-1]
        file_name = img_path.split('/')[-1]
        jsonData['img_path'][-1] = '/'.join(img_path.split('/')[:-2] + [file_name])
        save_dir = '/'.join(img_path.split('/')[:-2])
        
        mesh_save_dir = os.path.join(args.save_dir, 'mesh', args.set, save_dir)
        mesh_save_path = os.path.join(mesh_save_dir, file_name[:-4]+'.obj')
        os.makedirs(mesh_save_dir, exist_ok=True)
        mesh2save.export(mesh_save_path)
        
        img_save_dir = os.path.join(args.save_dir, 'image', args.set, save_dir)
        img_save_path = os.path.join(img_save_dir, file_name)
        os.makedirs(img_save_dir, exist_ok=True)
        PIL.Image.fromarray(img2save.astype('uint8')).save(img_save_path)
    
    jsonData = {k: np.stack(v, axis=0).tolist() for k,v in jsonData.items()}
    json_filename = os.path.join(args.save_dir, f'{args.set}_info.json')
    with open(json_filename,'w') as f:
        json.dump(jsonData, f)
    
    # f = open(json_filename, 'r')
    # content = f.read()

    # i = json.loads(content)
    # for k, v in i.items():
    #     print(k, v)

if __name__ == "__main__":
    main()
