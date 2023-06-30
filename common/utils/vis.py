# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path as osp
import cv2
import numpy as np
import matplotlib
matplotlib.use('tkagg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from config import cfg
from PIL import Image, ImageDraw
import trimesh
# import pyrender
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.renderer.mesh.shader import SoftDepthShader

def get_keypoint_rgb(skeleton):
    rgb_dict= {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]['name']

        if joint_name.endswith('thumb_null'):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith('thumb3'):
            rgb_dict[joint_name] = (255, 51, 51)
        elif joint_name.endswith('thumb2'):
            rgb_dict[joint_name] = (255, 102, 102)
        elif joint_name.endswith('thumb1'):
            rgb_dict[joint_name] = (255, 153, 153)
        elif joint_name.endswith('thumb0'):
            rgb_dict[joint_name] = (255, 204, 204)
        elif joint_name.endswith('index_null'):
            rgb_dict[joint_name] = (0, 255, 0)
        elif joint_name.endswith('index3'):
            rgb_dict[joint_name] = (51, 255, 51)
        elif joint_name.endswith('index2'):
            rgb_dict[joint_name] = (102, 255, 102)
        elif joint_name.endswith('index1'):
            rgb_dict[joint_name] = (153, 255, 153)
        elif joint_name.endswith('middle_null'):
            rgb_dict[joint_name] = (255, 128, 0)
        elif joint_name.endswith('middle3'):
            rgb_dict[joint_name] = (255, 153, 51)
        elif joint_name.endswith('middle2'):
            rgb_dict[joint_name] = (255, 178, 102)
        elif joint_name.endswith('middle1'):
            rgb_dict[joint_name] = (255, 204, 153)
        elif joint_name.endswith('ring_null'):
            rgb_dict[joint_name] = (0, 128, 255)
        elif joint_name.endswith('ring3'):
            rgb_dict[joint_name] = (51, 153, 255)
        elif joint_name.endswith('ring2'):
            rgb_dict[joint_name] = (102, 178, 255)
        elif joint_name.endswith('ring1'):
            rgb_dict[joint_name] = (153, 204, 255)
        elif joint_name.endswith('pinky_null'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('pinky3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('pinky2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('pinky1'):
            rgb_dict[joint_name] = (255, 153, 255)
        else:
            rgb_dict[joint_name] = (230, 230, 0)
        
    return rgb_dict

def vis_keypoints(img, kps, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad = 3, save_path=None):
    
    rgb_dict = get_keypoint_rgb(skeleton)
    _img = Image.fromarray(img.transpose(1,2,0).astype('uint8')) 
    draw = ImageDraw.Draw(_img)
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']
        
        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))
        #if i not in [0,1,20] and i not in [21,22,41]:
        #    continue
        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=rgb_dict[parent_joint_name], width=line_width)
        if score[i] > score_thr:
            draw.ellipse((kps[i][0]-circle_rad, kps[i][1]-circle_rad, kps[i][0]+circle_rad, kps[i][1]+circle_rad), fill=rgb_dict[joint_name])
        if score[pid] > score_thr and pid != -1:
            draw.ellipse((kps[pid][0]-circle_rad, kps[pid][1]-circle_rad, kps[pid][0]+circle_rad, kps[pid][1]+circle_rad), fill=rgb_dict[parent_joint_name])
    
    if save_path is None:
        _img.save(osp.join(cfg.vis_dir, filename))
    else:
        _img.save(osp.join(save_path, filename))

def vis_keypoints_together(img, all_kps, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad = 3, save_path=None):
    # all_kps:[com_num, Nj, 2]
    rgb_dict = get_keypoint_rgb(skeleton)
    img_ls = []
    com_num = all_kps.shape[0]
    for cid in range(com_num):
        kps = all_kps[cid]
        _img = Image.fromarray(img.transpose(1,2,0).astype('uint8'))
        draw = ImageDraw.Draw(_img)
        for i in range(len(skeleton)):
            joint_name = skeleton[i]['name']
            pid = skeleton[i]['parent_id']
            parent_joint_name = skeleton[pid]['name']
            
            kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
            kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))
            #if i not in [0,1,20] and i not in [21,22,41]:
            #    continue
            if score[i] > score_thr and score[pid] > score_thr and pid != -1:
                draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=rgb_dict[parent_joint_name], width=line_width)
            if score[i] > score_thr:
                draw.ellipse((kps[i][0]-circle_rad, kps[i][1]-circle_rad, kps[i][0]+circle_rad, kps[i][1]+circle_rad), fill=rgb_dict[joint_name])
            if score[pid] > score_thr and pid != -1:
                draw.ellipse((kps[pid][0]-circle_rad, kps[pid][1]-circle_rad, kps[pid][0]+circle_rad, kps[pid][1]+circle_rad), fill=rgb_dict[parent_joint_name])
        img_ls.append(_img)
    img2save = np.concatenate(img_ls, 1)  # [H, 2*W, 3]
    img2save = Image.fromarray(img2save.astype('uint8')) 
    if save_path is not None:
        img2save.save(osp.join(save_path, filename))
        
    return np.concatenate(img_ls, 1)

def vis_3d_keypoints(kps_3d, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad=3):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rgb_dict = get_keypoint_rgb(skeleton)
    
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        x = np.array([kps_3d[i,0], kps_3d[pid,0]])
        y = np.array([kps_3d[i,1], kps_3d[pid,1]])
        z = np.array([kps_3d[i,2], kps_3d[pid,2]])

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            ax.plot(x, z, -y, c = np.array(rgb_dict[parent_joint_name])/255., linewidth = line_width)
        if score[i] > score_thr:
            ax.scatter(kps_3d[i,0], kps_3d[i,2], -kps_3d[i,1], c = np.array(rgb_dict[joint_name]).reshape(1,3)/255., marker='o')
        if score[pid] > score_thr and pid != -1:
            ax.scatter(kps_3d[pid,0], kps_3d[pid,2], -kps_3d[pid,1], c = np.array(rgb_dict[parent_joint_name]).reshape(1,3)/255., marker='o')

    #plt.show()
    #cv2.waitKey(0)
    
    fig.savefig(osp.join(cfg.vis_dir, filename), dpi=fig.dpi)

def vis_meshverts_together(img, all_kps, hand_type, filename, save_path=None):
    # hand_type:[2,]
    img_ls = []
    com_num = len(all_kps)
    for cid in range(com_num):
        save_img = img.copy()
        ht = hand_type
        half_nv = all_kps[cid].shape[0] // 2
        for jidx, jproj in enumerate(all_kps[cid]):
            jx, jy = int(jproj[0]), int(jproj[1])
            if jx < 0:
                jx = 0
            if jx >= save_img.shape[1]:
                jx = save_img.shape[1] - 1
            if jy < 0:
                jy = 0
            if jy >= save_img.shape[0]:
                jy = save_img.shape[0] - 1
            if jidx < half_nv:
                if ht[0] > 0.5:
                    save_img[jy, jx, 0] = 0
                    save_img[jy, jx, 1] = 0
                    save_img[jy, jx, 2] = 255
            else:
                if ht[1] > 0.5:
                    save_img[jy, jx, 0] = 0
                    save_img[jy, jx, 1] = 255
                    save_img[jy, jx, 2] = 0
        img_ls.append(save_img)
    img2save = np.concatenate(img_ls, 1)  # [H, 2*W, 3]
    img2save = Image.fromarray(img2save.astype('uint8')) 
    if save_path is None:
        img2save.save(osp.join(cfg.vis_dir, filename))
    else:
        img2save.save(osp.join(save_path, filename))


def vis_meshverts_together_with_mask(img, mask, all_kps, hand_type, filename, save_path=None):
    # hand_type:[2,]
    # img:[H, W, 3], mask:[H, W]
    # print('valid:', np.sum((mask == 0) | (mask == 1)))
    # print('invalid:', np.sum((mask != 0) & (mask != 1)))
    # m = (mask != 0) & (mask != 1)
    img_ls = []
    com_num = len(all_kps)
    for cid in range(com_num):
        save_img = mask[:, :, None].repeat(3,axis=-1) * 255
        # save_img[m] = [125, 125, 125]
        ht = hand_type
        half_nv = all_kps[cid].shape[0] // 2
        invalid = 0
        for jidx, jproj in enumerate(all_kps[cid]):
            jx, jy = int(jproj[0]), int(jproj[1])
            if jx < 0:
                jx = 0
            if jx >= save_img.shape[1]:
                jx = save_img.shape[1] - 1
            if jy < 0:
                jy = 0
            if jy >= save_img.shape[0]:
                jy = save_img.shape[0] - 1
            if jidx < half_nv:
                if ht[0] > 0.5:
                    if mask[jy, jx] != 1.:
                        save_img[jy, jx, 0] = 255
                        save_img[jy, jx, 1] = 0
                        save_img[jy, jx, 2] = 0
                        invalid += 1
                    else:
                        save_img[jy, jx, 0] = 0
                        save_img[jy, jx, 1] = 0
                        save_img[jy, jx, 2] = 255
            else:
                if ht[1] > 0.5:
                    if mask[jy, jx] != 1.:
                        save_img[jy, jx, 0] = 255
                        save_img[jy, jx, 1] = 0
                        save_img[jy, jx, 2] = 0
                        invalid += 1
                    else:
                        save_img[jy, jx, 0] = 0
                        save_img[jy, jx, 1] = 255
                        save_img[jy, jx, 2] = 0
        # print(invalid)
        img_ls.append(save_img)
    img2save = np.concatenate(img_ls, 1)  # [H, 2*W, 3]
    img2save = Image.fromarray(img2save.astype('uint8')) 
    if save_path is None:
        img2save.save(osp.join(cfg.vis_dir, filename))
    else:
        img2save.save(osp.join(save_path, filename))
        

def vis_meshverts_together_with_sampleresults(img, mask, all_kps, samp_res, hand_type, filename, save_path=None):
    # hand_type:[2,]
    # img:[H, W, 3], mask:[H, W]
    # all_kps:[com_num, 778*2, 2]
    # samp_res:[com_num, 778*2]
    
    # print('valid:', np.sum((mask == 0) | (mask == 1)))
    # print('invalid:', np.sum((mask != 0) & (mask != 1)))
    # m = (mask != 0) & (mask != 1)
    img_ls = []
    com_num = len(all_kps)
    for cid in range(com_num):
        save_img = mask[:, :, None].repeat(3,axis=-1) * 255
        # save_img[m] = [125, 125, 125]
        ht = hand_type
        half_nv = all_kps[cid].shape[0] // 2
        invalid = 0
        for jidx, jproj in enumerate(all_kps[cid]):
            jx, jy = int(jproj[0]), int(jproj[1])
            if jx < 0:
                jx = 0
            if jx >= save_img.shape[1]:
                jx = save_img.shape[1] - 1
            if jy < 0:
                jy = 0
            if jy >= save_img.shape[0]:
                jy = save_img.shape[0] - 1
            if jidx < half_nv:
                if ht[0] > 0.5:
                    if samp_res[cid][jidx] != 0.:
                        save_img[jy, jx, 0] = 255
                        save_img[jy, jx, 1] = 0
                        save_img[jy, jx, 2] = 0
                        invalid += 1
                    else:
                        save_img[jy, jx, 0] = 0
                        save_img[jy, jx, 1] = 0
                        save_img[jy, jx, 2] = 255
            else:
                if ht[1] > 0.5:
                    if samp_res[cid][jidx] != 0.:
                        save_img[jy, jx, 0] = 255
                        save_img[jy, jx, 1] = 0
                        save_img[jy, jx, 2] = 0
                        invalid += 1
                    else:
                        save_img[jy, jx, 0] = 0
                        save_img[jy, jx, 1] = 255
                        save_img[jy, jx, 2] = 0
        # print(invalid)
        img_ls.append(save_img)
    img2save = np.concatenate(img_ls, 1)  # [H, 2*W, 3]
    img2save = Image.fromarray(img2save.astype('uint8')) 
    if save_path is None:
        img2save.save(osp.join(cfg.vis_dir, filename))
    else:
        img2save.save(osp.join(save_path, filename))


def vis_mask_overlap(gt_mask, pred_mask):
    # gt_mask:[H, W]
    # pred_mask:[H, W]
    
    gm = (gt_mask[:, :, None].repeat(3, axis=-1) == 1.)         # [H, W, 3]
    pm = (pred_mask[:, :, None].repeat(3, axis=-1) == 1.)       # [H, W, 3]
     
    a = (gm & pm)
    b = (gm | pm)
    iou = np.sum(a[:, :, 0], axis=(-1, -2)) / np.sum(b[:, :, 0], axis=(-1, -2))
    
    v1_r = ((a != b) & gm)[:, :, 0]
    v1_b = ((a != b) & pm)[:, :, 0]
    a = a * 255.
    a[v1_r, :] = [255, 0, 0]
    a[v1_b, :] = [255, 165, 0]
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(a, '%.4f'%(iou), (10, 20), font, 0.5, (0, 255, 0), 1)
    return a.astype(np.uint8), iou
    
    
# def render_mesh(img, verts, face, cam_param):
#     # img:[H, W, 3]
#     # verts:[nv, 3]
#     # face:[nf, 3]
#     # mesh
#     mesh = trimesh.Trimesh(verts, face)
#     rot = trimesh.transformations.rotation_matrix(
# 	np.radians(180), [1, 0, 0])
#     mesh.apply_transform(rot)
#     material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
#     mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
#     scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
#     scene.add(mesh, 'mesh')
    
#     focal, princpt = cam_param['focal'], cam_param['princpt']
#     camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
#     scene.add(camera)
 
#     # renderer
#     renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
   
#     # light
#     light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
#     light_pose = np.eye(4)
#     light_pose[:3, 3] = np.array([0, -1, 1])
#     scene.add(light, pose=light_pose)
#     light_pose[:3, 3] = np.array([0, 1, 1])
#     scene.add(light, pose=light_pose)
#     light_pose[:3, 3] = np.array([1, 1, 2])
#     scene.add(light, pose=light_pose)

#     # render
#     rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
#     rgb = rgb[:,:,:3].astype(np.float32)
#     valid_mask = (depth > 0)[:,:,None] # [H, W, 1]
#     # print(np.sum(depth == 0))
#     # save to image
#     return rgb * valid_mask + img * (1-valid_mask), valid_mask[:, :, 0]


def render_mesh(img, verts, faces, cam_param):
    # img:[H, W, 3]/[bs, H, W, 3]
    # verts:[nv, 3]/[bs, nv, 3]
    # face:[nf, 3]/[bs, nf, 3]
    # mesh
    
    v = verts.clone() if torch.is_tensor(verts) else torch.tensor(verts.astype(float)).float().clone()
    f = faces.clone() if torch.is_tensor(faces) else torch.tensor(faces.astype(float)).float().clone()
    i = img.clone() if torch.is_tensor(img) else torch.tensor(img).float().clone()
    fcl = cam_param['focal'].clone() if torch.is_tensor(cam_param['focal']) else torch.tensor(cam_param['focal']).float().clone()
    prp = cam_param['princpt'].clone() if torch.is_tensor(cam_param['princpt']) else torch.tensor(cam_param['princpt']).float().clone()
    if len(verts.shape) == 2:
        v = v[None]
        f = f[None]
        i = i[None]
        fcl, prp = fcl[None], prp[None]
    meshes = Meshes(verts=v, faces=f)
    device = meshes.device
    bs, nv = len(meshes.verts_list()), len(meshes.verts_list()[0])
    color = torch.ones(bs, nv, 3, device=device)
    color = color * 255.
    meshes.textures = TexturesVertex(verts_features=color)
    
    _, img_height, img_weight, _ = i.shape
    fcl[:, 0] = fcl[:, 0] * 2 / min(img_height, img_weight)
    fcl[:, 1] = fcl[:, 1] * 2 / min(img_height, img_weight)
    prp[:, 0] = - (prp[:, 0] - img_weight // 2) * 2 / img_weight
    prp[:, 1] = - (prp[:, 1] - img_height // 2) * 2 / img_height

    cameras = PerspectiveCameras(focal_length=-fcl, principal_point=prp, device=device)
    raster_settings = RasterizationSettings(
        image_size=(img_height, img_weight), 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    shader_rgb=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
    shader_depth=SoftDepthShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
    
    # renderer
    renderer_rgb = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=shader_rgb
    )
    
    renderer_depth = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=shader_depth
    )


    # render
    znear = 0.0
    zfar = 100000.0
    rendered_depth = renderer_depth(meshes, znear=znear, zfar=zfar)    # [bs, H, W, 1]
    rendered_image = renderer_rgb(meshes, znear=znear, zfar=zfar)      # [bs, H, W, 3]
    
    if len(verts.shape) == 2:
        rgb = rendered_image[0, ..., :3]             # [H, W, 3]
        valid_mask = (rendered_depth < zfar)[0]      # [H, W, 1]
        i = i[0]                                     # [H, W, 3]
    else:
        rgb = rendered_image[..., :3]            # [bs, H, W, 3]
        valid_mask = (rendered_depth < zfar)     # [bs, H, W, 1]
    
    rgb = rgb.cpu().numpy().astype(np.float32)
    valid_mask = valid_mask.cpu().numpy().astype(np.float32)
    i = i.cpu().numpy()
    
    return rgb * valid_mask + i * (1-valid_mask), valid_mask[..., 0]         # [H, W, 3]/[bs, H, W, 3], [H, W]/[bs, H, W]