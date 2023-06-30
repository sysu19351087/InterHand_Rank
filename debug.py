import torch
import os
import cv2

# preds = torch.load('/data1/linxiaojian_m2023/InterHand2.6M-main/cliff10_InterHand2.6M_0.1/result/train/ins.pth')

# preds['do_flip'] = preds.pop('do_filp')
# for k, v in preds.items():
#     print(k, v.shape)
# torch.save(preds, '/data1/linxiaojian_m2023/InterHand2.6M-main/cliff10_InterHand2.6M_0.1/result/train/ins.pth')
# root_path = '/data1/linxiaojian/InterHand2.6M-main/cliff10_InterHand2.6M_0.1/model_dump'
# cfg_dict_to_save = {
#     'train_ratio': 0.1,
#     'run_name': 'cliff10',
#     'backbone': 'intaghand',
#     'crop': False,
#     'init_zeros': False
# }
# for ckpt_path in os.listdir(root_path):
#     ckpt = torch.load(os.path.join(root_path, ckpt_path))
#     ckpt['cfg_dict_to_save'] = cfg_dict_to_save
#     torch.save(ckpt, os.path.join(root_path, ckpt_path))
#     print('save to', os.path.join(root_path, ckpt_path))
img_dir = "/data1/linxiaojian/Datasets/interhand/images"
mask_dir = "/data1/linxiaojian/Datasets/interhand/masks"
mode = 'train'
filename = "Capture3/0044_cup/cam410028/image19274.jpg"

img_path = os.path.join(img_dir, mode, filename)
mask_path = os.path.join(mask_dir, mode, filename)

ori_img = cv2.imread(img_path)
mask = cv2.imread(mask_path)

img2save = ori_img * (mask / 255)
cv2.imwrite('vis_mask.png', img2save)
cv2.imwrite('ori_image.png', ori_img)
