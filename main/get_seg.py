# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from PIL import Image
import PIL.Image
import numpy as np
import argparse
import json
from tqdm import tqdm
from typing import Any, Dict, List
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel.data_parallel import DataParallel
import os
import sys
current_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(current_path + '/../../common'))
sys.path.append(os.path.abspath(current_path + '/../../data/InterHand2.6M'))
from seg_dataset import Seg_Dataset
from seg_config import seg_cfg
from utils.vis import render_mesh
from segment_anything.utils.transforms import ResizeLongestSide

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

# parser.add_argument(
#     "--input",
#     type=str,
#     required=True,
#     help="Path to either a single input image or folder of images.",
# )

# parser.add_argument(
#     "--output",
#     type=str,
#     required=True,
#     help=(
#         "Path to the directory where masks will be output. Output will be either a folder "
#         "of PNGs per image or a single json with COCO-style masks."
#     ),
# )

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)

amg_settings.add_argument('--gpu', type=str, dest='gpu_ids')
amg_settings.add_argument('--set', type=str, dest='set')


# def write_masks_to_folder(image, masks: List[Dict[str, Any]], path: str) -> None:
#     header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
#     metadata = [header]
#     for i, mask_data in enumerate(masks):
#         mask = mask_data["segmentation"]
#         filename = f"{i}.png"
#         PIL.Image.fromarray(np.uint8(mask[:, :, None] * image)).save(os.path.join(path, filename))
#         #cv2.imwrite(os.path.join(path, filename), mask[:, :, None] * image)
#         mask_metadata = [
#             str(i),
#             str(mask_data["area"]),
#             *[str(x) for x in mask_data["bbox"]],
#             *[str(x) for x in mask_data["point_coords"][0]],
#             str(mask_data["predicted_iou"]),
#             str(mask_data["stability_score"]),
#             *[str(x) for x in mask_data["crop_box"]],
#         ]
#         row = ",".join(mask_metadata)
#         metadata.append(row)
#     metadata_path = os.path.join(path, "metadata.csv")
#     with open(metadata_path, "w") as f:
#         f.write("\n".join(metadata))

#     return


def write_masks_to_folder(image, boxes, masks: List[Dict[str, Any]], path: str) -> None:
    for i, (box, mask) in enumerate(zip(boxes, masks)):
        filename = f"{i}.png"
        img2show = image.copy() * mask[:, :, None]
        cv2.rectangle(img2show, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))
        PIL.Image.fromarray(np.uint8(img2show)).save(os.path.join(path, filename))
    return


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    # seg_cfg.set_args(args.gpu_ids)
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    cfg_dict = {
        'encoder_img_size': sam.image_encoder.img_size,
    }
    seg_cfg.set_args(args.gpu_ids, **cfg_dict)
    resize_transform = ResizeLongestSide(seg_cfg.encoder_img_size)
    # sam = DataParallel(sam).cuda()
    sam = sam.cuda()
    
    def collate_fn(x):
        return x
    
    def process(x):
        img = torch.zeros(len(x), 512, 334, 3)                             # [bs, H, W, 3]
        valid_verts = torch.stack([bi['valid_verts'] for bi in x], 0)        # [bs, 2*778, 3]
        valid_faces = torch.stack([bi['valid_faces'] for bi in x], 0)        # [bs, 2*nf, 3]
        render_focal = torch.stack([bi['render_focal'] for bi in x], 0)      # [bs, 2]
        render_princpt = torch.stack([bi['render_princpt'] for bi in x], 0)  # [bs, 2]
        cam_param = {'focal': render_focal, 'princpt': render_princpt}
        _, render_masks = render_mesh(img, valid_verts / 1000, valid_faces, cam_param)   # [bs, H, W], 1 or 0
        
        ret_x = []
        for i, ec in enumerate(x):
            new_ec = {'image': ec['image'], 
                      'original_size': ec['original_size'], 
                      'idx': ec['idx'],
                      'ori_image': ec['ori_image'],
                      }
            bbox = np.zeros((4,))
            render_mask = render_masks[i]
            if ec['need_rot'] == 1.:
                render_mask = cv2.rotate(render_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            bbox[0], bbox[1] = min(np.where(render_mask)[1]), min(np.where(render_mask)[0])
            bbox[2], bbox[3] = max(np.where(render_mask)[1]), max(np.where(render_mask)[0])
            ori_box = bbox.copy().astype(int)
            bbox = torch.tensor(bbox).to(ec['image'].device).unsqueeze(0)  # [1, 4]
            new_ec['boxes'] = resize_transform.apply_boxes_torch(bbox, ec['original_size'])   # xyxy
            new_ec['ori_box'] = ori_box
            new_ec['render_mask'] = render_mask
            
            ret_x.append(new_ec)
        return ret_x

    testset_loader = Seg_Dataset(args.set)
    batch_generator = DataLoader(dataset=testset_loader, batch_size=seg_cfg.num_gpus * seg_cfg.test_batch_size,
                                shuffle=True, num_workers=seg_cfg.num_thread, pin_memory=True, collate_fn=collate_fn)
    # input_boxs = [np.array([60,140,325,305])]
    # if not os.path.isdir(args.input):
    #     targets = [args.input]
    # else:
    #     targets = [
    #         f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
    #     ]
    #     targets = [os.path.join(args.input, f) for f in targets]

    # os.makedirs(args.output, exist_ok=True)

    # for t, b in zip(targets, input_boxs):
    #     print(f"Processing '{t}'...")
    #     image = cv2.imread(t)
    #     if image is None:
    #         print(f"Could not load '{t}' as an image, skipping...")
    #         continue
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    #     predictor.set_image(image)
    #     masks, _, _ = predictor.predict(
    #         point_coords=None,
    #         point_labels=None,
    #         box=b[None, :],
    #         multimask_output=False,
    #     )

    #     base = os.path.basename(t)
    #     base = os.path.splitext(base)[0]
    #     save_base = os.path.join(args.output, base)
    #     if output_mode == "binary_mask":
    #         os.makedirs(save_base, exist_ok=True)
    #         write_masks_to_folder(image, input_boxs, masks, save_base)
    #     else:
    #         save_file = save_base + ".json"
    #         with open(save_file, "w") as f:
    #             json.dump(masks, f)
    # print("Done!")
    mask_dir = '/data1/linxiaojian/Datasets/interhand/masks'
    for itr, batched_input in enumerate(tqdm(batch_generator)):
        for bid in range(len(batched_input)):
            for k, v in batched_input[bid].items():
                if torch.is_tensor(v):
                    batched_input[bid][k] = v.to(sam.device)
        
        # count = 0
        # for bid in range(len(batched_input)):
        #     img_path = testset_loader.datalist[batched_input[bid]['idx']]['img_path']
        #     filename = img_path[len(testset_loader.img_path)+1:]
        #     if os.path.exists(os.path.join(mask_dir, filename)):
        #         count += 1
        # if count == len(batched_input):
        #     continue
        batched_input = process(batched_input)
        batched_output = sam(batched_input, multimask_output=True)

        for bid in range(len(batched_output)):
            render_mask = batched_input[bid]['render_mask']     # [H, W], 0 or 1
            
            mask = np.zeros_like(render_mask)    # [H, W]
            overlap = 0
            for mid in range(len(batched_output[bid]['masks'][0])):
                ms = batched_output[bid]['masks'][0][mid].detach().cpu().numpy()  # [H, W], 0 or 1
                
                ol = (render_mask.astype(np.uint8) + ms.astype(np.uint8)) == 2        # [H, W]
                
                if np.sum(ol) >= overlap:
                    overlap = np.sum(ol)
                    mask = ms
                    
            img = batched_input[bid]['ori_image']      # [H, W, 3]
            
            img_path = testset_loader.datalist[batched_input[bid]['idx']]['img_path']
            filename = img_path[len(testset_loader.img_path)+1:]
            os.makedirs(os.path.join(mask_dir, os.path.dirname(filename)), exist_ok=True)
            mask_path = os.path.join(mask_dir, filename)

            # img2save = img * mask[:, :, None]
            # img2save = img * render_mask[:, :, None]
            img2save = mask[:, :] * 255
            # box = batched_input[bid]['ori_box']
            # cv2.rectangle(img2save, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))
            img2save = np.uint8(img2save)
            # print(np.sum((img2save == 0) | (img2save == 255)))
            PIL.Image.fromarray(img2save).save(mask_path)
        #     print(mask_path)
        # if itr > -1:
        #     break

if __name__ == "__main__":
    args = parser.parse_args()
    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    main(args)
