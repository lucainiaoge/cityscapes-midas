import os
import numpy as np
import cv2
import torch
from cityscapes_loader import CityscapesDataset

DEPTH_MAP_SUFFIX = "DepthMap"

class MidasHybrid(object):
    def __init__(self):
        self.model_type = "DPT_Hybrid"
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        self.midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def pred_depth_map(self, img):
        input_batch = self.midas_transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.unsqueeze(-1).cpu().numpy() #(B, H, 1)

if __name__ == "__main__":
    local_path = "./data/cityscapes"
    dataset_train = CityscapesDataset(local_path, split="train")
    dataset_val = CityscapesDataset(local_path, split="val")
    midas_predictor = MidasHybrid()

    for img, gt_seg, img_path in dataset_train:
        print("MiDaS processing image:", img_path)
        depth_map = midas_predictor.pred_depth_map(img) 
        depth_map = depth_map / depth_map.max()
        
        img_path = os.path.normpath(img_path)
        img_path_list = img_path.split(os.sep)
        img_filename = img_path_list[-1]
        depth_map_filename = img_filename[:-15] + DEPTH_MAP_SUFFIX + ".png"
        print("Depth map filename:", depth_map_filename)

        depth_map_dir_list = img_path_list[:-1]
        depth_map_dir_list[-3] = DEPTH_MAP_SUFFIX
        depth_map_dir = os.path.join(*depth_map_dir_list)

        if not os.path.exists(depth_map_dir):
            os.makedirs(depth_map_dir)
        print("Depth map path:", depth_map_dir)

        depth_map_path = os.path.join(depth_map_dir, depth_map_filename)

        cv2.imwrite(depth_map_path, depth_map*255)

