import torch
from cityscapes_loader import CityscapesDataset

class MidasHybrid(object):
    def __init__(self):
        self.model_type = "DPT_Hybrid"
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        self.midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def pred_depth_map(img):
        img_torch = torch.from_numpy(img).to(self.device)
        input_batch = self.midas_transform(img_torch).to(self.device)

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

    debug = True
    debug_num = 10
    debug_counter = 0
    if not os.path.exists("debug"):
        os.makedirs("debug")
    for img, gt_seg, img_path in dataset_train:
        print(img_path)
        depth_map = midas_predictor.pred_depth_map(img) 
        depth_map = depth_map / depth_map.max()

        cv2.imwrite("./debug/{}_img_debug.png".format(debug_counter), img)
        cv2.imwrite("./debug/{}_gt_seg_debug.png".format(debug_counter), gt_seg)
        cv2.imwrite("./debug/{}_depth_map_debug.png".format(debug_counter), depth_map)

        debug_counter += 1
        if debug_counter > debug_num:
            break
