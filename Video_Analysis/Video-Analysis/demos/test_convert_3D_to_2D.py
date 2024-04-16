import sys
sys.path.insert(1, "../Video-Analysis")

from VideoAnalysis.action.models.utils.utils import perspective_projection
import joblib
import os
import numpy as np
import torch

if __name__ == "__main__":
    PATH = "/home/hoang/Documents/VideoSynopsis/Code/Video-Analysis/data"
    files = os.listdir(os.path.join(PATH, "ava_val")) 
    pkl_file_test = os.path.join(PATH, "ava_val", files[0])
    data = joblib.load(pkl_file_test)

    BS = data["3d_joints"].shape[0]

    joints_3d = data["3d_joints"][:, 0]
    pred_cam = data["camera"][:, 0]
    rotation = np.eye(3)
    camera_center = np.zeros(shape=(BS, 2))
    focal_length = np.full(shape=(BS, 2), fill_value=5000)
    img_size = np.array(data["frame_size"])
    
    joints_3d = torch.from_numpy(joints_3d)
    rotation = torch.from_numpy(rotation).unsqueeze(0).expand(BS, -1, -1)
    pred_cam = torch.from_numpy(pred_cam)
    focal_length = torch.from_numpy(focal_length)
    img_size = torch.from_numpy(img_size)
    camera_center = torch.from_numpy(camera_center)

    joints_2d = perspective_projection(joints_3d, rotation, 
                                       translation=pred_cam,
                                       focal_length=focal_length/img_size,
                                       camera_center=camera_center)
    print(joints_2d[0].size())

    
    