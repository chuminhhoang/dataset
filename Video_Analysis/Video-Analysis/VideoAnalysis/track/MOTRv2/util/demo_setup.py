import json
import random
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from ultralytics import YOLO
from models.structures import Instances
import torchvision.transforms.functional as F

class ReadImage():
    def __init__(self) -> None:
        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def init_img(self, img):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img
    
class Detector(object):
    def __init__(self, args, model, vid):
        self.args = args
        self.detr = model
        self.vid = vid
        self.seq_num = os.path.basename(vid)
        self.predict_path = os.path.join(self.args.output_dir, args.exp_name)

        os.makedirs(self.predict_path, exist_ok=True)

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]

    def detect(self, prob_threshold=0.4, area_threshold=100, vis=False, objects=None, i = 0, track_instances=None, bboxes = None, conf = None):
        total_dts = 0
        total_occlusion_dts = 0
        lines = []
        augment = ReadImage()
        cur_img, ori_img = augment.init_img(objects)
        results = torch.cat((bboxes, conf), dim=1)
        proposals = results.reshape(-1, 5)
        cur_img, proposals = cur_img.cuda(), proposals.cuda()

        # track_instances = None
        if track_instances is not None:
            track_instances.remove('boxes')
            track_instances.remove('labels')
        seq_h, seq_w, _ = ori_img.shape

        res = self.detr.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
        track_instances = res['track_instances']

        dt_instances = deepcopy(track_instances)
        # filter det instances by score.
        dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
        dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

        total_dts += len(dt_instances)

        bbox_xyxy = dt_instances.boxes.tolist()
        identities = dt_instances.obj_idxes.tolist()
        conf = dt_instances.scores.tolist()
        save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
        track_array = []
        for xyxy, (track_id, conf) in zip(bbox_xyxy, zip(identities, conf)):
            if track_id < 0 or track_id is None:
                continue
            x1, y1, x2, y2 = xyxy
            track_array.append([x1, y1, x2, y2, float(conf), int(track_id)])
            self.dict_bboxes[str(track_id)].append([int(x1), int(y1), int(x2), int(y2)])
        print("totally {} dts {} occlusion dts".format(total_dts, total_occlusion_dts))
        return track_instances, ori_img, np.array(track_array)

class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=10):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.score_thresh)
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_thresh)
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_idxes[to_del] = -1
