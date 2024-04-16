import sys
import numpy as np
import cv2
from collections import defaultdict
from VideoAnalysis.apis.video_analyzer import VideoAnalyzer
def caculate_iou(grouth_truth, bbox):
    x1_min, y1_min, w, h = grouth_truth
    x1_max = x1_min+w
    y1_max = y1_min+h
    x2_min, y2_min, x2_max, y2_max = bbox
    intersection_x1 = max(x1_min, x2_min)
    intersection_y1 = max(y1_min, y2_min)
    intersection_x2 = min(x1_max, x2_max)
    intersection_y2 = min(y1_max, y2_max)
    
    # Calculate the areas of the bounding boxes and the intersection
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
    
    # Calculate the Union area
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou
def add_datatrain_feature(path, ground_truth_bbx, frame_id, video_analyze):
    frame = cv2.imread(path)
    x_min, y_min, w, h = ground_truth_bbx[0]
    x_max = x_min+w
    y_max = y_min+h
    
    bboxes, point = video_analyze.get_pose(frame[int(y_min):int(y_max), int(x_min):int(x_max)])
    point=point[:1, :, :]
    point[:, :, 0] = point[:, :, 0]+x_min
    point[:, :, 1] = point[:, :, 1]+y_min
    selected_point = np.array(point)

    return selected_point.flatten()

# def check(feature, bbox):
#     for i in range(len(feature)):
#         if(feature[6].all()==feature[6+1].all()):
#             print(bbox[i])
#             print(bbox[i+1])
#             print('------------')
#         else: print('no')
        
