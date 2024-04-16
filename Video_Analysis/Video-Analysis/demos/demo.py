import sys
import torch
import sys
sys.path.insert(1, '../Video-Analysis')
from VideoAnalysis.configs import default
from VideoAnalysis.apis.video_analyzer import VideoAnalyzer
import cv2
import numpy as np
import joblib
# from VideoAnalysis.track.MOTRv2.models import build_model
# from VideoAnalysis.track.MOTRv2.util.tool import load_model
# from VideoAnalysis.track.MOTRv2.util.demo_setup import ReadImage, Detector, RuntimeTrackerBase


if __name__ == '__main__':
    cfg = default._C.clone()
    video_analyzer = VideoAnalyzer(cfg)
    image = cv2.imread('/root/Hoang/Video_Analysis/Video-Analysis/data_image/27000.jpg')
    with open('/root/Hoang/Video_Analysis/Video-Analysis/data/ava/ava-train_-5KQ66BBWC4_000060_1_91.pkl', 'rb') as fp:
        data = joblib.load(fp)
    x_min, y_min, w, h = data['frame_bbox'][0]
    x_max = x_min+w
    y_max = y_min+h
    top, left = (x_min, y_max)
    cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    check_cropped_image = cropped_image
    bbox, point = video_analyzer.get_pose(cropped_image)
    dot_radius = 2 
    dot_color = (255, 0, 0)     
    for p in point[0]:
        x, y = p
        cv2.circle(cropped_image, (int(x), int(y)), dot_radius, dot_color, -1)
        print(x+x_min)
        print(y+y_min)
        cv2.circle(image, (int(x+x_min), int(y+y_min)), dot_radius, dot_color, -1)
    cv2.imwrite('/root/Hoang/Video_Analysis/a.jpg', cropped_image)
    cv2.imwrite('/root/Hoang/Video_Analysis/b.jpg', image)
#     # detr.track_embed.score_thr = cfg.TRACK.update_score_threshold
#     # device = cfg.TRACK.device
#     # detr.track_base = RuntimeTrackerBase(cfg.TRACK.score_threshold,  cfg.TRACK.score_threshold,  cfg.TRACK.miss_tolerance)
#     # checkpoint = torch.load(cfg.TRACK.resume, map_location='cpu')
#     # detr = load_model(detr, cfg.TRACK.resume).cuda()
#     # detr.eval()
#     # video_path = "/root/Hoang/MOTRv2/1_1.mp4"  # Change the extension based on the desired video format
#     # # fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # You can also try 'MJPG' or 'H264'
#     # # video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 720))
#     # cap = cv2.VideoCapture("/root/Hoang/MOTRv2/1_1.mp4" )
#     # if not cap.isOpened():
#     #     print("Không thể mở video")
#     # idx = 0
#     # det = Detector(args, model=detr, vid='aa')
#     # track_instance = None
#     # while True:
#     #     ret, frame = cap.read()
#     #     if not ret:
#     #         break
#     #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     #     idx+=1
#     #     if idx%2==0:
#     #         track_instance, ori_img = det.detect(args.score_threshold, objects = frame, i=idx, track_instances = track_instance) 
# import joblib 
# with open('/root/Hoang/Video_Analysis/Video-Analysis/data/ava/ava-train_-5KQ66BBWC4_000060_1_91.pkl', 'rb') as fp:
#     data = joblib.load(fp)
# print(data['action_label_psudo'])
# with open('/root/Hoang/Video_Analysis/a.txt', 'w') as w:
#     w.write(data)