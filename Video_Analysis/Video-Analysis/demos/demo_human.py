import sys
sys.path.insert(1, "../Video-Analysis")
from VideoAnalysis.detection.yolo8 import YOLO8
from VideoAnalysis.configs.default import _C
from VideoAnalysis.apis.video_analyzer import VideoAnalyzer
if __name__ == '__main__':
    cfg = _C.clone()
    cfg.DETECT.METHOD = 'normal'
    # cfg.DETECT.MODEL = 'weights/yolov8l_crowdhuman-human-only_50e_1024.pt'
    cfg.DETECT.SZ = (768, 1280)
    video_analyze = VideoAnalyzer(cfg)
    model = video_analyze.get_bbox
    track = video_analyze.get_track

    import cv2 as cv
    # cv.namedWindow("f", cv.WINDOW_NORMAL) 
    # cv.resizeWindow('f', 1280, 700)
    vid = cv.VideoCapture('/root/Hoang/Video_Analysis/Video-Analysis/VideoAnalysis/_145Aa_xkuE.mp4')

    count = 0
    while True:
        suc, frame = vid.read()
        if not suc:
            break
        
        if count % 1 == 0:
            result = model([frame])[0]
            tracklet = track(frame, result)
            print(tracklet)
            print('--------------')
            for r in result:
                x1, y1, x2, y2, conf, _ = r
                cv.rectangle(frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0,255,0),
                            2)
                cv.putText(frame, f'{conf:0.3f}', 
                        (int((x1)), int((y1-10))),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2, cv.LINE_AA)
            # cv.imshow('f', frame)
            cv.imwrite('/root/Hoang/Video_Analysis/Video-Analysis/check/{}.jpg'.format(str(count)), frame)
        count += 1
        
        if cv.waitKey(1) & 0xff == 27:
            break
