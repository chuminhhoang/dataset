import argparse
import atexit
import copy
import logging
import queue
import threading
import time
from abc import ABCMeta, abstractmethod

import cv2
import mmcv
import numpy as np
import torch
from mmengine import Config, DictAction
from mmengine.structures import InstanceData

from mmaction.structures import ActionDataSample

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this demo! ')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskInfo:
    """Wapper for a clip.

    Transmit data around three threads.

    1) Read Thread: Create task and put task into read queue. Init `frames`,
        `processed_frames`, `img_shape`, `ratio`, `clip_vis_length`.
    2) Main Thread: Get data from read queue, predict human bboxes and stdet
        action labels, draw predictions and put task into display queue. Init
        `display_bboxes`, `stdet_bboxes` and `action_preds`, update `frames`.
    3) Display Thread: Get data from display queue, show/write frames and
        delete task.
    """

    def __init__(self):
        self.id = -1

        # raw frames, used as human detector input, draw predictions input
        # and output, display input
        self.frames = None

        # stdet params
        self.processed_frames = None  # model inputs
        self.frames_inds = None  # select frames from processed frames
        self.img_shape = None  # model inputs, processed frame shape
        # `action_preds` is `list[list[tuple]]`. The outer brackets indicate
        # different bboxes and the intter brackets indicate different action
        # results for the same bbox. tuple contains `class_name` and `score`.
        self.action_preds = None  # stdet results

        # human bboxes with the format (xmin, ymin, xmax, ymax)
        self.display_bboxes = None  # bboxes coords for self.frames
        self.stdet_bboxes = None  # bboxes coords for self.processed_frames
        self.ratio = None  # processed_frames.shape[1::-1]/frames.shape[1::-1]

        # for each clip, draw predictions on clip_vis_length frames
        self.clip_vis_length = -1

    def add_frames(self, idx, frames, processed_frames):
        """Add the clip and corresponding id.

        Args:
            idx (int): the current index of the clip.
            frames (list[ndarray]): list of images in "BGR" format.
            processed_frames (list[ndarray]): list of resize and normed images
                in "BGR" format.
        """
        self.frames = frames
        self.processed_frames = processed_frames
        self.id = idx
        self.img_shape = processed_frames[0].shape[:2]

    def add_bboxes(self, display_bboxes):
        """Add correspondding bounding boxes."""
        self.display_bboxes = display_bboxes
        self.stdet_bboxes = display_bboxes.clone()
        self.stdet_bboxes[:, ::2] = self.stdet_bboxes[:, ::2] * self.ratio[0]
        self.stdet_bboxes[:, 1::2] = self.stdet_bboxes[:, 1::2] * self.ratio[1]

    def add_action_preds(self, preds):
        """Add the corresponding action predictions."""
        self.action_preds = preds

    def get_model_inputs(self, device):
        """Convert preprocessed images to MMAction2 STDet model inputs."""
        cur_frames = [self.processed_frames[idx] for idx in self.frames_inds]
        input_array = np.stack(cur_frames).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(device)
        datasample = ActionDataSample()
        datasample.proposals = InstanceData(bboxes=self.stdet_bboxes)
        datasample.set_metainfo(dict(img_shape=self.img_shape))

        return dict(
            inputs=input_tensor, data_samples=[datasample], mode='predict')


class StdetPredictor:
    """Wrapper for MMAction2 spatio-temporal action models.

    Args:
        config (str): Path to stdet config.
        ckpt (str): Path to stdet checkpoint.
        device (str): CPU/CUDA device option.
        score_thr (float): The threshold of human action score.
        label_map_path (str): Path to label map file. The format for each line
            is `{class_id}: {class_name}`.
    """

    def __init__(self, config, checkpoint, device, score_thr, label_map_path):
        self.score_thr = score_thr

        # load model
        config.model.backbone.pretrained = None
        # model = build_detector(config.model, test_cfg=config.get('test_cfg'))
        # load_checkpoint(model, checkpoint, map_location='cpu')
        # model.to(device)
        # model.eval()
        model = init_detector(config, checkpoint, device=device)
        self.model = model
        self.device = device

        # init label map, aka class_id to class_name dict
        with open(label_map_path) as f:
            lines = f.readlines()
        lines = [x.strip().split(': ') for x in lines]
        self.label_map = {int(x[0]): x[1] for x in lines}
        try:
            if config['data']['train']['custom_classes'] is not None:
                self.label_map = {
                    id + 1: self.label_map[cls]
                    for id, cls in enumerate(config['data']['train']
                                             ['custom_classes'])
                }
        except KeyError:
            pass

    def predict(self, task):
        """Spatio-temporval Action Detection model inference."""
        # No need to do inference if no one in keyframe
        if len(task.stdet_bboxes) == 0:
            return task

        with torch.no_grad():
            result = self.model(**task.get_model_inputs(self.device))
        scores = result[0].pred_instances.scores
        # pack results of human detector and stdet
        preds = []
        for _ in range(task.stdet_bboxes.shape[0]):
            preds.append([])
        for class_id in range(scores.shape[1]):
            if class_id not in self.label_map:
                continue
            for bbox_id in range(task.stdet_bboxes.shape[0]):
                if scores[bbox_id][class_id] > self.score_thr:
                    preds[bbox_id].append((self.label_map[class_id],
                                           scores[bbox_id][class_id].item()))

        # update task
        # `preds` is `list[list[tuple]]`. The outer brackets indicate
        # different bboxes and the intter brackets indicate different action
        # results for the same bbox. tuple contains `class_name` and `score`.
        task.add_action_preds(preds)

        return task


def abbrev(name):
        """Get the abbreviation of label name:

        'take (an object) from (a person)' -> 'take ... from ...'
        """
        while name.find('(') != -1:
            st, ed = name.find('('), name.find(')')
            name = name[:st] + '...' + name[ed + 1:]
        return name

def hex2color(h):
      """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
      return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))

if __name__ == "__main__":
    from ultralytics import YOLO
    import cv2
    import numpy as np
    import time

    import mmcv
    from mmengine import Config

    text_fontface=cv2.FONT_HERSHEY_SIMPLEX
    text_fontscale=0.5
    text_fontcolor=(255, 255, 255)  # white
    text_thickness=1
    text_linetype=1
    plate='03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
    plate = plate.split('-')
    plate = [hex2color(h) for h in plate]

    checkpoint = '/content/gdrive/MyDrive/Video-Analysis/weights/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth'
    config = '/content/mmaction2/configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py'
    config = Config.fromfile(config)

    detector = YOLO('/content/gdrive/MyDrive/Video-Analysis/weights/yolov8l_crowdhuman-human-only_50e_1024.pt')
    cap = cv2.VideoCapture('/content/gdrive/MyDrive/Video-Analysis/videos/out_first.mp4')


    device='cuda'
    action_score_thr=0.15
    # mmaction handler
    stdet_predictor = StdetPredictor(
        config=config,
        checkpoint=checkpoint,
        device=device,
        score_thr=action_score_thr,
        label_map_path='/content/mmaction2/tools/data/ava/label_map.txt')

    stdet_input_shortside=256
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    writer = cv2.VideoWriter('/content/gdrive/MyDrive/Video-Analysis/slowonly_test.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         25, (w, h))

    stdet_input_size = mmcv.rescale_size(
            (w, h), (stdet_input_shortside, np.Inf))

    display_size = (w, h)
    ratio = tuple(
            n / o for n, o in zip(stdet_input_size, display_size))

    img_norm_cfg = dict(
            mean=np.array(config.model.data_preprocessor.mean),
            std=np.array(config.model.data_preprocessor.std),
            to_rgb=False)

    clip_len = 8
    frame_interval = 8
    window_size = clip_len * frame_interval
    frame_start = window_size // 2 - (clip_len // 2) * frame_interval
    frames_inds = [
        frame_start + frame_interval * i for i in range(clip_len)
    ]
    predict_stepsize = 8
    buffer_size = window_size - predict_stepsize

    task = TaskInfo()
    task.ratio = ratio
    task.frames_inds = frames_inds

    frames = []
    processed_frames = []

    bboxes = []
    task.action_preds = []

    read_id = 1
    print('running...')
    begin = time.time()
    while True:
        suc, frame = cap.read()
        if not suc:
            print('End or Unable to read video .....')
            print('Average FPS: ', read_id / (time.time()-begin))
            break
        draw_frame = frame.copy()
        
        frames.append(frame)
        processed_frame = mmcv.imresize(
                            frame, stdet_input_size).astype(np.float32)
        _ = mmcv.imnormalize_(processed_frame,
                              **img_norm_cfg)
        processed_frames.append(processed_frame)

        if len(processed_frames) == window_size:
            # middle frame detection
            imgsz = (640, 640)
            conf = 0.51
            classes = (0,)
            verbose = False
            middle_frame = frames[window_size // 2]
            results = detector(frame,
                                imgsz=imgsz,
                                conf=conf,
                                classes=classes,
                                verbose=verbose, )

            results = [pred.boxes.data for pred in results]
            bboxes = results[0][:, :4]
            task.add_bboxes(bboxes)

            task.add_frames(read_id + 1, frames, processed_frames)
            stdet_predictor.predict(task)
            # print(read_id, '--------------------')
            # for pred in task.action_preds:
            #     pred = sorted(pred, key=lambda x: x[1], reverse=True)
            #     print(pred)
            # print('\n')

            processed_frames = processed_frames[-buffer_size:]
            frames = frames[-buffer_size:]
            
        max_labels_per_bbox = 5
        for bbox, preds in zip(bboxes, task.action_preds):
            preds = sorted(preds, key=lambda x: x[1], reverse=True)
            box = bbox.cpu().numpy().astype(np.int64)
            st, ed = tuple(box[:2]), tuple(box[2:])

            cv2.rectangle(draw_frame, st, ed, (0,255,0), 1)

            for k, (label, score) in enumerate(preds):
                if k >= max_labels_per_bbox:
                    break
                text = f'{abbrev(label)}: {score:.4f}'
                location = (0 + st[0], 18 + k * 18 + st[1])
                textsize = cv2.getTextSize(text, text_fontface,
                                            text_fontscale,
                                            text_thickness)[0]
                textwidth = textsize[0]
                diag0 = (location[0] + textwidth, location[1] - 14)
                diag1 = (location[0], location[1] + 2)
                cv2.rectangle(draw_frame, diag0, diag1, plate[k + 1], -1)
                cv2.putText(draw_frame, text, location, text_fontface,
                            text_fontscale, text_fontcolor,
                            text_thickness, text_linetype)
            
        
        writer.write(draw_frame)
        if cv2.waitKey(1) == 27:
            break
        read_id += 1


    writer.release()
    cap.release()