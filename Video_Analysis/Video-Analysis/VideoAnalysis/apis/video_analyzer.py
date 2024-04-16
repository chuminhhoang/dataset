import sys
sys.path.insert(1, "../Video-Analysis")

from VideoAnalysis.detection import detector_factory
# from VideoAnalysis.action.models import HMR2018Predictor, HMRFullConfig
# from VideoAnalysis.action.models.hmr.utils_dataset import process_image
# from VideoAnalysis.synopsis import SynopsisVideoProducer

from VideoAnalysis.utils import get_pylogger
LOGGER = get_pylogger(__name__)

import numpy as np
import yaml
import dacite
import torch
import torch.nn as nn
from collections import defaultdict 



class VideoAnalyzer(nn.Module):
    def __init__(self, cfg, task=None):
        super().__init__()
        self.cfg = cfg
        # self.setup_HMR()
        self.setup_action_detection()
        self.setup_pose_predictor()
        self.setup_detector()
        # self.setup_tracker()
        


    ################# SETUP MODEL #################
    def setup_detector(self):
        LOGGER.info(f'Setting up {self.cfg.DETECT.MODEL} detection model. Using {self.cfg.DETECT.WEIGHT}...')
        self.detector = detector_factory[self.cfg.DETECT.MODEL](self.cfg)


    def setup_HMR(self):
        with open(f"{self.cfg.ACTION.EXTRACT_FEAT.HMR.CONFIG}", "r") as stream:
            try:
                HMR_cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        LOGGER.info(f"Setting up HMR model. Loading {HMR_cfg['hmr']['hmar_path']}...")

        HMR_cfg = dacite.from_dict(data_class=HMRFullConfig, data=HMR_cfg)
        self.HMR = HMR2018Predictor(HMR_cfg)
        self.HMR.load_weights(HMR_cfg.hmr.hmar_path)


    def setup_pose_predictor(self):
        if self.cfg.POSE.MODEL == 'mmpose':
            LOGGER.info(f'Setting {self.cfg.POSE.MODEL} - {self.cfg.POSE.MODEL_CONFIG} pose predictor. \
                        USing {self.cfg.POSE.WEIGHT}...')
            from mmpose.apis import init_model, inference_bottomup
            self.pose_predictor = init_model(self.cfg.POSE.MODEL_CONFIG,
                                             device=self.cfg.POSE.DEVICES,
                                             checkpoint=self.cfg.POSE.WEIGHT)
    

    def setup_tracker(self):
        if self.cfg.TRACK.MODEL == "motrv2":
            from VideoAnalysis.track import MOTRv2
            LOGGER.info(f"Setting up {self.cfg.TRACK.MODEL} tracker...")
            self.tracker = MOTRv2(self.cfg)

    def setup_action_detection(self):
        import mmcv
        from mmengine.config import Config
        from mmdet.apis import inference_detector, init_detector
        if self.cfg.ACTION.MODEL == 'MAE':
            self.test_frame = []
            config = Config.fromfile(self.cfg.ACTION.config_slowfast)
            self.action_model = init_detector(config, self.cfg.ACTION.checkpoint_slowfast, device=self.cfg.ACTION.device) 
            from VideoAnalysis.action.mmaction2.tools.preprocess import TaskInfo
            self.task = TaskInfo()
            self.task.frames = []
            self.task.processed_frames = []
            self.clip_len = self.cfg.ACTION.clip_len
            frame_interval = self.cfg.ACTION.frame_interval
            predict_stepsize = self.cfg.ACTION.predict_stepsize
            self.buffer_size = int(self.clip_len - predict_stepsize)
            self.task.frames_inds = [i for i in range(0, self.clip_len, frame_interval)]
            self.img_norm_cfg = dict(
            mean=np.array([123.675, 116.28, 103.53]),
            std=np.array([58.395, 57.12, 57.375]),
            to_rgb=False)

    def setup_synopsis_video_producer(self):
        LOGGER.info('Setting up Synopsis Video Producer...')
        self.synopsis_video_producer = SynopsisVideoProducer(self.cfg)

    def setup_tracker_ground_plane(self, camparam_1, camparam_2, camparam_3):
        from VideoAnalysis.track.trackers.motrv2.caculate_uv import Mapper
        # example for type and data cam para input for UCMC track
        # campara_param = np.asarray([[0.00000, -1.00000, 0.00000],
        #                 [0.00000, 0.00000, -1.00000],
        #                 [1.00000, 0.00000, 0.00000]])
        # campara_param_1 = np.asarray([[0, 1600, 3624]])
        # campara_param_2 = np.asarray([[1200, 0, 960],  
        #                 [0, 1200, 540], 
        #                 [0, 0, 1 ]])
        self.mapper = Mapper(campara_param=camparam_1, campara_param_1 = camparam_2, campara_param_2 = camparam_3 , dataset=None)
    
    ################# GET RESULTS ################# 

    def get_bbox(self, image):
        bbox = self.detector(image)
        return bbox

    def get_pose(self, image):
        if self.cfg.POSE.MODEL == 'mmpose':
            from mmpose.apis import init_model, inference_bottomup
            pose_result = inference_bottomup(self.pose_predictor, image)
            keypoints_2D = pose_result[0].pred_instances.keypoints
            bboxes = pose_result[0].pred_instances.bboxes
            bbox_scores = pose_result[0].pred_instances.bbox_scores
            det_thresh = self.cfg.DETECT.CONF
            bboxes = bboxes[bbox_scores>det_thresh]
            keypoints_2D =  keypoints_2D[bbox_scores>det_thresh]
            bbox_scores = bbox_scores[bbox_scores>det_thresh]
            bboxes = np.concatenate([bboxes, bbox_scores[..., None]], axis=1)

        return bboxes, keypoints_2D

    def get_pose_ver_training(self, image):
        if self.cfg.POSE.MODEL == 'mmpose':
            from mmpose.apis import init_model, inference_bottomup
            pose_result = inference_bottomup(self.pose_predictor, image)
            keypoints_2D = pose_result[0].pred_instances.keypoints
            bboxes = pose_result[0].pred_instances.bboxes
            bbox_scores = pose_result[0].pred_instances.bbox_scores
            det_thresh = self.cfg.DETECT.CONF
            bboxes = bboxes[bbox_scores>det_thresh]
            keypoints_2D =  keypoints_2D[bbox_scores>det_thresh]
            bbox_scores = bbox_scores[bbox_scores>det_thresh]
            bboxes = np.concatenate([bboxes, bbox_scores[..., None]], axis=1)

        return bboxes, keypoints_2D
    def get_u_v_feature(self, bbox):
        y, R = self.mapper.mapto(bbox)
        return y, R
    
    def get_context_feature(self, image, idx):
        # h-w ? w-h not checked
        import mmcv
        w, h =  image.shape[:2]
        self.task.frames.append(mmcv.imresize(image, (w, h)))
        stdet_input_size = mmcv.rescale_size((w, h), (256, np.Inf))
        self.task.ratio = tuple(n / o for n, o in zip(stdet_input_size, (w, h)))
        processed_frame= mmcv.imresize(image, stdet_input_size).astype(np.float32)
        processed_frame = mmcv.imnormalize_(processed_frame, **self.img_norm_cfg)
        self.task.processed_frames.append(processed_frame)
        if  len(self.task.processed_frames) == self.clip_len:
            # campara_param have 3 items, 1-> RotationMatrices(3*3), 2-> TranslationVectors(1*3), 3-> IntrinsicMatrix(3*3)
            # self.dict_ground_plane = defaultdict(list)
            self.task.add_frames(idx, self.task.frames, self.task.processed_frames)
            mid_bbox = []
            for key in self.tracker.dict_bboxes.keys():
                mid_frame = (len(self.tracker.dict_bboxes[key])//2)
                for indx, bbox in enumerate(self.tracker.dict_bboxes[key]):
                    if indx==mid_frame:
                        # y, R = mapper.mapto(bbox)
                        # self.dict_ground_plane[key].append([y, R])
                        mid_bbox.append(bbox)
                        break
                    # check device used in here for cuda()
            self.task.add_bboxes(torch.Tensor(mid_bbox).to('cuda:0'))
            with torch.no_grad():
                # check device used in here for inputs
                preds, feats = self.action_model(**self.task.get_model_inputs('cuda:0'))
            self.tracker.dict_bboxes.clear()
            self.task.processed_frames = self.task.processed_frames[-self.buffer_size:]
            self.task.frames = self.task.frames[-self.buffer_size:]  
            return feats
    
    def get_context_feature_train_version(self, image, idx, bboxes):
        # h-w ? w-h not checked
        import mmcv
        import cv2
        w, h =  image.shape[:2]
        # replace append => list full
        self.test_frame.append(image)
        # for list in here
        self.task.frames.append(mmcv.imresize(image, (w, h)))
        stdet_input_size = mmcv.rescale_size((w, h), (256, np.Inf))
        self.task.ratio = tuple(n / o for n, o in zip(stdet_input_size, (w, h)))
        processed_frame= mmcv.imresize(image, stdet_input_size).astype(np.float32)
        processed_frame = mmcv.imnormalize_(processed_frame, **self.img_norm_cfg)
        self.task.processed_frames.append(processed_frame)
        self.buffer_size = 58
        mid_bbox = []
        if  len(self.task.processed_frames) == self.clip_len:
            # campara_param have 3 items, 1-> RotationMatrices(3*3), 2-> TranslationVectors(1*3), 3-> IntrinsicMatrix(3*3)
            # self.dict_ground_plane = defaultdict(list)
            self.task.add_frames(idx, self.task.frames, self.task.processed_frames)
            mid_bbox = bboxes[len(self.task.processed_frames)//2-1]  
            # test_img =  self.test_frame[len(self.task.processed_frames)//2-1]
            person_id = mid_bbox[:, 4]
            mid_bbox = mid_bbox[:, :4].tolist()
            self.task.add_bboxes(torch.Tensor(mid_bbox).to('cuda:0'))
            with torch.no_grad():
                # check device used in here for inputs
                a = self.action_model(**self.task.get_model_inputs('cuda:0'))
            self.tracker.dict_bboxes.clear()
            self.task.processed_frames = self.task.processed_frames[-self.buffer_size:]
            self.task.frames = self.task.frames[-self.buffer_size:] 
            self.test_frame = self.test_frame[-self.buffer_size:]
            if mid_bbox[0][0] == -1:
                return 1, []
            return a, person_id
        return None, None
    def get_track(self, image, proposals):
        torch.cuda.empty_cache()
        if self.cfg.TRACK.MODEL == "motrv2":
            tracklets = self.tracker.detect(image, proposals)
        return tracklets
    def reset(self):
        self.tracker.reset()

    def get_human_features(self, image, frame_name, t_, gt=1, ann=None, extra_data=None):
        bboxes, keypoints_2D = self.get_pose(image)
        
        NPEOPLE = bboxes.shape[0]
        if NPEOPLE == 0:
            return []

        # TODO: Track to reID and compute ground-point here
        tracklets = self.get_track(image, bboxes)

        img_height, img_width, _  = image.shape
        new_image_size            = max(img_height, img_width)
        top, left                 = (new_image_size - img_height)//2, (new_image_size - img_width)//2,
        
        ratio = 1.0/int(new_image_size)*self.cfg.render.res
        image_list = []
        center_list = []
        scale_list = []
        selected_ids = []

        for p_ in range(NPEOPLE):
            if bboxes[p_][2]-bboxes[p_][0]<self.cfg.phalp.small_w or bboxes[p_][3]-bboxes[p_][1]<self.cfg.phalp.small_h:
                continue
            p_image, center_, scale_, center_pad, scale_pad = self.get_croped_image(image, bboxes[p_], bboxes[p_]) # PHALP return both bboxes and bboxes_pad 
                                                                                                                   # as pred instance from MaskRCNN
            image_list.append(p_image)
            center_list.append(center_pad)
            scale_list.append(scale_pad)
            selected_ids.append(p_)
        
        BS = len(image_list)
        if BS == 0: return []
        
        with torch.no_grad():
            extra_args      = {}
            hmar_out        = self.HMAR(image_list.cuda(), **extra_args)

            pred_smpl_params, pred_joints_2d, pred_joints, pred_cam  = self.HMAR.get_3d_parameters(hmar_out['pose_smpl'], hmar_out['pred_cam'],
                                                                                                   center=(np.array(center_list) + np.array([left, top]))*ratio,
                                                                                                   img_size=self.cfg.render.res,
                                                                                                   scale=np.max(np.array(scale_list), axis=1, keepdims=True)*ratio)
            pred_smpl_params = [{k:v[i].cpu().numpy() for k,v in pred_smpl_params.items()} for i in range(BS)]
            pred_cam_ = pred_cam.view(BS, -1)
            pred_cam_.contiguous()
        
        detection_data_list = []
        for i, p_ in enumerate(selected_ids):
            detection_data = {
                    "bbox"            : np.array([bboxes[p_][0], bboxes[p_][1], (bboxes[p_][2] - bboxes[p_][0]), (bboxes[p_][3] - bboxes[p_][1])]), # xmin, ymin, w, h
                    "conf"            : bboxes[p_][-1],
                    "id"              : None,
                    
                    "center"          : center_list[i],
                    "scale"           : scale_list[i],
                    "smpl"            : pred_smpl_params[i],
                    "camera"          : pred_cam_[i].cpu().numpy(),
                    "camera_bbox"     : hmar_out['pred_cam'][i].cpu().numpy(),
                    "2d_joints"       : keypoints_2D[p_],
                    
                    "size"            : [img_height, img_width],
                    "img_path"        : frame_name,
                    "img_name"        : frame_name.split('/')[-1] if isinstance(frame_name, str) else None,
                    "class_name"      : 0,
                    "time"            : t_,

                    "ground_truth"    : gt[p_],
                    "annotations"     : ann[p_],
                    "extra_data"      : extra_data[p_] if extra_data is not None else None
            }
            detection_data_list.append(detection_data)
        
        return detection_data_list


    def get_croped_image(image, bbox, bbox_pad):
        center_      = np.array([(bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2])
        scale_       = np.array([(bbox[2] - bbox[0]), (bbox[3] - bbox[1])])

        center_pad   = np.array([(bbox_pad[2] + bbox_pad[0])/2, (bbox_pad[3] + bbox_pad[1])/2])
        scale_pad    = np.array([(bbox_pad[2] - bbox_pad[0]), (bbox_pad[3] - bbox_pad[1])])
        image_tmp    = process_image(image, center_pad, 1.0*np.max(scale_pad))

        return image_tmp, center_, scale_, center_pad, scale_pad
    
    def xyxy2xywhn(bboxes, img_shape):
        bboxes = bboxes.astype(np.float64)
        dw, dh = 1 / img_shape
        bboxes[:, 2] = (bboxes[:, 2] - bboxes[:, 0])
        bboxes[:, 3] = (bboxes[:, 3] - bboxes[:, 1])
        bboxes[:, ::2] =  bboxes[:, ::2]*dw
        bboxes[:, 1::2] = bboxes[:, 1::2]*dh
        return bboxes
    
        

