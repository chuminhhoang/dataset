from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# DETECTION TASK CONFIG
# -----------------------------------------------------------------------------
_C.DETECT = CN()

_C.DETECT.MODEL = 'yolov8'  # model path
_C.DETECT.WEIGHT = 'weights/yolov8l_crowdhuman-human-only_50e_1024.pt'
_C.DETECT.SZ = (768, 1280)  # input size
_C.DETECT.CONF = 0.4
_C.DETECT.CLASSES = (0, )   # None for all classes
_C.DETECT.METHOD = 'normal' # inference mode: normal
                            #                 grid
                            #                 silding_window   
_C.DETECT.NMS_THRESH = 0.65 # NMS threshold

# some other configs
_C.DETECT.DEVICES = 'cuda:0' # device
_C.DETECT.VERBOSE = False # whether to print the results


# used for grid inference mode
_C.DETECT.GRID = CN()
_C.DETECT.GRID.GRID_SIZE = 640 # int or tuple (W, H)

# used for sliding window mode 
_C.DETECT.SLIDING_WINDOW = CN()
_C.DETECT.SLIDING_WINDOW.WINDOW_SIZE = 640 # int or tuple
_C.DETECT.SLIDING_WINDOW.OVERLAP_WINDOW_SIZE_RATIO = 0.3 # float or tuple, must be in-range (0, 1)
_C.DETECT.SLIDING_WINDOW.NMS_THRESH = 0.65 # float, must be in-range (0, 1)


# -----------------------------------------------------------------------------
# ACTION DETECTION
# -----------------------------------------------------------------------------
_C.ACTION = CN()
_C.ACTION.MODEL = 'MAE'
_C.ACTION.clip_len = 64
_C.ACTION.frame_interval = 4
_C.ACTION.predict_stepsize = 8
_C.ACTION.device ='cuda:0'
_C.ACTION.checkpoint_slowfast = r'VideoAnalysis/action/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb_20230314-3dafab75.pth'
# /root/Hoang/Video_Analysis/Video-Analysis/VideoAnalysis/action/mmaction2/configs/detection/videomae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py
_C.ACTION.config_slowfast = r'VideoAnalysis/action/mmaction2/configs/detection/videomae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py'
# _C.ACTION.config_slowfast = r'VideoAnalysis/action/mmaction2/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py'
# -----------------------------------------------------------------------------
# TRACKING TASK CONFIG
# -----------------------------------------------------------------------------
_C.TRACK = CN()
_C.TRACK.MODEL = 'motrv2'
_C.TRACK.WEIGHT = r'VideoAnalysis/track/MOTR/motrv2_dancetrack.pth'
_C.TRACK.MODEL_CONFIG = r'.cache/configs/motrv2_dancetrack.yaml'
_C.TRACK.DEVICES = 'cuda:0'
 
# -----------------------------------------------------------------------------
# HMR TASK CONFIG
# -----------------------------------------------------------------------------


# /root/Hoang/Video-Analysis/VideoAnalysis/apis
# -----------------------------------------------------------------------------
# POSE TASK CONFIG
# -----------------------------------------------------------------------------
_C.POSE = CN()

_C.POSE.MODEL = 'mmpose'
_C.POSE.MODEL_CONFIG = 'VideoAnalysis/pose/mmpose/configs/body_2d_keypoint/rtmo/crowdpose/rtmo-l_16xb16-700e_body7-crowdpose-640x640.py'
_C.POSE.WEIGHT = 'weights/rtmo-l_16xb16-700e_body7-crowdpose-640x640-5bafdc11_20231219.pth'
_C.POSE.CONF = 0.3
_C.POSE.DEVICES = 'cuda:0'


# -----------------------------------------------------------------------------
# SYNOPSIS TASK CONFIG
# -----------------------------------------------------------------------------
_C.SYNOPSIS = CN()

_C.SYNOPSIS.START_FRAME = 0 # synopsis video starting frame index
_C.SYNOPSIS.MAX_LENGTH = 250 # maximum synopsis video length (in frames)
_C.SYNOPSIS.FPS = 30 # using to render synopsis video
                     # -1 if using source video FPS

_C.SYNOPSIS.OMEGA_A = 1 # activity weight
_C.SYNOPSIS.OMEGA_C = 100 # collisions weight
_C.SYNOPSIS.OMEGA_T = 1 # chronological weight
_C.SYNOPSIS.OMEGA_SD = 1 # smooth and deviation weight


# TUBE
_C.SYNOPSIS.TUBE = CN() # a tube is used to manage an object 

_C.SYNOPSIS.SAVENAME = 'default'

_C.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH = 48
_C.SYNOPSIS.TUBE.RATIO_SIZE_RANGE = [0.9,1.]
_C.SYNOPSIS.TUBE.RATIO_SPEED_RANGE = [0.3,2.]

_C.SYNOPSIS.TUBE.DEVIATION_SIGMA = 2. # emprically set as 2 
_C.SYNOPSIS.TUBE.DEVIATION_WEIGHT_V = 10000 # velocity weight in deviation term
_C.SYNOPSIS.TUBE.DEVIATION_WEIGHT_S = 10000 # size weight in deviation term
_C.SYNOPSIS.TUBE.SMOOTH_WEIGHT_V = 0.01 # velocity weight in smooth term
_C.SYNOPSIS.TUBE.SMOOTH_WEIGHT_S = 0.01 # size weight in smooth term


# Markov-chain Monte-Carlo
_C.SYNOPSIS.MCMC = CN()
_C.SYNOPSIS.MCMC.NUM_ITERATIONS = 300000 # maximum num iterations to compute MCMC


# -----------------------------------------------------------------------------
# COMMON CONFIG
# -----------------------------------------------------------------------------
_C.COMMON = CN()

_C.COMMON.MARKER_PATH = 'src/marker/video4' # path to store track logger information
                                            # using to generate sysnopsis information


