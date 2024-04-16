import os
import cv2
import glob    
import mmcv
import torch
import joblib
import logging
import numpy as np
import pandas as pd
from configs import default
from collections import defaultdict
logger = logging.getLogger(__name__)
from apis.video_analyzer import VideoAnalyzer

# 132 // 6 =22
def appearance_index(person_id):
    appearance_index_list = defaultdict(list)
    for i in person_id:
        appearance_index_list[str(i)] = np.zeros([132, 1, 1])
        for f_id in range(0, 132):
            appearance_index_list[str(i)][f_id] = f_id//6
    return appearance_index_list

def action_label_gt(df, person_id, start_frame):
    action_label_gt = defaultdict(list)
    for i_p in person_id:
        action_label_gt[str(i_p)] = np.zeros((165, 1, 81))  
    for f_ in range(start_frame, start_frame+165):
        new_df = df.loc[(df['frame'] == f_) & (df['label'] != 'Nan')]  
        if len(new_df) > 0:
            id_each_frame = new_df['person_id'].unique()
            for i_e in id_each_frame:
                temp_df = new_df.loc[new_df['person_id'] == i_e]
                list_label = temp_df['label'].unique().astype(int)
                action_label_gt[i_e][int(f_-start_frame), :, list_label] = 1
    return action_label_gt

# Todo extract_appearance
def extract_appearance(list_frame_data, df, id_person, start_frame, video_analyze):
    extract_appearance_list = defaultdict(list)
    extract_psudo_list = defaultdict(list)
    has_gt_list = defaultdict(list)
    
    for i_p in id_person:
        has_gt_list[str(i_p)] = np.zeros([165, 1 ,1])
        extract_psudo_list[str(i_p)] = np.zeros([23, 81])
        extract_appearance_list[str(i_p)] = np.zeros([23, 768])
        
    idx = -31
    appear_index = []
    list_middle_frame = []
    video_analyze.task.frames = []
    video_analyze.buffer_size = 58
    video_analyze.task.processed_frames = []
    while True:
        if idx > len(list_frame_data) - 32: 
            break
        frame = list_frame_data[idx + 31]
        w, h =  frame.shape[:2]
        video_analyze.task.frames.append(mmcv.imresize(frame, (w, h)))
        stdet_input_size = mmcv.rescale_size((w, h), (256, np.Inf))
        video_analyze.task.ratio = tuple(n / o for n, o in zip(stdet_input_size, (w, h)))
        processed_frame= mmcv.imresize(frame, stdet_input_size).astype(np.float32)
        processed_frame = mmcv.imnormalize_(processed_frame, **video_analyze.img_norm_cfg)
        video_analyze.task.processed_frames.append(processed_frame)
        
        if idx % 6 == 0 and idx <= 132 and idx >= 0:
            new_df = df.loc[(df['frame']== idx+start_frame) & (df['person_id']!='Nan')]
            bbox = np.array([[-1, -1, -1, -1, -1]])
            if len(new_df) > 0:
                    appear_index.append(idx//6)
                    new_df = new_df.drop_duplicates(subset = ['person_id'])
                    bbox = np.array(new_df[['x1', 'y1', 'x2', 'y2', 'person_id']])
                    list_person_id = np.array(new_df['person_id'])
                    for id_person in list_person_id:
                        if idx == 132: 
                            has_gt_list[str(id_person)][idx] = np.array(1)
                        elif idx % 30 != 0:
                            has_gt_list[str(id_person)][idx: idx+6] = np.ones([6, 1, 1])
                        else: 
                            has_gt_list[str(id_person)][idx] = 2
                            has_gt_list[str(id_person)][idx+1: idx+6] = np.ones([5, 1, 1])                                        
            list_middle_frame.append(bbox)
                
        if  len(video_analyze.task.processed_frames) == video_analyze.clip_len:
            video_analyze.task.add_frames(idx, video_analyze.task.frames, video_analyze.task.processed_frames)
            temp_value = list_middle_frame.pop(0)
            bbox = temp_value[:, :4]
            list_id = temp_value[:, 4]
            if not np.all(bbox[0]==-1):
                mid_bbox = bbox.tolist()
                video_analyze.task.add_bboxes(torch.Tensor(mid_bbox).to('cuda:0'))
                with torch.no_grad():
                    a = video_analyze.action_model(**video_analyze.task.get_model_inputs('cuda:0'))
                temp_count = appear_index.pop(0)
                for index, id_person in enumerate(list_id):
                    extract_appearance_list[str(id_person)][int(temp_count)] = a[1][index].cpu().numpy()
                    extract_psudo_list[str(id_person)][int(temp_count)] = a[0][0].pred_instances.scores[index].cpu().numpy()
                    
            video_analyze.task.processed_frames = video_analyze.task.processed_frames[-video_analyze.buffer_size:]
            video_analyze.task.frames = video_analyze.task.frames[-video_analyze.buffer_size:] 
            video_analyze.test_frame = video_analyze.test_frame[-video_analyze.buffer_size:]
        idx+=1
    return extract_appearance_list, extract_psudo_list, has_gt_list

def calculate_iou(bbox1, bbox2):
    bbox1[::2] = bbox1[::2] + bbox2[0]
    bbox1[1::2] = bbox1[1::2] + bbox2[1]
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])
    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / float(union_area)
    return iou

# get frame data
def gen_batch(path, batch = 196, start_frames=60):
    
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(0, length):
        ret, frame = cap.read()
        if i <= start_frames - 32: 
            continue
        frames.append(frame)
        if len(frames) == batch:
            yield frames, frames[-batch+32:]
            frames = frames[-batch+30:]
    cap.release()
    cv2.destroyAllWindows()
    
# Todo extract pose
def extract_pose(list_frame_data, df, person_id, start_frame, video_analyze):
    
    extract_pose_list = defaultdict(list)
    has_detection_list = defaultdict(list)
    for i_p in list_person_id:
        extract_pose_list[str(i_p)] = np.zeros([165, 1, 28])
        has_detection_list[str(i_p)] = np.zeros([165, 1 ,1])
    idx = 0
    bboxes = []
    while True:
        try:
            frame = list_frame_data[idx]
        except IndexError:
            break
        new_df = df.loc[(df['frame'] == idx+start_frame) & (df['person_id']!='Nan')]
        if len(new_df) > 0:
            new_df = new_df.drop_duplicates(subset = ['person_id'])
            bboxes = np.array(new_df[['x1', 'y1', 'x2', 'y2', 'person_id']])
            p_id = bboxes[:, 4]
            for t, id in enumerate(p_id):
                bbox = bboxes[int(t), :4]
                for index_bbox in range(len(bbox)):
                    if bbox[index_bbox] < 0: 
                        bbox[index_bbox] = 0
                frame_cropped = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                bbox_each_pose, pose = video_analyze.get_pose_ver_training(frame_cropped)
                pose = torch.tensor(pose)
                if len(pose) == 1:
                    has_detection_list[str(id)][idx] = 1
                    extract_pose_list[str(id)][idx] = pose.flatten(1)
                elif len(pose) > 1: 
                    max_iou = 0
                    result = 0
                    for b, p in zip(bbox_each_pose, pose):
                        iou = calculate_iou(b, bbox)
                        if iou > max_iou:
                            result = p
                            max_iou = iou
                    has_detection_list[str(id)][idx] = 1
                    extract_pose_list[str(id)][idx] = result[None, ...].flatten(1)
        idx+=1
    return extract_pose_list, has_detection_list

# Todo info dataset (frame_name, frame_size, frame_bbox, frame_conf)
# name: /datasets01/AVA/080720/frames/-5KQ66BBWC4/-5KQ66BBWC4_000002.jpg
def init_meta(df, height, width, person_id, start_frame, name):
    
    frame_name_list = defaultdict(list)
    frame_size_list = defaultdict(list)
    frame_id_list = defaultdict(list)
    name_vid = name
    for id in person_id:
        name_frame = []
        for frame in range(start_frame, start_frame+165):
             name_frame.append(name_vid+'_'+str(frame))
        frame_name_list[str(id)] = name_frame
        frame_size_list[str(id)] = np.full([165, 2], [height, width])
        frame_id_list[str(id)] = np.arange(start_frame, start_frame+165)[:, None, None]
    return frame_name_list, frame_size_list, frame_id_list

# bbox/score for validate
def xyxy_to_xywh(xyxy):
    """
    Convert XYXY format (x,y top left and x,y bottom right) to XYWH format (x,y top left point and width, height).
    :param xyxy: [X1, Y1, X2, Y2]
    :return: [X, Y, W, H]
    """
    if np.array(xyxy).ndim > 1 or len(xyxy) > 4:
        raise ValueError('xyxy format: [x1, y1, x2, y2]')
    w_temp = abs(xyxy[0] - xyxy[2])
    h_temp = abs(xyxy[1] - xyxy[3])
    return np.array([int(xyxy[0]), int(xyxy[1]), int(w_temp), int(h_temp)])

# convert to dictionary
def convert_into_dict_index(extract_appearance_list, extract_psudo_list):
    dict_extract_appearance =  defaultdict(list)
    dict_extract_psudo = defaultdict(list)
    for idx in range(0, 22):
        dict_extract_appearance[idx].append([extract_appearance_list[idx]])
        dict_extract_psudo[idx].append([extract_psudo_list[idx]])
    return dict_extract_appearance, dict_extract_psudo

# check full bbox
def extracting_bbox(df, start_frame, person_id):

    new_df = df.loc[(df['frame'] == start_frame)]
    bbox_dict_list = defaultdict(list)
    score_dict_list = defaultdict(list)
    
    for i_p in person_id:
        bbox_dict_list[str(i_p)] = np.full([165, 4], -1).tolist()
        for i in range(len(bbox_dict_list[str(i_p)])):
            bbox_dict_list[str(i_p)][i] = np.array([-1, -1, -1, -1])
        score_dict_list[str(i_p)] = np.full([165, 1], 0)
        
    for f_ in range(start_frame, start_frame+165):
        df_bbox = df.loc[(df['frame']==f_)]
        df_bbox = df_bbox.drop_duplicates(subset=['person_id'])
        if len(df_bbox) > 0:
            array_bbox = np.array(df_bbox[['x1', 'y1', 'x2', 'y2', 'person_id']])
            id_each_frame = array_bbox[:, 4]
            scores = np.ones([len(array_bbox)])
            for index, i_f in enumerate(id_each_frame): 
                bbox_dict_list[str(i_f)][f_-start_frame] = xyxy_to_xywh(array_bbox[index][:4])
                score_dict_list[str(i_f)][f_-start_frame] = scores[index]
    return bbox_dict_list, score_dict_list
        
        
if __name__ == "__main__":
    frame_length = 165
    cfg = default._C.clone()
    video_analyze = VideoAnalyzer(cfg) 
    root = 'VideoAnalysis'
    train_data_folder = 'VideoAnalysis/train_data'
    folder_video_15min = '/home/mq/data_disk2T/Hoang/AVA_Kinetics/videos_15min'
    videos_15min = os.listdir(folder_video_15min)
    pkl_data = os.listdir('{}/train_data'.format(root))
    for video_data in videos_15min:
        start_frame = 60
        list_frame_data = []
        video_name = video_data.split('.')[0]
        
        if not os.path.exists('{}/csv_data/output_{}.csv'.format(root, video_name)):
            continue
        csv_df = pd.read_csv('{}/csv_data/output_{}.csv'.format(root, video_name))
        logger.info(video_name)
        
        if video_name in pkl_data:
            data = [int(f_.split('_')[-3]) for f_ in os.listdir(os.path.join('{}/train_data'.format(root), video_name))]
            start_frame = max(data) + 30
        list_frame_data_batch = gen_batch(path='{}/{}'.format(folder_video_15min, video_data), batch=196, start_frames=60)
        check_start = 60
        
        for list_frame_data_middle_frame, list_frame_data_pose in list_frame_data_batch:
            # logger.info(start_frame)
            if start_frame + frame_length > int(csv_df['frame'].iloc[-1]): break
            
            if len(csv_df.loc[(csv_df['frame'] == start_frame)]) == 0: 
                start_frame += 30
                # check_start += 30
                # continue
            
            if check_start < start_frame:
                check_start += 30
                continue
            
            h, w = list_frame_data_middle_frame[0].shape[:2]
            selected_df = csv_df.loc[(csv_df['frame'] >= start_frame) & (csv_df['frame'] < start_frame + frame_length) & (csv_df['person_id']!='Nan')]
            list_person_id = selected_df['person_id'].unique()
            
            extract_appearance_list, extract_psudo_list, has_gt_list = extract_appearance(list_frame_data = list_frame_data_middle_frame, df = selected_df, id_person = list_person_id, \
                                                                                            start_frame = start_frame,video_analyze = video_analyze)
            
            extract_pose_list, has_detection_list = extract_pose(list_frame_data = list_frame_data_pose, df = selected_df, person_id = list_person_id, start_frame = start_frame, \
                                                                                            video_analyze = video_analyze)
            
            frame_name_list, frame_size_list, frame_id_list = init_meta(df = selected_df, height = h, width = w, person_id = list_person_id, start_frame = start_frame, \
                                                                                            name = video_name)
            # bbox/score for validate
            bbox_dict_list, score_dict_list =  extracting_bbox(df = selected_df, start_frame = start_frame, person_id = list_person_id)
            
            # appearance_index
            appearance_index_list = appearance_index(person_id = list_person_id)
            
            # action_label_gt
            action_label_gt_list = action_label_gt(df = selected_df, person_id = list_person_id, start_frame = start_frame)
            # fill pkl data
            
            logger.info(start_frame)
            for i_p in list_person_id:
                data = defaultdict(list)
                dict_extract_appearance, dict_extract_psudo = convert_into_dict_index(extract_appearance_list[str(i_p)], extract_psudo_list[str(i_p)])
                data['fid'].append(frame_id_list[str(i_p)])
                data['has_gt'].append(has_gt_list[str(i_p)])
                data['frame_bbox'].append(bbox_dict_list[str(i_p)])
                data['frame_size'].append(frame_size_list[str(i_p)])
                data['frame_conf'].append(score_dict_list[str(i_p)])
                data['frame_name'].append(frame_name_list[str(i_p)])
                data['action_label_psudo'].append(dict_extract_psudo)
                data['pose_shape'].append(extract_pose_list[str(i_p)])
                data['appearance_dict'].append(dict_extract_appearance)
                data['has_detection'].append(has_detection_list[str(i_p)])
                data['action_label_gt'].append(action_label_gt_list[str(i_p)])
                data['appearance_index'].append(appearance_index_list[str(i_p)])
                if not os.path.exists('{}/{}'.format(train_data_folder, video_name)):
                    os.makedirs('{}/{}'.format(train_data_folder, video_name))
                output = '{}/{}/{}_{}_{}_165.pkl'.format(train_data_folder, video_name, video_name, start_frame, i_p)
                joblib.dump(data, output)
            check_start += 30
            start_frame += 30
                      