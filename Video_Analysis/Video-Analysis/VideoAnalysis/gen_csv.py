import sys
import cv2
import glob
import torch
import numpy as np
import pandas as pd
from configs import default
from init_pkl import extracting_pkl
from collections import defaultdict
from apis.video_analyzer import VideoAnalyzer

def extract_data(path, loading_index, header):
    name_selected = path.split('/')[-1].split('.')[0]
    check_gt = list_gt.loc[(list_gt['Name']==name_selected)]
    idx = 0
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dict_mapper = defaultdict(list)
    label = defaultdict(list)
    check_id = 0
    header = header
    while True:
        ret, frame = cap.read()
        if ret is not True:
            break
        tracklet = []
        if check_id + 29 >= idx: 
            idx+=1
            continue
        else: 
            check_id = 0
        if idx<=loading_index: 
            idx+=1
            continue
        if (idx-60)%30==0 and idx>=60:
            v.clear()
            frame_idx = ((idx-60)//30)+902
            pre_df_frame = check_gt.loc[(check_gt['frame']==frame_idx)]
            if len(pre_df_frame) == 0:
                check_id = idx
                idx+=1
                continue
            # save_df_frame = df_frame
            df_frame = pre_df_frame.drop_duplicates(subset=['person_id'])
            arr = df_frame.to_numpy()
            indx_map = arr[:, 7]
            a = arr[:, 2:6]
            a[:, 1::2] = a[:, 1::2]*h
            a[:, ::2] = a[:, ::2]*w
            pre_df_frame['x1'] = pre_df_frame['x1']*w
            pre_df_frame['y1'] = pre_df_frame['y1']*h
            pre_df_frame['x2'] = pre_df_frame['x2']*w
            pre_df_frame['y2'] = pre_df_frame['y2']*h
            b = np.concatenate((a, indx_map[None, :].T), 1)
            x_arr, y_arr =a.shape
            ones = np.ones((x_arr, 2))
            a = np.concatenate((a, ones), 1)
            video_analyze.reset()    
            tracklet = video_analyze.get_track(frame, a)
            for i in range(len(tracklet)):
                dict_mapper[tracklet[i, 5]] = int(indx_map[i]) 
            pre_df_frame['frame'] = str(idx)
            pre_df_frame['person_id'] = pre_df_frame['person_id'].astype(int)
            pre_df_frame.to_csv('VideoAnalysis/csv_data/output_{}.csv'.format(name_selected), mode='a', index=False, header=header)
            header = False
        elif idx%1 ==0 and idx > 60:
            bbox = video_analyze.get_bbox(frame)
            tracklet = video_analyze.get_track(frame, bbox[0])
            count = 0
            if len(tracklet) == 0:
                temp_arr = np.array([name_selected, str(idx), -1, -1, -1, -1,'Nan', 'Nan'])
                temp = pd.DataFrame(temp_arr.reshape(1, -1), columns=['Name', 'frame', 'x1', 'y1', 'x2', 'y2', 'label', 'person_id'])
                temp.to_csv('VideoAnalysis/csv_data/output_{}.csv'.format(name_selected), mode='a', index=False, header=False)
            for t in tracklet:
                if dict_mapper[t[5]] != []:
                    track_id = dict_mapper[t[5]]
                else:
                    track_id = t[5]
                temp_arr = np.array([name_selected, str(idx), t[0], t[1], t[2], t[3],'Nan', int(track_id)])
                temp = pd.DataFrame(temp_arr.reshape(1, -1), columns=['Name', 'frame', 'x1', 'y1', 'x2', 'y2', 'label', 'person_id'])
                count+=1
                temp.to_csv('VideoAnalysis/csv_data/output_{}.csv'.format(name_selected), mode='a', index=False, header=header)
                header = False
        idx+=1

if __name__ == '__main__':
    cfg = default._C.clone()
    list_gt = pd.read_csv('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/ava_annotation.csv')
    video_analyze = VideoAnalyzer(cfg)
    ex_list = []
    for path_existed in glob.glob('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/csv_data/*'):  
        path_existed = path_existed.split('/')[-1].split('.')[0]
        ex_list.append(path_existed)
    for path in glob.glob('/home/mq/data_disk2T/Hoang/AVA_Kinetics/videos_15min/*'):
        name = path.split('/')[-1].split('.')[0]
        file_exist = len(list_gt.loc[(list_gt['Name']==name)])
        if file_exist > 0:       
            check = False
            for exist_name in ex_list:
                if name in exist_name:
                    check = True
                    print('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/csv_data/{}.csv'.format(exist_name))
                    df = pd.read_csv('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/csv_data/{}.csv'.format(exist_name))
                    last_index = df['frame'].to_numpy()[-1]
                    print(f'{name} is processed')
                    if int(last_index) >= 27000:
                        continue
                    else: 
                        extract_data(path, last_index, False)
                        extracting_pkl(path,'/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/csv_data/{}.csv'.format(exist_name), name)
            if check==False:
                extract_data(path, 0, True)
                extracting_pkl(path,'/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/csv_data/output_{}.csv'.format(name), name)

    
        
    