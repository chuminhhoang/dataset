import joblib
import numpy as np
import os
import pandas as pd
import shutil
from init_pkl import extracting_bbox_scores
check_exist = os.listdir('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/train_data')
print(check_exist)
for i in os.listdir('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/data_fix'):
    if i in check_exist:
        print(i)
        continue
    df = pd.read_csv('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/csv_data/output_{}.csv'.format(i))
    extracting_bbox_scores(df=df, frame_name=i, check = 0)
# from VideoAnalysis.init_pkl import action_label_gt
# data = joblib.load('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/data_fix/9IF8uTRrWAM/9IF8uTRrWAM_120_0_165.pkl')
# print(data['frame_bbox'])

# print(np.array(data['action_label_psudo'][0]))
# print('------------------------')
# data1 = joblib.load('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/validate_data/GozLjpMNADg/GozLjpMNADg_26700_1009_165.pkl')
# print(np.array(data1['action_label_psudo'][0][0]).shape)
# data2 = joblib.load('/home/mq/data_disk2T/Hoang/AVA_Kinetics/data/ava_train/ava-train_-5KQ66BBWC4_000930_1_128.pkl')
# print(data2['frame_bbox'])
# data1['frame_bbox'][0][0][::2] = data1['frame_bbox'][0][0][::2]/1920
# data1['frame_bbox'][0][0][1::2]= data1['frame_bbox'][0][0][1::2]/1080 
# print(data1['frame_bbox'][0][0])
# print(data1['frame_bbox'][0][0])
# print(data1['frame_size'][0][0])
# a = 231.12/1080
# print(a)
# 0.0,0.214,0.214,0.984,14,0
# , 0, 0, 0, 0, 0, 0]])]})
#   0,           0,           0])]})
# list_video = os.listdir('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/train_data/Ag-pXiLrd48')
# count = 0
# for i in list_video:
#     a = joblib.load('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/train_data/Ag-pXiLrd48/{}'.format(i))
#     if np.array(a['appearance_index'])[0,:,0,0][-1]==21:
#         shutil.move('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/train_data/Ag-pXiLrd48/{}'.format(i), '/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/data/ava_val')
# data = joblib.load('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/pkl_data/Ag-pXiLrd48/Ag-pXiLrd48_4980_34_165.pkl')
# # # list_map = [4, 5, 7, 8, 10, 11, 12, 14, 20, 24, 34, 52, 64, 80]
# for idx,  x in enumerate(np.array(data['has_gt'][0][:, 0, 0])):
#     if x >=1:
#         # print(x)
#         # print(idx)
#         # print(data['has_gt'][0][idx])
#         print(data['action_label_psudo'][0][29//6])
        # print(idx)
# print(data['has_gt'])
# count = 0
# print(len(os.listdir('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/pkl_data/Ag-pXiLrd48')))
# for i in os.listdir('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/pkl_data/Ag-pXiLrd48'):
#         count+=1
#         if i == 'GozLjpMNADg_17490_2627_165.pkl':
#             print(count)
#             break
# # from collections import defaultdict
# a = defaultdict(list)
# n = [1, 2, 3]
# a[n] = 0
# print(a)
# print(data['action_label_psudo'][0].keys())
# print(data.keys())
# print(data['appearance_index'])
# print(np.array(data1['frame_bbox']))
# # # idx = 0
# print(data.keys())
# print(data['action_label_psudo'])
# # print(np.array(data['has_gt'][0][:, 0, 0]))
# # print(data['action_label_gt'][0][:, 0, ])
# for i in range(165):
#     if data['has_gt'][0][:, 0, 0][i] ==2:
#         print(i)
#         print(data['action_label_gt'][0][:, 0, :][i])
# a = np.array([-1, -1, -1, -1, -1])
# print(np.all(a==-1)) 
# # print(np.array(data['has_gt'])[0, :, 0, 0][150])
# for i in range(len(np.array(data['has_gt'])[0, :, 0, 0])):
#     # print(np.array(data['has_gt'])[0, :, 0, 0][i])
#     if np.array(data['has_gt'])[0, :, 0, 0][i] >0:
#          print(data['action_label_psudo'])
    # idx+=1
# print(data['action_label_psudo'])
# a = np.array([4, 5, 7, 8, 10, 11, 12, 14, 20, 24, 34, 52, 64, 80])
# output = open('/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/stuffs/ava_valid_classes.npy', 'wb')
# np.save(output, a)
# print(type(a))
# del a
# data.clear()
# for idx, x in enumerate(list_map):
#     data[idx] = x
# output = open('/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/stuffs/ava_class_mapping.pkl', 'wb')
# joblib.dump(data, output)
