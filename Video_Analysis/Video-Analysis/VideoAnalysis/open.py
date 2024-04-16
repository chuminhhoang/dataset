import joblib
import numpy as np
import glob
import os
# os.remove('/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/data/ava_train/Ag-pXiLrd48_14370_125_165.pkl')
# Ag-pXiLrd48

# data = joblib.load('/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/data/ava_val/-OyDO1g74vc_90_0_165.pkl')
# # for i in data['has_gt']:
# #     print(i[0][0])
# #     if i[0][0] == 1:
# #         print(i)
# # print(data['appearance_dict'][0].keys())
# data = joblib.load('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/validate_data/G5Yr20A5z_Q/G5Yr20A5z_Q_360_10_165.pkl')
# print(data['frame_bbox'])
a = os.listdir('/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/data/ava_train')
count = 0
for name in a:
    count+=1
    if count < 90000:
        continue
    data = joblib.load('/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/data/ava_train/{}'.format(name))
    ava_pseudo_labels = data['action_label_psudo'][0]
    ava_pseudo_labels_ = np.zeros((132-0, 1, 81))
    appe_idx = data['appearance_index'][0][0:132]
    # ava_pseudo_vectors_ = np.zeros((132-0, 1,768))
    for i in range(len(appe_idx)):
            # Todo: half start and half end
            temp = np.array(ava_pseudo_labels[int(appe_idx[i][0][0])])
            if len(np.array(ava_pseudo_labels[int(appe_idx[i][0][0])]).shape) == 4:
                temp = temp.squeeze(0)
                ava_pseudo_labels_[i, 0, :] = temp[0, 0, :]
            else:
                ava_pseudo_labels_[i, 0, :] = np.array(ava_pseudo_labels[int(appe_idx[i][0][0])])[0, 0, :]
#     print(np.array(data['frame_bbox']).shape)
# #     if 'Ag-pXiLrd48' in name:
# #         print(name)
#         # os.remove('/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/data/ava_train/{}'.format(name))
# print(np.array(data['has_gt'][0]).shape)
# print(np.array(data['has_detection'][0]).shape)
# print(np.array(data['pose_shape'][0]).shape)
# print(np.array(data['action_label_psudo'][0][1])[0, 0, :].shape)
# print(np.array(data['appearance_dict'][0][1])[0, 0, :].shape)
# print(data['frame_bbox'])
# print(np.array(data['frame_size']).shape)
# print(np.array(data['frame_name']).shape)
# print(np.array(data['has_gt'][0]).shape)
# print(np.array(data['fid'][0]).shape)
# print(np.array(data['frame_conf']).shape)
# print(np.array(data['appearance_index'][0]).shape)
# print(np.array(data['action_label_gt'][0]).shape)
# # print('-----------------------------------------')
# data1 = joblib.load('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/train_data/Db19rWN5BGo/Db19rWN5BGo_150_22_165.pkl')
# print(np.array(data1['has_gt'][0]).shape)
# print(np.array(data1['has_detection'][0]).shape)
# print(np.array(data1['pose_shape'][0]).shape)
# print(np.array(data1['action_label_psudo'][0][1])[0, 0, :].shape)
# print(np.array(data1['appearance_dict'][0][1])[0, 0, :].shape)
# print(np.array(data1['frame_size']).shape)
# print(np.array(data1['frame_name']).shape)
# print(np.array(data1['has_gt'][0]).shape)
# print(np.array(data1['fid'][0]).shape)
# print(np.array(data1['frame_conf']).shape)
# print(np.array(data1['appearance_index'][0]).shape)
# print(np.array(data1['action_label_gt'][0]).shape)
# # train_data = os.listdir('/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/data/ava_train')
# # print(train_data[10264])
# # data = joblib.load('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/pkl_data/Db19rWN5BGo/Db19rWN5BGo_90_5_165.pkl')
# # print(np.array(data['action_label_psudo'][0][21]).shape)
# count = 0
# for i in train_data:
#     if 'Db19rWN5BGo' in i:
#         os.remove('/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/data/ava_train/{}'.format(i))
#         print(i)
# count = 0
# for i in train_data:
#     try:
#         data = joblib.load('/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/data/ava_train/{}'.format(i))
#     except:
#         os.remove('/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/data/ava_train/{}'.format(j))
#         continue
#     print(np.array(data['action_label_psudo'][0][21])[0, 0, :].shape)
#     count+=1
#     print(count)