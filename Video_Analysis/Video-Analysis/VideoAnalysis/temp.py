import pandas as pd
import joblib
import glob
import os
import numpy as np
# df = pd.read_csv('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/csv_data/output_G5Yr20A5z_Q.csv')
# print(df['frame'].iloc[-1])
data = joblib.load('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/train_data/Db19rWN5BGo/Db19rWN5BGo_330_33_165.pkl')
print(np.array(data['action_label_gt'][0].shape))
print(np.array(data['has_gt'][0]).shape)
print(np.array(data['has_detection'][0]).shape)
print(np.array(data['pose_shape'][0]).shape)
print(np.array(data['action_label_psudo'][0][1][0]).shape)
print(np.array(data['appearance_dict'][0][1][0]).shape)
print(np.array(data['frame_bbox']).shape)
# print(data['appearance_index'])
print(np.array(data['frame_size']).shape)
print(np.array(data['frame_name']).shape)
print(np.array(data['has_gt'][0]).shape)
print(np.array(data['fid'][0]).shape)
# print('-------------------------------')
# a = np.full([2, 165, 2, 2, 1, 1], 1)
# a = np.squeeze(a, axis=-1)
# a = np.squeeze(a, axis=-1)
# print(a.shape)
# count = 0
# for i in glob.glob('/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/data/ava_val/*'):
#     # if count < 149451:
#     #     count += 1
#     #     continue
#     try:
#         data = joblib.load(i)
#     except:
#         os.remove(i)
#         continue
#     count += 1
#     # if count > 200:
#     #     break
#     # print(np.array(data['has_gt'][0]).shape)
#     # print(np.array(data['has_detection'][0]).shape)
#     # print(np.array(data['pose_shape'][0]).shape)
#     # print(np.array(data['action_label_psudo'][0][1][0]).shape)
#     # print(np.array(data['appearance_dict'][0][1][0]).shape)
#     # print(np.array(data['frame_bbox']).shape)
#     # print(np.array(data['frame_size']).shape)
#     # print(np.array(data['frame_name']).shape)
#     # print(np.array(data['has_gt'][0]).shape)
#     # print(type(data['frame_bbox']))
#     if isinstance(data['frame_bbox'][0], list):
#         # print(data['frame_bbox'])
#         pass
#     else:
#         arr_bbox = []
#         for k in data['frame_bbox'][0]:
#             arr_bbox.append(np.array(k))
#         data['frame_bbox'] = [arr_bbox]
#         with open(i, 'wb') as f:
#             joblib.dump(data, f)
        
#         # bbox = np.array(data['frame_bbox'])[0][None, ...]
#         # del data['frame_bbox']
#         # data['frame_bbox'] = bbox.tolist()
#     # if np.array(data['frame_bbox']).shape[0] > 1:
#     #     bbox = np.array(data['frame_bbox'])[0][None, ...]
#     #     del data['frame_bbox']
#     #     data['frame_bbox'] = bbox.tolist()
#     #     print(bbox)
#     #     # print(np.array(data['frame_bbox']).shape)
#     #     with open(i, 'wb') as f:
#     #         joblib.dump(data, f)
#     # if len(np.array(data['appearance_index'][0]).shape) == 1:
#     #     del data['appearance_index']
#     #     appearance_index_list = np.full([132, 1, 1], 0)
#     #     for f_id in range(0, 132):
#     #         appearance_index_list[f_id] = f_id//6
#     #     index = appearance_index_list.tolist()
#     #     data['appearance_index'].append(index)  
#     #     conf = np.array(data['frame_conf'])
#     #     conf = np.squeeze(conf, axis=-1)
#     #     conf = conf.tolist()
#     #     del data['frame_conf'] 
#     #     data['frame_conf'] = conf
#     #     # exit()
#     #     with open(i, 'wb') as f:
#     #         joblib.dump(data, f)
#     #     with open(i, 'wb') as f:
#     #         joblib.dump(data, f)
#     # if len(np.array(data['fid']).shape) == 2:
#     #     a = np.array(data['fid'])[:, :, None, None]
#     #     del data['fid']
#     #     data['fid'] = a
#     #     with open(i, 'wb') as f:
#     #         joblib.dump(data, f)
#     # if len(np.array(data['fid']).shape) == 6:
#     #     a = np.array(data['fid'])
#     #     a = np.squeeze(a, axis=-1)
#     #     a = np.squeeze(a, axis=-1)
#     #     del data['fid']
#     #     data['fid'] = a
#     #     with open(i, 'wb') as f:
#     #         joblib.dump(data, f)
#     print(count)
#     # print(np.array(data['frame_conf']).shape)
#     # print(np.array(data['appearance_index'][0]))
print('-----------------------------------------')
data1 = joblib.load('/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/data/ava_val/-OyDO1g74vc_90_0_165.pkl')
print(np.array(data1['action_label_gt'][0].shape))
print(np.array(data1['has_gt'][0]).shape)
print(np.array(data1['has_detection'][0]).shape)
print(np.array(data1['pose_shape'][0]).shape)
print(np.array(data1['action_label_psudo'][0][1][0]).shape)
print(np.array(data1['appearance_dict'][0][1][0]).shape)
print(np.array(data1['frame_bbox']).shape)
# print(data['appearance_index'])
print(np.array(data1['frame_size']).shape)
print(np.array(data1['frame_name']).shape)
print(np.array(data1['has_gt'][0]).shape)
print(np.array(data1['fid'][0]).shape)
# for idx, i in enumerate(data['has_gt'][0][:, 0, 0]):
#     if i == 2:
#         print(data['action_label_gt'][0][idx])
# print(data['has_gt'])
# print(data['has_gt'])
# for idx,  x in enumerate(np.array(data['has_gt'][0][:, 0, 0])):
#     if x >=1:
# #         print(idx)
#         print(data['action_label_psudo'][0][idx//6])
#         print(idx)
# for idx, x in enumerate(np.array(data['has_detection'][0][:, 0, 0])):
#     if x == 1:
#         print(data['pose_shape'][0][int(idx)])