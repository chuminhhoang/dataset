import joblib
import numpy as np
import os
import pandas as pd
from collections import defaultdict

def statisttic(df_by_video, statistic_percent):
    label = df_by_video['label'].unique().astype(int)
    all_label = np.array(df_by_video['label']).astype(int).tolist()
    for i in label:
        count_label = all_label.count(i)
        statistic_percent[i] += count_label

DATA_DIR="/home/mq/data_disk2T/Hoang/AVA_Kinetics/video"
if __name__ == '__main__':
    train_folder = os.listdir('/home/mq/data_disk2T/Hoang/AVA_Kinetics/video')
    test = [i.split('.')[0] for i in train_folder]
    # df = pd.read_csv('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/VideoAnalysis/ava_annotation.csv', dtype={'Name':'string','frame':'string','x1':'string','y1':'string','x2':'string','y2':'string','label':'string','person_id':'string'})
    anno_folder = open('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/anno/ava_file_names_trainval_v2.1.txt', 'r')
    new_file = open('/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/anno/new.txt', 'w')
    while True:
    
        # Get next line from file
        line = anno_folder.readline()
        # if line is empty
        # end of file is reached
        
        if not line:
            break
        if line.split('.')[0] not in test:
            print(line)
            new_file.write('{}\n'.format(line.strip()))
 
    new_file.close()
    anno_folder.close()    
    # for i in df['Name'].unique():
    #     if i not in test:
            
    # statistic_percent = defaultdict(int)
    # for video in train_folder:
    #     print(video)
    #     df_by_video = df.loc[(df['Name']==video.split('.')[0])]
    #     statisttic(df_by_video, statistic_percent)
    # all_sum_label = 0
    # percent_each_label = defaultdict(float)
    # for k in statistic_percent.keys():
    #     all_sum_label += int(statistic_percent[k])
    # print(statistic_percent)
    # for n_k in statistic_percent.keys():
    #     percent_each_label[n_k] = (statistic_percent[n_k]/all_sum_label)
    # print(percent_each_label)