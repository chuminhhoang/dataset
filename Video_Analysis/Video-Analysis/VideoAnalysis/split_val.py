import os
import shutil
import random
import logging
import pandas as pd

def split_val():
    check_exit = 200
    root = 'VideoAnalysis/train_data'
    csv_path = 'VideoAnalysis/csv_data'
    video_ava_selected = 'BY3sZmvUp-0'
    df = pd.read_csv('{}/output_{}.csv'.format(csv_path, video_ava_selected))
    while check_exit > 0:
        start_frame = random.choice(range(60, 25000, 30))
        id_person_list = df.loc[(df['frame'] == start_frame)]['person_id'].tolist()
        for i_p in id_person_list:
            shutil.copy('{}/{}/{}_{}_{}_165.pkl'.format(root, video_ava_selected, video_ava_selected, start_frame, i_p), '/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/data/ava_test')
        check_exit -= 1

def split_train():
    train_data = os.listdir('VideoAnalysis/train_data')
    ava_val = os.listdir('/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/data/ava_val')
    for name_video in train_data:
        for train_pkl in os.listdir('VideoAnalysis/train_data/{}'.format(name_video)):
            if train_pkl not in ava_val:
                shutil.copy('VideoAnalysis/train_data/{}/{}'.format(name_video, train_pkl), '/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/data/ava_train')

if __name__ == '__main__':
    split_train()