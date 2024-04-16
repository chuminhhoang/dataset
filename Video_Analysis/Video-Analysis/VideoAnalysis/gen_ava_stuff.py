import os
import pandas as pd

def gen_val():
    csv_path = 'VideoAnalysis/csv_data'
    ava_val = os.listdir('/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/data/ava_val')
    df = pd.read_csv('VideoAnalysis/ava_annotation.csv', dtype = {'Name': str, 'frame': str, 'x1': str, 'y1': str, \
                                                                 'x2': str, 'y2': str, 'label': str, 'person_id': str})
    ava_val.sort()
    
    for val_pkl in ava_val:
        person_id = val_pkl.split('_')[2]
        name_video = val_pkl.split('_')[0]
        frame_start = int(int(val_pkl.split('_')[1])/30+900)
        selected_df = df.loc[(df['frame'] == str(frame_start)) & (df['Name'] == name_video) & (df['person_id'] == person_id)]
        selected_df.to_csv('/home/mq/data_disk2T/Hoang/AVA_Kinetics/LART/v.csv', mode = 'a', header=False, index=False)


           
if __name__ == '__main__':
    gen_val()        
        
        
