from typing import Any
from .sort import Sort

class TrackRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracker = Sort() # TODO, get the tracker from cfg

        self.start_end_time_infor = {} # used to store start_end time information of video

    def __update_track(self, dets):
        tracklets = self.tracker.update(dets)
        return tracklets
    

    def __update_start_end_time_infor(self, tracklets, frame_time):
        """Update start_end time information of the video

        Parameters:
        -----------
            tracklets, list[list]:
                contains infomation about position, size and ID
            frame_time, int:
                current frame count
        """
        for tracklet in tracklets:
            ID = int(tracklet[-1])
            
            if ID in self.start_end_time_infor: 
                self.start_end_time_infor[ID][-1] = frame_time # if ID already exists, update time end only
            else:
                self.start_end_time_infor[ID] = [frame_time, frame_time] # else init both time start and time end


    def __call__(self, dets, frame_time):
        tracklets = self.__update_track(dets)
        self.__update_start_end_time_infor(tracklets, frame_time)
        
        return tracklets