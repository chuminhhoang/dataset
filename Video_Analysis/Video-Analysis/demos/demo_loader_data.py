import dacite
import yaml
import sys
sys.path.append('../Video-Analysis/')
from VideoAnalysis.action.trainer.config import TRAINER
from VideoAnalysis.action.datamodule.datasets.ava import AVADataset
if __name__ == '__main__':
    with open(f"/root/Hoang/Video_Analysis/Video-Analysis/.cache/configs/trainer.yaml", "r") as stream:
            try:
                motrv2_cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc) 
    
    cfg = dacite.from_dict(data_class=TRAINER, data=motrv2_cfg)
    ava_dataloader = AVADataset(cfg, train=True)
    
    a = ava_dataloader
    input_data, output_data, meta_data, video_name = a.__getitem__(0)
    