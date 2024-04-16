from dataclasses import dataclass, field

@dataclass
class KineticsConfig:
    sampling_factor: int = 1
    num_action_classes: int = 400

@dataclass
class AVAConfig:
    sampling_factor: int = 1
    num_action_classes: int = 80
    num_valid_action_classes: int = 60
    gt_type: str = "all"
    head_dropout: float =0.0
    predict_valid: bool = True
    map_on: str = "AVA" # or "AVA-AK"

@dataclass
class PoseShape:
    dim: int = 229
    mid_dim: int = 229
    en_dim: int = 128

@dataclass
class Joints3D:
    dim: int = 135
    mid_dim: int = 256
    en_dim: int = 128

@dataclass
class Apperance:
    dim: int = 2048
    mid_dim: int = 512
    en_dim: int = 256


@dataclass
class ExtraFeat:
    pose_shape: PoseShape = field(default_factory=PoseShape)
    joints_3D: Joints3D = field(default_factory=Joints3D)
    apperance: Apperance = field(default_factory=Apperance)
    enable: str = 'joints_3D,apperance'

@dataclass
class TRAINER:
    train_dataset: str = "ava_train,kinetics_train"
    test_dataset: str = "ava_val"
    frame_length: int = 125
    max_people: int = 5
    test_batch_id: int = -1
    number_of_processes: int = 25
    full_seq_render: bool = False
    frame_rate_range: int = 1
    num_smpl_heads: int = 1
    action_space: str = "ava"
    loss_on_others_action: bool = True
    focal_length: int = 5000

    extra_feat: ExtraFeat = field(default_factory=ExtraFeat)
    kinetics: KineticsConfig = field(default_factory=KineticsConfig)
    ava: AVAConfig = field(default_factory=AVAConfig)

if __name__ == "__main__":
    from dataclasses import asdict
    import yaml

    cfg = asdict(TRAINER())
    with open('../../../.cache/configs/trainer.yaml', 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)
        


