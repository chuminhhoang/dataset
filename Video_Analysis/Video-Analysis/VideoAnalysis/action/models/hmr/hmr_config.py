from dataclasses import dataclass, field

# Config for HMAR
@dataclass
class SMPLHeadConfig:
    TYPE: str = 'basic'
    POOL: str = 'max'
    SMPL_MEAN_PARAMS: str = f".cache/smpl_mean_params.npz"
    IN_CHANNELS: int = 2048

@dataclass
class BackboneConfig:
    TYPE: str = 'resnet'
    NUM_LAYERS: int = 50
    MASK_TYPE: str = 'feat'

@dataclass
class TransformerConfig:
    HEADS: int = 1
    LAYERS: int = 1
    BOX_FEATS: int = 6

@dataclass
class ModelConfig:
    IMAGE_SIZE: int = 256
    SMPL_HEAD: SMPLHeadConfig = field(default_factory=SMPLHeadConfig)
    BACKBONE: BackboneConfig = field(default_factory=BackboneConfig)
    TRANSFORMER: TransformerConfig = field(default_factory=TransformerConfig)
    pose_transformer_size: int = 2048


@dataclass
class HMRConfig:
    hmar_path: str = f"weights/hmar_v2_resnet50.pth"

@dataclass
class SMPLConfig:
    MODEL_PATH: str = f".cache/models/smpl/"
    GENDER: str = 'neutral'
    MODEL_TYPE: str = 'smpl'
    NUM_BODY_JOINTS: int = 23
    JOINT_REGRESSOR_EXTRA: str = f".cache/SMPL_to_J19.pkl"
    TEXTURE: str = f".cache/texture.npz"


@dataclass
class HMRFullConfig:
    hmr: HMRConfig = field(default_factory=HMRConfig)
    SMPL: SMPLConfig = field(default_factory=SMPLConfig)
    MODEL: ModelConfig = field(default_factory=ModelConfig)

