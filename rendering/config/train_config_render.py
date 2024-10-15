import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List


@dataclass
class RenderConfig:
    grid_size: int = 2048
    radius: float = 1.2
    look_at_height = 0.2
    base_theta: float = 70

    fov_para: float = np.pi / 3 *0.9   
    remove_mesh_part_names: List[str] = field(default_factory=["MI_CH_Top"].copy)
    remove_unsupported_buffers: List[str] = field(default_factory=["filamat"].copy)
    n_views: int = 4  
    views_before: List[Tuple[float, float]] = field(default_factory=list)
    views_after: List[Tuple[float, float]] = field(default_factory=[[180, 30], [180, 150]].copy)
    alternate_views: bool = True
    calcu_uncolored_mode: str = "WarpGrid"  
    projection_mode: str = "Pinhole"  
    texture_interpolation_mode: str = 'bilinear'
    texture_default_color: List[float] = field(default_factory=[0.8, 0.1, 0.8].copy)
    texturify_blend_alpha: float = 1.0
    render_angle_thres: float = 68

    views_init: List[float] = field(default_factory=[0,1,2,3].copy)
    views_inpaint: List[Tuple[float, float]] = field(default_factory=[].copy)




@dataclass
class GuideConfig:
    shape_path: str = "xxx"
    initial_texture: Path = None
    texture_resolution: List[int] = field(default_factory=[2048, 2048].copy)  # h w
    append_direction: bool = True
    shape_scale: float = 0.6
    z_update_thr: float = 0.1
    strict_projection: bool = True
    force_run_xatlas: bool = False


@dataclass
class OptimConfig:
    """ Parameters for the optimization process """
    seed: int = 1
    lr: float = 1e-2
    train_step: int = 200


@dataclass
class LogConfig:
    exp_path = "xxx"
    full_eval_size: int = 100
    cache_path: str = "rendering_cache"


@dataclass
class TrainConfig:
    """ The main configuration for the coach trainer """
    log: LogConfig = field(default_factory=LogConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    guide: GuideConfig = field(default_factory=GuideConfig)


