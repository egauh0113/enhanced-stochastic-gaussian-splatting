# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
# Copyright 2026 Eugene Auh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------
# Original File: examples/simple_trainer.py
# Original Author: nerfstudio-project
# Original License: Apache-2.0
# Original Source: https://github.com/nerfstudio-project/gsplat
#
# Modifications: This file was modified by Eugene Auh in 2026,
#                to implement Enhanced Stochastic Gaussian Splatting.
# -----------------------------------------------------------------------------

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap

from enhanced_stochastic_gaussian_splatting.losses.multinormal import (
    KullbackLeiblerDivergenceMultivariateNormalLower as KLDivMultiNormal,
    KullbackLeiblerDivergenceSO3NormalBCHLower as KLDivSO3NormalBCH,
)
from enhanced_stochastic_gaussian_splatting.losses.enhanced_sc import (
    EnhancedSparsificationCurves,
)
from enhanced_stochastic_gaussian_splatting.losses.laplace_nll_loss import (
    laplace_nll_loss,
    calc_div,
    var2div,
)
from enhanced_stochastic_gaussian_splatting.utils.transforms import (
    Vec2Lower,
    init_cholesky_params,
    normalize_variance_image,
    algebra2quaternion as a2q,
    quaternion2algebra as q2a,
)


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(
        default_factory=lambda: [7_000, 16_000, 16_001, 30_000]
    )
    # Steps to save the model
    save_steps: List[int] = field(
        default_factory=lambda: [7_000, 16_000, 16_001, 30_000]
    )
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(
        default_factory=lambda: [7_000, 16_000, 16_001, 30_000]
    )
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 1  # 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.2
    # Far plane clipping distance
    far_plane: float = 200

    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.05
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1

    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15_000
    # Reset opacities every this steps
    reset_every: int = 3000
    # Refine GSs every this steps
    refine_every: int = 100

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = False

    # Iteration to switch to VI training.
    switch_to_vi: int = 16_000

    # Number of VI samples per forward pass.
    n_vi_sample: int = 8
    full_sample: bool = False

    # Weight for KL divergence loss.
    kl_lambda: float = 1e-2

    # Use AUSE or AUSC for uncertainty training.
    # If True, use AUSE (AUSC-Oracle). If False, use AUSC (AUSC-0).
    use_ause: bool = False
    aus_lambda: float = 5

    # Use Laplace NLL loss instead of L1 loss for uncertainty training.
    use_nll_loss: bool = True
    div_tilde: float = 0.002
    rep_method: Literal["mean", "median"] = "median"

    # Which GS parameters to include in VI.
    vi_means: bool = True
    vi_scales: bool = True
    vi_quats: bool = True
    vi_opacities: bool = True
    vi_sh0: bool = True
    vi_shN: bool = vi_sh0 and True  # Disable if SH0 is disabled.

    # Learning rate for the posterior parameters.
    posterior_lr: float = 1e-4

    # Stochastic GS parameters priors variances.
    prior_var_means: float = 1e-6
    prior_var_scales: float = 1e-6
    prior_var_quats: float = 1e-6
    prior_var_opacities: float = 1e-4
    prior_var_sh0: float = 1e-4
    prior_var_shN: float = 1e-4

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.prune_opa = self.prune_opa
            strategy.grow_grad2d = self.grow_grad2d
            strategy.grow_scale3d = self.grow_scale3d
            strategy.prune_scale3d = self.prune_scale3d

            strategy.refine_start_iter = int(self.refine_start_iter * factor)
            strategy.refine_stop_iter = int(self.refine_stop_iter * factor)
            strategy.reset_every = int(self.reset_every * factor)
            strategy.refine_every = int(self.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(self.refine_start_iter * factor)
            strategy.refine_stop_iter = int(self.refine_stop_iter * factor)
            strategy.refine_every = int(self.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    posterior_lr: float = 1e-4,
    prior_var_means: float = 1e-6,
    prior_var_scales: float = 1e-6,
    prior_var_quats: float = 1e-6,
    prior_var_opacities: float = 1e-6,
    prior_var_sh0: float = 1e-6,
    prior_var_shN: float = 1e-6,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
        # Covariances for VI.
        (
            "means_cov",
            torch.nn.Parameter(
                init_cholesky_params([prior_var_means] * 3, device=device).repeat(N, 1)
            ),
            posterior_lr if not posterior_lr == 0.0 else means_lr * scene_scale,
        ),
        (
            "scales_cov",
            torch.nn.Parameter(
                init_cholesky_params([prior_var_scales] * 3, device=device).repeat(N, 1)
            ),
            posterior_lr if not posterior_lr == 0.0 else scales_lr,
        ),
        (
            "quats_cov",
            torch.nn.Parameter(
                init_cholesky_params([prior_var_quats] * 3, device=device).repeat(N, 1)
            ),
            posterior_lr if not posterior_lr == 0.0 else quats_lr,
        ),
        (
            "opacities_cov",
            torch.nn.Parameter(
                init_cholesky_params([prior_var_opacities], device=device).repeat(N, 1)
            ),
            posterior_lr if not posterior_lr == 0.0 else opacities_lr,
        ),
    ]

    # color is SH coefficients.
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
    colors[:, 0, :] = rgb_to_sh(rgbs)
    params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
    params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))

    # Covariances for VI.
    params.append(
        (
            "sh0_cov",
            torch.nn.Parameter(
                init_cholesky_params([prior_var_sh0] * 3, device=device).repeat(N, 1, 1)
            ),
            posterior_lr if not posterior_lr == 0.0 else sh0_lr,
        )
    )
    params.append(
        (
            "shN_cov",
            torch.nn.Parameter(
                init_cholesky_params([prior_var_shN] * 3, device=device).repeat(
                    N, (sh_degree + 1) ** 2 - 1, 1
                )
            ),
            posterior_lr if not posterior_lr == 0.0 else shN_lr,
        )
    )

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


def compute_kl(
    pc: torch.nn.ParameterDict, cfg: Config, gs_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute the KL divergence loss between the posterior and prior distributions.

    Args:
        pc: The Gaussian primitives.
        cfg: The configuration to check which parameters are stochastic.
        gs_ids: The indices to compute KL divergence for. If None, compute for all GSs.

    Returns:
        kl_loss: The KL divergence loss.
    """
    if gs_ids is None:
        gs_ids = torch.arange(len(pc["means"]), device=pc["means"].device)
    else:
        assert gs_ids.max() < len(pc["means"]), "gs_ids out of range"

    # D_KL(posterior | prior)
    kl_loss = 0.0

    # Uncertainties on 3DSGS.
    kld = KLDivMultiNormal()

    # Position.
    if cfg.vi_means:
        kl_p = kld(
            pc["means"][gs_ids],
            pc["means_cov"][gs_ids],
            pc["means_prior"][gs_ids],
            pc["means_prior_cov"][gs_ids],
        ).mean()
        kl_loss += kl_p

    # Color.
    if cfg.vi_sh0:
        kl_c0 = kld(
            pc["sh0"][gs_ids],
            pc["sh0_cov"][gs_ids],
            pc["sh0_prior"][gs_ids],
            pc["sh0_prior_cov"][gs_ids],
        ).mean()
        kl_loss += kl_c0
    if cfg.vi_shN:
        kl_cN = kld(
            pc["shN"][gs_ids],
            pc["shN_cov"][gs_ids],
            pc["shN_prior"][gs_ids],
            pc["shN_prior_cov"][gs_ids],
        ).mean()
        kl_loss += kl_cN

    # Opacity.
    if cfg.vi_opacities:
        kl_o = kld(
            pc["opacities"][gs_ids],
            pc["opacities_cov"][gs_ids],
            pc["opacities_prior"][gs_ids],
            pc["opacities_prior_cov"][gs_ids],
        ).mean()
        kl_loss += kl_o

    # Scale.
    if cfg.vi_scales:
        kl_s = kld(
            pc["scales"][gs_ids],
            pc["scales_cov"][gs_ids],
            pc["scales_prior"][gs_ids],
            pc["scales_prior_cov"][gs_ids],
        ).mean()
        kl_loss += kl_s

    # Attitude.
    if cfg.vi_quats:
        kld_so3 = KLDivSO3NormalBCH()
        kl_R = kld_so3(
            pc["quats"],
            pc["quats_cov"],
            pc["quats_prior"],
            pc["quats_prior_cov"],
        ).mean()
        kl_loss += kl_R

    return kl_loss


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Flag indicating VI mode.
        self.is_vi = False

        # Model
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            posterior_lr=cfg.posterior_lr,
            prior_var_means=cfg.prior_var_means,
            prior_var_scales=cfg.prior_var_scales,
            prior_var_quats=cfg.prior_var_quats,
            prior_var_opacities=cfg.prior_var_opacities,
            prior_var_sh0=cfg.prior_var_sh0,
            prior_var_shN=cfg.prior_var_shN,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

        self.v2l = {
            "means": Vec2Lower(3).to(self.device),
            "scales": Vec2Lower(3).to(self.device),
            "quats": Vec2Lower(3).to(self.device),
            "opacities": Vec2Lower(1).to(self.device),
            "features": Vec2Lower(32).to(self.device),
            "colors": Vec2Lower(3).to(self.device),
            "sh0": Vec2Lower(3).to(self.device),
            "shN": Vec2Lower(3).to(self.device),
        }

    def sample_splats(
        self, pc: torch.nn.ParameterDict, gs_ids: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Add random noise to the Gaussian primitives for Monte Carlo sampling.

        Args:
            pc: The Gaussian primitives.
            gs_ids: The indices to sample.

        Returns:
            means: The sampled means.
            quats: The sampled quaternions.
            scales: The sampled scales.
            opacities: The sampled opacities.
            colors: The sampled colors.
        """
        # Return all if no gs_ids are provided.
        if gs_ids is None:
            gs_ids = torch.arange(len(pc["means"]), device=pc["means"].device)
        else:
            assert gs_ids.max() < len(pc["means"]), "gs_ids out of range"

        if gs_ids.device != pc["means"].device:
            gs_ids = gs_ids.to(pc["means"].device)

        # Random sampling.
        means = (
            pc["means"][gs_ids]
            + (
                self.v2l["means"](pc["means_cov"][gs_ids])
                @ torch.randn_like(pc["means"][gs_ids])[..., None]
            ).squeeze(-1)
            if self.cfg.vi_means
            else pc["means"][gs_ids]
        )  # [N, 3]
        quats = (
            a2q(
                q2a(pc["quats"][gs_ids])
                + (
                    self.v2l["quats"](pc["quats_cov"][gs_ids])
                    @ torch.randn_like(pc["quats"][gs_ids])[..., :3, None]
                ).squeeze(-1)
            )
            if self.cfg.vi_quats
            else pc["quats"][gs_ids]
        )  # [N, 4]. Random on Lie algebra.
        scales = torch.exp(
            pc["scales"][gs_ids]
            + (
                self.v2l["scales"](pc["scales_cov"][gs_ids])
                @ torch.randn_like(pc["scales"][gs_ids])[..., None]
            ).squeeze(-1)
            if self.cfg.vi_scales
            else pc["scales"][gs_ids]
        )  # [N, 3]
        opacities = (
            torch.sigmoid(
                pc["opacities"][gs_ids]
                + (
                    self.v2l["opacities"](pc["opacities_cov"][gs_ids])
                    .squeeze(-1)
                    .squeeze(-1)
                    * torch.randn_like(pc["opacities"][gs_ids])
                )
            )
            if self.cfg.vi_opacities
            else torch.sigmoid(pc["opacities"][gs_ids])
        )  # [N,]
        sh0 = (
            pc["sh0"][gs_ids]
            + (
                self.v2l["sh0"](pc["sh0_cov"][gs_ids])
                @ torch.randn_like(pc["sh0"][gs_ids])[..., None]
            ).squeeze(-1)
            if self.cfg.vi_sh0
            else pc["sh0"][gs_ids]
        )  # [N, 1, 3]
        shN = (
            pc["shN"][gs_ids]
            + (
                self.v2l["shN"](pc["shN_cov"][gs_ids])
                @ torch.randn_like(pc["shN"][gs_ids])[..., None]
            ).squeeze(-1)
            if self.cfg.vi_shN
            else pc["shN"][gs_ids]
        )  # [N, K-1, 3]
        colors = torch.cat([sh0, shN], 1)  # [N, K, 3]

        return means, quats, scales, opacities, colors

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        gs_ids: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means: Optional[Tensor] = None
        quats: Optional[Tensor] = None
        scales: Optional[Tensor] = None
        opacities: Optional[Tensor] = None
        colors: Optional[Tensor] = None

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model

        if gs_ids is None:
            gs_ids = torch.arange(
                len(self.splats["means"]), device=self.splats["means"].device
            )
        else:
            assert gs_ids.max() < len(self.splats["means"]), "gs_ids out of range"

        if self.is_vi:
            # Random sampling.
            means, quats, scales, opacities, colors = self.sample_splats(
                self.splats, gs_ids
            )
        else:
            means = self.splats["means"][gs_ids]  # [N, 3]
            # quats = F.normalize(self.splats["quats"][gs_ids], dim=-1)  # [N, 4]
            # rasterization does normalization internally
            quats = self.splats["quats"][gs_ids]  # [N, 4]
            scales = torch.exp(self.splats["scales"][gs_ids])  # [N, 3]
            opacities = torch.sigmoid(self.splats["opacities"][gs_ids])  # [N,]
            colors = torch.cat(
                [self.splats["sh0"][gs_ids], self.splats["shN"][gs_ids]], 1
            )  # [N, K, 3]

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps), dynamic_ncols=True)
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            yc_hat = []
            yd_hat = []
            gs_ids = torch.arange(len(self.splats["means"]), device=device)
            for rep in range(cfg.n_vi_sample if self.is_vi else 1):
                torch.manual_seed(hash(str(cfg) + str(step) + str(rep)))
                renders, alphas, info = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    gs_ids=gs_ids,
                    sh_degree=sh_degree_to_use,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                    masks=masks,
                )
                if renders.shape[-1] == 4:
                    colors, depths = renders[..., 0:3], renders[..., 3:4]
                else:
                    colors, depths = renders, None

                if cfg.use_bilateral_grid:
                    grid_y, grid_x = torch.meshgrid(
                        (torch.arange(height, device=self.device) + 0.5) / height,
                        (torch.arange(width, device=self.device) + 0.5) / width,
                        indexing="ij",
                    )
                    grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                    colors = slice(
                        self.bil_grids,
                        grid_xy.expand(colors.shape[0], -1, -1, -1),
                        colors,
                        image_ids.unsqueeze(-1),
                    )["rgb"]

                if cfg.random_bkgd:
                    bkgd = torch.rand(1, 3, device=device)
                    colors = colors + bkgd * (1.0 - alphas)

                yc_hat.append(colors)
                yd_hat.append(depths)

                if rep == 0:
                    info0 = info

                    if not cfg.full_sample and rep == 0:
                        # Active Gaussian IDs at the first sampling.
                        radii = info0["radii"].squeeze(0)  # [N, 2]
                        means2d = info0["means2d"].squeeze(0)  # [N, 2]
                        gs_ids = (
                            (radii > 0).all(dim=-1)
                            & (means2d[:, 0] + radii[:, 0] >= 0)
                            & (means2d[:, 0] - radii[:, 0] < info0["width"])
                            & (means2d[:, 1] + radii[:, 1] >= 0)
                            & (means2d[:, 1] - radii[:, 1] < info0["height"])
                        )  # [N,]

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info0,
            )

            # loss
            loss = 0.0

            image_stack = torch.stack(yc_hat, dim=-1)
            rep_image = (
                yc_hat[0]
                if not self.is_vi
                else (
                    image_stack.mean(dim=-1)
                    if cfg.rep_method == "mean" or not cfg.use_nll_loss
                    else image_stack.median(dim=-1).values
                )
            )
            var_image = (
                image_stack.var(dim=-1)
                if self.is_vi
                else torch.ones_like(rep_image) * 1e-8
            )
            if self.is_vi and cfg.use_nll_loss:
                div_image = (
                    var2div(var_image)
                    if cfg.rep_method == "mean"
                    else calc_div(image_stack)
                )
                nllloss = laplace_nll_loss(rep_image, pixels, div_image, full=True)
                ssimloss = 1.0 - fused_ssim(
                    rep_image.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)
                )
                ssimloss = ssimloss / len(yc_hat)
                loss += (
                    nllloss * (1.0 - cfg.ssim_lambda) * cfg.div_tilde
                    + ssimloss * cfg.ssim_lambda
                )
            else:
                l1loss = F.l1_loss(rep_image, pixels)
                ssimloss = 1.0 - fused_ssim(
                    rep_image.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)
                )
                loss += l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            if self.is_vi:
                # KL divergence and AUSE losses.
                if cfg.kl_lambda > 0.0:
                    klloss = compute_kl(pc=self.splats, cfg=cfg, gs_ids=gs_ids)
                    loss += klloss * cfg.kl_lambda
                if cfg.aus_lambda > 0.0:
                    sc = EnhancedSparsificationCurves(
                        image_stack.permute(0, 3, 1, 2, 4), pixels.permute(0, 3, 1, 2)
                    )  # (..., H, W, C, S) -> (..., C, H, W, S)
                    ausloss = sc.compute_AUSE() if cfg.use_ause else sc.compute_AUSC()
                    loss += ausloss * cfg.aus_lambda

            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            loss.backward()

            desc = (
                (
                    f"loss={loss.item():.3f}|"
                    f"L{l1loss.item() * (1.0 - cfg.ssim_lambda):.3f}+"
                    f"S{ssimloss.item() * cfg.ssim_lambda:.3f}| "
                    f"sh degree={sh_degree_to_use}| "
                )
                if not self.is_vi
                else (
                    (
                        f"loss={loss.item():.3f}|"
                        f"L{l1loss.item() * (1.0 - cfg.ssim_lambda):.3f}+"
                        f"S{ssimloss.item() * cfg.ssim_lambda:.3f}+"
                        f"K{(klloss.item() * cfg.kl_lambda) if cfg.kl_lambda > 0.0 else 0.0:.3f}+"
                        f"A{(ausloss.item() * cfg.aus_lambda) if cfg.aus_lambda > 0.0 else 0.0:.3f}| "
                        # f"sh degree={sh_degree_to_use}| "
                    )
                    if not cfg.use_nll_loss
                    else (
                        f"loss={loss.item():.3f}|"
                        f"L{nllloss.item() * (1.0 - cfg.ssim_lambda) * cfg.div_tilde:.3f}+"
                        f"S{ssimloss.item() * cfg.ssim_lambda:.3f}+"
                        f"K{(klloss.item() * cfg.kl_lambda) if cfg.kl_lambda > 0.0 else 0.0:.3f}+"
                        f"A{(ausloss.item() * cfg.aus_lambda) if cfg.aus_lambda > 0.0 else 0.0:.3f}| "
                        # f"sh degree={sh_degree_to_use}| "
                    )
                )
            )
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                if self.is_vi:
                    if cfg.kl_lambda > 0.0:
                        self.writer.add_scalar("train/klloss", klloss.item(), step)
                    if cfg.aus_lambda > 0.0:
                        self.writer.add_scalar("train/ausloss", ausloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                sh0 = self.splats["sh0"]
                shN = self.splats["shN"]

                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

            # Transient to Variational Inference.
            if step - 1 >= cfg.switch_to_vi:
                self.transition_to_vi(cfg)
            else:
                pass

    def transition_to_vi(self, cfg: Config) -> None:
        if self.is_vi:
            # print("Already in Variational Inference mode.")
            return

        # Transition the model to VI mode.
        print("Transitioning to Variational Inference mode.")
        print("Sample size:", cfg.n_vi_sample)
        print("Posterior LR:", cfg.posterior_lr)
        print("KLD weight:", cfg.kl_lambda)
        print("AUSE weight:" if cfg.use_ause else "AUSC weight:", cfg.aus_lambda)
        print("NLL diversity:", cfg.div_tilde) if cfg.use_nll_loss else None
        opts = [
            "means" if cfg.vi_means else None,
            "quats" if cfg.vi_quats else None,
            "scales" if cfg.vi_scales else None,
            "opacities" if cfg.vi_opacities else None,
            "sh0" if cfg.vi_sh0 else None,
            "shN" if cfg.vi_shN else None,
        ]
        print("VI options:", " ".join(o for o in opts if o))

        N = len(self.splats["means"])

        with torch.no_grad():
            # Copy the current parameters to the fixed prior parameters.
            if cfg.vi_means:
                self.splats["means_prior"] = (
                    self.splats["means"].detach().clone().requires_grad_(False)
                )
                self.splats["means_prior_cov"] = (
                    init_cholesky_params([cfg.prior_var_means] * 3, device=self.device)
                    .repeat(N, 1)
                    .requires_grad_(False)
                )
            if cfg.vi_opacities:
                self.splats["opacities_prior"] = (
                    self.splats["opacities"].detach().clone().requires_grad_(False)
                )
                self.splats["opacities_prior_cov"] = (
                    init_cholesky_params([cfg.prior_var_opacities], device=self.device)
                    .repeat(N, 1)
                    .requires_grad_(False)
                )
            if cfg.vi_sh0:
                self.splats["sh0_prior"] = (
                    self.splats["sh0"].detach().clone().requires_grad_(False)
                )
                self.splats["sh0_prior_cov"] = (
                    init_cholesky_params([cfg.prior_var_sh0] * 3, device=self.device)
                    .repeat(N, 1, 1)
                    .requires_grad_(False)
                )
            if cfg.vi_shN:
                self.splats["shN_prior"] = (
                    self.splats["shN"].detach().clone().requires_grad_(False)
                )
                self.splats["shN_prior_cov"] = (
                    init_cholesky_params([cfg.prior_var_shN] * 3, device=self.device)
                    .repeat(N, (cfg.sh_degree + 1) ** 2 - 1, 1)
                    .requires_grad_(False)
                )
            if cfg.vi_quats:
                self.splats["quats_prior"] = (
                    self.splats["quats"].detach().clone().requires_grad_(False)
                )
                self.splats["quats_prior_cov"] = (
                    init_cholesky_params([cfg.prior_var_quats] * 3, device=self.device)
                    .repeat(N, 1)
                    .requires_grad_(False)
                )
            if cfg.vi_scales:
                self.splats["scales_prior"] = (
                    self.splats["scales"].detach().clone().requires_grad_(False)
                )
                self.splats["scales_prior_cov"] = (
                    init_cholesky_params([cfg.prior_var_scales] * 3, device=self.device)
                    .repeat(N, 1)
                    .requires_grad_(False)
                )

        self.is_vi = True
        return

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": [], "ause_rmse": [], "ause_mae": []}
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            if step >= cfg.switch_to_vi:
                # Open VI for computing the uncertainty metrics.
                self.is_vi = True

                # Random sampling for uncertainty estimation.
                yc_hat = []
                gs_ids = torch.arange(len(self.splats["means"]), device=device)
                for rep in range(cfg.n_vi_sample):
                    torch.manual_seed(hash(str(cfg) + str(step) + str(rep)))
                    colors, alphas, info = self.rasterize_splats(
                        camtoworlds=camtoworlds,
                        Ks=Ks,
                        width=width,
                        height=height,
                        gs_ids=gs_ids,
                        sh_degree=cfg.sh_degree,
                        near_plane=cfg.near_plane,
                        far_plane=cfg.far_plane,
                        masks=masks,
                    )  # [1, H, W, 3]
                    colors = torch.clamp(colors, 0.0, 1.0)
                    colors = colors[..., :3]  # Take RGB channels
                    yc_hat.append(colors)

                    if not cfg.full_sample and rep == 0:
                        # Active Gaussian IDs at the first sampling.
                        radii = info["radii"].squeeze(0)  # [N, 2]
                        means2d = info["means2d"].squeeze(0)  # [N, 2]
                        gs_ids = (
                            (radii > 0).all(dim=-1)
                            & (means2d[:, 0] + radii[:, 0] >= 0)
                            & (means2d[:, 0] - radii[:, 0] < info["width"])
                            & (means2d[:, 1] + radii[:, 1] >= 0)
                            & (means2d[:, 1] - radii[:, 1] < info["height"])
                        )  # [N,]

                    imageio.imwrite(
                        f"{self.render_dir}/{stage}_step{step}_{i:04d}_sample{rep}.png",
                        (colors.squeeze(0).cpu().numpy() * 255).astype(np.uint8),
                    )

                image_stack = torch.stack(yc_hat, dim=-1)  # [1, H, W, 3, S]

                # Write representative images.
                color_mean = image_stack.mean(dim=-1)  # [1, H, W, 3]
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}_mean.png",
                    (color_mean.squeeze(0).cpu().numpy() * 255).astype(np.uint8),
                )

                color_median = image_stack.median(dim=-1).values  # [1, H, W, 3]
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}_median.png",
                    (color_median.squeeze(0).cpu().numpy() * 255).astype(np.uint8),
                )

                # Write error images
                mean_err = (color_mean - pixels).abs()  # [1, H, W, 3]
                mono_err = normalize_variance_image(mean_err)  # [1, H, W, 1]
                render_err = (
                    apply_float_colormap(mono_err).detach().cpu().squeeze(0).numpy()
                )  # [1, H, W, 3]
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}_err_mean.png",
                    (render_err * 255).astype(np.uint8),
                )

                median_err = (color_median - pixels).abs()  # [1, H, W, 3]
                mono_err = normalize_variance_image(median_err)  # [1, H, W, 1]
                render_err = (
                    apply_float_colormap(mono_err).detach().cpu().squeeze(0).numpy()
                )  # [1, H, W, 3]
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}_err_median.png",
                    (render_err * 255).astype(np.uint8),
                )

                # Write distribution images.
                colors_var = image_stack.var(dim=-1)  # [1, H, W, 3]
                mono_var = normalize_variance_image(colors_var)  # [1, H, W, 1]
                render_var = (
                    apply_float_colormap(mono_var).detach().cpu().squeeze(0).numpy()
                )  # [1, H, W, 3]
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}_dist_var.png",
                    (render_var * 255).astype(np.uint8),
                )

                colors_div = calc_div(image_stack)  # [1, H, W, 3]
                mono_div = normalize_variance_image(colors_div)  # [1, H, W, 1]
                render_div = (
                    apply_float_colormap(mono_div).detach().cpu().squeeze(0).numpy()
                )  # [1, H, W, 3]
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}_dist_div.png",
                    (render_div * 255).astype(np.uint8),
                )

                # Close VI for rendering the final image.
                self.is_vi = False

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            imageio.imwrite(
                f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                canvas,
            )

            pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors_p, pixels_p))
            metrics["ssim"].append(self.ssim(colors_p, pixels_p))
            metrics["lpips"].append(self.lpips(colors_p, pixels_p))
            if cfg.use_bilateral_grid:
                cc_colors = color_correct(colors, pixels)
                cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

            if step >= cfg.switch_to_vi:
                image_stack = image_stack.permute(0, 3, 1, 2, 4)

                sc_rmse = EnhancedSparsificationCurves(
                    image_stack,
                    pixels_p,
                    diff_to_error_fn=lambda x: x.square().mean(dim=-3),
                    summarizer_fn=lambda x: x.mean(dim=-1).sqrt().mean(),
                )
                metrics["ause_rmse"].append(sc_rmse.compute_AUSE())
                sc_rmse.save_sc(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}_sc_rmse.png",
                    plot_random=True,
                )
                sc_rmse.save_sc(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}_sc_rmse_raw.png",
                    plot_random=True,
                    normalize=False,
                    ylim=0.04,
                )

                sc_mae = EnhancedSparsificationCurves(
                    image_stack,
                    pixels_p,
                    diff_to_error_fn=lambda x: x.abs().mean(dim=-3),
                    summarizer_fn=lambda x: x.mean(dim=-1).mean(),
                )
                metrics["ause_mae"].append(sc_mae.compute_AUSE())
                sc_mae.save_sc(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}_sc_mae.png",
                    plot_random=True,
                )
                sc_mae.save_sc(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}_sc_mae_raw.png",
                    plot_random=True,
                    normalize=False,
                    ylim=0.03,
                )

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        if step >= cfg.switch_to_vi:
            ause_rmse = torch.stack(metrics["ause_rmse"]).mean()
            ause_mae = torch.stack(metrics["ause_mae"]).mean()
            print(
                f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f}, "
                f"AUSE_RMSE: {ause_rmse:.3f}, AUSE_MAE: {ause_mae:.3f} "
                f"Time: {ellipse_time:.3f}s/image "
                f"Number of GS: {len(self.splats['means'])}"
            )
        else:
            print(
                f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
                f"Time: {ellipse_time:.3f}s/image "
                f"Number of GS: {len(self.splats['means'])}"
            )

        stats = (
            {
                "psnr": psnr.item(),
                "ssim": ssim.item(),
                "lpips": lpips.item(),
                "ellipse_time": ellipse_time,
                "num_GS": len(self.splats["means"]),
            }
            if step < cfg.switch_to_vi
            else {
                "psnr": psnr.item(),
                "ssim": ssim.item(),
                "lpips": lpips.item(),
                "ause_rmse": ause_rmse.item(),
                "ause_mae": ause_mae.item(),
                "ellipse_time": ellipse_time,
                "num_GS": len(self.splats["means"]),
            }
        )
        # save stats as json
        with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"{stage}/{k}", v, step)
        self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=False),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=False),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)
