# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
# Copyright 2026 Eugene Auh

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------
# Original File: examples/simple_trainer_2dgs.py
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
from typing import Dict, List, Literal, Optional, Tuple
from pathlib import Path

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
from datasets.colmap import Dataset, Parser
from datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    apply_depth_colormap,
    colormap,
    knn,
    rgb_to_sh,
    set_random_seed,
)
from gsplat_viewer_2dgs import GsplatViewer, GsplatRenderTabState
from gsplat.rendering import rasterization_2dgs, rasterization_2dgs_inria_wrapper
from gsplat.strategy import DefaultStrategy
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
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None

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

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)
    revised_opacity: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Enable normal consistency loss. (Currently for 2DGS only)
    normal_loss: bool = True
    # Weight for normal loss
    normal_lambda: float = 5e-2
    # Iteration to start normal consistency regulerization
    normal_start_iter: int = 7_000

    # Distortion loss. (experimental)
    dist_loss: bool = True
    # Weight for distortion loss
    dist_lambda: float = 1e-2
    # Iteration to start distortion loss regulerization
    dist_start_iter: int = 3_000

    # Model for splatting.
    model_type: Literal["2dgs", "2dgs-inria", "esgs"] = "2dgs"

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    # Iteration to switch to VI training.
    switch_to_vi: int = 16_000

    # Number of VI samples per forward pass.
    n_vi_sample: int = 8

    # Weight for KL divergence loss.
    kl_lambda: float = 1e-2

    # Use AUSE or AUSC for uncertainty training.
    # If True, use AUSE (AUSC-Oracle). If False, use AUSC (AUSC-0).
    use_ause: bool = False
    aus_lambda: float = 5

    # Use Laplace NLL loss instead of L1 loss for uncertainty training.
    use_nll_loss: bool = True
    div_tilde: float = 0.002

    # Which GS parameters to include in VI.
    vi_means: bool = True
    vi_scales: bool = True
    vi_quats: bool = True
    vi_opacities: bool = True
    vi_sh0: bool = True
    vi_shN: bool = vi_sh0 and True  # Disable if SH0 is disabled.
    vi_features: bool = True  # NOT TESTED.
    vi_colors: bool = True  # NOT TESTED.

    # Learning rate for the posterior parameters.
    posterior_lr: float = 1e-4

    # Stochastic GS parameters priors variances.
    prior_var_means: float = 1e-6
    prior_var_scales: float = 1e-6
    prior_var_quats: float = 1e-6
    prior_var_opacities: float = 1e-4
    prior_var_sh0: float = 1e-4
    prior_var_shN: float = 1e-4
    prior_var_features: float = 1e-4  # NOT TESTED.
    prior_var_colors: float = 1e-4  # NOT TESTED.

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    posterior_lr: float = 1e-4,
    prior_var_means: float = 1e-6,
    prior_var_scales: float = 1e-6,
    prior_var_quats: float = 1e-6,
    prior_var_opacities: float = 1e-6,
    prior_var_sh0: float = 1e-6,
    prior_var_shN: float = 1e-6,
    prior_var_features: float = 1e-6,
    prior_var_colors: float = 1e-6,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    N = points.shape[0]
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
        # Covariances for VI.
        (
            "means_cov",
            torch.nn.Parameter(
                init_cholesky_params([prior_var_means] * 3, device=device).repeat(N, 1)
            ),
            posterior_lr if not posterior_lr == 0.0 else 1.6e-4 * scene_scale,
        ),
        (
            "scales_cov",
            torch.nn.Parameter(
                init_cholesky_params([prior_var_scales] * 2, device=device).repeat(N, 1)
            ),
            posterior_lr if not posterior_lr == 0.0 else 5e-3,
        ),
        (
            "quats_cov",
            torch.nn.Parameter(
                init_cholesky_params([prior_var_quats] * 3, device=device).repeat(N, 1)
            ),
            posterior_lr if not posterior_lr == 0.0 else 1e-3,
        ),
        (
            "opacities_cov",
            torch.nn.Parameter(
                init_cholesky_params([prior_var_opacities], device=device).repeat(N, 1)
            ),
            posterior_lr if not posterior_lr == 0.0 else 5e-2,
        ),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))

        # Covariances for VI.
        params.append(
            (
                "sh0_cov",
                torch.nn.Parameter(
                    init_cholesky_params([prior_var_sh0] * 3, device=device).repeat(
                        N, 1, 1
                    )
                ),
                posterior_lr if not posterior_lr == 0.0 else 2.5e-3,
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
                posterior_lr if not posterior_lr == 0.0 else 2.5e-3 / 20,
            )
        )
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

        # Covariances for VI.
        params.append(
            (
                "features_cov",
                torch.nn.Parameter(
                    init_cholesky_params(
                        [prior_var_features] * feature_dim, device=device
                    ).repeat(N, 1)
                ),
                posterior_lr if not posterior_lr == 0.0 else 2.5e-3,
            )
        )
        params.append(
            (
                "colors_cov",
                torch.nn.Parameter(
                    init_cholesky_params([prior_var_colors] * 3, device=device).repeat(
                        N, 1
                    )
                ),
                posterior_lr if not posterior_lr == 0.0 else 2.5e-3,
            )
        )

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size)}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


def compute_kl(splats: torch.nn.ParameterDict, cfg: Config) -> torch.Tensor:
    # D_KL(posterior | prior)
    kl_loss = 0.0

    # Uncertainties on 3DSGS.
    kld = KLDivMultiNormal()

    # Position.
    if cfg.vi_means:
        kl_p = kld(
            splats["means"],
            splats["means_cov"],
            splats["means_prior"],
            splats["means_prior_cov"],
        ).mean()
        kl_loss += kl_p

    # Color.
    if not cfg.app_opt:
        if cfg.vi_sh0:
            kl_c0 = kld(
                splats["sh0"],
                splats["sh0_cov"],
                splats["sh0_prior"],
                splats["sh0_prior_cov"],
            ).mean()
            kl_loss += kl_c0
        if cfg.vi_shN:
            kl_cN = kld(
                splats["shN"],
                splats["shN_cov"],
                splats["shN_prior"],
                splats["shN_prior_cov"],
            ).mean()
            kl_loss += kl_cN
    else:
        if cfg.vi_colors:
            kl_c0 = kld(
                splats["colors"],
                splats["colors_cov"],
                splats["colors_prior"],
                splats["colors_prior_cov"],
            ).mean()
            kl_loss += kl_c0
        if cfg.vi_features:
            kl_cN = kld(
                splats["features"],
                splats["features_cov"],
                splats["features_prior"],
                splats["features_prior_cov"],
            ).mean()
            kl_loss += kl_cN

    # Opacity.
    if cfg.vi_opacities:
        kl_o = kld(
            splats["opacities"],
            splats["opacities_cov"],
            splats["opacities_prior"],
            splats["opacities_prior_cov"],
        ).mean()
        kl_loss += kl_o

    # Scale.
    if cfg.vi_scales:
        kl_s = kld(
            splats["scales"][:, :2],
            splats["scales_cov"],
            splats["scales_prior"][:, :2],
            splats["scales_prior_cov"],
        ).mean()
        kl_loss += kl_s

    # Attitude.
    if cfg.vi_quats:
        kld_so3 = KLDivSO3NormalBCH()
        kl_R = kld_so3(
            splats["quats"],
            splats["quats_cov"],
            splats["quats_prior"],
            splats["quats_prior_cov"],
        ).mean()
        kl_loss += kl_R

    return kl_loss


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

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
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            posterior_lr=cfg.posterior_lr,
            prior_var_means=cfg.prior_var_means,
            prior_var_scales=cfg.prior_var_scales,
            prior_var_quats=cfg.prior_var_quats,
            prior_var_opacities=cfg.prior_var_opacities,
            prior_var_sh0=cfg.prior_var_sh0,
            prior_var_shN=cfg.prior_var_shN,
            prior_var_features=cfg.prior_var_features,
            prior_var_colors=cfg.prior_var_colors,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        self.model_type = cfg.model_type

        if self.model_type in ["2dgs", "esgs"]:
            key_for_gradient = "gradient_2dgs"
        else:
            key_for_gradient = "means2d"

        # Densification Strategy
        self.strategy = DefaultStrategy(
            verbose=False,
            prune_opa=cfg.prune_opa,
            grow_grad2d=cfg.grow_grad2d,
            grow_scale3d=cfg.grow_scale3d,
            prune_scale3d=cfg.prune_scale3d,
            # refine_scale2d_stop_iter=4000, # splatfacto behavior
            refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=(
                cfg.refine_stop_iter
                if not cfg.model_type == "esgs"
                else min(cfg.refine_stop_iter, cfg.switch_to_vi)
            ),
            reset_every=cfg.reset_every,
            refine_every=cfg.refine_every,
            absgrad=cfg.absgrad,
            revised_opacity=cfg.revised_opacity,
            key_for_gradient=key_for_gradient,
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()

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

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )

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
            "scales": Vec2Lower(2).to(self.device),
            "quats": Vec2Lower(3).to(self.device),
            "opacities": Vec2Lower(1).to(self.device),
            "features": Vec2Lower(32).to(self.device),
            "colors": Vec2Lower(3).to(self.device),
            "sh0": Vec2Lower(3).to(self.device),
            "shN": Vec2Lower(3).to(self.device),
        }

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        means: Optional[Tensor] = None
        quats: Optional[Tensor] = None
        scales: Optional[Tensor] = None
        opacities: Optional[Tensor] = None
        colors: Optional[Tensor] = None

        if self.model_type == "esgs" and self.is_vi:
            means = (
                self.splats["means"]
                + (
                    self.v2l["means"](self.splats["means_cov"])
                    @ torch.randn_like(self.splats["means"])[..., None]
                ).squeeze(-1)
                if self.cfg.vi_means
                else self.splats["means"]
            )  # [N, 3]
            quats = (
                a2q(
                    q2a(self.splats["quats"])
                    + (
                        self.v2l["quats"](self.splats["quats_cov"])
                        @ torch.randn_like(self.splats["quats"])[..., :3, None]
                    ).squeeze(-1)
                )
                if self.cfg.vi_quats
                else self.splats["quats"]
            )  # [N, 4]. Random on Lie algebra.
            scales = (
                torch.exp(
                    torch.cat(
                        [
                            self.splats["scales"][:, :2]
                            + (
                                self.v2l["scales"](self.splats["scales_cov"])
                                @ torch.randn(
                                    self.splats["scales"].shape[0],
                                    2,
                                    device=self.splats["scales"].device,
                                )[..., None]
                            ).squeeze(-1),
                            self.splats["scales"][:, 2:],
                        ],
                        dim=-1,
                    )
                )
                if self.cfg.vi_scales
                else torch.exp(self.splats["scales"])
            )  # [N, 3]. Random only on x and y axes.
            # TODO: check
            # scales = torch.exp(
            #     self.splats["scales"]
            #     + (
            #         self.v2l["scales"](self.splats["scales_cov"])
            #         @ torch.randn_like(self.splats["scales"])[..., None]
            #     ).squeeze(-1)
            # ) if self.cfg.vi_scales else torch.exp(self.splats["scales"])  # [N, 3]
            opacities = (
                torch.sigmoid(
                    self.splats["opacities"]
                    + (
                        self.v2l["opacities"](self.splats["opacities_cov"])
                        .squeeze(-1)
                        .squeeze(-1)
                        * torch.randn_like(self.splats["opacities"])
                    )
                )
                if self.cfg.vi_opacities
                else torch.sigmoid(self.splats["opacities"])
            )  # [N,]

            image_ids = kwargs.pop("image_ids", None)
            if self.cfg.app_opt:
                features = (
                    self.splats["features"]
                    + torch.einsum(
                        "...ij,...j->...i",
                        self.v2l["features"](self.splats["features_cov"]),
                        torch.randn_like(self.splats["features"]),
                    )
                    if self.cfg.vi_features
                    else self.splats["features"]
                )  # [N, F]
                colors = self.app_module(
                    features=features,
                    embed_ids=image_ids,
                    dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                    sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
                )
                colors = (
                    colors
                    + self.splats["colors"]
                    + torch.einsum(
                        "...ij,...j->...i",
                        self.v2l["colors"](self.splats["colors_cov"]),
                        torch.randn_like(self.splats["colors"]),
                    )
                    if self.cfg.vi_colors
                    else self.splats["colors"]
                )  # [N, 3]
                colors = torch.sigmoid(colors)
            else:
                sh0 = (
                    self.splats["sh0"]
                    + (
                        self.v2l["sh0"](self.splats["sh0_cov"])
                        @ torch.randn_like(self.splats["sh0"])[..., None]
                    ).squeeze(-1)
                    if self.cfg.vi_sh0
                    else self.splats["sh0"]
                )  # [N, 1, 3]
                shN = (
                    self.splats["shN"]
                    + (
                        self.v2l["shN"](self.splats["shN_cov"])
                        @ torch.randn_like(self.splats["shN"])[..., None]
                    ).squeeze(-1)
                    if self.cfg.vi_shN
                    else self.splats["shN"]
                )  # [N, K-1, 3]
                colors = torch.cat([sh0, shN], 1)  # [N, K, 3]
        else:
            means = self.splats["means"]  # [N, 3]
            # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
            # rasterization does normalization internally
            quats = self.splats["quats"]  # [N, 4]
            scales = torch.exp(self.splats["scales"])  # [N, 3]
            opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

            image_ids = kwargs.pop("image_ids", None)
            if self.cfg.app_opt:
                colors = self.app_module(
                    features=self.splats["features"],
                    embed_ids=image_ids,
                    dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                    sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
                )
                colors = colors + self.splats["colors"]
                colors = torch.sigmoid(colors)
            else:
                colors = torch.cat(
                    [self.splats["sh0"], self.splats["shN"]], 1
                )  # [N, K, 3]

        assert self.cfg.antialiased is False, "Antialiased is not supported for 2DGS"

        if self.model_type in ["2dgs", "esgs"]:
            (
                render_colors,
                render_alphas,
                render_normals,
                normals_from_depth,
                render_distort,
                render_median,
                info,
            ) = rasterization_2dgs(
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
                absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
        elif self.model_type == "2dgs-inria":
            renders, info = rasterization_2dgs_inria_wrapper(
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
                absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
            render_colors, render_alphas = renders
            render_normals = info["normals_rend"]
            normals_from_depth = info["normals_surf"]
            render_distort = info["render_distloss"]
            render_median = render_colors[..., 3]

        return (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        )

    def train(self):
        cfg = self.cfg
        device = self.device

        # Dump cfg.
        with open(f"{cfg.result_dir}/cfg.json", "w") as f:
            json.dump(vars(cfg), f)

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
            for rep in range(cfg.n_vi_sample if self.is_vi else 1):
                torch.manual_seed(hash(str(cfg) + str(step) + str(rep)))
                (
                    renders,
                    alphas,
                    normals,
                    normals_from_depth,
                    render_distort,
                    render_median,
                    info,
                ) = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=image_ids,
                    render_mode="RGB+ED" if cfg.depth_loss else "RGB+D",
                    distloss=self.cfg.dist_loss,
                )
                if renders.shape[-1] == 4:
                    colors, depths = renders[..., 0:3], renders[..., 3:4]
                else:
                    colors, depths = renders, None

                if cfg.random_bkgd:
                    bkgd = torch.rand(1, 3, device=device)
                    colors = colors + bkgd * (1.0 - alphas)

                yc_hat.append(colors)
                yd_hat.append(depths)

            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )
            masks = data["mask"].to(device) if "mask" in data else None
            if masks is not None:
                pixels = pixels * masks[..., None]
                yc_hat = [y * masks[..., None] for y in yc_hat]

            # loss
            loss = 0.0

            image_stack = torch.stack(yc_hat, dim=-1)
            mean_image = image_stack.mean(dim=-1) if self.is_vi else yc_hat[0]
            var_image = (
                image_stack.var(dim=-1)
                if self.is_vi
                else torch.ones_like(mean_image) * 1e-8
            )
            if self.is_vi and cfg.use_nll_loss:
                div_image = var2div(var_image)
                nllloss = laplace_nll_loss(mean_image, pixels, div_image, full=True)
                ssimloss = 1.0 - self.ssim(
                    pixels.permute(0, 3, 1, 2), mean_image.permute(0, 3, 1, 2)
                )
                ssimloss = ssimloss / len(yc_hat)
                loss += (
                    nllloss * (1.0 - cfg.ssim_lambda) * cfg.div_tilde
                    + ssimloss * cfg.ssim_lambda
                )
            else:
                l1loss = F.l1_loss(mean_image, pixels)
                ssimloss = 1.0 - self.ssim(
                    pixels.permute(0, 3, 1, 2), mean_image.permute(0, 3, 1, 2)
                )
                loss += l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            if self.is_vi:
                # KL divergence and AUSE losses.
                if cfg.kl_lambda > 0.0:
                    klloss = compute_kl(self.splats, cfg)
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

            if cfg.normal_loss:
                if step > cfg.normal_start_iter:
                    curr_normal_lambda = cfg.normal_lambda
                else:
                    curr_normal_lambda = 0.0
                # normal consistency loss
                normals = normals.squeeze(0).permute((2, 0, 1))
                normals_from_depth *= alphas.squeeze(0).detach()
                if len(normals_from_depth.shape) == 4:
                    normals_from_depth = normals_from_depth.squeeze(0)
                normals_from_depth = normals_from_depth.permute((2, 0, 1))
                normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
                normalloss = curr_normal_lambda * normal_error.mean()
                loss += normalloss

            if cfg.dist_loss:
                if step > cfg.dist_start_iter:
                    curr_dist_lambda = cfg.dist_lambda
                else:
                    curr_dist_lambda = 0.0
                distloss = render_distort.mean()
                loss += distloss * curr_dist_lambda

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
            # if cfg.depth_loss:
            #     desc += f"depth loss={depthloss.item():.6f}| "
            # if cfg.dist_loss:
            #     desc += f"dist loss={distloss.item():.6f}"
            # if cfg.pose_opt and cfg.pose_noise:
            #     # monitor the pose error if we inject noise
            #     pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
            #     desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
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
                if cfg.normal_loss:
                    self.writer.add_scalar("train/normalloss", normalloss.item(), step)
                if cfg.dist_loss:
                    self.writer.add_scalar("train/distloss", distloss.item(), step)
                if cfg.tb_save_image:
                    canvas = (
                        torch.cat([pixels, colors[..., :3]], dim=2)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=cfg.packed,
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

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                torch.save(
                    {
                        "step": step,
                        "splats": self.splats.state_dict(),
                    },
                    f"{self.ckpt_dir}/ckpt_{step}.pt",
                )

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
                self.render_traj(step)

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
            if self.model_type == "esgs" and step - 1 >= cfg.switch_to_vi:
                self.transition_to_vi(cfg)
            else:
                pass

        return

    def transition_to_vi(self, cfg: Config) -> None:
        if self.is_vi:
            # print("Already in Variational Inference mode.")
            return

        # Transition the model to VI mode.
        print("Transitioning to Variational Inference mode.")
        print("Sampling number:", cfg.n_vi_sample)
        print(f"Posterior LR: {cfg.posterior_lr}")
        print("KLD weight :", cfg.kl_lambda)
        print("AUSE weight:" if cfg.use_ause else "AUSC weight:", cfg.aus_lambda)
        print("NLL diversity:", cfg.div_tilde) if cfg.use_nll_loss else None
        opts = [
            "means" if cfg.vi_means else None,
            "quats" if cfg.vi_quats else None,
            "scales" if cfg.vi_scales else None,
            "opacities" if cfg.vi_opacities else None,
            "features" if cfg.vi_features and cfg.app_opt else None,
            "colors" if cfg.vi_colors and cfg.app_opt else None,
            "sh0" if cfg.vi_sh0 and not cfg.app_opt else None,
            "shN" if cfg.vi_shN and not cfg.app_opt else None,
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
            if not cfg.app_opt:
                if cfg.vi_sh0:
                    self.splats["sh0_prior"] = (
                        self.splats["sh0"].detach().clone().requires_grad_(False)
                    )
                    self.splats["sh0_prior_cov"] = (
                        init_cholesky_params(
                            [cfg.prior_var_sh0] * 3, device=self.device
                        )
                        .repeat(N, 1, 1)
                        .requires_grad_(False)
                    )
                if cfg.vi_shN:
                    self.splats["shN_prior"] = (
                        self.splats["shN"].detach().clone().requires_grad_(False)
                    )
                    self.splats["shN_prior_cov"] = (
                        init_cholesky_params(
                            [cfg.prior_var_shN] * 3, device=self.device
                        )
                        .repeat(N, (cfg.sh_degree + 1) ** 2 - 1, 1)
                        .requires_grad_(False)
                    )
            else:
                if cfg.vi_colors:
                    self.splats["colors_prior"] = (
                        self.splats["colors"].detach().clone().requires_grad_(False)
                    )
                    self.splats["colors_prior_cov"] = (
                        init_cholesky_params(
                            [cfg.prior_var_colors] * 3, device=self.device
                        )
                        .repeat(N, 1)
                        .requires_grad_(False)
                    )
                if cfg.vi_features:
                    self.splats["features_prior"] = (
                        self.splats["features"].detach().clone().requires_grad_(False)
                    )
                    self.splats["features_prior_cov"] = (
                        init_cholesky_params(
                            [cfg.prior_var_features] * 32, device=self.device
                        )
                        .repeat(N, 1)
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
                    init_cholesky_params([cfg.prior_var_scales] * 2, device=self.device)
                    .repeat(N, 1)
                    .requires_grad_(False)
                )

        self.is_vi = True
        return

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": [], "ause_rmse": [], "ause_mae": []}
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            if cfg.model_type == "esgs" and step >= cfg.switch_to_vi:
                # Open VI for computing the uncertainty metrics.
                self.is_vi = True

                # Random sampling for uncertainty estimation.
                yc_hat = []
                for rep in range(cfg.n_vi_sample):
                    torch.manual_seed(hash(str(cfg) + str(step) + str(rep)))
                    (
                        colors,
                        alphas,
                        normals,
                        normals_from_depth,
                        render_distort,
                        render_median,
                        _,
                    ) = self.rasterize_splats(
                        camtoworlds=camtoworlds,
                        Ks=Ks,
                        width=width,
                        height=height,
                        sh_degree=cfg.sh_degree,
                        near_plane=cfg.near_plane,
                        far_plane=cfg.far_plane,
                        render_mode="RGB+ED",
                        distloss=self.cfg.dist_loss,
                    )  # [1, H, W, 3]
                    colors = torch.clamp(colors, 0.0, 1.0)
                    colors = colors[..., :3]  # Take RGB channels
                    yc_hat.append(colors)

                    imageio.imwrite(
                        f"{self.render_dir}/val_{i:04d}_sample_{rep}_{step}.png",
                        (colors.squeeze(0).cpu().numpy() * 255).astype(np.uint8),
                    )

                image_stack = torch.stack(yc_hat, dim=-1)  # [1, H, W, 3, S]

                # write mean images
                color_mean = image_stack.mean(dim=-1)  # [1, H, W, 3]
                imageio.imwrite(
                    f"{self.render_dir}/val_{i:04d}_mean_{step}.png",
                    (color_mean.squeeze(0).cpu().numpy() * 255).astype(np.uint8),
                )

                # write error images
                colors_err = (color_mean - pixels).abs()  # [1, H, W, 3]
                mono_err = normalize_variance_image(colors_err)  # [1, H, W, 1]
                render_err = (
                    apply_float_colormap(mono_err).detach().cpu().squeeze(0).numpy()
                )  # [1, H, W, 3]
                imageio.imwrite(
                    f"{self.render_dir}/val_{i:04d}_err_{step}.png",
                    (render_err * 255).astype(np.uint8),
                )

                # write variance images
                colors_var = image_stack.var(dim=-1)  # [1, H, W, 3]
                mono_var = normalize_variance_image(colors_var)  # [1, H, W, 1]
                render_var = (
                    apply_float_colormap(mono_var).detach().cpu().squeeze(0).numpy()
                )  # [1, H, W, 3]
                imageio.imwrite(
                    f"{self.render_dir}/val_{i:04d}_var_{step}.png",
                    (render_var * 255).astype(np.uint8),
                )

                # Plot diversity histogram.
                import matplotlib.pyplot as plt

                colors_div = var2div(colors_var.mean(dim=-1))
                colors_div = colors_div.detach().cpu()
                hist, bin_edges = torch.histogram(
                    colors_div, bins=100, range=(0.0, 0.05)
                )
                fig, ax = plt.subplots(figsize=(4.5, 3))

                ax.set_axisbelow(True)
                ax.minorticks_on()
                ax.grid(which="major", alpha=0.75)
                ax.grid(which="minor", axis="x", alpha=0.5)
                ax.bar(bin_edges[:-1], hist, width=bin_edges[1] - bin_edges[0])
                ax.set_xlabel("Diversity")
                ax.set_ylabel("Frequency")

                fig.tight_layout()
                fig.savefig(f"{self.render_dir}/val_{i:04d}_divhist_{step}.png")
                plt.close(fig)

                # Close VI for rendering the final image.
                self.is_vi = False

            torch.cuda.synchronize()
            tic = time.time()
            (
                colors,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                _,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 3]
            colors = torch.clamp(colors, 0.0, 1.0)
            colors = colors[..., :3]  # Take RGB channels
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_{step}.png",
                (canvas * 255).astype(np.uint8),
            )

            # write median depths
            render_median = (render_median - render_median.min()) / (
                render_median.max() - render_median.min()
            )
            # render_median = render_median.detach().cpu().squeeze(0).unsqueeze(-1).repeat(1, 1, 3).numpy()
            render_median = (
                apply_float_colormap(render_median).detach().cpu().squeeze(0).numpy()
            )

            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_median_depth_{step}.png",
                (render_median * 255).astype(np.uint8),
            )

            # write normals
            normals = (normals * 0.5 + 0.5).squeeze(0).cpu().numpy()
            normals_output = (normals * 255).astype(np.uint8)
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_normal_{step}.png", normals_output
            )

            # write normals from depth
            normals_from_depth *= alphas.squeeze(0).detach()
            normals_from_depth = (normals_from_depth * 0.5 + 0.5).cpu().numpy()
            normals_from_depth = (normals_from_depth - np.min(normals_from_depth)) / (
                np.max(normals_from_depth) - np.min(normals_from_depth)
            )
            normals_from_depth_output = (normals_from_depth * 255).astype(np.uint8)
            if len(normals_from_depth_output.shape) == 4:
                normals_from_depth_output = normals_from_depth_output.squeeze(0)
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_normals_from_depth_{step}.png",
                normals_from_depth_output,
            )

            # write distortions

            render_dist = render_distort
            dist_max = torch.max(render_dist)
            dist_min = torch.min(render_dist)
            render_dist = (render_dist - dist_min) / (dist_max - dist_min)
            render_dist = (
                apply_float_colormap(render_dist).detach().cpu().squeeze(0).numpy()
            )
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_distortions_{step}.png",
                (render_dist * 255).astype(np.uint8),
            )

            pixels = pixels.permute(0, 3, 1, 2)  # [..., C, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [..., C, H, W]
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))
            if cfg.model_type == "esgs" and step >= cfg.switch_to_vi:
                image_stack = image_stack.permute(0, 3, 1, 2, 4)

                sc_rmse = EnhancedSparsificationCurves(
                    image_stack,
                    pixels,
                    diff_to_error_fn=lambda x: x.square().mean(dim=-3),
                    summarizer_fn=lambda x: x.mean(dim=-1).sqrt().mean(),
                )
                metrics["ause_rmse"].append(sc_rmse.compute_AUSC())
                sc_rmse.save_sc(
                    f"{self.render_dir}/val_{i:04d}_sc_rmse_{step}.png",
                    plot_random=True,
                )
                sc_rmse.save_sc(
                    f"{self.render_dir}/val_{i:04d}_sc_rmse_raw_{step}.png",
                    plot_random=True,
                    normalize=False,
                    ylim=0.04,
                )

                sc_mae = EnhancedSparsificationCurves(
                    image_stack,
                    pixels,
                    diff_to_error_fn=lambda x: x.abs().mean(dim=-3),
                    summarizer_fn=lambda x: x.mean(dim=-1).mean(),
                )
                metrics["ause_mae"].append(sc_mae.compute_AUSC())
                sc_mae.save_sc(
                    f"{self.render_dir}/val_{i:04d}_sc_mae_{step}.png", plot_random=True
                )
                sc_mae.save_sc(
                    f"{self.render_dir}/val_{i:04d}_sc_mae_raw_{step}.png",
                    plot_random=True,
                    normalize=False,
                    ylim=0.03,
                )
        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        if cfg.model_type == "esgs" and step >= cfg.switch_to_vi:
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
        # save stats as json
        stats = (
            {
                "psnr": psnr.item(),
                "ssim": ssim.item(),
                "lpips": lpips.item(),
                "ellipse_time": ellipse_time,
                "num_GS": len(self.splats["means"]),
            }
            if not (cfg.model_type == "esgs" and step >= cfg.switch_to_vi)
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
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds = self.parser.camtoworlds[5:-5]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _, surf_normals, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            surf_normals = (surf_normals - surf_normals.min()) / (
                surf_normals.max() - surf_normals.min()
            )

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

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

        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            render_mode="RGB+ED",
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "depth":
            # normalize depth to [0, 1]
            depth = render_median
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
        elif render_tab_state.render_mode == "normal":
            render_normals = render_normals * 0.5 + 0.5  # normalize to [0, 1]
            renders = render_normals.cpu().numpy()
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        else:
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        return renders


def main(cfg: Config):
    runner = Runner(cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
