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

import torch


USE_EXP_ON_DIAG = True

TRIL_IDXS = {
    1: torch.tril_indices(1, 1),
    2: torch.tril_indices(2, 2),
    3: torch.tril_indices(3, 3),
}


class Vec2Lower(torch.nn.Module):
    def __init__(self, D: int) -> None:
        super().__init__()
        self.D = D
        self.register_buffer("tril_idx", torch.tril_indices(self.D, self.D))

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Convert a vector representation of a lower-triangular matrix to the matrix form.

        Args:
            params: [..., D(D+1)/2] Vector representation of lower-triangular matrix.

        Returns:
            L:      [..., D, D] Lower-triangular matrix.
        """
        *batch_dims, _ = params.shape
        L = params.new_zeros(*batch_dims, self.D, self.D)

        if USE_EXP_ON_DIAG:
            idx = 0
            for i in range(self.D):
                for j in range(i + 1):
                    if i == j:
                        L[..., i, j] = torch.exp(params[..., idx])
                    else:
                        L[..., i, j] = params[..., idx]
                    idx += 1
        else:
            L[..., self.tril_idx[0], self.tril_idx[1]] = params
        return L


def init_cholesky_params(diag_vals: list, device="cpu") -> torch.Tensor:
    """Initialize [D(D+1)/2] Cholesky parameters given diagonal values.

    Args:
        diag_vals: [D] diagonal Entries of covariance.
        device:    Torch device.

    Returns:
        params:    [D(D+1)/2] Cholesky parameters.
    """
    D = len(diag_vals)
    num_params = D * (D + 1) // 2

    params = torch.zeros(num_params, dtype=torch.float32, device=device)
    diag_indices = [i * (i + 1) // 2 + i for i in range(D)]

    # params[diag_indices] = (
    #     torch.log(torch.tensor(diag_vals, dtype=torch.float32, device=device))
    #     if USE_EXP_ON_DIAG
    #     else torch.tensor(diag_vals, dtype=torch.float32, device=device)
    # )
    params[diag_indices] = (
        torch.log(torch.tensor(diag_vals, dtype=torch.float32, device=device)) * 0.5
        if USE_EXP_ON_DIAG
        else torch.sqrt(torch.tensor(diag_vals, dtype=torch.float32, device=device))
    )

    return params


def algebra2quaternion(w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert so(3) Lie algebra vector to quaternion.

    Args:
        w: [..., 3] so(3) Lie algebra vector.
        eps: Small value for numerical stability.

    Returns:
        q: [..., 4] Quaternion. (w, x, y, z)
    """
    theta = w.norm(dim=-1)
    q = torch.zeros(w.shape[:-1] + (4,), device=w.device, dtype=w.dtype)

    small = theta < eps
    large = ~small

    q[small, 0] = 1.0
    q[small, 1:] = 0.5 * w[small]

    theta_half = 0.5 * theta[large]
    q[large, 0] = torch.cos(theta_half)
    q[large, 1:] = (w[large] / theta[large].unsqueeze(-1)) * torch.sin(
        theta_half
    ).unsqueeze(-1)

    return q


def quaternion2algebra(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert quaternion to so(3) Lie algebra.

    Args:
        q: [..., 4] Quaternion. (w, x, y, z)
        eps: Small value for numerical stability.

    Returns:
        w: [..., 3] so(3) Lie algebra vector.
    """
    qw = q[..., 0]
    qv = q[..., 1:]
    qv_norm = qv.norm(dim=-1)

    theta = 2.0 * torch.atan2(qv_norm, qw)
    w = torch.zeros_like(qv)

    small = qv_norm < eps
    large = ~small

    w[small] = 2.0 * qv[small]
    w[large] = qv[large] * (theta[large] / qv_norm[large]).unsqueeze(-1)

    return w


def normalize_variance_image(var_image: torch.Tensor) -> torch.Tensor:
    """Normalize variance image to [0, 1] range for visualization.

    Args:
        var_image: [..., H, W, 3] Variance image.

    Returns:
        normalized_var: [..., H, W, 1] Normalized variance image.
    """
    mono_var = var_image.mean(dim=-1, keepdim=True)  # [..., H, W, 1]
    var_min = mono_var.min()
    var_max = mono_var.max()
    normalized_var = (mono_var - var_min) / (var_max - var_min + 1e-8)
    return normalized_var
