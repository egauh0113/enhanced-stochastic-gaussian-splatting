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

from ..utils.transforms import Vec2Lower, quaternion2algebra as q2a


class KullbackLeiblerDivergenceMultivariateNormalLower(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        mu_q: torch.Tensor,
        v_q: torch.Tensor,
        mu_p: torch.Tensor,
        v_p: torch.Tensor,
        center_free: bool = False,
    ) -> torch.Tensor:
        """Compute KL divergence D_KL(Q||P) where posterior Q ~ N(mu_q, sigma_q)
                                                  and prior P ~ N(mu_p, sigma_p).
            D_KL(Q||P) = 0.5 * (log(|Σp|/|Σq|) - D + tr(Σp.inv@Σq) + (μp-μq).T@Σp.inv@(μp-μq))
        Args:
            mu_q:   [..., D] Mean of distribution Q.
            v_q:    [..., D(D+1)/2] Lower-triangular Cholesky factor of covariance of distribution Q.
            mu_p:   [..., D] Mean of distribution P.
            v_p:    [..., D(D+1)/2] Lower-triangular Cholesky factor of covariance of distribution P.
        Returns:
            kl_div: [...,] KL divergence D_KL(Q||P).
        """
        D = 1 if mu_q.dim() == 1 else mu_q.shape[-1]

        # Lower-triangular Cholesky factors.
        v2l = Vec2Lower(D)
        L_q = v2l(v_q)
        L_p = v2l(v_p)

        # Stabilization.
        eps = torch.diagonal(L_q, dim1=-2, dim2=-1)
        L_q = L_q + torch.diag_embed(1e-6 * torch.ones_like(eps))
        L_p = L_p + torch.diag_embed(1e-6 * torch.ones_like(eps))

        # Log-det term: log(|L_p|/|L_q|) = 2 * sum(log(diag(L_p))) - 2 * sum(log(diag(L_q)))
        logdet_term = 2 * (
            torch.sum(torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1)), dim=-1)
            - torch.sum(torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1)), dim=-1)
        )

        # Trace term: solve_triangular(L_p, L_q) -> L_p^-1 @ L_q
        # trace_term = torch.sum(
        #     torch.linalg.solve_triangular(L_p, L_q, upper=False) ** 2, dim=[-2, -1]
        # )
        trace_term = torch.sum((torch.linalg.inv(L_p) @ L_q) ** 2, dim=[-2, -1])

        if not center_free:
            # Quadratic term.
            mu_diff = (
                (mu_p - mu_q).unsqueeze(-1).unsqueeze(-1)
                if D == 1
                else (mu_p - mu_q).unsqueeze(-1)
            )
            # quad_term = torch.sum(
            #     torch.linalg.solve_triangular(L_q, mu_diff, upper=False) ** 2, dim=[-2, -1]
            # )
            quad_term = torch.sum((torch.linalg.inv(L_q) @ mu_diff) ** 2, dim=[-2, -1])

            kl_div = 0.5 * (logdet_term - D + trace_term + quad_term)
        else:
            kl_div = 0.5 * (logdet_term - D + trace_term)

        return kl_div


class KullbackLeiblerDivergenceSO3NormalBCHLower(
    KullbackLeiblerDivergenceMultivariateNormalLower
):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        mu_q: torch.Tensor,
        v_q: torch.Tensor,
        mu_p: torch.Tensor,
        v_p: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute KL divergence D_KL(Q||P) where posterior Q ~ Normal(mu_q in so(3), sigma_q)
                                                  and prior P ~ Normal(mu_p in so(3), sigma_p).
            D_KL(Q||P) = 0.5 * (log(|Σp|/|Σq|) - 3 + tr(Σp.inv@Σq)
                                + (μq-Jlv_q@μp).T@Σp.inv@(μq-Jlv_q@μp))
        Args:
            mu_q:   [..., 4] Mean of distribution Q in quaternion.
            v_q:    [..., 6] Lower-triangular Cholesky factor of covariance of distribution Q.
            mu_p:   [..., 4] Mean of distribution P in quaternion.
            v_p:    [..., 6] Lower-triangular Cholesky factor of covariance of distribution P.
        Returns:
            kl_div: [...,] KL divergence D_KL(Q||P).
        """
        w_q = q2a(mu_q)
        w_p = q2a(mu_p)

        # Hat operation.
        w_p_hat = torch.zeros(
            w_p.shape[:-1] + (3, 3), device=w_p.device, dtype=w_p.dtype
        )
        w_p_hat[..., 0, 1] = -w_p[..., 2]
        w_p_hat[..., 0, 2] = w_p[..., 1]
        w_p_hat[..., 1, 0] = w_p[..., 2]
        w_p_hat[..., 1, 2] = -w_p[..., 0]
        w_p_hat[..., 2, 0] = -w_p[..., 1]
        w_p_hat[..., 2, 1] = w_p[..., 0]

        w_p_norm = w_p.norm(dim=-1)
        large = w_p_norm >= eps

        # Inverse left Jacobian of SO(3) group.
        I3 = torch.eye(3, device=w_p.device, dtype=w_p.dtype)
        J_l_inv_p = I3.expand(*w_p.shape[:-1], *I3.shape).clone()

        J_l_inv_p[large] = (
            I3
            - 0.5 * w_p_hat[large]
            + (
                1.0 / (w_p_norm[large] ** 2)
                - (1 + torch.cos(w_p_norm[large]))
                / (2 * w_p_norm[large] * torch.sin(w_p_norm[large]))
            )[..., None, None]
            * (w_p_hat[large] @ w_p_hat[large])
        )
        w_diff = w_p - (J_l_inv_p @ w_q.unsqueeze(-1)).squeeze(-1)

        return super().forward(w_diff, v_q, torch.zeros_like(w_diff), v_p)
