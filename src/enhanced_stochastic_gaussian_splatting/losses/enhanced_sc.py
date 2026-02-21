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

from .sparsification_curves import SparsificationCurves


class EnhancedSparsificationCurves(SparsificationCurves):
    def compute_AUSC(self, k: str = "us") -> torch.Tensor:
        r"""Compute Area Under the Sparsification Curve (AUSC).

        Args:
            k: Key of the prediction to evaluate.
        """
        sc = self._get_sparsification_curve(self.uncertainties[k], self.errors[k])
        ausc = self._compute_area_under_sparsification_curve(sc).abs()
        return ausc

    def save_sc(
        self,
        filename: str,
        k: str = "us",
        plot_random: bool = False,
        normalize: bool = True,
        ylim: float | None = None,
    ) -> None:
        r"""Save sparsification curve plot.

        Args:
            filename:    Path to save the plotted graph.
            k:           Key of the prediction to plot.
            plot_random: Whether to plot the random removal curve for reference.
            normalize:   Whether to normalize the curves by the maximum oracle error.
            ylim:        y-axis limit for the plot.
                         If None, it will be set 120% of the maximum value of oracle curve.
        """
        import matplotlib.pyplot as plt

        # Normalized plot.
        fig, ax = plt.subplots(figsize=(4.5, 3))
        x = self.percentage_ax.cpu().numpy() / self.HW

        # Calculate curves.
        oracle = self._get_sparsification_curve(self.errors[k], self.errors[k])
        sc = self._get_sparsification_curve(self.uncertainties[k], self.errors[k])

        ratio = max(oracle).item() if normalize else 1.0

        # Random curve.
        if plot_random:
            random = self._get_sparsification_curve(
                torch.rand_like(self.errors[k]), self.errors[k]
            )
            ax.plot(
                x,
                random.cpu().numpy() / ratio,
                label="Random",
                linestyle=":",
                color="red",
            )

        # Sparsification curve.
        ax.plot(
            x,
            sc.cpu().numpy() / ratio,
            label="Sparsification",
            linestyle="--",
            color="red",
        )

        # Oracle curve.
        ax.plot(
            x,
            oracle.cpu().numpy() / ratio,
            label="Oracle",
            linestyle="-",
            color="red",
        )

        ax.set_xlabel("Ratio of Removed Pixels")
        ax.set_ylabel("Normalized Error" if normalize else "Error")
        ax.legend(loc="lower left")
        ax.set_xlim(0, 1)
        ax.set_ylim(
            0,
            (
                ylim
                if ylim is not None
                else (1.2 if normalize else 1.2 * max(oracle).item())
            ),
        )
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)

        return
