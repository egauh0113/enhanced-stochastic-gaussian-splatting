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

import math

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.overrides import handle_torch_function, has_torch_function_variadic


class LaplaceNLLLoss(_Loss):
    r"""Laplace negative log likelihood loss.

    The targets are treated as samples from Laplace distributions with
    means and diversities predicted by the neural network. For a
    ``target`` tensor modelled as having Laplace distribution with a tensor
    of locations ``input`` and a tensor of positive diversities ``div``, the loss is:

    .. math::
        \text{loss} = \log\left(\text{max}\left(\text{div}, \text{eps}\right)\right)
        \ + \frac{|y - \mu|}{\text{max}\left(\text{div}, \text{eps}\right)} + \text{const.}
    where :attr:`eps` is used for stability. By default, the constant term of
    the loss function is omitted unless :attr:`full` is ``True``. If ``div`` is not the same
    size as ``input`` (due to a homoscedastic assumption), it must either have a final dimension
    of 1 or have one fewer dimension (with all other sizes being the same) for correct broadcasting.

    Args:
        full (bool, optional): include the constant term in the loss
            calculation. Default: ``False``.
        eps (float, optional): value used to clamp ``div`` (see note below), for
            stability. Default: 1e-6.
        reduction (str, optional): specifies the reduction to apply to the
            output:``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the output is the average of all batch
            member losses, ``'sum'``: the output is the sum of all batch member
            losses. Default: ``'mean'``.

    Shape:
        - Input: :math:`(N, *)` or :math:`(*)` where :math:`*` means any number of additional
          dimensions
        - Target: :math:`(N, *)` or :math:`(*)`, same shape as the input, or same shape as the input
          but with one dimension equal to 1 (to allow for broadcasting)
        - Diversity: :math:`(N, *)` or :math:`(*)`, same shape as the input, or same shape as the input but
          with one dimension equal to 1, or same shape as the input but with one fewer
          dimension (to allow for broadcasting), or a scalar value
        - Output: scalar if :attr:`reduction` is ``'mean'`` (default) or
          ``'sum'``. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same
          shape as the input

    Examples:
        TBD

    Note:
        The clamping of ``div`` is ignored with respect to autograd, and so the
        gradients are unaffected by it.

    Reference:
        TBD
    """

    __constants__ = ["full", "eps", "reduction"]
    full: bool
    eps: float

    def __init__(
        self, *, full: bool = False, eps: float = 1e-6, reduction: str = "mean"
    ) -> None:
        super().__init__(None, None, reduction)
        self.full = full
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor, div: Tensor | float) -> Tensor:
        """
        Runs the forward pass.
        """
        return laplace_nll_loss(
            input, target, div, full=self.full, eps=self.eps, reduction=self.reduction
        )


def laplace_nll_loss(
    input: Tensor,
    target: Tensor,
    div: Tensor | float,
    full: bool = False,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the Laplace negative log likelihood loss.

    See :class:`~laplace_nll_loss.LaplaceNLLLoss` for details.

    Args:
        input: Location of the Laplace distribution.
        target: Sample from the Laplace distribution.
        div: Tensor of positive diversity parameter(s), one for each of the locations
            in the input (heteroscedastic), or a single one (homoscedastic),
            or a positive scalar value to be used for all locations.
        full (bool, optional): Whether to include the constant term in the loss calculation. Default: ``False``.
        eps (float, optional): Value added to div, for stability. Default: 1e-6.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """
    if has_torch_function_variadic(input, target, div):
        return handle_torch_function(
            laplace_nll_loss,
            (input, target, div),
            input,
            target,
            div,
            full=full,
            eps=eps,
            reduction=reduction,
        )

    # Entries of div must non-negative
    if isinstance(div, float):
        if div < 0:
            raise ValueError("div has negative entry/entries")
        div = div * torch.ones_like(input)
    elif torch.any(div < 0):
        raise ValueError("div has negative entry/entries")

    # Check div size
    # If div.size == input.size, the case is heteroscedastic and no further checks are needed.
    # Otherwise:
    if div.size() != input.size():
        # If div is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
        # e.g. input.size = (10, 2, 3), div.size = (10, 2)
        # -> unsqueeze div so that div.shape = (10, 2, 1)
        # this is done so that broadcasting can happen in the loss calculation
        if input.size()[:-1] == div.size():
            div = torch.unsqueeze(div, -1)

        # This checks if div is broadcastable to the input and there is only one mismatched dimension.
        # This is also a homoscedastic case.
        # e.g. input.size = (10, 2, 3), div.size = (10, 2, 1)
        # or  input.size = (4, 3, 32, 32), div.size = (4, 1, 32, 32)
        elif (
            input.ndim == div.ndim
            and sum(y for x, y in zip(input.size(), div.size(), strict=True) if x != y)
            == 1
        ):  # Heteroscedastic case
            pass

        # If none of the above pass, then the size of div is incorrect.
        else:
            raise ValueError("div is of incorrect size")

    # Check validity of reduction method
    if reduction != "none" and reduction != "mean" and reduction != "sum":
        raise ValueError(reduction + " is not valid")

    # Clamp for stability
    div = div.clone()
    with torch.no_grad():
        div.clamp_(min=eps)
    # Calculate the loss
    # pyrefly: ignore [unsupported-operation]
    loss = torch.log(div) + torch.abs(input - target) / div
    if full:
        loss += math.log(2.0)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def calc_div(img: Tensor, eps: float = 1e-6) -> Tensor:
    """Calculate diversity for Laplace NLL loss using the mean absolute deviation from the median.

    Args:
        img: [..., S] Tensor of samples.
        eps: Small value for numerical stability.

    Returns:
             [...,] Tensor of diversity values.
    """
    # image_stack: [1, H, W, 3, S]
    median = img.median(dim=-1).values  # [1, H, W, 3]
    div = torch.mean(torch.abs(img - median.unsqueeze(-1)), dim=-1).clamp(min=eps)  # [1, H, W, 3]
    return div


def var2div(var: Tensor, eps: float = 1e-6) -> Tensor:
    """Convert variance to diversity for Laplace NLL loss.

    Args:
        var: [...,] Variance tensor.
        eps: Small value for numerical stability.

    Returns:
             [...,] Diversity tensor.
    """
    return torch.sqrt(var.clamp(min=eps) * 0.5)
