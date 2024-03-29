"""
Random helpers
"""

from __future__ import annotations
import torch
import math
import warnings


def no_grad_trunc_normal_(tensor: torch.Tensor, mean, std, a, b, *, generator=None):
    """
    Code copied and adopted from torch.nn.init._no_grad_trunc_normal_,
    to support the extra `generator` argument (https://github.com/pytorch/pytorch/issues/98200).

    Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

    :param tensor:
    :param mean:
    :param std:
    :param a:
    :param b:
    :param generator:
    :return: tensor
    """

    def _norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        _lower = _norm_cdf((a - mean) / std)
        _upper = _norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2*lower-1, 2*upper-1].
        tensor.uniform_(2 * _lower - 1, 2 * _upper - 1, generator=generator)  # noqa

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
