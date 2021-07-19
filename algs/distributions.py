import math
import torch
from torch import distributions as pyd
from numbers import Number



"""
distributions used for the policy
"""


class StableNormal(torch.distributions.Normal):
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        log_scale = math.log(self.scale) if isinstance(self.scale, Number) else self.scale.log()
        return - 0.5 * ((value - self.loc)  / self.scale )**2 - log_scale - math.log(math.sqrt(2 * math.pi))
        return output




class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1, threshold=20):
        super().__init__(cache_size=cache_size)
        self.softplus = torch.nn.Softplus(threshold=threshold)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2. * (math.log(2.) - x - self.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale, threshold=20, stable=False):
        self.loc = loc
        self.scale = scale

        if stable:
            self.base_dist = StableNormal(loc, scale)
        else:
            self.base_dist = pyd.Normal(loc, scale)

        transforms = [TanhTransform(threshold=threshold)]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
