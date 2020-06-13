__all__ = ["mcmc", "convergetest", "credregion", "sig", "ess"]

from .mcmc import mcmc
from .gelman_rubin import convergetest
from .credible_region import credregion, sig, ess

# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__ ):
        del locals()[varname]
del(varname)
