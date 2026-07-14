"""Exceptions used across the DFTR-Filter implementation."""


class IncorrectConstantsException(Exception):
    """Raised when algorithm constants violate their documented constraints."""


class IncorrectInputException(Exception):
    """Raised when user-supplied problem data (x0, bounds, ...) is inconsistent."""


class FailedSimulation(Exception):
    """Raised when the user's cost/constraint function returns NaN/Inf."""


class IllPoisedModel(Exception):
    """Raised when the interpolation system is singular/too ill-conditioned to trust."""


class PoisednessIsZeroException(Exception):
    """Raised when every Lagrange polynomial vanishes identically on the ball (duplicate points)."""


class RestorationFailed(Exception):
    """Raised when Restore certifies an infeasible stationary point (Theorem 1, case (a))."""
