import numpy as np
from .data import SIZE


def quality(window):
    if window.shape != (SIZE, 3):
        return "BAD_SHAPE"
    if not np.isfinite(window).all():
        return "NONFINITE"
    if (np.ptp(window, axis=0) == 0).any():
        return "CONST_AXIS"
    return ""
