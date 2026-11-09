import numpy as np

NAMES = [f"{axis}_{stat}" for axis in "xyz" for stat in ("rms", "std", "range", "absmax")]


def features(window):
    rms = np.sqrt(np.mean(np.square(window), axis=0))
    return np.column_stack((rms, window.std(axis=0), np.ptp(window, axis=0),
                            np.abs(window).max(axis=0))).ravel()


def magnitude(features_matrix):
    return np.linalg.norm(features_matrix[:, ::4], axis=1).reshape(-1, 1)


# Each 0.5 s window has 2 Hz FFT spacing. Mean removal excludes the DC offset.
BAND_STARTS = (2, 25, 50, 100, 200, 400, 800, 1600)
BAND_ENDS = (*BAND_STARTS[1:], 3200)
SPECTRAL_NAMES = [
    f"{axis}_power_share_{lo}_{hi}Hz"
    for axis in "xyz" for lo, hi in zip(BAND_STARTS, BAND_ENDS)
]


def band_powers(window):
    from .data import HZ
    centered = window - window.mean(axis=0)
    taper = np.hanning(len(window))[:, None]
    spectrum = np.fft.rfft(centered * taper, axis=0)
    power = np.abs(spectrum) ** 2
    power[1:-1] *= 2.0
    freqs = np.fft.rfftfreq(len(window), 1 / HZ)
    # The last band includes the Nyquist bin. All other right edges are open.
    starts = np.searchsorted(freqs, BAND_STARTS)
    bands = np.add.reduceat(power, starts, axis=0)
    total = bands.sum(axis=0)
    shares = np.divide(bands, total, out=np.zeros_like(bands), where=total > 0)
    return shares.T.ravel()
