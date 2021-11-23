import numpy as np


def rms(signal: np.ndarray):
    return np.sqrt(np.mean(np.power(signal, 2)))


def db2mag(magnitude_db: np.ndarray):
    return np.power(10, magnitude_db / 20)


def mag2db(magnitude: np.ndarray):
    return 20 * np.log10(magnitude)
