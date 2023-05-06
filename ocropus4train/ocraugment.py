import ocrodeg
import scipy
import scipy.ndimage
import scipy.ndimage as ndi
from ocrlib.ocrhelpers import *


def maybe(p):
    return random.uniform(0, 1) < p


def aniso(a):
    aniso = 1.0 + random.uniform(-0.05, 0.05)
    return ocrodeg.transform_image(a, aniso=aniso)


def distort(a):
    sigma = 10 ** random.uniform(-0.3, 1.0)
    mag = 10 ** random.uniform(-0.1, 0.5)
    noise = ocrodeg.bounded_gaussian_noise(a.shape, sigma, mag)
    return ocrodeg.distort_with_noise(a, noise)


def normalize(a):
    a = a.astype(np.float32)
    a -= np.amin(a)
    a /= max(1e-3, np.amax(a))
    return a


def height_normalize(a, h):
    zoom = float(h) / a.shape[0]
    if zoom < 1.0:
        a = ndi.zoom(a, zoom, order=1)
    return a.clip(0, 1)


def autoinvert(a):
    lo, hi = np.amin(a), np.amax(a)
    if a.mean() > (lo + hi) / 2.0:
        a = 1.0 - a
    return a


def make_noise(shape, sigmas):
    result = None
    for sigma, mag in sigmas:
        noise = np.random.normal(0, 1.0, shape)
        noise = scipy.ndimage.gaussian_filter(noise, sigma)
        noise /= np.amax(np.abs(noise))
        if result is None:
            result = noise * mag
        else:
            result += noise * mag
    result /= np.amax(np.abs(result))
    return result


def threshold(a):
    a = normalize(a)
    return (a > 0.5).astype(np.float32)


def noisify(a, noise=[(0.1, 0.5), (1.0, 1.0), (5.0, 0.5)]):
    a = a.astype(np.float32)
    a = normalize(a)
    level = random.uniform(0.05, 0.4)
    a += make_noise(a.shape, noise) * level
    a = normalize(a)
    return a
