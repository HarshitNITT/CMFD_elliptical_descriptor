from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve

from gaussian_filter import gaussian_filter
from gaussian_pyramid import createPyramid
from DOG_Pyramid import differenceOfGaussian_Pyramid
from keypoints import get_keypoints
from orientation import orientationAssignment
from descriptors import get_local_descriptors

class Improved_SIFT(object):
    def __init__(self, im, s=3, num_octave=4, s0=1.3, sigma=1.6, r_th=10, t_c=0.03, w=16):
        self.im = convolve(rgb2gray(im), gaussian_filter(s0))
        self.s = s
        self.sigma = sigma
        self.num_octave = num_octave
        self.t_c = t_c
        self.R_th = (r_th+1)**2 / r_th
        self.w = w

    def get_features(self):
        gaussian_pyr = createPyramid(self.im, self.num_octave, self.s, self.sigma)
        DOG_Pyramid = differenceOfGaussian_Pyramid(gaussian_pyr)
        kp_pyr = get_keypoints(DOG_Pyramid, self.R_th, self.t_c, self.w)
        feats = []

        for i, DoG_octave in enumerate(DOG_Pyramid):
            kp_pyr[i] = orientationAssignment(kp_pyr[i], DoG_octave)
            feats.append(get_local_descriptors(kp_pyr[i], DoG_octave))

        self.kp_pyr = kp_pyr
        self.feats = feats

        return feats
