import numpy as np

def differenceOfGaussian_Octave(Octave_layer):
    octave = []

    for i in range(1, len(Octave_layer)):
        octave.append(Octave_layer[i] - Octave_layer[i-1])

    return np.concatenate([o[:,:,np.newaxis] for o in octave], axis=2)

def differenceOfGaussian_Pyramid(Gaussian_Pyramid):
    pyr = []

    for Octave_in_pyramid in Gaussian_Pyramid:
        pyr.append(differenceOfGaussian_Octave(Octave_in_pyramid))

    return pyr
    