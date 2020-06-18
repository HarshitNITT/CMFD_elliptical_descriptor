import numpy as np
from gaussian_filter import gaussianFilter
from scipy.ndimage.filters import convolve

def createOcatave(firstLevel, s, sigma):
	octave = [firstLevel]
	#generating different ocataves to form pyramid
	k = 2**(1/s)
	gaussianKernel = gaussianFilter(k * sigma)

	for i in range(s+2):
		nextLevel = convolve(octave[-1], gaussianKernel)
		
		octave.append(nextLevel)

	return octave

def createPyramid(image, octaveNum, s, sigma):
    pyramid = []
    #generating different ocataves pyramid
    for _ in range(octaveNum):
        octave = createOcatave(image, s, sigma)
        pyramid.append(octave)

        image = octave[-3][::2, ::2]

    return pyramid