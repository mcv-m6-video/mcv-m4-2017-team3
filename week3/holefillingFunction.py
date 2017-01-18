import numpy as np
from scipy import ndimage



def holefilling(image, fourConnectivity = True):
    inputImage = np.copy(image)
    h_max = np.max(inputImage * 2.0)
    data_mask = np.isfinite(inputImage)
    inside_mask = ndimage.binary_erosion(data_mask, structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.bool))
    edge_mask = (data_mask & ~inside_mask)

    outputImage = np.copy(inputImage)
    outputImage[inside_mask] = h_max

    # Array for storing previous iteration
    outputOldImage = np.copy(inputImage)
    outputOldImage.fill(0)

    # Connectivity structuring element
    if fourConnectivity:
        el = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.bool)
    else:
        el = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.bool)

    while not np.array_equal(outputOldImage , outputImage):
        outputOldImage = np.copy(outputImage)
        outputImage = np.maximum(inputImage, ndimage.grey_erosion(outputImage, size=(3, 3), footprint=el))

    return outputImage