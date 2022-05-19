from PIL import Image
import numpy as np
from enum import Enum

class ConversionType(Enum):
    Linearization = 0
    Nonlinearization = 1


matrixChromaticAdaptationD65ToD50 = [
    [1.06411421, 0.09733737, 0.01444089],
    [-0.00467724, 0.97567698, 0.0047212 ],
    [-0.00639813, -0.02213695, 0.750636  ]]


matrixChromaticAdaptationD65ToD70 = [
    [0.97343386, -0.03742433, -0.00698047],
    [0.00177458, 1.00727875, -0.00261299],
    [0.00325271, 0.01176267, 1.13034218]]

imageBitdepth = 8


def Clip(pixel, min, max):
    return min if pixel < min else (max if pixel > max else pixel)


def ConvertSRGBGamma(conversionType, value, bitdepth):
    if (conversionType == ConversionType.Linearization):
        value = Clip(value, 0, 2 ** bitdepth - 1)
        value /= 2 ** bitdepth - 1
        if (value <= 0.04045):
            returnValue = 25 * value / 323
        else:
            returnValue = (200 * value + 11) / 211
            returnValue = returnValue ** (12/5)
    elif (conversionType == ConversionType.Nonlinearization):
        value = Clip(value, 0.0, 1.0)
        if (value <= 0.0031308):
            returnValue = 323 * value / 25
        else:
            returnValue = 211 * (value ** (5/12)) - 11
            returnValue /= 200
        returnValue *= 2 ** bitdepth - 1
        returnValue = int(returnValue)
    else:
        returnValue = 0
    return returnValue
    

def ConvertSRGBGammaRGB(conversionType, RGB, bitdepth):
    returnR = ConvertSRGBGamma(conversionType, RGB[0], bitdepth)
    returnG = ConvertSRGBGamma(conversionType, RGB[1], bitdepth)
    returnB = ConvertSRGBGamma(conversionType, RGB[2], bitdepth)
    return np.array([returnR, returnG, returnB], dtype=float)


def MultiplexMatrix(image, matrix):
    imageBuffer = Image.new('RGB', image.size) # it generates the PIL.Image.Image object
    for y_pos in range(image.size[1]):
        for x_pos in range(image.size[0]):
            rgb = image.getpixel((x_pos, y_pos)) # the cordinate is type, it returns pixel value of tuple
            linearRGB = ConvertSRGBGammaRGB(ConversionType.Linearization, rgb, imageBitdepth)
            linearRGBPostMatrixManipulation = matrix @ linearRGB
            rgbPostMatrixMultiplex = ConvertSRGBGammaRGB(ConversionType.Nonlinearization, linearRGBPostMatrixManipulation, imageBitdepth)
            imageBuffer.putpixel((x_pos, y_pos), (int(rgbPostMatrixMultiplex[0]), int(rgbPostMatrixMultiplex[1]), int(rgbPostMatrixMultiplex[2]))) # the cordinate is type, the pixel value is tuple
    return imageBuffer


def main():
    filename = 'lena_512x512.bmp'
    imageFromFile = Image.open(filename) # it generates the PIL.Image.Image object
    matrixSet = [matrixChromaticAdaptationD65ToD50, matrixChromaticAdaptationD65ToD70]
    for matrix in matrixSet:
        imageBuffer = MultiplexMatrix(imageFromFile, matrix)
        imageBuffer.show()


if __name__ == "__main__":
    main()