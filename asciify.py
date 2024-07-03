import math
import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance

WHITE = 224
ASCII = " `.',_-:~\"^!;/\\<>)(+|=v?L]7[x{tzcrTi}sYfl*jInuy4JF1aVoekh2XZwAC963PmKE%@U5pHqSbd#O&GDNRW$80gQBM"
VALUES = [
    0.0,
    4.44,
    6.44,
    7.54,
    9.97,
    10.4,
    11.32,
    12.85,
    13.28,
    15.47,
    15.6,
    16.17,
    16.36,
    19.49,
    19.71,
    19.88,
    19.88,
    20.78,
    20.82,
    21.09,
    22.08,
    23.72,
    23.82,
    24.05,
    24.56,
    25.85,
    26.15,
    26.24,
    26.35,
    26.4,
    26.53,
    26.53,
    26.62,
    26.63,
    26.69,
    26.74,
    27.09,
    27.29,
    27.92,
    28.18,
    28.29,
    28.76,
    29.47,
    29.77,
    30.06,
    30.09,
    30.09,
    30.58,
    30.97,
    31.21,
    31.45,
    31.9,
    32.19,
    32.94,
    33.67,
    33.88,
    34.88,
    35.22,
    35.62,
    36.12,
    36.76,
    37.22,
    37.38,
    37.46,
    37.71,
    38.46,
    38.71,
    38.95,
    39.17,
    39.24,
    39.33,
    39.76,
    39.92,
    40.41,
    40.74,
    40.97,
    40.99,
    41.18,
    41.4,
    41.62,
    42.32,
    42.96,
    43.76,
    44.27,
    44.36,
    44.67,
    45.14,
    45.97,
    47.19,
    47.27,
    47.33,
    47.58,
    48.56,
    48.97,
    49.38,
]
SOLIDASCII = " ░▒▓█"
SOLIDVALUES = [0.0, 22.02, 45.46, 127.68, 176.79]
EDGEASCII = "_/|\\_/|\\_"
COLORRESET = "\033[0m"
CHARASPECT = 0.5

def loadImage(path):
    image = Image.open(path)
    image = image.convert("RGB")
    # does nothing rn, change to adjust brightness
    # TODO: Add CLI brightness adjustment
    image = ImageEnhance.Brightness(image).enhance(1)
    return image

def downscaleImage(input, charsAcross):
    charsTall = charsAcross * CHARASPECT * input.height / input.width
    input = input.resize((int(charsAcross), int(charsTall)))
    return input

def printProgressBar(
    iteration,
    total,
    length=50,
):
    percent = round(100 * (iteration / float(total)), 2)
    filledLength = int(length * iteration // total)
    bar = "█" * filledLength + "-" * (length - filledLength)
    print(f"\r |{bar}| {percent}%", end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def valueLookup(value, useSolid=False):
    asciiSet = SOLIDASCII if useSolid else ASCII
    valueSet = SOLIDVALUES if useSolid else VALUES
    value = value / WHITE * max(valueSet)
    for i in range(len(valueSet)):
        if value < valueSet[i]:
            return asciiSet[i - 1]
    return asciiSet[-1]

def getColorEscaper(color):
    r = math.floor(color[0] / WHITE * 5)
    g = math.floor(color[1] / WHITE * 5)
    b = math.floor(color[2] / WHITE * 5)
    code = 36 * r + 6 * g + b + 16
    return f"\033[38;5;{math.floor(code)}m"

# TODO thin edges with thresholded downscaling
def getEdgeMap(image, size):
    image = image.astype(np.float32)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    sobelx = cv2.Scharr(image, cv2.CV_32F, 1, 0)
    sobely = cv2.Scharr(image, cv2.CV_32F, 0, 1)
    gradient = np.hypot(sobelx, sobely)
    angle = np.vectorize(
        lambda x, y: round((math.atan2(y, x) + math.pi) / math.pi * 4)
    )(sobely, sobelx)
    angle = np.where(gradient > np.max(gradient) * 0.1, angle, -1)

    angle = cv2.resize(angle, size, interpolation=cv2.INTER_NEAREST)
    
    return angle

def convertImage(image, charsAcross, useEdges=False, useColor=False, useSolid=False):
    downscaled = downscaleImage(image, charsAcross)
    edgeMap = getEdgeMap(np.array(image.convert("L")), (downscaled.width, downscaled.height))
    # plt.imshow(edgeMap)
    # plt.show()
    
    output = ""
    pixels = downscaled.getdata()
    lastColor = None
    for i in range(len(pixels)):
        if i % downscaled.width == 0:
            output += "\n"
        printProgressBar(i, len(pixels))
        if useColor:
            thisColor = getColorEscaper(pixels[i])
            if lastColor != thisColor:
                output += COLORRESET
                output += thisColor
                lastColor = thisColor

        if useEdges and edgeMap[i // downscaled.width][i % downscaled.width] >= 0:
            output += "\033[1m" 
            output += EDGEASCII[int(edgeMap[i // downscaled.width][i % downscaled.width])] 
            output += "\033[22m"
        else:
            output += valueLookup(np.array(pixels[i]).mean(), useSolid)
    output += COLORRESET
    return output

def main():
    args = sys.argv

    if not len(args) > 2:
        # inputFilename = sg.popup_get_file('Select an image to convert to ASCII', file_types=(("Image Files", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff"),))
        print("Usage: python main.py <input_image> <horizontal_resolution> [<tags>]")
        print("Tags:")
        print("-e: Use edge detection")
        print("-c: Use color")
        print("-s: Use just solid ASCII chars")
        exit()

    inputFilename = args[1]
    #outputFilename = "output/" + inputFilename.split(".")[0] + "_ASCII.txt"
    input = loadImage(inputFilename)
    output = convertImage(input, int(args[2]), "-e" in args, "-c" in args, "-s" in args)
    print(output)

if __name__ == "__main__":
    main()
