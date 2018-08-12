# Colourfullness: https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
# Perimeter: https://stackoverflow.com/questions/48786232/how-can-i-calculate-the-perimeter-of-an-object-in-an-image
from PIL import Image
from PIL import ImageFilter
from matplotlib import pyplot as plt
from imutils import build_montages
from imutils import paths
import shannon_entropy
import cv2
import numpy as np
import argparse
import imutils
import os
from skimage.measure import label, regionprops
import skimage.io as io

def getSize(filename):
  st = os.stat(filename)
  return st.st_size

def image_colorfulness(imagePath):
  image = cv2.imread(imagePath)
  image = imutils.resize(image, width=250)
	# split the image into its respective RGB components
  (B, G, R) = cv2.split(image.astype("float"))
 
	# compute rg = R - G
  rg = np.absolute(R - G)
 
	# compute yb = 0.5 * (R + G) - B
  yb = np.absolute(0.5 * (R + G) - B)
 
	# compute the mean and standard deviation of both `rg` and `yb`
  (rbMean, rbStd) = (np.mean(rg), np.std(rg))
  (ybMean, ybStd) = (np.mean(yb), np.std(yb))
 
	# combine the mean and standard deviations
  stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
  meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
 
	# derive the "colorfulness" metric and return it
  return stdRoot + (0.3 * meanRoot)

# ------------------------------ MY METHOD --------------------
fileNames = ['Animals_001_h','Animals_002_v','Animals_003_h']

for fileName in fileNames:
  fileSize = getSize(fileName + '.jpg')

  # * Shannon
  img = Image.open(fileName + '.jpg')
  shannonResult = shannon_entropy.shannon_entropy(img) - 1.56

  # ** Canny
  img = cv2.imread(fileName + '.jpg', 0)

  edges = cv2.Canny(img, 255, 300)
  cannyFilename = fileName + '_canny.jpg'
  cv2.imwrite(cannyFilename, edges)
  cannonResult = getSize(cannyFilename)

  # *** Kronrod

  ## Colorfullness
  fileColorfullness = image_colorfulness(fileName + '.jpg')

  ## Perimeter
  img = io.imread(fileName + '.jpg')
  bw = img[:,:,0] > 230
  regions = regionprops(bw.astype(int))
  filePerimeter = regions[0].perimeter

  print "{0}, size(raw): {1}, shannon: {2}, size(canny): {3}, colorfullness: {4}, perimeter: {5}".format(fileName, fileSize, shannonResult, cannonResult, fileColorfullness, filePerimeter)
