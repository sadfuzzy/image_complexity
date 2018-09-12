# Colourfullness: https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
# Perimeter: https://stackoverflow.com/questions/48786232/how-can-i-calculate-the-perimeter-of-an-object-in-an-image
# Saliency: https://www.pyimagesearch.com/2018/07/16/opencv-saliency-detection/
from PIL import Image
from PIL import ImageFilter
from imutils import build_montages
from imutils import paths
from skimage.measure import label, regionprops
import shannon_entropy
import cv2
import imutils
import os
import csv
import skimage.io as io
import numpy as np

def getSize(filename):
  st = os.stat(filename)
  return st.st_size

def imageColorfulness(imagePath):
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
# fileNames = ['Animals_001_h','Animals_002_v','Animals_003_h']
fileNames =  ['-1.1','-1.2','-1.3','-2.1','-2.2','-2.3','-3.1','-3.2','-3.3','-4.1']
imagesPath = 'pics/'
balancingKoeff = 1.56

with open('results.csv', 'w', newline='') as csvfile:
  results = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

  results.writerow(['Image', 'File size', 'Shannon', 'Canny', 'SaliencyMap', 'SaliencyTresh', 'Kronrod', 'Perimeter'])

  for fileName in fileNames:
    fileName = imagesPath + fileName
    fileNameJpg = fileName + '.jpg'

    # * File size
    fileSize = getSize(fileNameJpg)

    # * Shannon
    img = Image.open(fileNameJpg)
    shannonResult = shannon_entropy.shannon_entropy(img) - balancingKoeff

    # ** Canny
    img = cv2.imread(fileNameJpg, 0)

    edges = cv2.Canny(img, 255, 300)
    cannyFilename = fileName + '_canny.jpg'
    cv2.imwrite(cannyFilename, edges)
    cannonResult = getSize(cannyFilename)

    # ** Saliency
    # initialize OpenCV's static saliency spectral residual detector and
    # compute the saliency map
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)

    # if we would like a *binary* map that we could process for contours,
    # compute convex hull's, extract bounding boxes, etc., we can
    # additionally threshold the saliency map
    threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    cv2.imwrite(fileName + '_saliency_map.jpg', saliencyMap)
    saliencyMap = getSize(fileName + '_saliency_map.jpg')
    cv2.imwrite(fileName + '_saliency_thresh.jpg', threshMap)
    saliencyThresh = getSize(fileName + '_saliency_thresh.jpg')

    # *** Kronrod

    ## Colorfullness
    fileColorfullness = imageColorfulness(fileNameJpg)

    ## Perimeter
    img = io.imread(fileNameJpg)
    bw = img[:,:,0] > 230
    regions = regionprops(bw.astype(int))
    filePerimeter = regions[0].perimeter

    E = shannonResult + saliencyMap + saliencyThresh + fileColorfullness + filePerimeter

    print("{}, size(raw): {}, shannon: {}, size(canny): {}, saliency map: {}, saliency threshold: {}, colorfullness: {}, perimeter: {}, E: {}".format(fileName, fileSize, shannonResult, cannonResult, saliencyMap, saliencyThresh, fileColorfullness, filePerimeter, E))

    results.writerow([fileName, fileSize, shannonResult, cannonResult, saliencyMap, saliencyThresh, fileColorfullness, filePerimeter])
