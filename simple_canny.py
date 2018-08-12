# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 21:31:17 2015
This is the basic Canny Edge Detection code supplied by the
original OpenCV-Python Tutorials.
@author: Johnny
"""

import cv2
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFilter

for img_name in ['flowerA','flowerB','flowerC','messi']:
    img = cv2.imread(img_name + '.jpg', 0)
    
    height, width = img.shape[:2]
    max_height = 500
    max_width = 500

    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width/float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    #plt.subplot(121), plt.imshow(img, cmap='gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122), plt.imshow(edges, cmap='gray')
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    
    # edges = cv2.Canny(img, 223, 90)
    # edges0 = cv2.Canny(img, 100, 200)

    edges0 = cv2.Canny(img, 255, 300)
    cv2.imwrite(img_name + '_edges100200noblur.png', edges0)
    
    #img = cv2.bilateralFilter(img,9,75,75)
    img = cv2.GaussianBlur(img, (5, 5), 2)
    #cv2.imwrite(img_name + '_gauss.png', img)
    
    edges = cv2.Canny(img, 100, 200)
    #cv2.imwrite(img_name + '_edges100200.png', edges)
    edges1 = cv2.Canny(img, 200, 100)
    #cv2.imwrite(img_name + '_edges200100.png', edges1)
    
    #blur = cv2.bilateralFilter(img,9,75,75)
    #cv2.imwrite(img_name + '_blur.png', blur)
    #blur_edge = cv2.bilateralFilter(edges,9,75,75)
    #blur_edge = cv2.bilateralFilter(edges,9,100,100)

    