import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFilter

for img_name in os.listdir(os.getcwd()):
    img = cv2.imread(img_name)
    if img is None:
        continue

    height, width = img.shape[:2]

    max_height = 1024
    max_width = 1280

    if height > max_height or width > max_width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width/float(width) < scaling_factor:
            scaling_factor = max_width / float(width)

        # resize image
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    cv2.imwrite(img_name, img)
