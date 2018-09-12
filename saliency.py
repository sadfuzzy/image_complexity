import cv2
 
# load the input image
image = cv2.imread('/home/denis/code/tata/images/Animals_001_h.jpg')

saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
(success, saliencyMap) = saliency.computeSaliency(image)
saliencyMap = (saliencyMap * 255).astype("uint8")
# cv2.imshow("Image", image)
cv2.imwrite('saliencyMap.jpg', saliencyMap)

saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(image)
 
# if we would like a *binary* map that we could process for contours,
# compute convex hull's, extract bounding boxes, etc., we can
# additionally threshold the saliency map
threshMap = cv2.threshold(saliencyMap, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 
# show the images

cv2.imwrite('threshMap.jpg', threshMap)