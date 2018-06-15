# import the necessary packages
import numpy as np
import cv2

white = [255,255,255]
black = [0,0,0]
blue = [ 255, 0, 0]
 
def max_rgb_filter(img):
	# split the img into its BGR components
	(B, G, R) = cv2.split(img)
 
	# find the maximum pixel intensity values for each
	# (x, y)-coordinate,, then set all pixel values less
	# than M to zero
	M = np.maximum(np.maximum(R, G), B)
	R[R < M] = 0
	G[G < M] = 0
	B[B < M] = 0
 
	# merge the channels back together and return the img
	return cv2.merge([B, G, R])



def getContour(img):
	# find the red color game in the img
	upper = np.array(white)
	lower = np.array(white)
	mask = cv2.inRange(img, lower, upper)
	
	# find contours in the masked img and keep the largest one
	#(_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	#(cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	(cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

	return cnts

def getFeature(img):
	
	cnts = getContour(img)
	result = []

	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		#approx = cv2.approxPolyDP(c, 0.05 * peri, True)
		approx = cv2.approxPolyDP(c, 0.005 * peri, True)
		result.append(approx)
	
	return result
 

def filterBlue(img):
	#upper = np.array([255,0, 0])
	#lower = np.array([10, 0, 0])
	#return cv2.inRange(img, lower, upper)
	thresh = 25000
	img = img - np.array(blue)
	#img = np.abs(img)
	img = img**2
			
	for i in range(img.shape[0]):
   		for j in range(img.shape[1]):	

			t = sum(img[i,j])	

			if t > thresh:
				img[i,j] = black
			else:
				img[i,j] = white
	return img





def reload(img):
	name = "t.jpg"
	cv2.imwrite( name, img);
	return cv2.imread(name)

# load the games img
image = cv2.imread("109.png")
#print img.shape
#print filterBlue(img)
img = filterBlue(image)

image = reload(img)
img = image
#contours = getContour(img)
#cv2.drawContours(img, contours, -1, white, 1)

#img = reload(img)

#cv2.imwrite( "t.jpg", img);
#img = cv2.imread("t.jpg")



kernel = np.ones((1,5),np.uint8)
img = cv2.dilate(img,kernel,iterations = 10)


contours = getFeature(img)
#cv2.drawContours(img, contours, -1, white, 1)

#img = reload(img)

#cv2.imwrite( "t.jpg", img);
#img = cv2.imread("t.jpg")

mask = np.zeros_like(img)
cv2.drawContours(mask, contours, 3, white, -1)
out = np.zeros_like(img)
out[ mask == white] = image[mask == white]

img = out

cv2.imwrite( "result.jpg", img );

#cv2.imshow("Image", img)
#cv2.waitKey(0)
