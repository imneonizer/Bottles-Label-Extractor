import cv2
import os
import numpy as np
from PIL import Image



if not os.path.exists('temp_photos'):
    os.makedirs('temp_photos')

if not os.path.exists('temp_photos'):
    os.makedirs('labels')

if not os.path.exists('cropped'):
    os.makedirs('cropped/merge')

left = cv2.imread('labels/left_label.png')
front = cv2.imread('labels/front_label.png')
right = cv2.imread('labels/right_label.png')

def resize(img,name, scale_percent):
	name = name
	#print('Original Dimensions : ',img.shape)
	# percent of original size default = 15
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	# resize image
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	#print('Resized Dimensions : ',resized.shape)
	cv2.imwrite('temp_photos/resized '+str(name)+'.png', resized)
	#cv2.imshow('resized '+str(name), resized)
	return resized

def draw_cont(image,name,cont_width):
	name = name
	inputImage = image
	inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(inputImageGray,150,200,apertureSize = 3)
	minLineLength = 0
	maxLineGap = 0
	lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength,maxLineGap)
	for x in range(0, len(lines)):
		for x1,y1,x2,y2 in lines[x]:
		    #cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
		    pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
		    cv2.polylines(inputImage, [pts], True, (0,255,0), cont_width)

	#cv2.imshow("Trolley_Problem_Result", inputImage)
	cv2.imwrite('temp_photos/contour_'+str(name)+ '.png', inputImage)
	#cv2.imwrite('lines.png', inputImage)
	#cv2.imshow('edge', edges)
	return inputImage


def auto_crop(image,orig_image,name):
	name = name
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)

	edged = cv2.Canny(blurred, 1, 1)
	#edged = cv2.Canny(image, 100, 250)
	#cv2.imshow('canny',edged)
	(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	idx = 0
	for c in cnts:
		perimeter = cv2.arcLength(c,True)
		area = cv2.contourArea(c)
		M = cv2.moments(c)

		x,y,w,h = cv2.boundingRect(c)
		if w>190 and h>190:
			idx+=1
			#new_img=image[y:y+h,x:x+w]
			#Adjusting according to bottle
			new_img=orig_image[y+adjusting_height_top:y+h-adjusting_height_bot,x+adjusting_width_left:x+w-adjusting_width_right]
			cv2.imwrite('cropped/'+str(name)+str(idx) + '.png', new_img)
	#cv2.imshow('crop '+name,image)
	ret_img = cv2.imread('cropped/'+str(name)+'1.png')
	#cv2.imshow('cropped '+str(name), ret_img)
	return ret_img

#resizing images =======================
scale_percent = 50
left = resize(left,'left', scale_percent)
front = resize(front,'front', scale_percent)
right = resize(right,'right', scale_percent)
#=======================================

#drawing contours=======================
cont_width = 100
left_drawn = draw_cont(left.copy(), 'left', cont_width)
front_drawn = draw_cont(front.copy(), 'front', cont_width)
right_drawn = draw_cont(right.copy(), 'right', cont_width)
#=======================================

#finding contours and cropping images===
adjusting_height_top = 460 # manually cropping top
adjusting_height_bot = 80 # manually cropping bottom
adjusting_width_left = 19 # manually cropping bottom
adjusting_width_right = 0 # manually cropping bottom
label_left = auto_crop(left_drawn, left, 'left') # contour drawn image, original image, name
label_front = auto_crop(front_drawn, front, 'front') # contour drawn image, original image, name
label_right = auto_crop(right_drawn, right, 'right') # contour drawn image, original image, name
#=======================================

#Stacking===============================
#numpy_horizontal = np.hstack((label_left, label_front, label_right))
#numpy_horizontal_concat = np.concatenate((label_left, label_front), axis=1)
#cv2.imshow('Numpy Horizontal Concat', numpy_horizontal)
#cv2.imwrite('cropped/merged.png', numpy_horizontal)
#=======================================

#unwrap=====================================================================
def resize_auto(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]
    #print('label original size:')
    #print(h,w)
    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image
        # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    #(h, w) = resized.shape[:2]
    #print(h,w)
    # return the resized image
    return resized



left = cv2.imread('cropped/left1.png')
new = resize_auto(image=left,width=430)
cv2.imwrite('cropped/left_mask_fitted'+'.png', new)

front = cv2.imread('cropped/front1.png')
new = resize_auto(image=front,width=430)
cv2.imwrite('cropped/front_mask_fitted'+'.png', new)

right = cv2.imread('cropped/right1.png')
new = resize_auto(image=right,width=430)
cv2.imwrite('cropped/right_mask_fitted'+'.png', new)


#merging masks ===========================================================
def merge_mask(input_image_path, watermark_image_path, position):
    base_image = Image.open(input_image_path)
    watermark = Image.open(watermark_image_path)
 
    # add watermark to your image
    base_image.paste(watermark, position)
    #base_image.show()
    base_image.save(watermark_image_path)
    return base_image

def merge_unwrapped():
	#to crop margins
	#x,y,w,h = 66,125,449,440
	x,y,w,h = 50,125,449,450 #test

	label_left = cv2.imread("cropped/merge/left_mask_fitted.png")
	label_left = label_left[y:h, x:w]

	label_front = cv2.imread("cropped/merge/front_mask_fitted.png")
	label_front = label_front[y:h, x:w]

	label_right = cv2.imread("cropped/merge/right_mask_fitted.png")
	label_right = label_right[y:h, x:w]

	#Stacking===============================
	numpy_horizontal = np.hstack((label_left, label_front, label_right))
	numpy_horizontal_concat = np.concatenate((label_left, label_front), axis=1)
	#cv2.imshow('Numpy Horizontal Concat', numpy_horizontal)
	cv2.imwrite('merged.png', numpy_horizontal)
	return numpy_horizontal
	#=======================================

#adding cropped photos to mask for unwrapping
mask = 'cropped/mask.jpg'
label = 'cropped/left_mask_fitted.png'
merged = merge_mask(mask, label, position=(80,270))

label = 'cropped/front_mask_fitted.png'
merged = merge_mask(mask, label, position=(80,270))

label = 'cropped/right_mask_fitted.png'
merged = merge_mask(mask, label, position=(80,270))
#===========================================================================


#unwrapping=================================================================

direction = 'left'
f1= open("temp_photos/temp_variable_left.txt","w+")
f1.write(str(direction))
f1.close()

direction = 'front'
f2= open("temp_photos/temp_variable_front.txt","w+")
f2.write(str(direction))
f2.close()


direction = 'right'
f3= open("temp_photos/temp_variable_right.txt","w+")
f3.write(str(direction))
f3.close()

#unwrapping label with external codes
import unwrap_labels
del unwrap_labels

#numpy stacking=============================================================
_merged_ = merge_unwrapped()
cv2.imshow('Unwrapped labels', _merged_)
cv2.waitKey(0)
#===========================================================================