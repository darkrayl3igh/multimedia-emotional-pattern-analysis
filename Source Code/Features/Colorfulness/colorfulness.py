import cv2
from scipy import fftpack
import numpy as np
from matplotlib import pyplot as plt
from itertools import count

target = open('colorfulness.txt', 'w')

for count in range (9800) : 
	videoTitle = ("ACCEDE%05i.mp4" % count)
	print(videoTitle)
	cap = cv2.VideoCapture(videoTitle)
	while not cap.isOpened():
	    cap = cv2.VideoCapture(videoTitle)
	    cv.waitKey(1000)
	    print ("Wait for the header")

	pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)

	colorsum1 = 0
	colorpixel1 = 0

	lightsum2 = 0
	lightpixel2 = 0

	while True:
	    flag, imga = cap.read()
	    if flag:
			img = cv2.resize(imga, (0,0), fx=0.5, fy=0.5) 
			img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			cv2.imshow('video', img)

			i = 0
			sum = 0
			w = len(img)
			h = len(img[0])
			rglist = [None] * w * h
			yblist = [None] * w * h	
			for x in range(0,w):
				for y in range(0,h):
					rglist[i] = ( ((img.item(x,y,2)) - (img.item(x,y,1)) ) );  
					yblist[i] = ( (0.5 * ((img.item(x,y,2)) + (img.item(x,y,1))) - (img.item(x,y,0))) );
					sum = sum + img2.item(x,y)
					i = i+1
			stdRG = np.std(rglist)
			meanRG = np.mean(rglist)

			stdYB = np.std(yblist)
			meanYB = np.mean(yblist)

			stdRGYB = ((stdRG)**2 + (stdYB)**2)**0.5
		    	meanRGYB = ((meanRG)**2 + (meanYB)**2)**0.5

			C = stdRGYB + 0.3*meanRGYB

			colorsum1 = colorsum1 + C
			colorpixel1 = colorpixel1 + 1

			
			sum = sum / (w*h)
			lightsum2 = lightsum2 + sum
			lightpixel2 = lightpixel2 + 1


			pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)

	    else:
			cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
			print ("Frame is not ready")

	    if cv2.waitKey(10) == 27:
			break
	    if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
			break
	
	target.write(str(colorsum1/colorpixel1))
	target.write('\t')
	target.write(str(lightsum2/lightpixel2))
	target.write('\n')

target.close()
