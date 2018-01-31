import sys, cv2, os
import numpy as np
from phone_finder_library import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
'''
	Takes a single command line argument which is a path to a
	folder with labeled images and labels.txt
'''
folder = sys.argv[1]								#user input: DIR path of images & label file
locations = open(folder + '/labels.txt', "r") 		#get labels
Num_Neg = 500										#parameter definition, 
													#Number of Negative Samples to extract from each image
samples = []
labels = []

for file in os.listdir(folder):						#for each image in DIR
    if file.endswith(".jpg"):						#get name of image
		img = cv2.imread(folder + '/' + file)		#load image
		[Col, Row] = find_label(locations, file)	#get label associated with image
		[y,x,c] = img.shape							#get dimensions of imageS
		px = int(x*Col)
		py = int(y*Row)								#get image of phone
													#get 500 images of background
		phone, no_phone = get_training_data(img, px, py, x, y, Num_Neg)
		if(phone.any()):							#Augment the Positive Sample Set
			for Rot in range(360):
				M = cv2.getRotationMatrix2D((25,25),Rot,1)
				Positive_Sample = cv2.warpAffine(phone,M,(50,50))
													#extract features from positive samples
				hog_phone = feature_extractor(Positive_Sample)	
				samples.append(hog_phone)
				labels.append(1)

		for i in range(Num_Neg):					#extract features from negative samples
			hog_no_phone = feature_extractor(no_phone[0:50,0:50,0:3,i])
			samples.append(hog_no_phone)
			labels.append(0)			

													#Convert to Numpy Objects
samples = np.float32(samples)
samples = samples.reshape(samples.shape[0], samples.shape[1])
labels = np.array(labels)
													#randomize order
shuffle = np.random.permutation(len(samples))
samples = samples[shuffle]
labels = labels[shuffle]  

model = train_SVM(samples, labels)					#train model

print 'Modeled Trained'
 

	


