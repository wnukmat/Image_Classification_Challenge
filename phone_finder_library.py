import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from sklearn import svm, cluster
from sklearn.externals import joblib

def file_len(fname):
	'''
	finds lenth of file
	'''
	fname.seek(0)
	for i, l in enumerate(fname):
		pass
	
	return i + 1
	
def find_label(fname, im_name):
	'''
	returns label for image name
	'''
	fname.seek(0)
	for line in fname: 
		Fields = line.split(" ")
		if (Fields[0] == im_name):
			
			return float(Fields[1]), float(Fields[2])
			
def get_training_data(im, labelx, labely, x, y, Num_Neg):
	'''
	Outout:
		Hit:		50x50 pixel image containing phone
		Miss:		50x50 pixel examples (Num_Neg) randomly selected 
						from image, not containing phone
	'''
	# fig,ax = plt.subplots(1)
	# ax.imshow(im)	
	
	# Cut Out Negative Training Examples

	Miss = np.zeros((50,50,3,Num_Neg))
	Neg_Example = 0
	while(Neg_Example < Num_Neg):
		rx = random.randint(25,465)
		ry = random.randint(25,301)
		# Consider it a negative sample if center is 20 pixels away from center of phone
		#	Classifier can see examples with phone but that don't meet the 0.05 normalized distance threshold
		if(((rx<=labelx-20)and(ry<=labely-20)) or ((rx>=labelx+20)and(ry>=labely+20)) or
				((rx<=labelx-20)and(ry>=labely+20)) or ((rx>=labelx+20)and(ry<=labely-20))):
			
			Miss[:,:,:,Neg_Example] = im[ry-25:ry+25,rx-25:rx+25]
			
			Neg_Example += 1
			# rect = patches.Rectangle((rx-23,ry-23),46,46,linewidth=1,edgecolor='r',facecolor='none')
			# ax.add_patch(rect)

	# Cut Out Positive Training Example
	valid = np.logical_and( np.logical_and((labely-25 >=0), (labely+25 < y)), \
									np.logical_and((labelx-25 >=0), (labelx+25 < x)))
	
	Hit = np.zeros((50,50,3))
	if(valid):
		Hit = im[labely-25:labely+25,labelx-25:labelx+25]
		# rect = patches.Rectangle((labelx-23,labely-23),46,46,linewidth=1,edgecolor='g',facecolor='none')
		# ax.add_patch(rect)
		
	# plt.title('Generating Sample Set')
	# plt.show()
	# cv2.waitKey(0)
	return Hit, Miss
	
def feature_extractor(im):
	'''
	Histogram of Gradient features extracted from input image
	expects a 50x50 pixel image as input
	'''
	winSize = (50,50)		# corresponding to the size of the input image choosen
	blockSize = (20,20)		# 2x Cellsize, parameter handling illumination variations
	blockStride = (10,10)	# 50% Blocksize, normalization factor	
	cellSize = (10,10)		# dimensionality reduction factor to capture highly informative features
	nbins = 9				# default recogmendation of N. Dalal
	 
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
		
	return hog.compute(im.astype(np.uint8))
	
	
	
def train_SVM(samples, labels):
	# Create SVM Classifier
	clf = svm.SVC()
	clf.fit(samples, labels) 
	joblib.dump(clf, 'trained_phone_finder.pkl', compress=9)
	# print 'error: ', 1-clf.score(samples, labels)
	
	return clf
	
def cluster_positive_detections(Positive_Class_Set):
	MoreClusters = True
	k = 1
	while(MoreClusters == True):
		# print 'Clustering: K = ' + str(k)
		MoreClusters = False
		k_means = cluster.KMeans(n_clusters=k)
		k_means.fit(Positive_Class_Set) 
		
		for i in range(len(k_means.labels_)):
			dist = np.linalg.norm(Positive_Class_Set[i]-k_means.cluster_centers_[k_means.labels_[i]] )
			if(dist > 10):
				MoreClusters = True

		k += 1

	majority_class = max(set(k_means.labels_), key=list(k_means.labels_).count)
	
	return k_means.cluster_centers_[majority_class]

	
	
	
	
	