import sys, cv2, os
import numpy as np
from phone_finder_library import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sklearn import svm, cluster
from sklearn.externals import joblib


folder = sys.argv[1]
image = cv2.imread(folder) 
print 'Image Loaded'
Padded = cv2.copyMakeBorder(image,25,25,25,25,cv2.BORDER_REFLECT)

fig,ax = plt.subplots(1)
ax.imshow(image)	

clf = joblib.load('trained_phone_finder.pkl')
print 'Model Loaded'
[y,x,c] = image.shape

Positive_Class_Set = []

print 'Scanning Image'
for i in range(0,x,2):
	if(i%50==0):
		print '... ',
	for j in range(0,y,2):
		Test_Im = Padded[j:j+50,i:i+50]

		features = feature_extractor(Test_Im)
		classification = clf.predict(features.reshape(1,features.shape[0]))
		if(classification == [1]):
			Positive_Class_Set.append([i,j])
			# rect = patches.Rectangle((i-25,j-25),50,50,linewidth=1,edgecolor='r',facecolor='none')
			# ax.add_patch(rect)

print ''
print 'Positive Class Set Found'
print 'Clustering Duplicate Detections'

phone = cluster_positive_detections(Positive_Class_Set)
print 'Phone Detected at: ', phone[0]/x, phone[1]/y
rect = patches.Rectangle((phone[0]-25,phone[1]-25),50,50,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)

plt.title('Phone Detection')
plt.show()
cv2.waitKey(0)		
		








