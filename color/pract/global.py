import cv2
import numpy as np
import os
import glob
import mahotas as mt
import csv
from colorDescriptor import ColorDescriptor
from sklearn.svm import LinearSVC

cd = ColorDescriptor((8, 12, 3))

train_path  = "train"
train_names = os.listdir(train_path)

train_features = []
train_labels   = []

with open('index2.csv') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
         	train_features.append(row[1:])
         	train_labels.append(row[0])

clf_svm = LinearSVC(random_state=9)

clf_svm.fit(train_features, train_labels)

test_path = "test"
for file in glob.glob(test_path + "/*.jpg"):
	image = cv2.imread(file)
	features = cd.describe(image)
	prediction = clf_svm.predict(np.array(features).reshape(1, -1))[0]
	subdirname = os.path.basename(os.path.dirname(prediction))
	cv2.putText(image, subdirname, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
	print("Prediction - {}".format(subdirname))
	cv2.imshow("Test_Image", image)
	cv2.waitKey(0)

