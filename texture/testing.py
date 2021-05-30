import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC


def extract_features(image):
	textures = mt.features.haralick(image)
	ht_mean  = textures.mean(axis=0)
	return ht_mean

train_path  = "train"
train_names = os.listdir(train_path)

train_features = []
train_labels   = []


for train_name in train_names:
	cur_path = train_path + "/" + train_name
	cur_label = train_name
	i = 1
	for file in glob.glob(cur_path + "/*.jpg"):
		image = cv2.imread(file)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		features = extract_features(gray)
		train_features.append(features)
		train_labels.append(cur_label)
		i += 1


clf_svm = LinearSVC(random_state=9)
clf_svm.fit(train_features, train_labels)


test_path = "test"
for file in glob.glob(test_path + "/*.jpg"):
	image = cv2.imread(file)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	features = extract_features(gray)
	prediction = clf_svm.predict(features.reshape(1, -1))[0]
	cv2.putText(image, prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
	cv2.imshow("Test_Image", image)
	cv2.waitKey(0)

