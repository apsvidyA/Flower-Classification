from zernikemoments import ZernikeMoments
from imutils.paths import list_images
from skimage.feature import canny
import numpy as np
import argparse
import glob
import imutils
import os
import cv2

desc = ZernikeMoments(21)
output = open("index2.csv", "w")

train_path  = "train"
train_names = os.listdir(train_path)

for train_name in train_names:
	cur_path = train_path + "/" + train_name
	cur_label = train_name
	i = 1

	for file in glob.glob(cur_path + "/*.jpg"):
		imageID = file[file.rfind("/") + 1:]
		image = cv2.imread(file)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		edges = canny(gray)
		features = desc.describe(edges)
		features = [str(f) for f in features]
		output.write("%s,%s\n" % (file, ",".join(features)))

output.close()
