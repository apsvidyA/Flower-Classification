from searcher import Searcher
from zernikemoments import ZernikeMoments
from skimage.feature import canny
import numpy as np
import argparse
import glob
import imutils
import cv2
import csv

desc = ZernikeMoments(21)

test_path = "test"
for file in glob.glob(test_path + "/*.jpg"):
	image = cv2.imread(file)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edges = canny(gray)
	queryFeatures = desc.describe(edges)
	searcher = Searcher("index2.csv")
	results = searcher.search(queryFeatures)
	cv2.imshow("Query", image)

	for (score, resultID) in results:
		result = cv2.imread(resultID)
		cv2.imshow("Result", result)
		cv2.waitKey(0)

