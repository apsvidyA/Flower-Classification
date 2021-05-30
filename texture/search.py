from textureDescriptor import TextureDescriptor
from searcher import Searcher
import glob
import argparse
import cv2

test_path = "test"
for file in glob.glob(test_path + "/*.jpg"):
	image = cv2.imread(file)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	features = TextureDescriptor.extract_features(gray)
	searcher = Searcher("index2.csv")
	results = searcher.search(features)
	cv2.imshow("Query", image)

	for (score, resultID) in results:
		result = cv2.imread(resultID)
		cv2.imshow("Result", result)
		cv2.waitKey(0)
