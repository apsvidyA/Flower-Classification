from colorDescriptor import ColorDescriptor
from searcher import Searcher
import argparse
import cv2

cd = ColorDescriptor((8, 12, 3))

query = cv2.imread("queries/query2.jpg")
features = cd.describe(query)

searcher = Searcher("index.csv")
results = searcher.search(features)

cv2.imshow("Query", query)

for (score, resultID) in results:
	result = cv2.imread("prac/" + resultID)
	cv2.imshow("Result", result)
	cv2.waitKey(0)
