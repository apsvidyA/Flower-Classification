from scipy.spatial import distance as dist
import csv
class Searcher:
	def __init__(self, indexPath):
		self.indexPath = indexPath
	def search(self, queryFeatures,limit=10):
		results = {}
		with open(self.indexPath) as f:
			reader=csv.reader(f)
			for row in reader:
				features=[float(x) for x in row[1:]]
				d = dist.euclidean(features,queryFeatures)
				results[row[0]] = d
			f.close()
		results = sorted([(v, k) for (k, v) in results.items()])
		return results[:limit]
