import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC

class TextureDescriptor:
	def extract_features(image):
		textures = mt.features.haralick(image)
		ht_mean  = textures.mean(axis=0)
		return ht_mean
