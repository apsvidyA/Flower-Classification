from textureDescriptor import TextureDescriptor
import os
import glob
import cv2

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
		features = TextureDescriptor.extract_features(gray)
		features = [str(f) for f in features]
		output.write("%s,%s\n" % (file, ",".join(features)))

output.close()
