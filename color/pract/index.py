from colorDescriptor import ColorDescriptor
import os
import glob
import cv2


cd = ColorDescriptor((8, 12, 3))

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
		features = cd.describe(image)
		features = [str(f) for f in features]
		output.write("%s,%s\n" % (file, ",".join(features)))

output.close()



