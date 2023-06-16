from Detector import *

modelURL="http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
imagepath="test/2.png"
vidpath="test/london.mp4"
classFile="coco.names"
detect=Detector()
detect.readClasses(classFile)
detect.downloadModel(modelURL)
detect.loadModel()
detect.predictImage(imagepath)
#detect.predictVideo(vidpath,threshold=0.5)