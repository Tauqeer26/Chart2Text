import cv2,time,os ,tensorflow as tf
import numpy as np 
from tensorflow.python.keras.utils.data_utils import get_file
np.random.seed(123)

class Detector:
    def __init__(self):
        pass
    def readClasses(self,classesFilePath):
        with open(classesFilePath,'r') as f:
            self.classesList=f.read().splitlines()
        self.colorList=np.random.uniform(low=0,high=255,size=(len(self.classesList),3))

        print(len(self.classesList),len(self.colorList))
    def downloadModel(self,modelURL):
        fileName=os.path.basename(modelURL)
        self.modelName=fileName[:fileName.index('.')]
        self.cacheDir="./pretrained_models"
        os.makedirs(self.cacheDir,exist_ok=True)
        get_file(fname=fileName,origin=modelURL,cache_dir=self.cacheDir,cache_subdir="checkpoints",extract=True)
    def loadModel(self):
        print("Loading Model" +self.modelName)
        #tf.keras.backend.clear_session()
        self.model=tf.saved_model.load(os.path.join(self.cacheDir,"checkpoints",self.modelName,"saved_model"))
        print("Model "+ self.modelName+ "loaded succesfully....")
    
    def createBoundingBox(self,image):
        inputTensor=cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
        inputTensor=tf.convert_to_tensor(inputTensor,dtype=tf.uint8)
        inputTensor=inputTensor[tf.newaxis,...]

        detections=self.model(inputTensor)
        bboxs=detections['detection_boxes'][0].numpy()
        classIndexes=detections['detection_classes'][0].numpy().astype(np.int32)
        classScores=detections['detection_scores'][0].numpy()

        imH, imW, imC=image.shape
        bboxIdx=tf.image.non_max_suppression(bboxs,classScores,max_output_size=50,iou_threshold=0.5,score_threshold=0.5)
        print(bboxIdx)
        if len(bboxIdx)!=0:
            for i in bboxIdx:
                bbox=tuple(bboxs[i].tolist())
                classConfidence=round(100*classScores[i])
                classIndex=classIndexes[i]
                classLabelText=self.classesList[classIndex].upper()
                classColor=self.colorList[classIndex]
                displayText='{}: {}%'.format(classLabelText,classConfidence)

                ymin,xmin,ymax,xmax=bbox
                xmin,xmax,ymin,ymax=(xmin*imW,xmax*imW,ymin*imH,ymax*imH)
                xmin,xmax,ymin,ymax=int(xmin),int(xmax),int(ymin),int(ymax)

                cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color=classColor,thickness=1)
                cv2.putText(image,displayText,(xmin,ymin-10),cv2.FONT_HERSHEY_COMPLEX,1,classColor,2)
                
                ##############################
                linewidth=min(int((xmax-xmin)*0.2),int((ymax-ymin)*0.2))

                cv2.line(image,(xmin,ymin),(xmin+linewidth,ymin),classColor,thickness=5)
                cv2.line(image,(xmin,ymin),(xmin,ymin+linewidth),classColor,thickness=5)
                
                cv2.line(image,(xmax,ymin),(xmax-linewidth,ymin),classColor,thickness=5)
                cv2.line(image,(xmax,ymin),(xmax,ymin+linewidth),classColor,thickness=5)
                ##############################

                cv2.line(image,(xmin,ymax),(xmin+linewidth,ymax),classColor,thickness=5)
                cv2.line(image,(xmin,ymax),(xmin,ymax-linewidth),classColor,thickness=5)
                
                cv2.line(image,(xmax,ymax),(xmax-linewidth,ymax),classColor,thickness=5)
                cv2.line(image,(xmax,ymax),(xmax,ymax-linewidth),classColor,thickness=5)
                #print(displayText)
        return image


    def predictImage(self,imagepath):
        image=cv2.imread(imagepath)
        bboximage=self.createBoundingBox(image)

        cv2.imwrite(self.modelName+ ".jpg",bboximage)

        cv2.imshow("Results",bboximage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predictVideo(self,vidpath,threshold=0.5):
        cap=cv2.VideoCapture(vidpath)
        if (cap.isOpened()==False):
            print("Error opening video")
            return 
        (success,image)=cap.read()
        starttime=0
        while success:
            currentTime=time.time()
            fps=1/(currentTime-starttime)
            starttime=currentTime
            bboxImage=self.createBoundingBox(image)

            cv2.putText(bboxImage,"FPS: "+str(int(fps)),(20,70),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2 )
            cv2.imshow("Result",bboxImage)
            key=cv2.waitKey(1)&0xFF
            if key==ord('q'):
                break
            (success,image)=cap.read()
        cv2.destroyAllWindows()



