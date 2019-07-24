from PIL import Image
import DataAugmentor
import ModelCreator
import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential

# The dimensions for the Augmented Images to be Saved
imgScalDim = (150,150)


# Make it false incase you only want to train the Model (ie. Data Already Generated)
dataGenerated = False
# Make it false incase you just want to load the weights and predict outputs
needToTrainModel = True


# The Method Finds in Blob in the image by filtering them out through some threshold and saves it as a new Image
def generateImagesFromBlobs(imageDirectoryPath):    
    for tempImage in os.listdir(imageDirectoryPath):
        imageFullPath = os.path.join(imageDirectoryPath,tempImage)
        imagName = Path(imageFullPath).stem
        # The full Image path to save the blobs into
        savingImagPath = os.path.join(imageDirectoryPath,imagName) + '{}.png'
        img = cv2.imread(imageFullPath)
        height, width, channels = img.shape
        # Maximally Stable Extremal Regions (MSER) used as a feature detector
        mser = cv2.MSER_create()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        gray_img = img.copy()
        # Detecting the Contours and saving them as bounding Boxes
        msers, bboxes = mser.detectRegions(gray)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in msers]
        cv2.polylines(gray_img, hulls, 1, (0, 255, 0),1)
        imageIndex = 0
        # Saving all the hull regions found in the base image limiting by some offsets
        for contour in hulls:
            x,y,w,h = cv2.boundingRect(contour)
            if w > (width / 15) and h > (height / 15) :
                imageIndex += 1
                resizedImage = cv2.resize(img[y:y+h,x:x+w], imgScalDim, interpolation = cv2.INTER_AREA)
                cv2.imwrite(savingImagPath.format(imageIndex), resizedImage)
        os.remove(imageFullPath)            

# Generating Extra Image data by augmneting the blob images
def generateAllAugmentedImages(dataAugmentor,imageDirectoryPath):
    for imageName in os.listdir(imageDirectoryPath):        
        imageFullPath = os.path.join(imageDirectoryPath,imageName)
        imgSaveName = os.path.splitext(imageFullPath)[0]
        print('Augmenting {} ....'.format(imgSaveName))
        image = Image.open(imageFullPath)
        dataAugmentor.flipImageAndSave(image,imgSaveName)        
        dataAugmentor.addNoiseAndSave(image,image.width,image.height,3,imgSaveName)
        os.remove(imageFullPath)


# Place the document filename to verify
# The document should be in the Solution directory.
documentToVerify = ''

def CheckPredictionsByTheModel():
    modelPath = os.getcwd() + '\TrainedModel.h5'
    leNet5.load_weights(modelPath)

    imageToVerify = os.path.join(baseImageFolderPath,documentToVerify)
    img = cv2.imread(imageToVerify)
    height, width, channels = img.shape

    # Value Can be Tuned in depending on the ize and Type of Document
    documentSignStampIndex = ( height * width ) / 64

    # Maximally Stable Extremal Regions (MSER) used as a feature detector
    mser = cv2.MSER_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray_img = img.copy()
    # Detecting the Contours and saving them as bounding Boxes
    msers, bboxes = mser.detectRegions(gray)
    hulls = [cv2.convexHull(pImg.reshape(-1, 1, 2)) for pImg in msers]
    cv2.polylines(gray_img, hulls, 1, (0, 255, 0),1)     

    #Getting stamps and signs

    hasSign = False
    hasStamp = False

    for relevantHulls in hulls:
        x,y,w,h = cv2.boundingRect(contour)
        if (w * h) >= documentSignStampIndex:
            resizedImage = cv2.resize(img[y:y+h,x:x+w], imgScalDim, interpolation = cv2.INTER_AREA)
            output = leNet5.predict_classes(np.array(resizedImage))
            classes = np.argmax(predictions, axis=1)
            if classes == 0:
                hasSign = True
            elif classes == 1:
                hasStamp = True
        if hasSign & hasStamp:
            print('It is a Valid Document. (Both Signature and Stamp found !')
            break;

    if hasSign is False or hasStamp is False:
        print('It is a InValid Document. (Either of Signature and Stamp NOT found !')

# Entry Point
if __name__ == '__main__':

    baseImageFolderPath = os.path.join(os.getcwd() + '\BaseImages')
    if dataGenerated is False:
        print('Generating Dataset BLOB ....')      
        generateImagesFromBlobs(baseImageFolderPath)
        augmentor = DataAugmentor.DataAugmentor()
        print('Augmenting Blob Dataset ....')
        generateAllAugmentedImages(augmentor,baseImageFolderPath)
        totalImagesGenerated = len(os.listdir(baseImageFolderPath))
        print('Dataset Generated Properly ... Total Image Size : {}'.format(totalImagesGenerated))


    if needToTrainModel:
        # Initilising the Training Process .....
        # Creating the output/target labels across the Image Dataset
        print('Initializing LeNet5 ... ')
        imageFilePaths = [os.path.join(baseImageFolderPath,imgName) for imgName in os.listdir(baseImageFolderPath)]
        inputData = []
        outputData = []
        randomIndexArray = np.random.permutation(len(imageFilePaths))
        for item in randomIndexArray:
            inputData.append(np.array(Image.open(imageFilePaths[item])))
            if imageFilePaths[item].find('Sign') != -1:
                outputData.append('Sign')
            elif imageFilePaths[item].find('Stamp') != -1:
                outputData.append('Stamp')
            elif imageFilePaths[item].find('Text') != -1:
                outputData.append('Text')
            else :
                print('No Class Found for Image {}'.format(imageFilePaths[item]))

        print('Training Data Obtained Output Size = {}'.format(len(outputData)))

        # 70:30 Train-Test Split (Can even be done using Sci-kit Learn)
        trainingSampleCount = (int)(len(inputData) * 0.7)
        xTrainingArray = inputData[:trainingSampleCount]
        yTrainingArray = np.array(outputData[:trainingSampleCount])
        xTestArray = inputData[trainingSampleCount:]
        yTestArray = np.array(outputData[trainingSampleCount:])

        # Converting Target Variables to Categorical Format
        encoder = LabelBinarizer()
        yTrainingArray = encoder.fit_transform(yTrainingArray)
        yTestArray = encoder.fit_transform(yTestArray)

        # Model Intantiation and Training
        modelInstance = ModelCreator.ModelCreator()
        leNet5 = modelInstance.CreateModel()
        leNet5.fit(np.array(xTrainingArray),np.array(yTrainingArray),100,200)
        leNet5.evaluate(np.array(xTestArray),np.array(yTestArray),100,100)

        modelPath = os.getcwd() + '\TrainedModel.h5'
        leNet5.save_weights(modelPath)
    
    # Un-Comment it for predicting.
    #CheckPredictionsByTheModel()


    


    