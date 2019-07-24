from PIL import Image
import numpy as np


class DataAugmentor(object):
    ## The Class adds basic Augmentation to the Dataset.
    ## Baically the Dataset is all generated from 3 different Images Consisting of Classes of 
    #   1. Signatures
    #   2. Stamps
    #   3. Document Texts

    # This methods inverts the image at 180 Deg.
    def flipImageAndSave(self,image,filename):
        tempImage = np.array(image)
        newImage = Image.fromarray(np.fliplr(tempImage))
        newFileName = filename + "_Flipped.png"
        newImage.save(newFileName)


    # This method adds random noise to the image dataset
    def addNoiseAndSave(self,image,width,height,depth,filename):
        tempImage = np.array(image)
        noise = np.random.randint(5, size = (200, 200, 4), dtype = 'uint8')
        for i in range(height-1):
            for j in range(width-1):
                for k in range(depth):
                    if (tempImage[i][j][k] != 255):
                        # Adding noise value to the intensities across all the channels
                        tempImage[i][j][k] += noise[i][j][k]
        newImage = Image.fromarray(tempImage)
        newFileName = filename + "_Noised.png"
        newImage.save(newFileName)
