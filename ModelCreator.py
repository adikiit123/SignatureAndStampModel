from keras.initializers import TruncatedNormal
from keras.losses import categorical_crossentropy
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import SGD

class ModelCreator(object):

    def CreateModel(self):
        deepModel = Sequential()
        # 1st Convolution Kernel followed by Max Pool Layer with a Dropout Prob of 0.4
        convKernel1 = Conv2D(32, (5, 5), activation='relu', input_shape=(150, 150, 3))
        maxPool1 = MaxPooling2D(pool_size=(2, 2))
        dropOut1 = Dropout(0.4)

        # The Input Shape for 2nd Convolutional Layer is Calculated by : 
        # WidthTransformed = (Width - Spatial Index + 2 * Padding) / Stride
        # HeightTransformed = (Height - Spatial Index + 2 * Padding) / Stride
        # 2nd Convolution Kernel followed by Max Pool Layer with a Dropout Prob of 0.4
        convKernel2 = Conv2D(64, (5, 5), activation='relu')
        maxPool2 = MaxPooling2D(pool_size=(2, 2))
        dropOut2 = Dropout(0.4)

        deepModel.add(convKernel1)
        deepModel.add(maxPool1)
        deepModel.add(dropOut1)
        deepModel.add(convKernel2)
        deepModel.add(maxPool2)
        deepModel.add(dropOut2)

        # Flattening the network
        deepModel.add(Flatten())

        # ANN with FC Layer and O/P Layer
        fullyConnected1 = Dense(128, activation = 'sigmoid')
        fullyConnected2 = Dense(64,activation = 'sigmoid')
        outputLayer = Dense(3,activation = 'softmax')

        deepModel.add(fullyConnected1)
        deepModel.add(fullyConnected2)
        deepModel.add(outputLayer)

        # Nesterov Accelerated Gradient Descent 
        sgdOptimizer = sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        deepModel.compile(sgdOptimizer,loss = categorical_crossentropy,metrics = ['accuracy'])
        print(deepModel.summary())

        return deepModel

