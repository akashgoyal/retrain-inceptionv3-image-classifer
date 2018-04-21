import os, math

# import keras libs
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model,load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping


def create_model(model_name,training_type,num_classes):
    #Initializing the image width and image height : This will be updated as per the model which is going to be used
    img_width,img_height = 224,224

    #InceptionV3
    if (model_name.lower() == 'inceptionv3'):
            img_width,img_height = 299,299
            if(training_type == 'train_all'):
                model = applications.inception_v3.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
                model.layers.pop()
                for layer in model.layers:
                	layer.trainable = True
            top_model = Sequential()
            top_model.add(Flatten(input_shape=model.output_shape[1:]))
            top_model.add(Dense(1024, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(512, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(num_classes, activation='softmax'))
            #Final model
            model_final = Model(inputs = model.input, outputs = top_model(model.output))

    return model_final,img_width,img_height

def test_model(model_name,save_loc,test_generator):
    #load the saved model
    model_loc  = save_loc + model_name + ".h5"
    model = load_model(model_loc)

    #predict
    logits = model.predict_generator(test_generator,verbose = 1)

    return logits
