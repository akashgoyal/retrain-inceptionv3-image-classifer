import glob
import os
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

#load the keras libraries
from keras.layers import Dropout, Input, Dense, Activation,GlobalMaxPooling2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.callbacks import ModelCheckpoint

import models_keras

def test_model(data_dir_train,data_dir_test,batch_size,model_name,save_loc,results_loc):

    DATA_TRAIN = data_dir_train
    DATA_TEST = data_dir_test
    BATCH_SIZE = batch_size
    num_classes = len(os.listdir(DATA_TRAIN))

    nb_test_samples = sum([len(files) for r, d, files in os.walk(DATA_TEST)])

    print("Test Samples :",nb_test_samples)
    print("Number of Classes :",num_classes)

    #differenent sources from where the models are being initialized
    keras_models= ['inceptionv3']
    img_width,img_height = 299,299

    test_datagen = ImageDataGenerator(rescale = 1./255)

    test_generator = test_datagen.flow_from_directory(
    	DATA_TEST,
    	target_size = (img_height, img_width),
    	class_mode = None)

    #class_mode = None is used for testing but remember the data must be inside a sub_directory inside the test main directory

    logits = models_keras.test_model(model_name, save_loc, test_generator)
    logits_pd = pd.DataFrame(logits)
    save_results = results_loc + model_name + "_results.csv"
    logits_pd.to_csv(save_results,index=False)

    return_string = "Result saved at : " + save_results
    return return_string
