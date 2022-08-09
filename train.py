
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input
from tensorflow.keras.layers import Flatten, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
import tensorflow.keras as keras
import tensorflow
import datetime

import numpy as np
import os
import argparse
import cv2
import glob
from os import walk
from os.path import join


import tensorflow as tf
import tensorflow.keras.backend as K
from typeguard import typechecked
from sklearn.utils import class_weight 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow_addons
import matplotlib.pyplot as plt

from utils import countDirectories, preprocessLambda, factorBy

def plot_metrics(history, output):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color='blue', label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                color='orange', linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend();
    
    plt.savefig(output)



def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    """
    lr = 1e-2 #* 0.5e-3
    if epoch > 180:
        lr *= 0.5e-5
    elif epoch > 160:
        lr *= 1e-5
    elif epoch > 120:
        lr *= 1e-4
    elif epoch > 100:
        lr *= 1e-3
    elif epoch > 80:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    """
    lr = 1e-2 #* 0.5e-3
    if epoch > 60:
        lr *= 0.5e-3
    elif epoch > 50:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 30:
        lr *= 1e-1
    
    print('Learning rate: ', lr)
    return lr



preprocessing_function_255 = factorBy(1.0/255.0)
preprocessing_function_minus_plus_1 = preprocessLambda()

class Classifier():
    def __init__(self, images_path, train_test_split=0.80, batch_size=2, batch_size_ft=2, epochs=200, epochs_ft=10, hidden_layer_size=128, input_shape = (300,300,3) ):
        self.images_path = images_path
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.batch_size_ft = batch_size_ft
        self.epochs = epochs
        self.epochs_ft = epochs_ft
        self.hidden_layer_size = hidden_layer_size
        self.input_shape = input_shape

        self.num_classes, self.num_files = countDirectories(self.images_path)

        self.steps_per_epoch =  np.floor(self.num_files / self.batch_size )
        self.preprocessing_function = None

    def buildNetwork(self):

        self.base_model = keras.applications.InceptionResNetV2( #densenet201, inceptionv3, vgg16 ...
            weights='imagenet',  # Load weights pre-trained on ImageNet.
            input_shape=self.input_shape,
            include_top=False,
            classes=self.num_classes)
        
        #self.base_model = keras.applications.EfficientNetV2M( #densenet201, inceptionv3, vgg16 ...
        #    weights='imagenet',  # Load weights pre-trained on ImageNet.
        #    input_shape=self.input_shape,
        #    include_top=False,
        #    classes=self.num_classes)
        

        #Choose one for the model you are using...
        self.preprocessing_function = preprocessing_function_255  
        #self.preprocessing_function = preprocessing_function_minus_plus_1 

        self.base_model.trainable = False
        inputs = keras.Input(shape=self.input_shape)

        x = self.base_model(inputs, training=False)

        #features_vector = keras.layers.GlobalMaxPooling2D()(x)#this flattens also
        
        #self.embedding = keras.Model(inputs, features_vector)
        #self.embedding.save_weights(self.model_path+"embedding_"+self.save_weights_name) #load this weights in case of an error...
        #self.embedding.save(self.model_path+"embedding_"+self.save_model_name)

        
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization()(x)
        #a conv2d that is trained...
        print("the shape")
        print(x.shape)
        #(None, 9, 34, 1536)
        x = Conv2D(filters=x.shape[3] * 4 ,
          kernel_size=3,
          #padding="same",
          padding="valid",
          activation='relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization()(x)
        print("the shape2")
        print(x.shape)
        
        #features_vector = keras.layers.GlobalAveragePooling2D()(x)#this flattens also
        features_vector = keras.layers.GlobalMaxPooling2D()(x)#this flattens too
        #features_vector = keras.layers.Flatten()(x)
        print(features_vector.shape)


        #x = Flatten()(x)
        x = keras.layers.Dropout(0.2)(features_vector)  # Regularize with dropout
        x = Dense(self.hidden_layer_size)(x)
        #x = Dense(self.hidden_layer_size, activation="tanh")(x)
        x = BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        #x = keras.layers.Dropout(0.2)(x)
        #x = Dense(self.hidden_layer_size//2, activation="tanh")(x)
        x = Dense(self.hidden_layer_size//2)(x)
        x = BatchNormalization()(x)

        x = keras.layers.LeakyReLU(alpha=0.1)(x)

        #x = keras.layers.Dropout(0.2)(x)
        #x = BatchNormalization()(x)
        #x = Dense(self.hidden_layer_size//4, activation="tanh")(x)
        #x = keras.layers.Dropout(0.2)(x)
        outputs = Dense(self.num_classes,
                        activation='softmax')(x)
        self.model = keras.Model(inputs, outputs)


        METRICS = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'), 
            keras.metrics.Accuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
            'acc'
        ]

        self.model.compile(loss=tensorflow_addons.losses.focal_loss.SigmoidFocalCrossEntropy(),
                    optimizer=Adam(lr=lr_schedule(0)),
                    #metrics=['acc'])
                    metrics = METRICS)

        self.model.summary()

        try:
            self.model.load_weights("weights.h5")
            print("Weights loaded")

        except:
            print("Weights not loaded")

    def train(self):
        train_datagen = ImageDataGenerator(
        #rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=180,
        validation_split=1.0 -self.train_test_split,
        preprocessing_function=self.preprocessing_function)


        early_stopping = tf.keras.callbacks.EarlyStopping(
                                    monitor='val_prc', 
                                    verbose=1,
                                    patience=10,
                                    mode='max',
                                    restore_best_weights=True)
        
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = '%s_model.{epoch:03d}.h5' % 'inception'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        checkpoint = ModelCheckpoint(filepath=filepath,
                                    monitor='val_acc',
                                    verbose=1,
                                    save_best_only=True)

        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                    cooldown=0,
                                    patience=5,
                                    min_lr=0.5e-6)

        log_dir = "./" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping, tensorboard_callback]

        train_generator = train_datagen.flow_from_directory(
                                                            self.images_path,
                                                            target_size=self.input_shape[:2],
                                                            batch_size=self.batch_size,
                                                            class_mode='categorical',
                                                            subset='training')
        valid_generator = train_datagen.flow_from_directory(
                                                            self.images_path,
                                                            target_size=self.input_shape[:2],
                                                            batch_size=self.batch_size,
                                                            class_mode='categorical',
                                                            subset='validation')
        print(train_generator.classes)
        class_weights = class_weight.compute_class_weight(
           class_weight='balanced',
            classes = np.unique(train_generator.classes), 
            y=train_generator.classes)
        train_class_weights = dict(enumerate(class_weights))

        history= self.model.fit(x=train_generator,
                    verbose=1,
                    epochs=self.epochs,
                    validation_data=valid_generator,
                    steps_per_epoch=self.steps_per_epoch,
                    callbacks=callbacks,
                    class_weight=train_class_weights,
                    shuffle=True)

        try:
            scores = self.model.evaluate(x =valid_generator,
                                    batch_size=self.batch_size,
                                    verbose=0)
            print('Test loss fine tunning:', scores[0])
            print('Test accuracy:', scores[1])

        except:
            print("Error in fine tunning score part")

        print(history)
        self.model.save_weights('weights.h5') 
        self.model.save('model.h5')
        plot_metrics(history, "history.png")

    def fineTunning(self):
        #Fine tunning
        self.base_model.trainable = True

        METRICS = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'), 
            keras.metrics.Accuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
            'acc'
        ]

        self.model.compile(loss=tensorflow_addons.losses.focal_loss.SigmoidFocalCrossEntropy(),
                    optimizer=Adam(lr=1e-8),
                    #metrics=['acc'])
                    metrics = METRICS)

        self.model.summary()



        train_datagen = ImageDataGenerator(
        #rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=180,
        preprocessing_function=self.preprocessing_function)


        early_stopping = tf.keras.callbacks.EarlyStopping(
                                    monitor='val_prc', 
                                    verbose=1,
                                    patience=10,
                                    mode='max',
                                    restore_best_weights=True)
        
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = '%s_model.{epoch:03d}.h5' % 'inception_ft'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        checkpoint = ModelCheckpoint(filepath=filepath,
                                    monitor='val_acc',
                                    verbose=1,
                                    save_best_only=True)
        

        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                    cooldown=0,
                                    patience=5,
                                    min_lr=0.5e-6)

        log_dir = "./" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping, tensorboard_callback]

        train_generator = train_datagen.flow_from_directory(
                                                            self.images_path,
                                                            target_size=self.input_shape[:2],
                                                            batch_size=self.batch_size_ft,
                                                            class_mode='categorical',
                                                            subset='training')
        valid_generator = train_datagen.flow_from_directory(
                                                            self.images_path,
                                                            target_size=self.input_shape[:2],
                                                            batch_size=self.batch_size,
                                                            class_mode='categorical',
                                                            subset='validation')
        print(train_generator.classes)
        class_weights = class_weight.compute_class_weight(
           class_weight='balanced',
            classes = np.unique(train_generator.classes), 
            y=train_generator.classes)
        train_class_weights = dict(enumerate(class_weights))


        history= self.model.fit(x=train_generator,
                    verbose=1,
                    epochs=self.epochs_ft,
                    validation_data=valid_generator,
                    steps_per_epoch=self.steps_per_epoch,
                    callbacks=callbacks,
                    class_weight=train_class_weights,
                    shuffle=True)
        
        try:
            scores = self.model.evaluate(x =train_generator,
                                    batch_size=self.batch_size,
                                    verbose=0)
            print('Test loss fine tunning:', scores[0])
            print('Test accuracy:', scores[1])

        except:
            print("Error in fine tunning score part")

        print(history)
        self.model.save_weights('weights_ft.h5') 
        self.model.save('model_ft.h5')
        plot_metrics(history, "history_ft.png")

    def do(self):
        self.buildNetwork()
        self.train()
        self.fineTunning()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classifier")


    parser.add_argument("-he", "--target_height", type=int, default=300, help='target height')
    parser.add_argument("-wi", "--target_width", type=int, default=300, help='target width')
    parser.add_argument("-co", "--components", type=int, default=3, help='target components')
    parser.add_argument('-ip', '--images_path', type=str, default=r'Honey.AI_PracticalTest_Dataset', help="path to the images directory with images splitted by class")
    parser.add_argument('-tts', '--train_test_split', type=float, default=0.9, help="percentage of train samples")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-bsf", "--batch_size_ft", type=int, default=32, help="batch size fine tunning")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="epochs")
    parser.add_argument("-ef", "--epochs_ft", type=int, default=1, help="epochs fine tunning")
    parser.add_argument("-hls", "--hidden_layer_size", type=int, default=1536, help="hidden layer size")

    
    args = parser.parse_args()
    print(args)

    nth_classifier = Classifier(args.images_path, args.train_test_split, args.batch_size, args.batch_size_ft, args.epochs, args.epochs_ft, args.hidden_layer_size, (args.target_height, args.target_width, args.components))

    nth_classifier.do() 
