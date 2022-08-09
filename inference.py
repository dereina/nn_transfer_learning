import argparse
from email.policy import default

import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

import shutil
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
import time


from tensorflow.keras.models import load_model
import tensorflow_addons #needed for the model to load with sigmoidfocalloss
from os.path import join
tf.get_logger().setLevel('ERROR')  

from utils import countDirectories, preprocessLambda, factorBy

preprocessing_function_255 = factorBy(1.0/255.0)
preprocessing_function_minus_plus_1 = preprocessLambda()

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print(__file__)
print(os.getcwd())
print(sys.argv[0])
print(os.path.dirname(sys.argv[0]))
print(tf.__version__)



class Inference:
    def __init__(self, input, output, model, weights, list_classes, confidence):
        self.input = input
        self.output = output
        self.model = model
        self.weights = weights
        self.int_to_string = {}
        for i, c in enumerate(list_classes):
            self.int_to_string[i] = c
        
        self.confidence = confidence
   
        self.classifier_model = load_model(self.model)
        self.classifier_model.load_weights(self.weights)
        

    def infiere(self):
        path_t = self.output + "/confidence_"+str(self.confidence)
        path_tb = self.output + "/confidence_"

        try:
            shutil.rmtree(path_t);
            shutil.rmtree(path_tb);
        
        except:
            pass

        try:
            os.mkdir(path_t, 0x755 );
            os.mkdir(path_tb, 0x755 );
        
        except:
            pass

        for root, dirs, files in os.walk(self.input, topdown=False):
            for name in files:
                if name == "Thumbs.db":
                    continue
                
                print(os.path.join(root, name))
                ia = cv2.imread(os.path.join(root, name))
                if ia is None:
                    print("ERROR")
                    continue

                print(ia.shape)
                ia = cv2.cvtColor(ia, cv2.COLOR_BGR2RGB)
                #cv2.imshow("image", ia)
                #cv2.waitKey()
                img1 = preprocessing_function_minus_plus_1(ia)
                img2 = preprocessing_function_255(ia)
                to_predict = np.array([ia])
                to_predict = to_predict.astype('float32') / 255 #input normalization or network won't work...

                out_y = np.array([[1,0,0,0,0,0]])
                t1 = time.time()
                out_y = self.classifier_model.predict(to_predict)
                print("inference time: ", time.time() -t1)
                index = np.argmax(out_y[0])
                print(index)
                path = self.output + "/"
                if self.confidence <= out_y[0][index]:
                    path = path_t + "/"

                else:
                    path = path_tb + "/"
                
                try:
                    os.mkdir(path, 0x755 );
                
                except:
                    pass
                
                path += self.int_to_string[index]
                try:
                    os.mkdir(path, 0x755 );
                
                except:
                    pass

                
                cv2.imwrite(path + "/"+str(out_y[0][index])+"_"+name, ia)
                
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    default_inou = r'Honey.AI_PracticalTest_Dataset'

    parser.add_argument('-o', '--output', type=str, default=default_inou, help="path to the output directory")
    parser.add_argument("-i", "--input", type=str, default=default_inou, help="path to the input directory")

    parser.add_argument("-m", "--model", type=str, default=r'model.h5', help="path to the input model")
    parser.add_argument("-w", "--weights", type=str, default=r'weights.h5', help="path to the input weights")
    parser.add_argument("-c", "--confidence", type=float, default=0.5, help="probability confindence")
    


    parser.add_argument("-lc", "--list_classes", nargs="+", default=["Brassica", "Cardus", "Cistus sp",  "Citrus sp","Erica.m", "Erycalyptus sp", "Heliantus annuus", "Lavndula", "Pinus", "Rosamarinus officinalis", "Taraxacum", "Tilia"])

    args = parser.parse_args()
    print(args)
    for root, dirs, files in os.walk(args.input, topdown=True):
        for i, dir in enumerate(dirs):
            input = join(root, dir)
            inference = Inference(input, input, args.model, args.weights, args.list_classes, args.confidence)
            inference.infiere()
        if len(files)>0:
            input = root
            inference = Inference(input, input, args.model, args.weights, args.list_classes, args.confidence)
            inference.infiere()
        break
