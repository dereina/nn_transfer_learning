# deep_learning

The training deals with imbalanced data by using a SigmoidFocalCrossEntropy loss function, class weights for weighted updates and image data augmentation. It uses  a Transfer Learning aproac., There are many networks and every one needs a preprocessing function usually to convert from [0 255] to [0, 1] or [-1 1], helper functions for doing that are provided. 

# Train, see default arguments in the script

  usage: train.py [-h] [-he TARGET_HEIGHT] [-wi TARGET_WIDTH] [-co COMPONENTS] [-ip IMAGES_PATH] [-tts TRAIN_TEST_SPLIT] [-bs BATCH_SIZE]
                  [-bsf BATCH_SIZE_FT] [-e EPOCHS] [-ef EPOCHS_FT] [-hls HIDDEN_LAYER_SIZE]

  Image Classifier

  optional arguments:
    -h, --help            show this help message and exit
    -he TARGET_HEIGHT, --target_height TARGET_HEIGHT
                          target height
    -wi TARGET_WIDTH, --target_width TARGET_WIDTH
                          target width
    -co COMPONENTS, --components COMPONENTS
                          target components
    -ip IMAGES_PATH, --images_path IMAGES_PATH
                          path to the images directory with images splitted by class
    -tts TRAIN_TEST_SPLIT, --train_test_split TRAIN_TEST_SPLIT
                          percentage of train samples
    -bs BATCH_SIZE, --batch_size BATCH_SIZE
                          batch size
    -bsf BATCH_SIZE_FT, --batch_size_ft BATCH_SIZE_FT
                          batch size fine tunning
    -e EPOCHS, --epochs EPOCHS
                          epochs
    -ef EPOCHS_FT, --epochs_ft EPOCHS_FT
                          epochs fine tunning
    -hls HIDDEN_LAYER_SIZE, --hidden_layer_size HIDDEN_LAYER_SIZE
                          hidden layer size
                        
# Inference, see default arguments in the script

  usage: inference.py [-h] [-o OUTPUT] [-i INPUT] [-m MODEL] [-w WEIGHTS] [-c CONFIDENCE] [-lc LIST_CLASSES [LIST_CLASSES ...]]

  optional arguments:
    -h, --help            show this help message and exit
    -o OUTPUT, --output OUTPUT
                          path to the output directory
    -i INPUT, --input INPUT
                          path to the input directory
    -m MODEL, --model MODEL
                          path to the input model
    -w WEIGHTS, --weights WEIGHTS
                          path to the input weights
    -c CONFIDENCE, --confidence CONFIDENCE
                          probability confindence
    -lc LIST_CLASSES [LIST_CLASSES ...], --list_classes LIST_CLASSES [LIST_CLASSES ...]
