# mrcnn_py2_tensorflow
Usage
====
1. Download the original Mask_RCNN from https://github.com/matterport/Mask_RCNN.git
2. Annotate your images with VGG Image Annotator (2.0.8), you can download it from http://www.robots.ox.ac.uk/~vgg/software/via/

(Under python3 environment)

3. Put train.py into the Mask_RCNN/samples/balloon folder
4. Modify the train.py file to fit your data files
![image](https://github.com/himlen1990/toolbox/blob/master/mrcnn_py2_tensorflow_converter/IMG/illustration1.png)
![image](https://github.com/himlen1990/toolbox/blob/master/mrcnn_py2_tensorflow_converter/IMG/illustration2.png)
5. Start Training (using the command like python3 train.py train --dataset=/path/to/balloon/dataset --weights=coco)
6. Once the trianing step finished, there should be some h5 files under the folder Mask_RCNN/logs/xxx..
7. Modify the tensorflow_model_converter.py file then run python3 tensorflow_model_converter.py, after that you should get a log folder

(Under python2 environment)

8. Now you can modify the main function of the tensorflow_test.py file and test the trained model by runing tensorflow_test.py under python2 environment
 