# Finding Tiny Faces
By [Peiyun Hu](https://cs.cmu.edu/~peiyunh), [Deva Ramanan](https://cs.cmu.edu/~deva)


## Setup: 
Download and compile Matconvnet as a submodule. Make sure it passes all test cases after compilation. Feel free to refer to my compilation code as in `matconvnet/compile.m`. 

## Demo
We provide a demo script to run our detector on an input image and visualize the detections, as in `minimal_demo.m`. By default, this script takes images under `demo/data` and outputs detections to `demo/visual`. 

## Training

### Extra setup for training:
Download [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) and place its data and annotations under `data/widerface`, following such structure: 
- `data/widerface/wider_face_train.mat` (annotations for training set)
- `data/widerface/WIDER_train` (images for training set)

### Training code
Coming soon.
