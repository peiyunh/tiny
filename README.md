# [Finding Tiny Faces](https://arxiv.org/abs/1612.04402)
By [Peiyun Hu](https://cs.cmu.edu/~peiyunh), [Deva Ramanan](https://cs.cmu.edu/~deva)

![Demo result](https://raw.githubusercontent.com/peiyunh/tiny/master/demo/visual/selfie.png)

## Setup: 

### Matconvnet
Clone this project with the `--recursive` option so that you have [my fork of Matconvnet](https://github.com/peiyunh/matconvnet/tree/9822ec97f35cf5a56ae22707cc1c04e0d738e7db) downloaded as a submodule. Make sure it passes all test cases after compilation. Feel free to refer to my compilation code as in `matconvnet/compile.m`. 

## Demo
We provide a demo script to run our detector on an input image and visualize the detections, as in `minimal_demo.m`. By default, this script takes images under `demo/data` and outputs detections to `demo/visual`. 

## Training

### Extra setup for training:
Download [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) and place its data and annotations under `data/widerface`, following such structure: 
- `data/widerface/wider_face_train.mat` (annotations for training set)
- `data/widerface/WIDER_train` (images for training set)

### Training code
Coming soon.
