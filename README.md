# Finding Tiny Faces
By [Peiyun Hu](https://cs.cmu.edu/~peiyunh), [Deva Ramanan](https://cs.cmu.edu/~deva)


## Setup: 
- Download and compile Matconvnet as a submodule. Make sure it passes all test cases after compilation.
- Download our model HR-res101 (99MB) or HR-res50 (32MB). Place the model under models/ 

## Testing
We provide a demo script to run our detector on an input image and output the detections, as in minimal_demo.m. By default, this script takes images under demo/data and outputs detections to demo/visual. 

## Training

#### Extra setup for training:
- Download WIDER FACE dataset and place its data and annotations under data/widerface/, following this structure: 
  - data/widerface/wider_face_train.mat (annotations for training set)
  - data/widerface/WIDER_train (images for training set)

#### Training
Coming soon.
