![Demo result](https://raw.githubusercontent.com/peiyunh/tiny/master/selfie.png)

# Finding Tiny Faces
By Peiyun Hu and Deva Ramanan at Carnegie Mellon University. 

## Introduction
We develop a face detector (Tiny Face Detector) that can find ~800 faces out of ~1000 reportedly present, by making use of novel characterization of scale, resolution, and context to find small objects. Can you confidently identify errors? 

Tiny Face Detector was initially described in an [arXiv tech report](https://arxiv.org/abs/1612.04402). 

In this repo, we provide a MATLAB implementation of Tiny face detector, including both training and testing code. A demo script is also provided. 

### Citing us
If you find our work useful in your research, please consider citing: 
```latex
@article{hu2016finding,
  title={Finding Tiny Faces},
  author={Hu, Peiyun and Ramanan, Deva},
  journal={arXiv preprint arXiv:1612.04402},
  year={2016}
}
```

## Installation 
Clone the repo recursively so you have [my fork of MatConvNet](https://github.com/peiyunh/matconvnet/tree/9822ec97f35cf5a56ae22707cc1c04e0d738e7db). 
```zsh
git clone --recursive git@github.com:peiyunh/tiny.git
```

Compile MatConvNet by running following commands in MATLAB: 
```Matlab
>> cd matconvnet/;
>> addpath matlab/; 
>> vl_compilenn('enableImreadJpeg', true, 'enableGpu', true, 'cudaRoot', [cuda_dir],...
                'cudaMethod', 'nvcc', 'enableCudnn', true, 'cudnnRoot', [cudnn_dir]);
```

Compile our MEX function in MATLAB: 
```Matlab
>> cd utils/;
>> compile_mex; 
```

Download [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) and unzip data and annotation files to `data/widerface` such that: 
```zsh
$ ls data/widerface
wider_face_test.mat   wider_face_train.mat    wider_face_val.mat
WIDER_test/           WIDER_train/            WIDER_val/
```

## Demo
We provide a minimal demo `tiny_face_detector.m` that runs our detector on an single input image and output face detections: 
```Matlab
function bboxes = tiny_face_detector(image_path, output_path, prob_thresh, nms_thresh, gpu_id)
```

Here is a command you can run to reproduce our detection results on the world's largest selfie: 
```Matlab 
>> bboxes = tiny_face_detector('data/demo/selfie.jpg', './selfie.png', 0.5, 0.1, 1)
```

## Training 
To train a ResNet101-based Tiny Face Detector, run following command in MATLAB: 
```Matlab
>> hr_res101('train');           % which calls cnn_widerface.m
```

After training, run the following command to test on the validation set: 
```Matlab
>> hr_res101('test');            % which calls cnn_widerface_test_AB.m 
```

Finally, run the following command to evaluate the trained models: 
```Matlab
>> hr_res101('eval');            % which calls cnn_widerface_eval.m
```

Please refer to `scripts/hr_res101.m` for more details on how training/testing/evaluation is configured. 

### Clustering
We derive canonical bounding box shapes by K-medoids clustering (`cluster_rects.m`). For reproducibility, we provide our clustering results in `data/widerface/RefBox_N25.mat`. We also provide the version after template resolution analysis in `data/widerface/RefBox_N25_scaled.mat` (Fig. 8 in our paper).

### Evaluation
We provide both our own version of evaluation script (`cnn_widerface_eval.m`) and official evaluation script (`eval_tools/`). Our version consistently produces slightly lower numbers than the official one, but runs much faster. 
