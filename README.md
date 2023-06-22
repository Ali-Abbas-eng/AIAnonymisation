# AI-based Data Redaction for GDPR Compliance

## Overview
The General Data Protection Regulations (GDPR) enforces rules relating to the protection of natural persons with regards to the processing of personal data and rules relating to the free movement of personal data. This is done to protect fundamental rights and freedoms of natural persons and in particular their right to the protection of personal data. To comply with GDPR, companies have to redact information that exposes personal details about individuals as stated in the GDPR manual. These details include human faces and license plates.

In light of this, AI can do a great job in determining the position of such information in available data. Our project aims to create a pipeline to pre-process data using AI by automatically determining the location of faces and/or license plates in an image.

![sample predictions of the faster_rcnn_R_50_C4_3x](assets/predictions.png)


## Project Plan
In order to get a model that can accurately annotate images that might or might not contain faces and/or license plates, the process involves the following steps:
- [x] Dataset selection 
- [x] Data preparation 
- [x] Model selection
- [x] Training
- [x] Evaluation
- [x] Candidate Model Selection
- [ ] Train Candidate Model on a Larger Data Sample

### Plan Details
#### Dataset Selection
Face Detection Datasets : 
- [x] CelebA (**205, 999** annotated faces), [CelebA Dataset Website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
- [x] Wider Face (**393, 703** annotated faces) [Wider Face Dataset Website](http://shuoyang1213.me/WIDERFACE/).
- [x] FDDB (**5, 171** annotated faces) [FDDB Dataset Website](http://vis-www.cs.umass.edu/fddb/).

License Plates Detection Dataset: 
-[x] Chinese City Parking Data (350k annotated license plates)[CCPD 2019 Dataset Website](https://github.com/detectRecog/CCPD).

#### Data Preparation
To download and clean the datasets you can use the dataset_instantiation.py file in the following pattern:

    `python src/dataset_instantiation.py    --dataset_name <DATASET_NAME>

                                            --download (optional) use if you don't have the dataset on your device
                                            
                                            --extract (optional) use if you have the dataset only in compressed format
                                            
                                            --splits (optional) paths to JSON files in which data splits information will be saved in COCO format
                                            
                                            --proportions (optional) use to determine the proportion of each split w.r.t original dataset
                                            
                                            --clear_cache (optional), use this flag to delete downloaded dataset files after extraction` 
**Please Note**: DATASET_NAME can be one of the following celeba, ccpd2019, fddb, wider_face_train, or wider_face_val
### Model Selection
Any of the models available on the [Detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-object-detection-baselines) **Under COCO Object Detection Baselines** can be used to instantiate a model for training.

### Training
Training can be done with the help of train.py file in the following pattern

    `python src/train.py    --network_base_name MODEL_NAME (in the model zoo page it's "<table_name>_<row value in the name column")

                            --train_files path/to/first_train_file.json path/to/second_train_file.json ...
                                                                                    
                            --valid_files path/to/first_validation_file.json path/to/second_validation_file.json ...
                                                                                   
                            --output_directory path/to/OUTPUT_DIRECTORY (there will an additional folder with the same name of the network_base_name)
                                                                                    
                            --decay_frequency the interval at which learning rate will be decreased (default 10_000)
                                                                                    
                            --decay_gamma the factor with which leanring rate will be multiplied each decay_freq (default 0.5)
                                                                                   
                            --initial_leanring_rate LEARNING_RATE (step) (default 0.00025)
                                                                                   
                            --train_steps number of training steps (**NOT epochs**) (default 50_000)
                                                                                   
                            --eval_steps interval of training steps after which an evaluation process will start (default 10_000)
                                                                                   
                            --batch_size number of images in one step (default 2)
                                                                                    
                            --minimum_learning_rate the value after which learning rate decay will be obsolete (default 1e-5)
                                                                                    
                            --freeze_at the index of the last layer to be frozen (default 2)
                                                                                   
                            --roi_head number of region of interest heads in the output layer (default 256)`
**Example of the network_base_name would be faster_rcnn_R_50_C4_3x**
**`Example Command: python src/train.py --network_base_name faster_rcnn_R_50_C4_3x --train_files data/raw/WiderFaceTrain.json --valid_files data/raw/WiderFaceVal.json --decay_freq 2000 --decay_gamma 0.8 --output_directory output/trial_010 --initial_learning_rate 0.00025 --train_steps 25000 --eval_steps 2000 --batch_size 2 --min_learning_rate 0.000001 --freeze_at 2 --roi_heads 256`**

### Environment Setup
-[] TBD

#### Training Parameters
The parameters are fixed through all training experiments:
- learning rate: 0.00025 (decay at the 30kth step by a factor of 0.7)
- number of steps: 60k
- batch size: 1

#### Evaluation Summary 
Summary of training results showing the best performing model (for an intuition of which model is more promising for longer training schedule):

|           Model           | Localization Loss  | Classification Loss | Total Loss  | Average Precision (AP) |
|:-------------------------:|:------------------:|:-------------------:|:-----------:|:----------------------:|
| faster_rcnn_R_101_FPN_3x  |       0.0411       |       0.0168        |   0.0636    |          70.6          |
|  faster_rcnn_R_50_DC5_3x  |       0.0353       |       0.0114        |   0.0566    |         72.11          |
|  faster_rcnn_R_101_C4_3x  |       0.0373       |       0.0142        |   0.0744    |         71.76          |
|   retinanet_R_50_FPN_3x   |       0.153        |       0.0247        |    0.182    |         71.08          |
|  faster_rcnn_R_50_C4_3x   |       0.0332       |       0.0119        |   0.0789    |       **75.72**        |
|  retinanet_R_101_FPN_3x   |       0.126        |       0.0198        |    0.156    |         68.98          |



## References
1. [Art. 1 GDPR – Subject-matter and objectives - General Data Protection Regulation (GDPR) (gdpr-info.eu)](https://gdpr-info.eu/art-1-gdpr/)
2. https://github.com/understand-ai/anonymizer
3. [Large-scale CelebFaces Attributes (CelebA) Dataset, Ziwei Liu, Ping Luo, Xiaogang Wang, Xiaoou Tang, Multimedia Laboratory, The Chinese University of Hong Kong (cuhk.edu.hk)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
4. [GitHub - detectRecog/CCPD: [ECCV 2018] CCPD: a diverse and well-annotated dataset for license plate detection and recognition](https://github.com/detectRecog/CCPD)