# struct2depth_dataprep

I provide the code to prepare data for [struct2depth](https://github.com/tensorflow/models/tree/master/research/struct2depth) model introduced by the paper *Depth Prediction Without the Sensors: Leveraging Structure for Unsupervised Learning from Monocular Videos, V. Casser, S. Pirk, R. Mahjourian, A. Angelova, AAAI Conference on Artificial Intelligence, 2019.* 

![](melbourne_estimation.mov)

First, make sure your tensorflow version is <=1.14. I encountered some problems while running their model with later versions.

**Data:**
You should download a driving video(eg. a KITTI sequence). Here I provide sample images in *data_preprocessed* folder.

**Masks:**
- To prepare instance labels, install matterport's Mask R-CNN implementation from [here](https://github.com/matterport/Mask_RCNN). Also get the pretrained model "mask_rcnn_coco.h5" released in [this link](https://github.com/matterport/Mask_RCNN/releases/tag/v2.0). 
- This Mask R-CNN implementation is not enough. Download the "save_image" addition from [here](https://github.com/matterport/Mask_RCNN/commit/bc8f148b820ebd45246ed358a120c99b09798d71).
See the Get_masks.ipynb, which applies following steps:
1. Infer the masks and only take the objects which are "car, person, bike, truck, motorcycle" (only relevant objects that can move).
2. Give a separate instance label to each mask and save it as a black&white photo(HxWx1) where the channel refers to label. 
3. Consecutive frames(let's say a sequence of 3) should have same objects with same labels. To fix that, run alignment code provided by authors. 
4. Save them as 'xxx-seg.png'. Make sure they are in the same directory or fix the directories in the code.
I provide samples in this repo.

**Run model:**
You can run training/inference of the model either with masks or without. Get the [KITTI pretrained model](https://drive.google.com/file/d/1mjb4ioDRH8ViGbui52stSUDwhkGrDXy8/view?usp=sharing).
- For training, there is an extra image processing step (see gen_data_kitti). It resizes frames to 416x128 and combines in triplets. Do it both for depth and RGB images. See the gen_data_kitti.py which I have modified accordingly.
- For inference, you can either feed images one by one, or in triplet format. 
- When running the code, set "--size_constraint_weight 0" which turns the weak-prior estimation off, because apparently there are some issues on that part and result in zero division errors.


**Credits**  
I got some help from these repos:
- https://github.com/ferdyandannes/struct2depth_train 
- https://github.com/amanrajdce/struct2depth_pub

