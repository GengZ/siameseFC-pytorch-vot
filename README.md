## SiamFC-PyTorch-VOT

* This is the PyTorch (0.3.1) implementation of SiamFC tracker [1], which was originally <a href="https://github.com/bertinetto/siamese-fc">implemented</a> using MatConvNet [2].
* In this project, we obtain slightly better results on VOT-16 and VOT-17 dataset.
* **This project is originally forked from [HengLan's implementation](https://github.com/HengLan/SiamFC-PyTorch), which is with PyTorch 0.4.0 for OTB dataset.**

## Why fork and tinker?

*   Make small modificatoins for using with VOT toolkit. (Run into some errors when directly using Heng's implementation.)
*   Make small changes for better convergence during training (in my case).

## Goal

* Ready-to-go version for using with VOT toolkit.
* As a baseline for related Siamese Tracker re-implementation.
* Several design choices tested.
* A more compact implementation of SiamFC [1].
* Reproduce the results of SiamFC [1] in VOT-2016 challenge (SiamFC-A), and in VOT-2017 challenge.

## Requirements

* Python 2.7.12

* Python-opencv 3.2.0

* PyTorch 0.3.1

* Numpy 1.14.2

* Other packages listed in requirements.txt

  ***The results using packages of other version than above not guaranteed.***

## Data curation 

* Download <a href="http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz">ILSVRC15</a>, and unzip it (let's assume that `$ILSVRC2015_Root` is the path to your ILSVRC2015)
* Move `$ILSVRC2015_Root/Data/VID/val` into `$ILSVRC2015_Root/Data/VID/train/`, so we have five sub-folders in `$ILSVRC2015_Root/Data/VID/train/`
* Move `$ILSVRC2015_Root/Annotations/VID/val` into `$ILSVRC2015_Root/Annotations/VID/train/`, so we have five sub-folders in `$ILSVRC2015_Root/Annotations/VID/train/`
* Generate image crops
  * cd `$SiamFC-PyTorch/ILSVRC15-curation/` (Assume you've downloaded the rep and its path is `$SiamFC-PyTorch`)
  * change `vid_curated_path` in `gen_image_crops_VID.py` to save your crops
  * run `$python gen_image_crops_VID.py`, then you can check the cropped images in your saving path (i.e., `vid_curated_path`). It takes a day or two for image crops generation.

* Generate imdb for training and validation
  * cd `$SiamFC-PyTorch/ILSVRC15-curation/`
  * change `vid_root_path` and `vid_curated_path` to your custom path in `gen_imdb_VID.py`
  * run `$python gen_imdb_VID.py`, then you will get two json files `imdb_video_train.json` (~ 430MB) and `imdb_video_val.json` (~ 28MB) in current folder, which are used for training and validation.

## Train

* cd `$SiamFC-PyTorch/Train/`
* Change `data_dir`, `train_imdb` and `val_imdb` to your custom cropping path, training and validation json files.
* run `$python run_Train_SiamFC.py`
* **some notes for training:**
  * the options for training are in `Config.py`
  * each epoch (50 in total) may take 6 minuts (Nvidia Titan Pascal, num_worker=8 in my case)

## Tracking

* cd `$SiamFC-PyTorch/Tracking/`
* Take a look at `Config.py` first, which contains all parameters for tracking
* Change `self.net_base_path` to the path saving your trained models
* Change `self.net` to indicate whcih model you want for evaluation, and I've uploaded a trained model `SiamFC_45_model.pth` in this rep (located in $SiamFC-PyTorch/Train/model/)
* The default parameters I use for my results is as listed in `Config.py`.
* Copy all the files under `$SiamFC-PyTorch/Train/matlab` to `$VOT-Workspace`. And modify paths in all those files. (Don't panic, just few lines in each file.)
* Run VOT evaluation as described in VOT toolkit documentation.

## Results

On VOT-16 dataset, reproduced model achieves **EAO 0.24**  ***vs***  **EAO 0.24**, which is SiamFC-A as listed in VOT-16 challenge paper.

On VOT-17 dataset, reproduced model achieves **EAO 0.20** ***vs*** **EAO 0.19**, which is given by VOT-17 challenge paper. (I also use the VOT-17 results from original paper, and compare results using toolkit)

## References

[1] L. Bertinetto, J. Valmadre, J. F. Henriques, A. Vedaldi, and P. H. Torr. Fully-convolutional siamese networks for object tracking. In ECCV Workshop, 2016.

[2] A. Vedaldi and K. Lenc. Matconvnet – convolutional neural networks for matlab. In ACM MM, 2015.

[3]https://github.com/HengLan/SiamFC-PyTorch


