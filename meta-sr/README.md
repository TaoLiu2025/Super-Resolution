# Meta-SR
Official implementation of **Meta-SR: A Magnification-Arbitrary Network for Super-Resolution(CVPR2019)(PyTorch)**
  
[Paper](https://arxiv.org/pdf/1903.00875.pdf)

This code is built on [Meta-SR(PyTorch)](https://github.com/XuecaiHu/Meta-SR-Pytorch.git)


# Contribution
* The original master branch has fearful bug! But in this code the bugs has been fixed
* Rewrite the dataloader. 
* Prepare dataset python script
* Modify the original RDB net to FSRCNN/Quick-SR net.
* Fix GPU out of memory bug by dividing whole image to patches.
# Requirements

* See Dockerfile


# Train and Test

##  prepare  dataset
   * Download the dataset [DIV2K](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip) and [test dataset](https://pan.baidu.com/s/1OwkaanEm9HyTqofoqt6Rjg) fetch code: w3hk [GoogleDrive](https://drive.google.com/open?id=14BW1q3_i6FRoq7PwwQ-81GbXWph6934x)
   * Use prepare_dataset.py script to get train and test dataset. 
## train 
```
cd /Meta-SR-Pytorch 
python main.py --model metacnn --save metacnn --ext bin --lr_decay 200 --epochs 1000 --n_GPUs 4 --batch_size 16 
```
## test 
```
python main.py --model metacnn --save metacnn --ext sep --pre_train ./experiment/metardn/model/model_best.pt --test_only --data_test Set5  --n_GPUs 1 --test_whole 
```
