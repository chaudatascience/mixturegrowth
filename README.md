# MixtureGrowth: Growing Neural Networks by Recombining Learned Parameters

This repo contains a Pytorch implementation for [MixtureGrowth](https://arxiv.org/pdf/2311.04251.pdf). 

If you find this code useful in your research then please cite

    @InProceedings{phamMixtureGrowth2024,
         author={Chau Pham and Piotr Teterwak and Soren Nelson and Bryan A. Plummer},
         title={MixtureGrowth: Growing Neural Networks by Recombining Learned Parameters},
         booktitle={IEEE Winter Conference on Applications of Computer Vision (WACV)},
         year={2024}}


This code was tested using pytorch v2.0 and python 3.10.

### Installation
To set up the environment, clone this repo and run:
    
    pip install -r requirements.txt


### Training New Models


You can train a model using scripts in `scripts/cifar/template_mixing` and `scripts/imagenet/template_mixing` with the following command:

    ./scripts/[dataset]/template_mixing/ours.sh <NUM GPUS> <DATASET> <TAG> <ADDITIONAL ARGUMENTS>

 For example, for CIFAR-100:

    ./scripts/cifar/template_mixing/ours.sh 1 cifar100 first_run --template_size 0.5,0.5,1 --growth_epochs 200 --ensemble_train_epochs 200 --epochs 440

Thi will fully train a WRN-28-5 on CIFAR-100 for 200 epochs (network 1), then train another WRN-28-5 on CIFAR-100 for 200 epochs (network 2). Finally, it combines networks 1 and 2, and grows to a WRN-28-10. The fully grown network is then further trained for an additional 40 epochs.

Similarly, for CIFAR-10:

    ./scripts/cifar/template_mixing/ours.sh 1 cifar10 run_cifar10 --template_size 0.5,0.5,1 --growth_epochs 200 --ensemble_train_epochs 200 --epochs 440 

For ImageNet, modify the dataset path in `scripts/imagenet/template_mixing/ours_bank.sh` and run:

    ./scripts/imagenet/template_mixing/ours_bank.sh 2 imagenet run_imagenet --template_size 0.5,0.5,1 --growth_epochs 90 --ensemble_train_epochs 74 --epochs 178  


You can see a listing and description of many parameter settings with:

    python main.py --help
    
Some key arguments would be:

| arguments  | description |
| ------------- | ------------- |
| --template_size  | [list] indicates the size of template at each growth step |
| --growth_epochs  | At which epoch we start to grow |
| --ensemble_train_epochs  | Number of epochs we continue to train before growing to the fully grown network|
| --epochs  | Total training epochs |

### Evaluation
You can test a model using:

    ./scripts/cifar/template_mixing/ours_test.sh <NUM GPUS> <DATASET> <TAG> <ADDITIONAL ARGUMENTS>
   

For example: 

    ./scripts/cifar/template_mixing/ours_test.sh 1 cifar100 first_run