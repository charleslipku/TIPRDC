## TIPRDC: Task-Independent Privacy-Respecting Data Crowdsourcing Framework for Deep Learning with Anonymized Intermediate Representations
### Introduction
---
This repo is the PyTorch code for our Best Student Paper in KDD 2020 ([TIPRDC: Task-Independent Privacy-Respecting Data Crowdsourcing Framework for Deep Learning with Anonymized Intermediate Representations](https://dl.acm.org/doi/abs/10.1145/3394486.3403125), [slides](https://sites.duke.edu/angli/files/2020/12/KDD20_AngLi.pdf), [video](https://youtu.be/60FPYqXmYgU)).

#### Abstract
The success of deep learning partially benefits from the availability of various large-scale datasets. These datasets are often crowdsourced from individual users and contain private information like gender, age, etc. The emerging privacy concerns from users on data sharing hinder the generation or use of crowdsourcing datasets and lead to hunger of training data for new deep learning applications. One naive solution is to pre-process the raw data to extract features at the user-side, and then only the extracted features will be sent to the data collector. Unfortunately, attackers can still exploit these extracted features to train an adversary classifier to infer private attributes. Some prior arts leveraged game theory to protect private attributes. However, these defenses are designed for known primary learning tasks, the extracted features work poorly for unknown learning tasks. To tackle the case where the learning task may be unknown or changing, we present TIPRDC, a task-independent privacy-respecting data crowdsourcing framework with anonymized intermediate representation. The goal of this framework is to learn a feature extractor that can hide the privacy information from the intermediate representations; while maximally retaining the original information embedded in the raw data for the data collector to accomplish unknown learning tasks. We design a hybrid training method to learn the anonymized intermediate representation: (1) an adversarial training process for hiding private information from features; (2) maximally retain original information using a neural-network-based mutual information estimator. We extensively evaluate TIPRDC and compare it with existing methods using two image datasets and one text dataset. Our results show that TIPRDC substantially outperforms other existing methods. Our work is the first task-independent privacy-respecting data crowdsourcing framework.


<p align="center">
  <img src="https://github.com/charleslipku/TIPRDC/blob/main/overview.png">
   <b>The overview of TIPRDC.</b><br>
</p>

<p align="center">
  <img src="https://github.com/charleslipku/TIPRDC/blob/main/framework.png">
   <b>The hybrid learning method for training the fea- ture extractor.</b><br>
</p>



### Dependencies
---
Tested stable dependencies:
* PyTorch 1.0
* Python 3.6
* Numpy
* [visdom](https://github.com/facebookresearch/visdom)

### Dataset
---
* Image Dataset
  * [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  * [Labeled Faces in the Wild (LFW) Dataset](http://vis-www.cs.umass.edu/lfw/)
* Text Dataset
  * [Dialectal Tweets Dataset](http://slanglab.cs.umass.edu/TwitterAAE/)

### How to Train from Scratch
---
- ***Image/*** is the folder for training on the image dataset.
    - **VGG16.py** is the vanila VGG model.
    - **SegmentVGG16.py** partition the VGG into 2 sub-networks: *feature extractor* and *classifier*.
    - **MutualInformation.py** is the network for estimating the mutual information. 
    - **pre_train.py** is the module for pre-training. It will output a slightly pre-trained feature extractor and a classifier.
    - **main.py** is the core file for initializing the training process. It will call **train_decoder.py** and **train_extractor.py** to perform the adversarial training for achiving the target utility-privacy tradeoff. 
    - **decoder.py** contains a decoder to simulate the reconstruction attacks performed by the adversary, who aims to reconstruct the input image from intermediate representations. 
    - **MS_SSIM.py** defines the evaluation metric for the similarity between the raw images and the images that are  reconstructed by the attacker.

- ***Text/*** is the folder for training on the text dataset.
    - **LSTM.py** contains a feature extractor module, a classifier module, and a mutual information module. They are the counterparts of **SegmentVGG16.py** and **MUtualInformation.py** in **Image** folder.
    - **data_handler.py** will tag the raw data in the *DIAL* dataset.
    - **mydataset.py** works for the PyTorch Dataloader.
    - **pre_train.py** will get a slightly pre-trained text feature extractor and a classifier.
    - **main.py** is the core file for initializing the training process. It works together with **train_extractor.py** to run the adversarial training on text classification task.
