## TIPRDC: Task-Independent Privacy-Respecting Data Crowdsourcing Framework for Deep Learning with Anonymized Intermediate Representations
### Introduction
---
This repo is the PyTorch code for our Best Student Paper in KDD 2020 ([TIPRDC: Task-Independent Privacy-Respecting Data Crowdsourcing Framework for Deep Learning with Anonymized Intermediate Representations](https://dl.acm.org/doi/abs/10.1145/3394486.3403125), [slides](https://sites.duke.edu/angli/files/2020/12/KDD20_AngLi.pdf), [video](https://youtu.be/60FPYqXmYgU)).

#### Abstract
The success of deep learning partially benefits from the availability of various large-scale datasets. These datasets are often crowdsourced from individual users and contain private information like gender, age, etc. The emerging privacy concerns from users on data sharing hinder the generation or use of crowdsourcing datasets and lead to hunger of training data for new deep learning applications. One naive solution is to pre-process the raw data to extract features at the user-side, and then only the extracted features will be sent to the data collector. Unfortunately, attackers can still exploit these extracted features to train an adversary classifier to infer private attributes. Some prior arts leveraged game theory to protect private attributes. However, these defenses are designed for known primary learning tasks, the extracted features work poorly for unknown learning tasks. To tackle the case where the learning task may be unknown or changing, we present TIPRDC, a task-independent privacy-respecting data crowdsourcing framework with anonymized intermediate representation. The goal of this framework is to learn a feature extractor that can hide the privacy information from the intermediate representations; while maximally retaining the original information embedded in the raw data for the data collector to accomplish unknown learning tasks. We design a hybrid training method to learn the anonymized intermediate representation: (1) an adversarial training process for hiding private information from features; (2) maximally retain original information using a neural-network-based mutual information estimator. We extensively evaluate TIPRDC and compare it with existing methods using two image datasets and one text dataset. Our results show that TIPRDC substantially outperforms other existing methods. Our work is the first task-independent privacy-respecting data crowdsourcing framework.

![The overview of TIPRDC.](https://github.com/charleslipku/TIPRDC/blob/main/overview.png "The overview of TIPRDC.")



### Dependencies
---
Tested stable dependencies:
* PyTorch 1.0
* Python 3.6
* Numpy
* [visdom](https://github.com/facebookresearch/visdom)

### Dataset
---


### How to Train from Scratch
---
