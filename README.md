# PML-Project-24-25
Authors: Gaetano Tedesco, Patrick Jensen, Carsten Jørgensen

## General 
This repository contains the code for the project of the course Probabilistic Machine Learning at UCPH. 

The project is divided in two parts:
* Exercise A: Implementation of the DDPM model and comparison of the models
* Exercise B: Gaussian Process 

## Installation
Before running the code, make sure you have the following packages installed:
* torch==2.3.0
* torchvision==0.18.0
* tocheval==0.0.7
* tqdm==4.66.5
* numpy==1.26.4
* matplotlib==3.9.2
* scipy==1.14.1
* sklearn==1.5.1
* pyro==1.9.1

In order to install the packages, you can download the `requirements.txt` file form the repo and use the following command:
```pip install -r requirements.txt```

## Exercise A
The files relative to exercise A are:
* `ddpms.py`: file containing all the implementation of the DDPM models produced
* `utils.py`: file containing all the utility functions used in the exercise (backbone networks definition, training loop, evaluation functions and support functions)
* `model_training.ipynb`: notebook executing the training loops for all the models
* `model_comparison.ipynb`: notebook executing the evaluation of the models (in our case generating the data for the models and computing FID)
* `visual_comparison.ipynb`: notebook executing the visual comparison of the models
* `model_checkpoints`: folder containing the checkpoints of the models trained. The checkpoints are meant as shortcut to run the model comparison (both numerical and visual) without retraining the models.

## Exercise B
There are 3 notebooks covering exercise B:

* fitting_MAP.final.ipynb
* fitting_MCMC_final.ipynb
* learning.final.ipynb

In fitting_MAP.final.ipynb we use gradient descent to find the hyperparameters of the kernel. In one of the last cells we output an array of test log-likelihood values.

All the code for using MCMC/NUTS to compute hyperparameters are in fitting_MCMC_final.ipynb.

The test log-likelihoods outputed in one of the last cells in fitting_MAP.final.ipynb needs to be pasted into fitting_MCMC_final.ipynb so we can create the boxplot comparing the two method.

fitting_MAP.final.ipynb also outputs kernel parameters. These are needed in one of the first cells in learning.final.ipynb for the integral constraint part.

 
