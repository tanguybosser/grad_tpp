# Preventing Conflicting Gradients in Neural Marked Temporal Point Processes.

This repository contains the code base to reproduce the main experiments of the paper 'Preventing Conflicting Gradients in Neural Marked Temporal Point Processes' under review for NEURIPS2024. 

The base code is built on the implementations of ["Neural Temporal Point Processes For Modeling Electronic Health Records"](https://github.com/babylonhealth/neuralTPPs), (Enguehard et. al., 2020) and ["On the Predictive accuracy of Neural Temporal Point Process Models for Continuous-time Event Data"](https://github.com/tanguybosser/ntpp-tmlr2023), (Bosser and Ben Taieb, 2023). We thank the authors for sharing their valuable code. 

## Abstract 

Neural Marked Temporal Point Processes (MTPP) are flexible models to capture complex temporal inter-dependencies between labeled events. These models inherently learn two predictive distributions: one for the arrival times of events and another for the types of events, also known as marks. In this study, we demonstrate that learning an MTPP model can be framed as a two-task learning problem, where both tasks share a common set of trainable parameters that are optimized jointly. Furthermore, we show that this can lead to conflicting gradients during training, where task-specific gradients are pointing in opposite directions. When such conflicts arise, following the average gradient can be detrimental to the learning of each individual tasks, resulting in overall degraded performance. To overcome this issue, we introduce novel parametrizations for neural MTPP models that allow for separate modeling and training of each task, effectively avoiding the problem of conflicting gradients. Through experiments on multiple real-world event sequence datasets, we demonstrate the benefits of our framework compared to the original model formulations.

## About

This repository allows to train and evaluate various neural TPP models following an encoder/decoder framework. The implementations of the different encoders can be found within the ```tpps/models/encoders``` folder, while the decoders are located in ```tpps/models/decoders```. For a specified combination of encoder and decoder, the computation of the negative log-likelihood and related quantities (density, intensity, cdf...) is carried out in ```tpps/models/enc_dec.py```. The train, validation and test metrics, and model checkpoints are finally stored within a directory specified by the user. See the 'Usage' section below for more details.   

## Installation

run the following command to setup the environment:

```shell script
conda env create -f environment.yml
conda activate tpp_grad
```

## Usage

### Training a model

Commands to train the different models used in our experiments can be found in the `runs` folder. According to where you want the results and checkpoints to be saved, you will need to change the "--save-results-dir" and "--save-check-dir" arguments. Additionaly, depending on where you decide to store the datasets, the "--load-from-dir" path needs to be adjusted. 

To get the complete list of training arguments, run the following command:

```
python3 scripts/train.py -h
```

## Data
The preprocessed datasets, as well as the splits divisions, can be found at [this link](https://github.com/tanguybosser/ntpp-tmlr2023). Place the 'data' folder (located within the 'processed' folder) at the top level of this repository. 


