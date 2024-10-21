# High-Fidelity Transfer of Functional Priors for Wide Bayesian Neural Networks by Learning Activations

## Introduction
This repository contains code accompanying the submission `High-Fidelity Transfer of Functional Priors for Wide Bayesian Neural Networks by Learning Activations`. 

## Abstract
Function-space priors in Bayesian Neural Networks provide a more intuitive approach to embedding beliefs directly into the model’s output, thereby enhancing regularization, uncertainty quantification, and risk-aware decision-making. However, imposing function-space priors on BNNs is challenging. We address this task through optimization techniques that explore how trainable activations can accommodate complex priors and match intricate target function distributions. We discuss critical learning challenges, including identifiability, loss construction, and symmetries that arise in this context. Furthermore, we enable evidence maximization to facilitate model selection by conditioning the functional priors on additional hyperparameters. Our empirical findings demonstrate that even BNNs with a single wide hidden layer, when equipped with these adaptive trainable activations and conditioning strategies, can effectively achieve high-fidelity function-space priors, providing a robust and flexible framework for enhancing Bayesian neural network performance.

## Requirements
To setup environment execute `bash prepare_environment.sh`

## Code 
 - [Running experiments](bnngp/run_experiments.sh)
 - [Notebooks](bnngp/notebooks/)
 - [Testing code](bnngp/run_experiments_test.sh)
 
## Acknowledgements and Licence
This research is part of the project No. 2022/45/P/ST6/02969 co-funded by the National Science Centre and the European Union Framework Programme for Research and Innovation Horizon 2020 under the Marie Skłodowska-Curie grant agreement No. 945339. For the purpose of Open Access, the author has applied a CC-BY public copyright licence to any Author Accepted Manuscript (AAM) version arising from this submission. 
<p align="center">
  <img src="fig/eu_flag.jpg" width="350" />
  <img src="fig/ccby_licence.png" width="350" />
</p>

This research was in part funded by National Science Centre, Poland, 2022/45/N/ST6/03374.

We gratefully acknowledge Polish high-performance computing infrastructure PLGrid (HPC Center: ACK Cyfronet AGH) for providing computer facilities and support within computational grant no. PLG/2023/016302.