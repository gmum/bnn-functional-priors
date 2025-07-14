# Revisiting the Equivalence of Bayesian Neural Networks and Gaussian Processes: On the Importance of Learning Activations.

## Introduction
This repository contains code accompanying the paper [Revisiting the Equivalence of Bayesian Neural Networks and Gaussian Processes: On the Importance of Learning Activations](https://arxiv.org/abs/2410.15777). Please see [the project website](https://bayes.ii.uj.edu.pl/).

## Abstract
Gaussian Processes (GPs) provide a convenient framework for specifying function-space priors, making them a natural choice for modeling uncertainty. In contrast, Bayesian Neural Networks (BNNs) offer greater scalability and extendability but lack the advantageous properties of GPs. This motivates the development of BNNs capable of replicating GP-like behavior. However, existing solutions are either limited to specific GP kernels or rely on heuristics.
We demonstrate that trainable activations are crucial for effective mapping of GP priors to wide BNNs. Specifically, we leverage the closed-form 2-Wasserstein distance for efficient gradient-based optimization of reparameterized priors and activations. Beyond learned activations, we also introduce trainable periodic activations that ensure global stationarity by design, and functional priors conditioned on GP hyperparameters to allow efficient model selection.
Empirically, our method consistently outperforms existing approaches or matches performance of the heuristic methods, while offering stronger theoretical foundations.

## Requirements
To setup environment execute `bash prepare_environment.sh`

## Code 
 - [Running experiments](bnngp/run_experiments.sh)
 - [Notebooks](bnngp/notebooks/)
 - [Testing code](bnngp/run_experiments_test.sh)
 
## Acknowledgements and Licence
This research is part of the project No. 2022/45/P/ST6/02969 co-funded by the National Science Centre and the European Union Framework Programme for Research and Innovation Horizon 2020 under the Marie Skłodowska-Curie grant agreement No. 945339. For the purpose of Open Access, the author has applied a CC-BY public copyright licence to any Author Accepted Manuscript (AAM) version arising from this submission. 
<p align="center">
  <img src="fig/eu_flag.jpg" width="50" />
  <img src="fig/ccby_licence.png" width="90" />
</p>

This research was in part funded by National Science Centre, Poland, 2022/45/N/ST6/03374.

We gratefully acknowledge Polish high-performance computing infrastructure PLGrid (HPC Center: ACK Cyfronet AGH) for providing computer facilities and support within computational grant no. PLG/2023/016302.
