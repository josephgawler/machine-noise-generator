# Machine Noise Generator (GAN)

#### Contributers: Joseph Gawler, Narayan Sharma, Julian Savini, Qinrong Cui

The goal of this project is to leverage the MIMII dataset (https://zenodo.org/records/3384388) to train an adversarial machine learning model.

Machine Learning is often used to process the sound of machines, and determine if the machine is functioning properly. One of the major limitations of this process is creating a dataset to train a model. This is because the cost of breaking machines, or wearing machines down such that they no longer work properly, is an extremely expensive process, IF these machines are expensive to construct. Therefore, we want to build a GAN model to generate synthetic sounds, imitating a machine that either functions properly or not. We hope our GAN can capture the nuances of what sounds indicate a machine is functioning properly. If time remains, we’d also like to build a transformer model afterwards, that can take the synthetic data, and identify if the machine is working properly or not.
