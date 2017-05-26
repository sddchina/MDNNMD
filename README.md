MDNNMD
===============================
An implementation of multimodal deep neural network, a new model for human breast cancer prognosis prediction.

Requirements
========================
    python 2.7
    TensorFilow 1.0
    scikit-learn 0.18
    cuda 8.0
Usage
========================
python MDNNMD.py


Parameters of MDNNMD
=====================
The Parameters of MDNNMD are in our configuration file configuration.txt. The descriptions of these parameters of MDNNMD are provided below:

    =================================================================================================
    | PARAMETER NAME       | DESCRIPTION                                                            |
    =================================================================================================
    |α,β,γ                 |α,β,γ are three damping factors used to balance the contribution for    |
    |                      |each DNN model. Here the sum of three damping factors should be equal 1.|
    -------------------------------------------------------------------------------------------------
    |K                     |The number of fold with cross validation experiment or an index file.   |
    -------------------------------------------------------------------------------------------------    
    |         D1           |The data file of gene expression profile                                |
    -------------------------------------------------------------------------------------------------
    |NetConf.lambda_T      |The tuning parameter of network regularization, which is used to balance|
    -------------------------------------------------------------------------------------------------
    |                      |the fitness of the model (first term) and the smoothness of the scores  |
    -------------------------------------------------------------------------------------------------
    |                      |of connected genes (second term). The default number is 0.1.            |
    -------------------------------------------------------------------------------------------------

