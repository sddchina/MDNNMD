MDNNMD
===============================
An implementation of multimodal deep neural network, a new model for human breast cancer prognosis prediction.

Reference
========================

Our manuscipt titled with "A multimodal deep neural network for human breast cancer prognosis prediction by integrating multi-dimensional data" has been accepted by IEEE/ACM Transactions on Computational Biology and Bioinformatics. If you find MDNNMD useful in your research, please consider citing:

Sun, D., Wang, M., & Li, A. (2018). A multimodal deep neural network for human breast cancer prognosis prediction by integrating multi-dimensional data. IEEE/ACM Transactions on Computational Biology and Bioinformatics.

Requirements
========================
    [python 2.7](https://www.python.org/downloads/)
    [TensorFilow 1.0](https://www.tensorflow.org/install/)
    [scikit-learn 0.18](http://scikit-learn.org/stable/)
    [cuda 8.0](https://developer.nvidia.com/cuda-downloads)
Usage
========================
python MDNNMD.py


Parameters of MDNNMD
=====================
The Parameters of MDNNMD are in our configuration file `mdnnmd.conf`. The descriptions of these parameters of MDNNMD are provided below:

    =================================================================================================
    | PARAMETER NAME       | DESCRIPTION                                                            |
    =================================================================================================
    |       α,β,γ          |α,β,γ are three damping factors used to balance the contribution for    |
    |                      |each DNN model. Here the sum of three damping factors should be equal 1.|
    -------------------------------------------------------------------------------------------------
    |         K            |the number of fold with cross validation experiment or an index file.   |
    -------------------------------------------------------------------------------------------------    
    |         D1           |the data file of gene expression profile                                |
    -------------------------------------------------------------------------------------------------
    |         D2           |the data file of copy number alteration profile.                        |
    -------------------------------------------------------------------------------------------------
    |         D3           |the data file of clinical information                                   |
    -------------------------------------------------------------------------------------------------
    |        LABEL         |the predict label of breast cancer patients with 1 or 0.                |
    -------------------------------------------------------------------------------------------------
    |      batch_size      |mini-batch size.                                                        |
    -------------------------------------------------------------------------------------------------
    |         bne          |batch normalization epsilon.                                            |
    -------------------------------------------------------------------------------------------------
    |   active_function    |active_function in our MDNNMD model, choose tanh or relu.               |
    -------------------------------------------------------------------------------------------------
    
Output files of MDNNMD
=====================
The descriptions of output files of MDNNMD are provided below:

    ====================================================================================================================
    | VARIABLE NAME         |                                   DESCRIPTION                                            |
    ====================================================================================================================
    | Prediction_score.txt  |The final prediction score of all samples with 10 fold cross validation experiment.       |
    |                       |The output of the MDNNMD with a softmax function.                                         |
    --------------------------------------------------------------------------------------------------------------------
    | Prediction_labels.txt |The prediction labels represent long-term patients with 0 and short-term patients with 1. |
    --------------------------------------------------------------------------------------------------------------------    

Contact
=====================
Author: Dongdong Sun @HILAB  
Maintainer: Dongdong Sun  
Mail: sddchina@mail.ustc.edu.cn
Date: 2017-5-30  
Health Informatics Lab, School of Information Science and Technology, University of Science and Technology of China

