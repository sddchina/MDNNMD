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
------------------------

Parameters of MDNNMD
=====================
The Parameters of MDNNMD are in our configuration file configuration.txt. The descriptions of these parameters of MDNNMD are provided below:

    =================================================================================================
    | PARAMETER NAME       | DESCRIPTION                                                            |
    =================================================================================================
    |CompLeastProportion   |Least sample proportion included in each components, which represents   |
    |                      |minimum proportion of the samples in every components given by the      |
    |                      |mCGfinder. The default proportion is set to 15%.                        |
    -------------------------------------------------------------------------------------------------
    |maxCompoent           |Maximum number of components, which denotes the number of components    |
    |                      |given components given by mCGfinder at most. The default number is 5.   |
    -------------------------------------------------------------------------------------------------
    |NetConf.lambda_T      |The tuning parameter of network regularization, which is used to balance|
    |                      |the fitness of the model (first term) and the smoothness of the scores  |
    |                      |of connected genes (second term). The default number is 0.1.            |
    -------------------------------------------------------------------------------------------------

