# Prediction of Drug-Target Affinity Based on Residual Structure Graph Convolutional Network

## Dataset
All data used in this paper are publicly available can be accessed here:

Davis and KIBA: https://github.com/hkmztrk/DeepDTA/tree/master/data

## Requirements  

matplotlib==3.5.2  
pandas==1.2.4  
torch_geometric==2.1.0 
CairoSVG==2.5.2  
torch==1.11.0  
tqdm==4.64.0  
opencv_python==4.5.1.48  
networkx==2.8.7  
numpy==1.21.5  
ipython==8.3.0  
rdkit==2022.9.1 
scikit_learn==1.1.2 

## Descriptions of folders and files in the RGraphDTA repository

* **regression** folder includes the source code of RGraphDTA for regression tasks in the davis and KIBA datasets and this folder have included raw data.
  + **data** folder contains raw data of the davis and KIBA datasets.
  + **log** folder includes the source codes to record the training process.
  + **save** folder, The weights obtained from training the model will be saved here.
  + **metrics.py** contains a series of metrics to evalute the model performances.
  + **model.py**, the implementation of RGraphDTA can be found here.
  + **preprocessing.py**, a file that preprocesses the raw data into graph format and should be executed before model trianing.
  + **test.py**, test a trained model and print the results.
  + **train.py**, train RGraphDTA model.
  + **utils.py** file includes useful tools for model training.

* **visualization** folder includes the source code for visualization of a trained model. Note that this folder incudes a pretrained model for visualization. 
  * **visualization_RGNN.py**, this file includes algorithms that can produce heatmaps to reveal how RGraphDTA makes decisions. The core of this file is GradAAM class that takes a model and a module (layer) that you want to visualize as input where the module is chosen as the last layer of RGNN in our experiments. 

## Step-by-step running:  


### 1. regression folder

- First, cd RGraphDTA/regression, and run preprocessing.py using  
  `python preprocessing.py`  

  Running preprocessing.py convert the raw data into graph format.

  In this step, the preprocessed davis and KIBA datasets will be obtained.

- Second, run train.py using `python train.py` to train MGraphDTA.

  If you want to change the dataset for training, modify the dataset parameters in params.

  Explanation of parameters

  - --lr: learning rate, default =  5e-4
  - --batch_size: default = 512

- To test a trained model please run test.py using

  `python test.py`

  This will return the MSE, CI, and  R2 performance in the test set.


### 2. Visualization using Grad-AAM

- First, cd RGraphDTA/visualization, and run preprocessing.py using  
  `python preprocessing.py`  
 
- Second, run visualization_RGNN.py using  
  `python visualization_RGNN.py`  
  and the visualization results will save in RGraphDTA/visualization/results folders. 
  and the RGraphDTA/visualization/pretrained_model folder contains a pre-trained model that can be used to produce heatmaps. If you want to test Grad-AAM in your own model, please replace this pre-trained model with your own one and modify the path in the visualization_RGNN.py file.

