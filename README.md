# Continually learning social appropriateness of robot actions
This repo contains all source code for the work of Jonas Tjomsland, Sinan Kalkan and Hatice Gunes on Continually learning social appropriateness of robot actions [1]. It also includes components for extracting uncertainty estimates in predictions.

To run the code, create a new virtual environment with python 3.6.10, cd into the project folder and run "pip install -r requirements.txt"

## The code of this project can be seperated in three parts:
### Scripts for data structuring and data analysis of the created and labelled MANNERS-DB dataset:
#### The scripts in this part are found in the data folder and includes:
- data_analysis.ipynb\
Containing all statistical data analysis
- data_structuring.ipynb\
Cleaning and structuring the data from the Prolific crowd-labelling platform before saving it as csv.
### An extension of the Uncertainty-guided continual learning work by Ebrahimi et al. [2]:
#### The scripts in this part includes:
- UCB_modified.py\
The original work of Ebrahimi et al. with modifications to the loss function and some minor modifications to record the training process. The "loss" method is our 
work which fascilitates for regression, makes sure the correct output from the model is used to compute the loss and allow for aleatoric uncertainty to be obtained.
- Dataloaders\
The structure of the dataloaders used by Ebrahimi et al. is kept, but modifications are made to fit the MANNERS-DB dataset presented in our work.
- training.py\
The structure of the training script used by Ebrahimi et al. is kept, but modifications are made to fit the parameters and dataloaders necessary for our work.
### An evaluation of the predictive performance:
#### The scripts in this part includes:
- Evaluation.ipynb\
This is the main evaluation notebook where epistemic uncertainty is obtained and performance is evaluated.
- temp_eval.ipynb\
A notebook used to investigate the model performance at different stages of the Continual learning.


[1] Tjomsland J, Kalkan S, Gunes H. Mind Your Manners! A Dataset and A Continual Learning Approach for Assessing Social Appropriateness of Robot Actions. arXiv preprint arXiv:2007.12506. 2020 Jul 24.
[2] Ebrahimi, Sayna, et al. "Uncertainty-guided continual learning with bayesian neural networks." arXiv preprint arXiv:1906.02425 (2019).

