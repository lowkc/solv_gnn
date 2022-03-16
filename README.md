# Graph neural networks for solvation free energy prediction

## Installation
Dependencies: ```pytorch==1.6.0```, ```dgl==0.5.0```, ```rdkit=2020.03.5``` 

## Train the model
Train on a GPU using the ```run_training.py``` script. CPU training is also possible, just make sure to modify the code accordingly. Hyperparameter optimisation is available via Optuna in ```run_optuna.py```.

### Dataset
The CompSolv-Exp dataset was used to train our model. The data (excluding proprietary MNSol data) can be found in the ESI of the Vermeire and Green (2021) Chem. Eng. J. paper: <https://doi.org/10.1016/j.cej.2021.129307>

## Predict using a pre-trained model
Check out the notebook ```example_prediction_and_interaction_maps.ipynb``` for examples on how to load a trained model and how to visualise the weights of the interaction map.

To predict the solvation free energy for a list of solute-solvent pairs, you can load in the file using the code:
```python prediction.py --dataset-file='/path/to/prediction/.csv' --model-path=/path/to/saved/model/dir``` 
Example files for DAAQ solubility (as in the paper) are in the *trained_models/data*, and a zipped pre-trained model checkpoint can be downloaded in release v0.1.0.