# Graph neural networks for solvation free energy prediction

## Installation
Dependencies: ```pytorch==1.6.0```, ```dgl==0.5.0```, ```rdkit=2020.03.5``` 

## Train the model
Check out the notebook ```example_prediction_and_interaction_maps.ipynb``` for examples on how to load a trained model and how to visualise the weights of the interaction map.

## Predict using a pre-trained model
To predict the solvation free energy for a list of solute-solvent pairs, you can load in the file using the code:
```python prediction.py --dataset-file='/path/to/prediction/.csv' --model-path=/path/to/saved/model/dir```
