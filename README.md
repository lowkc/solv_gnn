# Graph neural networks for solvation free energy prediction

## Installation
Dependencies: ```pytorch==1.6.0```, ```dgl==0.5.0```, ```rdkit=2020.03.5```

## Train the model
Check out the notebook for examples.

## Pre-trained model
To predict the solvation free energy for a list of solute-solvent pairs, you can load in the file using the code:
```python prediction.py --dataset-file='/path/to/prediction/.csv' --model-path=/path/to/saved/model/dir```
