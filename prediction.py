import sys, os
import time
import warnings
import argparse
from pathlib import Path
import pandas as pd
from rdkit import Chem

from gnn.prediction.prediction import predict_from_file
from gnn.data.dataset import load_mols_labels
from gnn.utils import (
    load_checkpoints,
    save_checkpoints,
    seed_torch,
    pickle_dump,
    yaml_dump,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error


def parse_args():
    parser = argparse.ArgumentParser(description="PredictOutputFile")

    # input files
    parser.add_argument('--dataset-file', type=str, default=None)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--dielectric-constants', type=str, default=None)

    # output dir
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU index. None to use CPU."
    )
    parser.add_argument("--output", type=str, default="predictions.txt")
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    if args.model_path == None or args.dataset_file == None:
        raise ValueError('Please provide the paths to the saved model file and .csv of SMILES strings.')

    if args.dielectric_constants is not None:
        dc_file = Path(args.dielectric_constants)      
    else:
        dc_file = None

    
    pred_vals, true_vals = predict_from_file(model_path=args.model_path,
                  dataset_file=args.dataset_file,
                  solvent_extra_features=dc_file,
                  device=args.gpu)
    
    df = pd.DataFrame({'Predictions': pred_vals, 'True': true_vals})
    df.to_csv(args.output, index=False)

    mae = mean_absolute_error(true_vals, pred_vals)
    rmse = mean_squared_error(true_vals, pred_vals, squared=False)

    print("Test MAE: {:12.6e} \n".format(mae))
    print("Test RMSE: {:12.6e} \n".format(rmse))

if __name__ == "__main__":
    main()