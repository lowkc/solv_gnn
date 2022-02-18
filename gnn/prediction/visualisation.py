import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from gnn.data.dataloader import DataLoaderSolvation
from gnn.data.dataset import SolvationDataset
from gnn.prediction.load_model import load_model, load_dataset
from gnn.utils import seed_torch, to_path, load_scalers
from pathlib import Path

def eval_with_weights(model_path, data_file, solute_extra_features=None, solvent_extra_features=None, device=None, return_df=False):
    """
    Load the given model and make predictions on the compounds in data_file.
    This function allows you to visualise the weights of the interaction maps.
    """
    model = load_model(model_path, device=device, pretrained=True)

    state_dict_objs = {"model": model}
    solute_scaler, solvent_scaler, label_scaler = load_scalers(state_dict_objs, save_dir=model_path, filename="best_checkpoint.pkl")

    dataset = load_dataset(model_path, data_file, solute_extra_features=solute_extra_features,
                             solvent_extra_features=solvent_extra_features)
    dataset.normalize_features(solute_scaler, solvent_scaler)
    data_loader = DataLoaderSolvation(dataset, batch_size=100, shuffle=False)
    model.to(device)
    model.eval()
    
    nodes = ["atom", "bond", "global"]

    targets = []
    preds = []
    ids = []
    solute_weights = []
    solvent_weights = []
    
    with torch.no_grad():
        for solute_batched_graph, solvent_batched_graph, label in data_loader:
            solute_feats = {nt: solute_batched_graph.nodes[nt].data["feat"] for nt in nodes}
            solvent_feats = {nt: solvent_batched_graph.nodes[nt].data["feat"] for nt in nodes}
            target = torch.squeeze(label["value"])
            target_id = label["id"]
            solvent_norm_atom = label["solvent_norm_atom"]
            solvent_norm_bond = label["solvent_norm_bond"]
            solute_norm_atom = label["solute_norm_atom"]
            solute_norm_bond = label["solute_norm_bond"]
            
            if device is not None:
                solute_feats = {k: v.to(device) for k, v in solute_feats.items()}
                solvent_feats = {k: v.to(device) for k, v in solvent_feats.items()}
                target = target.to(device)
                solute_norm_atom = solute_norm_atom.to(device)
                solute_norm_bond = solute_norm_bond.to(device)
                solvent_norm_atom = solvent_norm_atom.to(device)
                solvent_norm_bond = solvent_norm_bond.to(device)

            pred = model(solute_batched_graph, solvent_batched_graph, solute_feats, 
                     solvent_feats, solute_norm_atom, solute_norm_bond, 
                     solvent_norm_atom, solvent_norm_bond)
            pred = pred.view(-1)
            pred = label_scaler.inverse_transform(pred.cpu()).to(device)
            
            preds.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
            ids.append(target_id)
            
            solu_wt, solv_wt = model.visualise_attn_weights(solute_batched_graph, solvent_batched_graph, solute_feats, 
                     solvent_feats, solute_norm_atom, solute_norm_bond, 
                     solvent_norm_atom, solvent_norm_bond)
            
            solute_weights.append(solu_wt)
            solvent_weights.append(solv_wt)
        
    predictions = np.concatenate(preds)
    targets = np.concatenate(targets)
    ids = np.concatenate(ids)
    solute_weights = np.concatenate(solute_weights)
    solvent_weights = np.concatenate(solvent_weights)

    if return_df:
        df = pd.DataFrame({
        "id": ids,
        "target": targets,
        "prediction": predictions,
        "solute_weights": solute_weights,
        "solvent_weights": solvent_weights,
        })
        df.to_csv('weights.csv', index=False)

    return ids, targets, predictions, solute_weights, solvent_weights