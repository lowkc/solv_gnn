import torch
import numpy as np
from gnn.data.dataloader import DataLoaderSolvation
from gnn.utils import load_scalers
from gnn.prediction.load_model import load_model, load_dataset

def predict_from_file(model_path, dataset_file, solute_extra_features=None, solvent_extra_features=None, device=None):
    '''
    Make predictions for a file containing solute and solvent pair SMILES strings.
    '''
    prediction = get_prediction(model_path, dataset_file, solute_extra_features, solvent_extra_features, device)
    
    return prediction


def get_prediction(model_path, dataset_file, solute_extra_features=None, solvent_extra_features=None, device=None):
    model = load_model(model_path, device=device)
    model.to(device)
    dataset = load_dataset(model_path, dataset_file, solute_extra_features, solvent_extra_features)

    feature_names = ["atom", "bond", "global"]
    state_dict_objs = {"model": model}
    solute_scaler, solvent_scaler, label_scaler = load_scalers(state_dict_objs, save_dir=model_path, filename="best_checkpoint.pkl")
    dataset.normalize_features(solute_scaler, solvent_scaler)

    data_loader = DataLoaderSolvation(dataset, batch_size=100, shuffle=False)

    # evaluate
    predictions = evaluate(model, feature_names, data_loader, label_scaler, device)
    return predictions

def evaluate(model, nodes, data_loader, label_scaler, device=None):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for solute_batched_graph, solvent_batched_graph, label in data_loader:
            solute_feats = {nt: solute_batched_graph.nodes[nt].data["feat"] for nt in nodes}
            solvent_feats = {nt: solvent_batched_graph.nodes[nt].data["feat"] for nt in nodes}
            solvent_norm_atom = label["solvent_norm_atom"]
            solvent_norm_bond = label["solvent_norm_bond"]
            solute_norm_atom = label["solute_norm_atom"]
            solute_norm_bond = label["solute_norm_bond"]
            
            mean = label_scaler.mean
            stdev = label_scaler.std

            if device is not None:
                solute_feats = {k: v.to(device) for k, v in solute_feats.items()}
                solvent_feats = {k: v.to(device) for k, v in solvent_feats.items()}
                solute_norm_atom = solute_norm_atom.to(device)
                solute_norm_bond = solute_norm_bond.to(device)
                solvent_norm_atom = solvent_norm_atom.to(device)
                solvent_norm_bond = solvent_norm_bond.to(device)
            
            pred = model(solute_batched_graph, solvent_batched_graph, solute_feats, 
                     solvent_feats, solute_norm_atom, solute_norm_bond, 
                     solvent_norm_atom, solvent_norm_bond)
            pred = pred.view(-1).cpu().numpy()
            pred = (pred * stdev + mean)
            predictions.append(pred)
            labels.append(label['value'].numpy())

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels).ravel()

    return predictions, labels
