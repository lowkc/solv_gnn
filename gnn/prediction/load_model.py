import torch
import yaml
import gnn
import tarfile
import shutil
import tempfile
import os
from pathlib import Path
from gnn.model.gated_solv_network import GatedGCNSolvationNetwork, InteractionMap
from gnn.data.dataset import SolvationDataset, get_dataset_species, load_mols_labels

from gnn.utils import (
    load_checkpoints,
    check_exists,
    to_path,
    yaml_load,
    read_rdkit_mols_from_file,
    pickle_load
)


def _check_species(molecules, state_dict_filename):
    if isinstance(molecules, (str, Path)):
        check_exists(molecules)
        mols = read_rdkit_mols_from_file(molecules)
    else:
        mols = molecules
    
    species = get_dataset_species(mols)

    supported_species = torch.load(str(state_dict_filename))["species"]
    not_supported = []
    for s in species:
        if s not in supported_species:
            not_supported.append(s)
    if not_supported:
        not_supported = ",".join(not_supported)
        supported = ",".join(supported_species)
        raise ValueError(
            f"Model trained with a dataset having species: {supported}; cannot make "
            f"predictions for molecule containing species: {not_supported}. "
        )


def load_dataset(model_path, dataset_file, solute_extra_features=None, solvent_extra_features=None):
    state_dict_filename = to_path(model_path).joinpath("dataset_state_dict.pkl")
    molecules, labels = load_mols_labels(dataset_file)
    _check_species(molecules, state_dict_filename)

    graphers = pickle_load(os.path.join(model_path, 'graphers.pkl')) 

    dataset = SolvationDataset(
        solute_grapher = graphers[0],
        solvent_grapher = graphers[1],
        molecules = molecules,
        labels = labels,
        solvent_extra_features=solvent_extra_features,
        solute_extra_features=solute_extra_features,
        feature_transformer=True,
        label_transformer=True,
        state_dict_filename=state_dict_filename
    )

    return dataset


def load_model(model_path, checkpoint_path=None, pretrained=True, device=None):
    model_path = to_path(model_path)
    with open(model_path.joinpath("train_args.yaml"), "r") as f:
        model_args = yaml.load(f, Loader=yaml.Loader)

    if model_args.attention_map == False:
        model_type = GatedGCNSolvationNetwork
    else:
        model_type = InteractionMap

    model = model_type(
           solute_in_feats=model_args.solute_feature_size,
           solvent_in_feats=model_args.solvent_feature_size,
           embedding_size=model_args.embedding_size,
           gated_num_layers=model_args.gated_num_layers,
           gated_hidden_size=model_args.gated_hidden_size,
           gated_num_fc_layers=model_args.gated_num_fc_layers,
           gated_graph_norm=model_args.gated_graph_norm,
           gated_batch_norm=model_args.gated_batch_norm,
           gated_activation=model_args.gated_activation,
           gated_residual=model_args.gated_residual,
           gated_dropout=model_args.gated_dropout,
           num_lstm_iters=model_args.num_lstm_iters,
           num_lstm_layers=model_args.num_lstm_layers,
           set2set_ntypes_direct=model_args.set2set_ntypes_direct,
           fc_num_layers=model_args.fc_num_layers,
           fc_hidden_size=model_args.fc_hidden_size,
           fc_batch_norm=model_args.fc_batch_norm,
           fc_activation=model_args.fc_activation,
           fc_dropout=model_args.fc_dropout,
           outdim=1,
           conv="GatedGCNConv",
            )
    
    if checkpoint_path is None:
        checkpoint_path = model_path
    checkpoint_path = to_path(checkpoint_path)

    if pretrained:
        if device == None:
            load_checkpoints(
            {"model": model},
            map_location=torch.device("cpu"),
            filename=checkpoint_path.joinpath("best_checkpoint.pkl"),
        )
        else:
            load_checkpoints(
            {"model": model},
            map_location=torch.device("cuda:0"),
            filename=checkpoint_path.joinpath("best_checkpoint.pkl"),
        )

    return model
