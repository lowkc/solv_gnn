import os
import time
import pickle
import yaml
import random
import torch
import dgl
import logging
import warnings
import sys
import shutil
import itertools
import copy
from pathlib import Path
import numpy as np
from typing import List, Any
from rdkit import Chem
from rdkit.Chem import BondType, AllChem
from rdkit.Geometry import Point3D
from gnn.data.transformers import HeteroGraphFeatureStandardScaler, StandardScaler

logger = logging.getLogger(__name__)

def reset_weights(model):
    '''
    Rest model weights to avoid weight leakage.
    '''
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def check_exists(path, is_file=True):
    p = to_path(path)
    if is_file:
        if not p.is_file():
            raise ValueError(f"File does not exist: {path}")
    else:
        if not p.is_dir():
            raise ValueError(f"File does not exist: {path}")


def create_directory(path, path_is_directory=False):
    p = to_path(path)
    if not path_is_directory:
        dirname = p.parent
    else:
        dirname = p
    if not dirname.exists():
        os.makedirs(dirname)


def pickle_dump(obj, filename):
    create_directory(filename)
    with open(to_path(filename), "wb") as f:
        pickle.dump(obj, f)


def pickle_load(filename):
    with open(to_path(filename), "rb") as f:
        obj = pickle.load(f)
    return obj


def yaml_dump(obj, filename):
    create_directory(filename)
    with open(to_path(filename), "w") as f:
        yaml.dump(obj, f, default_flow_style=False)


def yaml_load(filename):
    with open(to_path(filename), "r") as f:
        obj = yaml.safe_load(f)
    return obj


def stat_cuda(msg):
    print("-" * 10, "cuda status:", msg, "-" * 10)
    print(
        "allocated: {}M, max allocated: {}M, cached: {}M, max cached: {}M".format(
            torch.cuda.memory_allocated() / 1024 / 1024,
            torch.cuda.max_memory_allocated() / 1024 / 1024,
            torch.cuda.memory_cached() / 1024 / 1024,
            torch.cuda.max_memory_cached() / 1024 / 1024,
        )
    )


def seed_torch(seed=35, cudnn_benchmark=False, cudnn_deterministic=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    dgl.random.seed(seed)


def save_checkpoints(
    state_dict_objects, misc_objects, scaler_objects, is_best, msg=None, save_dir=None, filename="checkpoint.pkl"
):
    """
    Save checkpoints for all objects for later recovery.
    Args:
        state_dict_objects (dict): A dictionary of objects to save. The object should
            have state_dict() (e.g. model, optimizer, ...)
        misc_objects (dict): plain python object to save
        scaler_objects (dict): plain python dictionary, contains scaler information
        filename (str): filename for the checkpoint
    """
    objects = copy.copy(misc_objects)
    objects.update(scaler_objects)
    for k, obj in state_dict_objects.items():
        objects[k] = obj.state_dict()
    torch.save(objects, os.path.join(save_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, filename), os.path.join(save_dir, "best_checkpoint.pkl"))
        if msg is not None:
            logger.info(msg)



def load_checkpoints(state_dict_objects, save_dir=None, map_location=None, filename="checkpoint.pkl"):
    """
    Load checkpoints for all objects for later recovery.
    Args:
        state_dict_objects (dict): A dictionary of objects to save. The object should
            have state_dict() (e.g. model, optimizer, ...)
    """
    if save_dir is not None:
        checkpoints = torch.load(os.path.join(save_dir,str(filename)), map_location)
    else:
        checkpoints = torch.load(filename, map_location)
    for k, obj in state_dict_objects.items():
        state_dict = checkpoints.pop(k)
        obj.load_state_dict(state_dict)
    return checkpoints


def load_scalers(state_dict_objects, save_dir=None, map_location=None, filename="checkpoint.pkl"):
    """
    Load checkpoints for all objects for later recovery.
    Args:
        state_dict_objects (dict): A dictionary of objects to save. The object should
            have state_dict() (e.g. model, optimizer, ...)
    """
    if save_dir is not None:
        checkpoints = torch.load(os.path.join(save_dir,str(filename)), map_location)
    else:
        checkpoints = torch.load(filename, map_location)
    for k, obj in state_dict_objects.items():
        state_dict = checkpoints.pop(k)
        obj.load_state_dict(state_dict)
    
    solute_feature_scaler = HeteroGraphFeatureStandardScaler(
        mean = checkpoints['solute_features_scaler']['means'],
        std = checkpoints['solute_features_scaler']['stds']
        )
    solvent_feature_scaler = HeteroGraphFeatureStandardScaler(
        mean = checkpoints['solvent_features_scaler']['means'],
        std = checkpoints['solvent_features_scaler']['stds']
        )
    label_scaler = StandardScaler(
        mean=checkpoints['label_scaler']['means'],
        std = checkpoints['label_scaler']['stds']
        )

    return solute_feature_scaler, solvent_feature_scaler, label_scaler


def smiles_to_rdkit_mol(s, add_H=True):
    """
    Convert a smiles string to rdkit molecule.
    3D coords are created using RDkit: embedding then MMFF force filed (or UFF force
     field).
    Args:
        s (str): smiles of the molecule
        add_H (bool): whether to add H to the molecule
    Returns:
        rdkit mol
    """
    m = Chem.MolFromSmiles(s)
    if m is None:
        raise RdkitMolCreationError(f"smiles: {s}")

    if add_H:
        m = Chem.AddHs(m)

    m.SetProp("_Name", s)

    return m

def read_rdkit_mols_from_file(filename, format="smiles"):
    """
    Read molecules given in a file, supported format include `smiles`, `inchi`, `sdf`,
    and `pdb`.
    Args:
        filename (str or pathlib.Path): name of the file
        format (str): format of the molecules in the file
    Returns:
        (list): rdkit molecules
    """
    def from_smiles(filename):

        supp_0 = Chem.SmilesMolSupplier(filename, delimiter=',', smilesColumn=0, sanitize=True)
        supp_1 = Chem.SmilesMolSupplier(filename, delimiter=',', smilesColumn=1, sanitize=True)

        solvent_molecules = []
        solute_molecules = []
        for s in supp_0:
            try:
                m = smiles_to_rdkit_mol(s)
            except RdkitMolCreationError:
                m = None
            solute_molecules.append(m)
        for s in supp_1:
            try:
                m = smiles_to_rdkit_mol(s)
            except RdkitMolCreationError:
                m = None
            solvent_molecules.append(m)

        return [solute_molecules, solvent_molecules]
    
    filename = str(to_path(filename))
    if format == "smiles":
        return from_smiles(filename)

def to_path(path):
    return Path(path).expanduser().resolve()

def list_split_by_size(data: List[Any], sizes: List[int]) -> List[List[Any]]:
    """
    Split a list into `len(sizes)` chunks with the size of each chunk given by `sizes`.
    This is a similar to `np_split_by_size` for a list. We cannot use
    `np_split_by_size` for a list of graphs, because DGL errors out if we convert a
    list of graphs to an array of graphs.
    Args:
        data: the list of data to split
        sizes: size of each chunk.
    Returns:
        a list of list, where the size of each inner list is given by `sizes`.
    Example:
        >>> list_split_by_size([0,1,2,3,4,5], [1,2,3])
        >>>[[0], [1,2], [3,4,5]]
    """
    assert len(data) == sum(
        sizes
    ), f"Expect len(array) be equal to sum(sizes); got {len(data)} and {sum(sizes)}"

    indices = list(itertools.accumulate(sizes))

    new_data = []
    a = []
    for i, x in enumerate(data):
        a.append(x)
        if i + 1 in indices:
            new_data.append(a)
            a = []

    return new_data
