import itertools
import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, OrderedDict
from rdkit import Chem
from gnn.data.transformers import HeteroGraphFeatureStandardScaler, StandardScaler
from gnn.utils import to_path, list_split_by_size

logger = logging.getLogger(__name__)

def get_dataset_species(molecules):
    """
    Get all the species of atoms appearing in the the molecules.
    Args:
        molecules (list): rdkit molecules
    Returns:
        list: a sequence of species string
    """
    system_species = set()
    for mol in molecules:
        if mol[0] is None or mol[1] is None:
            continue
        species_solute = [a.GetSymbol() for a in mol[0].GetAtoms()]
        species_solvent = [a.GetSymbol() for a in mol[1].GetAtoms()]
        system_species.update(species_solute)
        system_species.update(species_solvent)

    return sorted(system_species)

class BaseDataset:
    """
    Base dataset class.
   Args:
    grapher (BaseGraph): grapher object that build different types of graphs:
        `hetero`, `homo_bidirected` and `homo_complete`.
        For hetero graph, atom, bond, and global state are all represented as
        graph nodes. For homo graph, atoms are represented as node and bond are
        represented as graph edges.
    molecules (list or str): rdkit molecules. If a string, it should be the path
        to the sdf file of the molecules.
    labels (list or str): each element is a dict representing the label for a bond,
        molecule or reaction. If a string, it should be the path to the label file.
    extra_features (list or str or None): each element is a dict representing extra
        features provided to the molecules. If a string, it should be the path to the
        feature file. If `None`, features will be calculated only using rdkit.
    feature_transformer (bool): If `True`, standardize the features by subtracting the
        means and then dividing the standard deviations.
    label_transformer (bool): If `True`, standardize the label by subtracting the
        means and then dividing the standard deviations. More explicitly,
        labels are standardized by y' = (y - mean(y))/std(y), the model will be
        trained on this scaled value. However for metric measure (e.g. MAE) we need
        to convert y' back to y, i.e. y = y' * std(y) + mean(y), the model
        prediction is then y^ = y'^ *std(y) + mean(y), where ^ means predictions.
        Then MAE is |y^-y| = |y'^ - y'| *std(y), i.e. we just need to multiple
        standard deviation to get back to the original scale. Similar analysis
        applies to RMSE.
    state_dict_filename (str or None): If `None`, feature mean and std (if
        feature_transformer is True) and label mean and std (if label_transformer is True)
        are computed from the dataset; otherwise, they are read from the file.
    """
    def __init__(
        self,
        grapher,
        molecules,
        labels,
        extra_features=None,
        feature_transformer=False,
        label_transformer=False,
        dtype="float32",
        state_dict_filename=None,
    ):
        if dtype not in ["float32", "float64"]:
            raise ValueError(f"`dtype {dtype}` should be `float32` or `float64`.")

        self.grapher = grapher
        self.molecules = (
            to_path(molecules) if isinstance(molecules, (str, Path)) else molecules
        )
        self.raw_labels = to_path(labels) if isinstance(labels, (str, Path)) else labels
        self.extra_features = (
            to_path(extra_features)
            if isinstance(extra_features, (str, Path))
            else extra_features
        )
        self.feature_transformer = feature_transformer
        self.label_transformer = label_transformer
        self.dtype = dtype
        self.state_dict_filename = state_dict_filename

        self.graphs = None
        self.labels = None
        self._feature_size = None
        self._feature_name = None
        self._feature_scaler_mean = None
        self._feature_scaler_std = None
        self._label_scaler_mean = None
        self._label_scaler_std = None
        self._species = None
        self._failed = None

        self._load()

    @property
    def feature_size(self):
        """
        Returns a dict of feature size with node type as the key.
        """
        return self._feature_size

    @property
    def feature_name(self):
        """
        Returns a dict of feature name with node type as the key.
        """
        return self._feature_name

    def get_feature_size(self, ntypes):
        """
        Get feature sizes.
        Args:
              ntypes (list of str): types of nodes.
        Returns:
             list: sizes of features corresponding to note types in `ntypes`.
        """
        size = []
        for nt in ntypes:
            for k in self.feature_size:
                if nt in k:
                    size.append(self.feature_size[k])
        # TODO more checks needed e.g. one node get more than one size
        msg = f"cannot get feature size for nodes: {ntypes}"
        assert len(ntypes) == len(size), msg

        return size

    @property
    def failed(self):
        """
        Whether an entry (molecule, reaction) fails upon converting using rdkit.
        Returns:
            list of bool: each element indicates whether a entry fails. The size of
                this list is the same as the labels, each one corresponds a label in
                the same order.
            None: is this info is not set
        """
        return self._failed

    def state_dict(self):
        d = {
            "feature_size": self._feature_size,
            "feature_name": self._feature_name,
            "feature_scaler_mean": self._feature_scaler_mean,
            "feature_scaler_std": self._feature_scaler_std,
            "label_scaler_mean": self._label_scaler_mean,
            "label_scaler_std": self._label_scaler_std,
            "species": self._species,
        }

        return d

    def load_state_dict(self, d):
        self._feature_size = d["feature_size"]
        self._feature_name = d["feature_name"]
        self._feature_scaler_mean = d["feature_scaler_mean"]
        self._feature_scaler_std = d["feature_scaler_std"]
        self._label_scaler_mean = d["label_scaler_mean"]
        self._label_scaler_std = d["label_scaler_std"]
        self._species = d["species"]

    def _load(self):
        """Read data from files and then featurize."""
        raise NotImplementedError

    @staticmethod
    def get_molecules(molecules):
        if isinstance(molecules, Path):
            path = str(molecules)
            supp = Chem.SDMolSupplier(path, sanitize=True, removeHs=False)
            molecules = [m for m in supp]
        return molecules

    @staticmethod
    def build_graphs(grapher, molecules, features, species):
        """
        Build DGL graphs using grapher for the molecules.
        Args:
            grapher (Grapher): grapher object to create DGL graphs
            molecules (list): rdkit molecules
            features (list): each element is a dict of extra features for a molecule
            species (list): chemical species (str) in all molecules
        Returns:
            list: DGL graphs
        """

        graphs = []
        for i, (m, feats) in enumerate(zip(molecules, features)):
            if m is not None:
                g = grapher.build_graph_and_featurize(
                    m, extra_feats_info=feats, dataset_species=species
                )
                # add this for check purpose; some entries in the sdf file may fail
                g.graph_id = i
            else:
                g = None

            graphs.append(g)

        return graphs

    def __getitem__(self, item):
        """Get data point with index
        Args:
            item (int): data point index
        Returns:
            g (DGLGraph or DGLHeteroGraph): graph ith data point
            lb (dict): Labels of the data point
        """
        g, lb, = self.graphs[item], self.labels[item]
        return g, lb

    def __len__(self):
        """
        Returns:
            int: length of dataset
        """
        return len(self.graphs)

    def __repr__(self):
        rst = "Dataset " + self.__class__.__name__ + "\n"
        rst += "Length: {}\n".format(len(self))
        for ft, sz in self.feature_size.items():
            rst += "Feature: {}, size: {}\n".format(ft, sz)
        for ft, nm in self.feature_name.items():
            rst += "Feature: {}, name: {}\n".format(ft, nm)
        return rst

class BondDataset(BaseDataset):
    def _load(self):

        logger.info("Start loading dataset")

        # get molecules, labels, and extra features
        molecules = self.get_molecules(self.molecules)
        raw_labels = self.get_labels(self.raw_labels)
        if self.extra_features is not None:
            extra_features = self.get_features(self.extra_features)
        else:
            extra_features = [None] * len(molecules)

        # get state info
        if self.state_dict_filename is not None:
            logger.info(f"Load dataset state dict from: {self.state_dict_filename}")
            state_dict = torch.load(str(self.state_dict_filename))
            self.load_state_dict(state_dict)

        # get species
        if self.state_dict_filename is None:
            species = get_dataset_species(molecules)
            self._species = species
        else:
            species = self.state_dict()["species"]
            assert species is not None, "Corrupted state_dict file, `species` not found"

        graphs = self.build_graphs(self.grapher, molecules, extra_features, species)

        self.graphs = []
        self.labels = []
        self._failed = []
        for i, g in enumerate(graphs):
            if g is None:
                self._failed.append(True)
            else:
                self.graphs.append(g)
                lb = {}
                for k, v in raw_labels[i].items():
                    if k == "value":
                        v = torch.tensor(v, dtype=getattr(torch, self.dtype))
                    elif k in ["bond_index", "num_bonds_in_molecule"]:
                        v = torch.tensor(v, dtype=torch.int64)
                    lb[k] = v
                self.labels.append(lb)
                self._failed.append(False)

        # this should be called after grapher.build_graph_and_featurize,
        # which initializes the feature name and size
        self._feature_name = self.grapher.feature_name
        self._feature_size = self.grapher.feature_size
        logger.info("Feature name: {}".format(self.feature_name))
        logger.info("Feature size: {}".format(self.feature_size))

        # feature transformers
        if self.feature_transformer:

            if self.state_dict_filename is None:
                feature_scaler = HeteroGraphFeatureStandardScaler(mean=None, std=None)
            else:
                assert (
                    self._feature_scaler_mean is not None
                ), "Corrupted state_dict file, `feature_scaler_mean` not found"
                assert (
                    self._feature_scaler_std is not None
                ), "Corrupted state_dict file, `feature_scaler_std` not found"

                feature_scaler = HeteroGraphFeatureStandardScaler(
                    mean=self._feature_scaler_mean, std=self._feature_scaler_std
                )

            if self.state_dict_filename is None:
                self._feature_scaler_mean = feature_scaler.mean
                self._feature_scaler_std = feature_scaler.std

            self.graphs = feature_scaler(self.graphs)
            logger.info("Feature scaler mean: {}".format(feature_scaler.mean))
            logger.info("Feature scaler std: {}".format(feature_scaler.std))

        # label transformers
        if self.label_transformer:

            # normalization
            values = torch.cat([lb["value"] for lb in self.labels])  # 1D tensor

            if self.state_dict_filename is None:
                mean = torch.mean(values)
                std = torch.std(values)
                self._label_scaler_mean = mean
                self._label_scaler_std = std
            else:
                assert (
                    self._label_scaler_mean is not None
                ), "Corrupted state_dict file, `label_scaler_mean` not found"
                assert (
                    self._label_scaler_std is not None
                ), "Corrupted state_dict file, `label_scaler_std` not found"
                mean = self._label_scaler_mean
                std = self._label_scaler_std

            values = (values - mean) / std

            # update label
            sizes = [len(lb["value"]) for lb in self.labels]
            lbs = torch.split(values, split_size_or_sections=sizes)
            for i, lb in enumerate(lbs):
                sz = len(lb)
                self.labels[i]["value"] = lb
                self.labels[i]["scaler_mean"] = mean.repeat(sz)
                self.labels[i]["scaler_stdev"] = std.repeat(sz)

            logger.info(f"Label scaler mean: {mean}")
            logger.info(f"Label scaler std: {std}")

        logger.info("Finish loading {} labels...".format(len(self.labels)))

    @staticmethod
    def get_labels(labels):
        if isinstance(labels, Path):
            labels = yaml_load(labels)
        return labels

    @staticmethod
    def get_features(features):
        if isinstance(features, Path):
            features = yaml_load(features)
        return features

class MoleculeDataset(BaseDataset):
    def __init__(
        self,
        grapher,
        molecules,
        labels,
        extra_features=None,
        feature_transformer=True,
        label_transformer=True,
        properties=["atomization_energy"],
        unit_conversion=True,
        dtype="float32",
    ):
        self.properties = properties
        self.unit_conversion = unit_conversion
        super(MoleculeDataset, self).__init__(
            grapher=grapher,
            molecules=molecules,
            labels=labels,
            extra_features=extra_features,
            feature_transformer=feature_transformer,
            label_transformer=label_transformer,
            dtype=dtype,
        )

    def _load(self):

        logger.info("Start loading dataset")

        # read label and feature file
        raw_labels, extensive = self._read_label_file()
        if self.extra_features is not None:
            features = yaml_load(self.extra_features)
        else:
            features = [None] * len(raw_labels)

        # build graph for mols from sdf file
        molecules = self.get_molecules(self.molecules)
        species = get_dataset_species(molecules)

        self.graphs = []
        self.labels = []
        natoms = []
        for i, (mol, feats, lb) in enumerate(zip(molecules, features, raw_labels)):

            if i % 100 == 0:
                logger.info("Processing molecule {}/{}".format(i, len(raw_labels)))

            if mol is None:
                continue

            # graph
            g = self.grapher.build_graph_and_featurize(
                mol, extra_feats_info=feats, dataset_species=species
            )
            # we add this for check purpose, because some entries in the sdf file may fail
            g.graph_id = i
            self.graphs.append(g)

            # label
            lb = torch.tensor(lb, dtype=getattr(torch, self.dtype))
            self.labels.append({"value": lb, "id": i})

            natoms.append(mol.GetNumAtoms())

        # this should be called after grapher.build_graph_and_featurize,
        # which initializes the feature name and size
        self._feature_name = self.grapher.feature_name
        self._feature_size = self.grapher.feature_size
        logger.info("Feature name: {}".format(self.feature_name))
        logger.info("Feature size: {}".format(self.feature_size))

        # feature and label transformer
        if self.feature_transformer:
            feature_scaler = HeteroGraphFeatureStandardScaler()
            self.graphs = feature_scaler(self.graphs)
            logger.info("Feature scaler mean: {}".format(feature_scaler.mean))
            logger.info("Feature scaler std: {}".format(feature_scaler.std))

        if self.label_transformer:
            labels = np.asarray([lb["value"].numpy() for lb in self.labels])
            natoms = np.asarray(natoms, dtype=np.float32)

            scaled_labels = []
            scaler_mean = []
            scaler_std = []

            label_scaler_mean = []
            label_scaler_std = []

            for i, is_ext in enumerate(extensive):
                if is_ext:
                    # extensive labels standardized by the number of atoms in the
                    # molecules, i.e. y' = y/natoms
                    lb = labels[:, i] / natoms
                    mean = np.zeros(len(lb))
                    std = natoms
                    label_scaler_mean.append(None)
                    label_scaler_std.append("num atoms")
                else:
                    # intensive labels standardized by y' = (y - mean(y))/std(y)
                    scaler = StandardScaler()
                    lb = labels[:, [i]]  # 2D array of shape (N, 1)
                    lb = scaler(lb)
                    lb = lb.ravel()
                    mean = np.repeat(scaler.mean, len(lb))
                    std = np.repeat(scaler.std, len(lb))
                    label_scaler_mean.append(scaler.mean)
                    label_scaler_std.append(scaler.std)
                scaled_labels.append(lb)
                scaler_mean.append(mean)
                scaler_std.append(std)

            scaled_labels = torch.tensor(
                np.asarray(scaled_labels).T, dtype=getattr(torch, self.dtype)
            )
            scaler_mean = torch.tensor(
                np.asarray(scaler_mean).T, dtype=getattr(torch, self.dtype)
            )
            scaler_std = torch.tensor(
                np.asarray(scaler_std).T, dtype=getattr(torch, self.dtype)
            )

            for i, (lb, m, s) in enumerate(zip(scaled_labels, scaler_mean, scaler_std)):
                self.labels[i]["value"] = lb
                self.labels[i]["scaler_mean"] = m
                self.labels[i]["scaler_stdev"] = s

            logger.info("Label scaler mean: {}".format(label_scaler_mean))
            logger.info("Label scaler std: {}".format(label_scaler_std))

        logger.info("Finish loading {} labels...".format(len(self.labels)))

    def _read_label_file(self):
        """
        Returns:
            rst (2D array): shape (N, M), where N is the number of lines (excluding the
                header line), and M is the number of columns (exluding the first index
                column).
            extensive (list): size (M), indicating whether the corresponding data in
                rst is extensive property or not.
        """

        rst = pd.read_csv(self.raw_labels, index_col=0)
        rst = rst.to_numpy()

        # supported property
        supp_prop = OrderedDict()
        supp_prop["atomization_energy"] = {"uc": 1.0, "extensive": True}

        if self.properties is not None:

            for prop in self.properties:
                if prop not in supp_prop:
                    raise ValueError(
                        "Property '{}' not supported. Supported ones are: {}".format(
                            prop, supp_prop.keys()
                        )
                    )
            supp_list = list(supp_prop.keys())
            indices = [supp_list.index(p) for p in self.properties]
            rst = rst[:, indices]
            convert = [supp_prop[p]["uc"] for p in self.properties]
            extensive = [supp_prop[p]["extensive"] for p in self.properties]
        else:
            convert = [v["uc"] for k, v in supp_prop.items()]
            extensive = [v["extensive"] for k, v in supp_prop.items()]

        if self.unit_conversion:
            rst = np.multiply(rst, convert)

        return rst, extensive

class SolvationDataset():
    def __init__(
        self,
        solute_grapher,
        molecules,
        labels,
        solvent_grapher=None,
        solvent_extra_features=None,
        solute_extra_features=None,
        feature_transformer=True,
        label_transformer=True,
        dtype="float32",
        state_dict_filename=None,
    ):
    
        if dtype not in ["float32", "float64"]:
            raise ValueError(f"`dtype {dtype}` should be `float32` or `float64`.")

        # Graphers - option to have a separate one for solute and solvent.
        self.solute_grapher = solute_grapher
        if solvent_grapher is None:
            self.solvent_grapher = solute_grapher
        else:          
            self.solvent_grapher = solvent_grapher

        self.molecules = (
            to_path(molecules) if isinstance(molecules, (str, Path)) else molecules
        )
        self.raw_labels = to_path(labels) if isinstance(labels, (str, Path)) else labels
        self.solute_extra_features = (
            to_path(solute_extra_features)
            if isinstance(solute_extra_features, (str, Path))
            else solute_extra_features
        )
        self.solvent_extra_features = (
            to_path(solvent_extra_features)
            if isinstance(solvent_extra_features, (str, Path))
            else solvent_extra_features
        )
        self.feature_transformer = feature_transformer
        self.label_transformer = label_transformer
        self.dtype = dtype
        self.state_dict_filename = state_dict_filename

        self.graphs = None
        self.labels = None

        self.label_scaler = None
        self.featuer_scaler = None

        self._solute_feature_size = None
        self._solute_feature_name = None
        self._solvent_feature_size = None
        self._solvent_feature_name = None
        self._species = None
        self._failed = None

        self._load()

    @staticmethod
    def get_labels(labels):
        if isinstance(labels, Path):
            df = pd.read_csv(labels, skipinitialspace=True, usecols=['DeltaGsolv'], index_col=None)
            labels = np.asarray(df)
        return labels

    @staticmethod
    def get_molecules(molecules):
        if isinstance(molecules, Path):
            path = str(molecules)
            # Load in file containing: solute SMILES, solvent SMILES, deltaGsolv value
            supp_0 = Chem.SmilesMolSupplier(path, delimiter=',', smilesColumn=0, sanitize=True)
            supp_1 = Chem.SmilesMolSupplier(path, delimiter=',', smilesColumn=1, sanitize=True)
            original_len = len(supp_1)

            zipped = zip(supp_0, supp_1)
            molecules = []

            for z in zipped:
                if z[0] is not None and z[1] is not None:
                    molecules.append((Chem.AddHs(z[0]), Chem.AddHs(z[1])))

            logger.info('Removed {} invalid SMILES.'.format(original_len - len(molecules)))
        return molecules

    @staticmethod
    def get_features(features):
        if isinstance(features, Path):
            features = pd.read_csv(features, index_col=None, header=None)
            features = np.asarray(features)
        return features
    

    def _load(self):
        logger.info("Start loading dataset.")
        # Get molecules, labels, atom types, and extra features
        molecules = self.get_molecules(self.molecules)
        raw_labels = self.get_labels(self.raw_labels)
        
        if self.state_dict_filename is not None:
            logger.info(f"Load dataset state dict from: {self.state_dict_filename}")
            state_dict = torch.load(str(self.state_dict_filename))
            self.load_state_dict(state_dict)
            species = self.state_dict()["species"]
            assert species is not None, "Corrupted state_dict file, `species` not found"
        else:
            species = get_dataset_species(molecules)
            self._species = species

        if self.solute_extra_features is not None:
            solute_features = self.get_features(self.solute_extra_features)
        else:
            solute_features = [None] * len(raw_labels)

        if self.solvent_extra_features is not None:
            solvent_features = self.get_features(self.solvent_extra_features)
        else:
            solvent_features = [None] * len(raw_labels)

        self.graphs = []
        self.labels = []

        for i, (mol, solu_feats, solv_feats, lb) in enumerate(zip(molecules, solute_features, solvent_features, raw_labels)):
            if i % 100 == 0:
                logger.info("Processing molecule {}/{}".format(i, len(raw_labels)))
            
            if mol is not None:
                # make molecular graph
                solute = mol[0]
                solvent = mol[1]
                g_solute = self.solute_grapher.build_graph_and_featurize(solute, extra_feats_info=solu_feats, dataset_species=species)
                g_solvent = self.solvent_grapher.build_graph_and_featurize(solvent, extra_feats_info=solv_feats, dataset_species=species)

                # added for checking purposes
                g_solute.graph_id = i
                g_solvent.graph_id = i
                g = [g_solute, g_solvent]
                lb = torch.tensor(lb, dtype=getattr(torch, self.dtype))
                self.labels.append({"value": lb, "id": i})
            else:
                continue
            self.graphs.append(g)

        # this should be called after grapher.build_graph_and_featurize,
        # which initializes the feature name and size

        self._solute_feature_name = self.solute_grapher.feature_name
        self._solute_feature_size = self.solute_grapher.feature_size
        self._solvent_feature_name = self.solvent_grapher.feature_name
        self._solvent_feature_size = self.solvent_grapher.feature_size

        logger.info("Solute feature names: {}".format(self.feature_names[0]))
        logger.info("Solute feature size: {}".format(self.feature_sizes[0]))
        logger.info("Solvent feature names: {}".format(self.feature_names[1]))
        logger.info("Solvent feature size: {}".format(self.feature_sizes[1]))
        logger.info("Finish loading {} labels...".format(len(self.labels)))


    def normalize_features(self, solute_scaler: HeteroGraphFeatureStandardScaler = None,
                                 solvent_scaler: HeteroGraphFeatureStandardScaler = None,
                                 replace_nan_token: int = 0):
        """
        If a StandardScaler` is provided, it is used to perform the normalization.
        Otherwise, a StandardScaler` is first fit to the features in this dataset
        and is then used to perform the normalization.
        """
        num_graphs = self.graphs
        solute_graphs = list(itertools.chain.from_iterable(self.graphs))[::2]
        solvent_graphs = list(itertools.chain.from_iterable(self.graphs))[1::2]

        if solute_scaler is None and solvent_scaler is None:
            solute_scaler = HeteroGraphFeatureStandardScaler()
            solvent_scaler = HeteroGraphFeatureStandardScaler()

            solute_graphs = solute_scaler.transform(solute_graphs)
            solvent_graphs = solvent_scaler.transform(solvent_graphs)

            # Split back into lists of two
            #self.graphs = list_split_by_size(graphs, [2]*len(num_graphs))
            self.graphs = list(zip(solute_graphs, solvent_graphs))
        
        else:
            solute_graphs = solute_scaler.transform(solute_graphs)
            solvent_graphs = solvent_scaler.transform(solvent_graphs)
            self.graphs = list(zip(solute_graphs, solvent_graphs))

        return solute_scaler, solvent_scaler

    def normalize_labels(self):
        """
        Normalizes the targets of the dataset using a StandardScaler.
        Returns a StandardScaler fitted to the targets which can be used for predictions.
        """
        labels = np.asarray([lb["value"].numpy() for lb in self.labels])
        scaled_labels = []
        #scaler_mean = []
        #scaler_std = []

        label_scaler = StandardScaler()
        lb = labels.reshape(-1,1) # 2D array of shape (N, 1)
        lb = label_scaler.transform(lb)
        mean = np.repeat(label_scaler.mean, len(lb))
        std = np.repeat(label_scaler.std, len(lb))
        lb = lb.ravel()

        scaled_labels.append(lb)
        #scaler_mean.append(mean)
        #scaler_std.append(std)
        scaled_labels = torch.tensor(np.asarray(scaled_labels).T, dtype=getattr(torch, self.dtype))
        #scaler_mean = torch.tensor(np.asarray(scaler_mean).T, dtype=getattr(torch, self.dtype))
        #scaler_std = torch.tensor(np.asarray(scaler_std).T, dtype=getattr(torch, self.dtype))

        #for i, (lb, m, s) in enumerate(zip(scaled_labels, scaler_mean, scaler_std)):
        for i, lb in enumerate(scaled_labels):
            self.labels[i]["value"] = lb
            #self.labels[i]["scaler_mean"] = m
            #self.labels[i]["scaler_stdev"] = s

        return label_scaler

    @property
    def feature_sizes(self):
        """
        Returns a dict of feature size with node type as the key.
        """
        return self._solute_feature_size, self._solvent_feature_size
    
    @property
    def feature_names(self):
        """
        Returns a dict of feature name with node type as the key.
        """
        return self._solute_feature_name, self._solvent_feature_name

    def get_feature_size(self, ntypes):
        """
        Get feature sizes.
        Args:
              ntypes (list of str): types of nodes.
        Returns:
             list: sizes of features corresponding to note types in `ntypes`.
        """
        size = []
        for nt in ntypes:
            for k in self.solute_feature_size:
                if nt in k:
                    size.append(self.solute_feature_size[k])
        # TODO more checks needed e.g. one node get more than one size
        msg = f"cannot get feature size for nodes: {ntypes}"
        assert len(ntypes) == len(size), msg

        return size

    @property
    def failed(self):
        """
        Whether an entry (molecule, reaction) fails upon converting using rdkit.
        Returns:
            list of bool: each element indicates whether a entry fails. The size of
                this list is the same as the labels, each one corresponds a label in
                the same order.
            None: is this info is not set
        """
        return self._failed

    def state_dict(self):
        d = {
            "solute feature_size": self._solute_feature_size,
            "solute feature_name": self._solute_feature_name,
            "solvent feature_size": self._solvent_feature_size,
            "solvent feature_name": self._solvent_feature_name,
            "species": self._species,
        }

        return d

    def load_state_dict(self, d):
        self._solute_feature_size = d["solute feature_size"]
        self._solute_feature_name = d["solute feature_name"]
        self._solvent_feature_size = d["solvent feature_size"]
        self._solvent_feature_name = d["solvent feature_name"]
        self._species = d["species"]

    def __getitem__(self, item):
        """Get data point with index
        Args:
            item (int): data point index
        Returns:
            g (DGLGraph or DGLHeteroGraph): graph ith data point
            lb (dict): Labels of the data point
        """
        g, lb, = self.graphs[item], self.labels[item]
        return g, lb

    def __len__(self):
        """
        Returns:
            int: length of dataset
        """
        return len(self.graphs)

    def __repr__(self):
        rst = "Dataset " + self.__class__.__name__ + "\n"
        rst += "Length: {}\n".format(len(self))

        for ft, sz in self.feature_sizes[0].items():
            rst += "Solvent feature: {}, size: {}\n".format(ft, sz)
        for ft, sz in self.feature_sizes[1].items():
            rst += "Solute feature: {}, size: {}\n".format(ft, sz)

        for ft, nm in self.feature_names[0].items():
            rst += "Solute feature: {}, name: {}\n".format(ft, nm)
        for ft, nm in self.feature_names[1].items():
            rst += "Solvent feature: {}, name: {}\n".format(ft, nm)
        return rst


class Subset(SolvationDataset):
    def __init__(self, dataset, indices):
        self.dtype = dataset.dtype
        self.dataset = dataset
        self.indices = indices
        self.labels = [self.dataset.labels[j] for j in indices]
        self.graphs = [self.dataset.graphs[j] for j in indices]
    
    @property
    def feature_sizes(self):
        return self.dataset.feature_sizes

    @property
    def feature_names(self):
        return self.dataset.feature_names

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)



def load_mols_labels(file):

    df = pd.read_csv(file, skipinitialspace=True, usecols=['DeltaGsolv'], index_col=None)
    labels = np.asarray(df)

    supp_0 = Chem.SmilesMolSupplier(file, delimiter=',', smilesColumn=0, sanitize=True)
    supp_1 = Chem.SmilesMolSupplier(file, delimiter=',', smilesColumn=1, sanitize=True)
    original_len = len(supp_1)

    zipped = zip(supp_0, supp_1)
    molecules = []

    for i, z in enumerate(zipped):
        if z[0] is not None and z[1] is not None:
            molecules.append((Chem.AddHs(z[0]), Chem.AddHs(z[1])))
        else:
            print(i, z[0], z[1])

    if original_len != len(molecules):    
        logger.info('Removed {} invalid SMILES.'.format(original_len - len(molecules)))

    return molecules, labels
    

def train_validation_test_split(dataset, validation=0.1, test=0.1, random_seed=0):
    """
    Split a dataset into training, validation, and test set.
    The training set will be automatically determined based on `validation` and `test`,
    i.e. train = 1 - validation - test.
    Args:
        dataset: the dataset
        validation (float, optional): The amount of data (fraction) to be assigned to
            validation set. Defaults to 0.1.
        test (float, optional): The amount of data (fraction) to be assigned to test
            set. Defaults to 0.1.
        random_seed (int, optional): random seed that determines the permutation of the
            dataset. Defaults to 0.
    Returns:
        [train set, validation set, test_set]
    """
    assert validation + test < 1.0, "validation + test >= 1"
    size = len(dataset)
    num_val = int(size * validation)
    num_test = int(size * test)
    num_train = size - num_val - num_test

    if random_seed is not None:
        np.random.seed(random_seed)
    idx = np.random.permutation(size)
    train_idx = idx[:num_train]
    val_idx = idx[num_train : num_train + num_val]
    test_idx = idx[num_train + num_val :]
    return [
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
          ]
