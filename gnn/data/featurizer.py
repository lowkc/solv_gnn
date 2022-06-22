"""
Featurise a molecule heterograph of atom, bond, and global nodes with RDKit.
"""

from dgl.batch import _batch_feat_dicts
import torch
import os
import itertools
from collections import defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem import rdEHTTools
from rdkit import RDConfig
from rdkit.Chem.rdchem import GetPeriodicTable
from openbabel import pybel
from xtb.libxtb import VERBOSITY_MUTED
from xtb.interface import Calculator, Param

class BaseFeaturizer:
    def __init__(self, dtype="float32"):
        if dtype not in ["float32", "float64"]:
            raise ValueError(
                "`dtype` should be `float32` or `float64`, but got `{}`.".format(dtype)
            )
        self.dtype = dtype
        self._feature_size = None
        self._feature_name = None

    @property
    def feature_size(self):
        """
        Returns:
            an int of the feature size.
        """
        return self._feature_size
    
    @property
    def feature_name(self):
        """
        Returns:
            a list of the names of each feature. Should be of the same length as 'feature size'.
        """
        return self._feature_name

    def __call__(self, mol, **kwargs):
        """
        Returns:
            A dictionary of the features.
        """
        raise NotImplementedError

class BondFeaturizer(BaseFeaturizer):
    """
    Base featurize all bonds in a molecule.
    The bond indices will be preserved, i.e. feature i corresponds to atom i.
    The number of features will be equal to the number of bonds in the molecule,
    so this is suitable for the case where we represent bond as graph nodes.
    Args:
        length_featurizer (str or None): the featurizer for bond length.
        length_featurizer_args (dict): a dictionary of the arguments for the featurizer.
            If `None`, default values will be used, but typically not good because this
            should be specific to the dataset being used.
    """
    def __init__(
        self, length_featurizer=None, length_featurizer_args=None, dtype="float32"
    ):
        super().__init__(dtype)
        self._feature_size = None
        self._feature_name = None

        if length_featurizer == "bin":
            if length_featurizer_args is None:
                length_featurizer_args = {"low": 0.0, "high": 2.5, "num_bins": 10}
            self.length_featurizer = DistanceBins(**length_featurizer_args)
        elif length_featurizer == "rbf":
            if length_featurizer_args is None:
                length_featurizer_args = {"low": 0.0, "high": 2.5, "num_centers": 10}
            self.length_featurizer = RBF(**length_featurizer_args)
        elif length_featurizer is None:
            self.length_featurizer = None
        else:
            raise ValueError(
                "Unsupported bond length featurizer: {}".format(length_featurizer)
            )

class BondAsNodeFeaturizerMinimum(BondFeaturizer):
    """
    Featurize all bonds in a molecule.
    Do not use bond type info.
    See Also:
        BondAsEdgeBidirectedFeaturizer
    """

    def __call__(self, mol, **kwargs):
        """
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object
        Returns
        -------
            Dictionary for bond features
        """

        # Note, this needs to be set such that single atom molecule works
        num_feats = 7

        num_bonds = mol.GetNumBonds()

        if num_bonds == 0:
            ft = [0.0 for _ in range(num_feats)]
            if self.length_featurizer:
                ft += [0.0 for _ in range(len(self.length_featurizer.feature_name))]
            feats = [ft]

        else:

            ring = mol.GetRingInfo()
            allowed_ring_size = [3, 4, 5, 6, 7]

            feats = []
            for u in range(num_bonds):
                bond = mol.GetBondWithIdx(u)

                ft = [
                    int(bond.IsInRing()),
                ]

                for s in allowed_ring_size:
                    ft.append(ring.IsBondInRingOfSize(u, s))

                ft.append(int(bond.GetBondType() == Chem.rdchem.BondType.DATIVE))

                if self.length_featurizer:
                    at1 = bond.GetBeginAtomIdx()
                    at2 = bond.GetEndAtomIdx()
                    atoms_pos = mol.GetConformer().GetPositions()
                    bond_length = np.linalg.norm(atoms_pos[at1] - atoms_pos[at2])
                    ft += self.length_featurizer(bond_length)

                feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = ["in_ring"] + ["ring size"] * 5 + ["dative"]
        if self.length_featurizer:
            self._feature_name += self.length_featurizer.feature_name

        return {"feat": feats}

class BondAsNodeFeaturizerFull(BondFeaturizer):
    """
    Featurize all bonds in a molecule.
    The bond indices will be preserved, i.e. feature i corresponds to atom i.
    The number of features will be equal to the number of bonds in the molecule,
    so this is suitable for the case where we represent bond as graph nodes.
    See Also:
        BondAsEdgeBidirectedFeaturizer
    """

    def __init__(
        self,
        length_featurizer=None,
        length_featurizer_args=None,
        dative=False,
        dtype="float32",
    ):
        super().__init__(
            length_featurizer, length_featurizer_args, dtype
        )
        self.dative = dative

    def __call__(self, mol, **kwargs):
        """
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object
        Returns
        -------
            Dictionary for bond features
        """

        # Note, this needs to be set such that single atom molecule works
        if self.dative:
            num_feats = 12
        else:
            num_feats = 11

        num_bonds = mol.GetNumBonds()

        if num_bonds == 0:
            ft = [0.0 for _ in range(num_feats)]
            if self.length_featurizer:
                ft += [0.0 for _ in range(len(self.length_featurizer.feature_name))]
            feats = [ft]

        else:
            ring = mol.GetRingInfo()
            allowed_ring_size = [3, 4, 5, 6, 7]

            feats = []
            for u in range(num_bonds):
                bond = mol.GetBondWithIdx(u)

                ft = [
                    int(bond.IsInRing()),
                    int(bond.GetIsConjugated()),
                ]
                for s in allowed_ring_size:
                    ft.append(ring.IsBondInRingOfSize(u, s))

                allowed_bond_type = [
                    Chem.rdchem.BondType.SINGLE,
                    Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE,
                    Chem.rdchem.BondType.AROMATIC,
                ]
                if self.dative:
                    allowed_bond_type.append(Chem.rdchem.BondType.DATIVE)
                ft += one_hot_encoding(bond.GetBondType(), allowed_bond_type)

                if self.length_featurizer:
                    at1 = bond.GetBeginAtomIdx()
                    at2 = bond.GetEndAtomIdx()
                    atoms_pos = mol.GetConformer().GetPositions()
                    bond_length = np.linalg.norm(atoms_pos[at1] - atoms_pos[at2])
                    ft += self.length_featurizer(bond_length)

                feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = (
            ["in_ring", "conjugated"]
            + ["ring size"] * 5
            + ["single", "double", "triple", "aromatic"]
        )
        if self.dative:
            self._feature_name += ["dative"]
        if self.length_featurizer:
            self._feature_name += self.length_featurizer.feature_name

        return {"feat": feats}

class AtomFeaturizerMinimum(BaseFeaturizer):
    """
    Featurize atoms in a molecule.
    Mimimum set of info without hybridization info.
    """

    def __call__(self, mol, **kwargs):
        """
        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object
            Also `extra_feats_info` should be provided as `kwargs` as additional info.
        Returns:
            Dictionary of atom features
        """
        try:
            species = sorted(kwargs["dataset_species"])
        except KeyError as e:
            raise KeyError(
                "{} `dataset_species` needed for {}.".format(e, self.__class__.__name__)
            )
        try:
            feats_info = kwargs["extra_feats_info"]
        except KeyError as e:
            raise KeyError(
                "{} `extra_feats_info` needed for {}.".format(e, self.__class__.__name__)
            )

        feats = []

        ring = mol.GetRingInfo()
        allowed_ring_size = [3, 4, 5, 6, 7]
        num_atoms = mol.GetNumAtoms()
        for i in range(num_atoms):
            ft = []
            atom = mol.GetAtomWithIdx(i)

            ft.append(atom.GetTotalDegree())
            ft.append(int(atom.IsInRing()))
            ft.append(atom.GetTotalNumHs(includeNeighbors=True))

            ft += one_hot_encoding(atom.GetSymbol(), species)

            for s in allowed_ring_size:
                ft.append(ring.IsAtomInRingOfSize(i, s))

            feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = (
            ["total degree", "is in ring", "total H"]
            + ["chemical symbol"] * len(species)
            + ["ring size"] * 5
        )
        return {"feat": feats}

class AtomFeaturizerFull(BaseFeaturizer):
    """
    Featurize atoms in a molecule.
    The atom indices will be preserved, i.e. feature i corresponds to atom i.
    """

    def __call__(self, mol, **kwargs):
        """
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object
        Returns
        -------
            Dictionary for atom features
        """
        try:
            species = sorted(kwargs["dataset_species"])
        except KeyError as e:
            raise KeyError(
                "{} `dataset_species` needed for {}.".format(e, self.__class__.__name__)
            )

        feats = []
        is_donor = defaultdict(int)
        is_acceptor = defaultdict(int)

        fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        mol_feats = mol_featurizer.GetFeaturesForMol(mol)
        _,res = rdEHTTools.RunMol(mol)
        huckel_charges = list(res.GetAtomicCharges())

        for i in range(len(mol_feats)):
            if mol_feats[i].GetFamily() == "Donor":
                node_list = mol_feats[i].GetAtomIds()
                for u in node_list:
                    is_donor[u] = 1
            elif mol_feats[i].GetFamily() == "Acceptor":
                node_list = mol_feats[i].GetAtomIds()
                for u in node_list:
                    is_acceptor[u] = 1

        ring = mol.GetRingInfo()
        allowed_ring_size = [3, 4, 5, 6, 7]
        num_atoms = mol.GetNumAtoms()
        for u in range(num_atoms):
            ft = [is_acceptor[u], is_donor[u]]
            ft.append(huckel_charges[u])

            atom = mol.GetAtomWithIdx(u)
            ft.append(atom.GetTotalDegree())

            ft.append(atom.GetTotalValence())

            ft.append(int(atom.GetIsAromatic()))
            ft.append(int(atom.IsInRing()))

            ft.append(atom.GetTotalNumHs(includeNeighbors=True))

            ft += one_hot_encoding(atom.GetSymbol(), species)

            ft += one_hot_encoding(
                atom.GetHybridization(),
                [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2,
                ],
            )

            for s in allowed_ring_size:
                ft.append(ring.IsAtomInRingOfSize(u, s))

            feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = (
            [
                "acceptor",
                "donor",
                "huckel partial charge",
                "total degree",
                "total valence",
                "is aromatic",
                "is in ring",
                "num total H",
            ]
            + ["chemical symbol"] * len(species)
            + ["hybridization"] * 4
            + ["ring size"] * 5
        )

        return {"feat": feats}


def atom_lone_pairs(atom):
    atom_num = atom.GetAtomicNum()
    dv = Chem.PeriodicTable.GetDefaultValence(Chem.GetPeriodicTable(), atom_num) # default valence
    nlp = Chem.PeriodicTable.GetNOuterElecs(Chem.GetPeriodicTable(), atom_num) - dv
    # subtract the charge to get the true number of lone pair electrons
    nlp -= atom.GetFormalCharge()
    return nlp

class SolventAtomFeaturizer(BaseFeaturizer):
    """
    Featurize atoms in a molecule.
    The atom indices will be preserved, i.e. feature i corresponds to atom i.
    """
    def __init__(self, partial_charges=None, dtype="float32"):
        super().__init__(dtype)
        self.partial_charges = partial_charges

    def __call__(self, mol, **kwargs):
        """
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object
        Returns
        -------
            Dictionary for atom features
        """
        try:
            species = sorted(kwargs["dataset_species"])
        except KeyError as e:
            raise KeyError(
                "{} `dataset_species` needed for {}.".format(e, self.__class__.__name__)
            )

        feats = []

        ring = mol.GetRingInfo()
        allowed_ring_size = [3, 4, 5, 6, 7]

        HDonorSmarts = Chem.MolFromSmarts('[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]')
        HAcceptorSmarts = Chem.MolFromSmarts('[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' +
                                     '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' +
                                     '$([nH0,o,s;+0])]')

        hbond_acceptors = sum(mol.GetSubstructMatches(HAcceptorSmarts), ())
        hbond_donors = sum(mol.GetSubstructMatches(HDonorSmarts), ())

        if self.partial_charges is not None:
            # Get Huckel partial charges
            if self.partial_charges != "crippen":
                x = AllChem.EmbedMolecule(mol, useRandomCoords=True)
                if x == 0:
                    AllChem.UFFOptimizeMolecule(mol)
                elif x == -1:
                    mol_smiles = Chem.MolToSmiles(mol)
                    pybel_mol = pybel.readstring("smi", mol_smiles)
                    pybel_mol.make3D(forcefield='uff', steps=100)
                    pdb = pybel_mol.write("pdb")
                    mol = Chem.MolFromPDBBlock(pdb)
                    
                if self.partial_charges == "huckel":
                    _, res = rdEHTTools.RunMol(mol)
                    pcharges = list(res.GetAtomicCharges())
                    
                elif self.partial_charges == "xtb": # uses xtb not mulliken
                    if mol.GetNumAtoms() != 167:
                        try:
                            atoms = np.zeros(mol.GetNumAtoms())
                            pos = np.zeros((mol.GetNumAtoms(), 3))
                            for i, atom in enumerate(mol.GetAtoms()):
                                positions = mol.GetConformer().GetAtomPosition(i)
                                atoms[i] = atom.GetAtomicNum()
                                pos[i] = (positions.x, positions.y, positions.z)
                            calc = Calculator(Param.GFN1xTB, atoms, pos)
                            calc.set_verbosity(VERBOSITY_MUTED)
                            res = calc.singlepoint()
                            pcharges = res.get_charges()
                        except Exception as e:
                            pcharges = np.zeros(mol.GetNumAtoms())
                    else:
                        _, res = rdEHTTools.RunMol(mol)
                        pcharges = list(res.GetAtomicCharges())
                
                elif self.partial_charges == "xtb_crippen":
                    if mol.GetNumAtoms() != 167:
                        try:
                            atoms = np.zeros(mol.GetNumAtoms())
                            pos = np.zeros((mol.GetNumAtoms(), 3))
                            for i, atom in enumerate(mol.GetAtoms()):
                                positions = mol.GetConformer().GetAtomPosition(i)
                                atoms[i] = atom.GetAtomicNum()
                                pos[i] = (positions.x, positions.y, positions.z)
                            calc = Calculator(Param.GFN1xTB, atoms, pos)
                            calc.set_verbosity(VERBOSITY_MUTED)
                            res = calc.singlepoint()
                            pcharges = res.get_charges()
                        except Exception as e:
                            pcharges = np.zeros(mol.GetNumAtoms())
                    else:
                        _, res = rdEHTTools.RunMol(mol)
                        pcharges = list(res.GetAtomicCharges())

                    mrContribs = rdMolDescriptors._CalcCrippenContribs(mol) 
                    crippen_vals = np.array([y for x,y in mrContribs])

            
            else: # Calculate the atomic polarisability using the Crippen scheme.
                mrContribs = rdMolDescriptors._CalcCrippenContribs(mol) 
                pcharges = np.array([y for x,y in mrContribs])
            
            pcharges = np.nan_to_num(pcharges, posinf=0, neginf=0) # Replace any NaN with 0
            if (sum(pcharges) > 100) or (sum(pcharges< -100)):
                pcharges = np.zeros(mol.GetNumAtoms())
        
        
        num_atoms = mol.GetNumAtoms()
        for u in range(num_atoms):
            ft = []

            atom = mol.GetAtomWithIdx(u)

            ft.append(atom.GetTotalDegree())
            if self.partial_charges is not None:
                ft.append(pcharges[u])
            else:
                ft.append(atom.GetFormalCharge())
            
            ft.append(int(atom.GetIsAromatic()))
            ft.append(int(atom.IsInRing()))

            #ft.append(Chem.PeriodicTable.GetRvdw(Chem.GetPeriodicTable(), atom.GetAtomicNum())) # vdW radius
            ft.append(atom_lone_pairs(atom)) # Number of lone pairs

            ft.append(atom.GetTotalNumHs(includeNeighbors=True))

            if u in hbond_acceptors:
                ft.append(1)
            else: 
                ft.append(0)

            if u in hbond_donors:
                ft.append(1)
            else:
                ft.append(0)

            ft += one_hot_encoding(atom.GetSymbol(), species)

            ft += one_hot_encoding(
                atom.GetHybridization(),
                [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2,
                ],
            )

            for s in allowed_ring_size:
                ft.append(ring.IsAtomInRingOfSize(u, s))

            feats.append(ft)

            if self.partial_charges == "xtb_crippen":
                ft.append(crippen_vals[u])

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = (
            [
                "total degree",
                "partial/formal charge",
                "is aromatic",
                "is in ring",
                #"van der Waals radius",
                "num lone pairs",
                "num total H",
                "H bond acceptor",
                "H bond donor",
            ]
            + ["chemical symbol"] * len(species)
            + ["hybridization"] * 4
            + ["ring size"] * 5
        )
        if self.partial_charges == "xtb_crippen":
            self._feature_name.append("atomic polarisability")

        return {"feat": feats}


class SolventGlobalFeaturizer(BaseFeaturizer):
    """
    Featurize the global state of a molecules using number of H-bond acceptors, number of H-bond donors,
    molecular weight, and optionally charge and solvent environment.
    Args:
        allowed_charges (list, optional): charges allowed the the molecules to take.
        volume (bool, optional): include the molecular volume (rdkit calculated) of the molecule.
        dielectric_constant (optional): include the dielectric constant of the solvent. This is read in from a separate file.
    """

    def __init__(self, allowed_charges=None, dielectric_constant=None, mol_volume=False, mol_refract=False,
                 dtype="float32"):
        super().__init__(dtype)
        self.allowed_charges = allowed_charges
        self.mol_volume = mol_volume
        self.dielectric_constant = dielectric_constant
        self.mol_refract = mol_refract

    def __call__(self, mol, **kwargs):

        pt = GetPeriodicTable()
        g = [
            mol.GetNumAtoms(),
            mol.GetNumBonds(),
            sum([pt.GetAtomicWeight(a.GetAtomicNum()) for a in mol.GetAtoms()]),
        ]

        if self.allowed_charges is not None or self.dielectric_constant is not None or self.mol_volume is not False:
            # Read these values from an additional file
            try:
                feats_info = kwargs["extra_feats_info"]
            except KeyError as e:
                raise KeyError(
                    "{} `extra_feats_info` needed for {}.".format(
                        e, self.__class__.__name__
                    )
                )
            
            if self.dielectric_constant is not None:
                g += [feats_info]

            if self.allowed_charges is not None:
                g += one_hot_encoding(feats_info["charge"], self.allowed_charges)

            if self.mol_volume:
                try:
                    AllChem.EmbedMolecule(mol)
                    AllChem.MMFFOptimizeMolecule(mol) # MMFF94
                    g += [AllChem.ComputeMolVolume(mol)] 
                except ValueError as e:
                    mol_smiles = Chem.MolToSmiles(mol)
                    pybel_mol = pybel.readstring("smi", mol_smiles)
                    pybel_mol.make3D(forcefield='mmff94', steps=100)
                    pdb = pybel_mol.write("pdb")
                    rd_mol = Chem.MolFromPDBBlock(pdb)
                    g += [AllChem.ComputeMolVolume(rd_mol)]

            if self.mol_refract:
                _, mr = AllChem.CalcCrippenDescriptors(mol)
                g += [mr]

        feats = torch.tensor([g], dtype=getattr(torch, self.dtype))

        self._feature_size = feats.shape[1]
        self._feature_name = ["num atoms", "num bonds", "molecule weight"]
        if self.dielectric_constant is not None:
            self._feature_name += ["dielectric constant"]
        if self.allowed_charges is not None:
            self._feature_name += ["charge one hot"] * len(self.allowed_charges)
        if self.mol_volume is not False:
            self._feature_name += ["molecular volume"]
        if self.mol_refract is not False:
            self._feature_name += ["molecular refractivity"]

        return {"feat": feats}


class GlobalFeaturizer(BaseFeaturizer):
    """
    Featurize the global state of a molecules using number of atoms, number of bonds,
    molecular weight, and optionally charge and solvent environment.
    Args:
        allowed_charges (list, optional): charges allowed the the molecules to take.
        volume (bool, optional): include the molecular volume (rdkit calculated) of the molecule.
        dielectric_constant (optional): include the dielectric constant of the solvent. This is read in from a separate file.
    """

    def __init__(self, allowed_charges=None, dielectric_constant=None, mol_volume=False, 
                 dtype="float32"):
        super().__init__(dtype)
        self.allowed_charges = allowed_charges
        self.mol_volume = mol_volume
        self.dielectric_constant = dielectric_constant

    def __call__(self, mol, **kwargs):

        pt = GetPeriodicTable()
        g = [
            mol.GetNumAtoms(),
            mol.GetNumBonds(),
            sum([pt.GetAtomicWeight(a.GetAtomicNum()) for a in mol.GetAtoms()]),
        ]

        if self.allowed_charges is not None or self.dielectric_constant is not None or self.mol_volume is not False:
            # Read these values from an additional file
            try:
                feats_info = kwargs["extra_feats_info"]
            except KeyError as e:
                raise KeyError(
                    "{} `extra_feats_info` needed for {}.".format(
                        e, self.__class__.__name__
                    )
                )
            
            if self.dielectric_constant is not None:
                g += [feats_info]

            if self.allowed_charges is not None:
                g += one_hot_encoding(feats_info["charge"], self.allowed_charges)

            if self.mol_volume:
                try:
                    AllChem.EmbedMolecule(mol)
                    AllChem.MMFFOptimizeMolecule(mol) # MMFF94
                    g += [AllChem.ComputeMolVolume(mol)] 
                except ValueError as e:
                    mol_smiles = Chem.MolToSmiles(mol)
                    pybel_mol = pybel.readstring("smi", mol_smiles)
                    pybel_mol.make3D(forcefield='mmff94', steps=100)
                    pdb = pybel_mol.write("pdb")
                    rd_mol = Chem.MolFromPDBBlock(pdb)
                    g += [AllChem.ComputeMolVolume(rd_mol)]

        feats = torch.tensor([g], dtype=getattr(torch, self.dtype))

        self._feature_size = feats.shape[1]
        self._feature_name = ["num atoms", "num bonds", "molecule weight"]
        if self.dielectric_constant is not None:
            self._feature_name += ["dielectric constant"]
        if self.allowed_charges is not None:
            self._feature_name += ["charge one hot"] * len(self.allowed_charges)
        if self.mol_volume is not False:
            self._feature_name += ["molecular volume"]

        return {"feat": feats}

def one_hot_encoding(x, allowable_set):
    """One-hot encoding.
    Parameters
    ----------
    x : str, int or Chem.rdchem.HybridizationType
    allowable_set : list
        The elements of the allowable_set should be of the
        same type as x.
    Returns
    -------
    list
        List of int (0 or 1) where at most one value is 1.
        If the i-th value is 1, then we must have x == allowable_set[i].
    """
    return list(map(int, list(map(lambda s: x == s, allowable_set))))


def multi_hot_encoding(x, allowable_set):
    """Multi-hot encoding.
    Args:
        x (list): any type that can be compared with elements in allowable_set
        allowable_set (list): allowed values for x to take
    Returns:
        list: List of int (0 or 1) where zero or more values can be 1.
            If the i-th value is 1, then we must have allowable_set[i] in x.
    """
    return list(map(int, list(map(lambda s: s in x, allowable_set))))

class DistanceBins(BaseFeaturizer):
    """
    Put the distance into a bins. As used in MPNN.
    Args:
        low (float): lower bound of bin. Values smaller than this will all be put in
            the same bin.
        high (float): upper bound of bin. Values larger than this will all be put in
            the same bin.
        num_bins (int): number of bins. Besides two bins (one smaller than `low` and
            one larger than `high`) a number of `num_bins -2` bins will be evenly
            created between [low, high).
    """

    def __init__(self, low=2.0, high=6.0, num_bins=10):
        super(DistanceBins, self).__init__()
        self.num_bins = num_bins
        self.bins = np.linspace(low, high, num_bins - 1, endpoint=True)
        self.bin_indices = np.arange(num_bins)

    @property
    def feature_size(self):
        return self.num_bins

    @property
    def feature_name(self):
        return ["dist bins"] * self.feature_size

    def __call__(self, distance):
        v = np.digitize(distance, self.bins)
        return one_hot_encoding(v, self.bin_indices)


class RBF(BaseFeaturizer):
    """
    Radial basis functions.
    e(d) = exp(- gamma * ||d - mu_k||^2), where gamma = 1/delta
    Parameters
    ----------
    low : float
        Smallest value to take for mu_k, default to be 0.
    high : float
        Largest value to take for mu_k, default to be 4.
    num_centers : float
        Number of centers
    """

    def __init__(self, low=0.0, high=4.0, num_centers=20):
        super(RBF, self).__init__()
        self.num_centers = num_centers
        self.centers = np.linspace(low, high, num_centers)
        self.gap = self.centers[1] - self.centers[0]

    @property
    def feature_size(self):
        return self.num_centers

    @property
    def feature_name(self):
        return ["rbf"] * self.feature_size

    def __call__(self, edge_distance):
        """
        Parameters
        ----------
        edge_distance : float
            Edge distance
        Returns
        -------
        a list of RBF values of size `num_centers`
        """
        radial = edge_distance - self.centers
        coef = -1 / self.gap
        return list(np.exp(coef * (radial ** 2)))
