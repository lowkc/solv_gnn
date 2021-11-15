import numpy as np
import itertools
import dgl

class SolvationNetwork:
    def __init__(
        self, solute, solvent, atom_mapping=None, bond_mapping=None, id=None
    ):
        """
        A class to represent a chemical reaction in reaction network.
        Args:
            reactants (list): integer indices of reactants
            products (list): integer indices of reactants
            atom_mapping (list of dict): each dict is an atom mapping from product to
                reactant
            bond_mapping (list of dict): each dict is a bond mapping from product to
                reactant
            id (int or str): unique identifier of the reaction
        Attrs:
            init_reactants (list): reactants indices in the global molecule pool. Not
                supposed to be changed.
            init_products (list): products indices in the global molecule pool. Not
                supposed to be changed.
            reactants (list): reactants indices in the subset molecule pool.
                Could be changed.
            products (list): products indices in the subset molecule pool.
                Could be changed.
        """