from typing import ForwardRef
import torch
import itertools
import numpy as np
import dgl
from gnn.model.gated_mol import GatedGCNMol

class GatedGCNSolvationNetwork(GatedGCNMol):
    def forward(self, solute_graph, solvent_graph, solute_feats, solvent_feats,
     solute_norm_atom=None, solute_norm_bond=None, solvent_norm_atom=None, solvent_norm_bond=None):
        """
        Args:
            graph (DGLHeteroGraph or BatchedDGLHeteroGraph): (batched) molecule graphs
            feats (dict): node features with node type as key and the corresponding features as value
            norm_atom (2D tensor or None): graph norm for atom
            norm_bond (2D tensor or None): graph norm for bond

        Returns:
            2D tensor: of shape(N, M), where M = outdim.
        """
        # embed the solute and solvent
        solute_feats = self.embedding(solute_feats)
        solvent_feats = self.embedding(solvent_feats)

        # pass through gated layer
        for layer in self.gated_layers:
            solute_feats = layer(solute_graph, solute_feats, solute_norm_atom, solute_norm_bond)
            solvent_feats = layer(solvent_graph, solvent_feats, solvent_norm_atom, solvent_norm_bond)

        # readout layer - set2set
        solute_feats = self.readout_layer(solute_graph, solute_feats) # 100 * hidden_dim
        solvent_feats = self.readout_layer(solvent_graph, solvent_feats) # 100 * hidden_dim

        # concatenate
        feats = torch.cat([solute_feats, solvent_feats], dim=1) # 200 * hidden_dim

        # fc
        for layer in self.fc_layers:
            feats = layer(feats)

        return feats
    
    def feature_before_fc(self, solute_graph, solvent_graph, solute_feats, solvent_feats, norm_atom=None, norm_bond=None):
        """
        Get the features before the final fully-connected.
        This is used for feature visualization.
        """
        # embed the solute and solvent
        solute_feats = self.embedding(solute_feats)
        solvent_feats = self.embedding(solvent_feats)

        # pass through gated layer
        for layer in self.gated_layers:
            solute_feats = layer(solute_graph, solute_feats, norm_atom, norm_bond)
            solvent_feats = layer(solvent_graph, solvent_feats, norm_atom, norm_bond)

        # readout layer - set2set
        solute_feats = self.readout_layer(solute_graph, solute_feats) # 100 * hidden_dim
        solvent_feats = self.readout_layer(solvent_graph, solvent_feats) # 100 * hidden_dim

        # concatenate
        feats = torch.cat((solute_feats, solvent_feats)) # 200 * hidden_dim

        return feats