from typing import ForwardRef
import torch
import itertools
import numpy as np
import dgl
from torch._C import device
from gnn.model.gated_mol import GatedGCNMol, AttentionGCN
import torch.nn.functional as F

class SelfInteractionMap(AttentionGCN):
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

        solute_feats = self.solute_embedding(solute_feats)
        solvent_feats = self.solvent_embedding(solvent_feats)

        # pass through gated layer
        for layer in self.gated_layers:
            solute_feats = layer(solute_graph, solute_feats, solute_norm_atom, solute_norm_bond)
            solvent_feats = layer(solvent_graph, solvent_feats, solvent_norm_atom, solvent_norm_bond)

        fts_solu = _split_batched_output_atoms(solute_graph, solute_feats["atom"]) # 22 * 64
        fts_solv = _split_batched_output_atoms(solvent_graph, solvent_feats["atom"]) # 23 * 64

        updated_solute_atom_fts = []
        updated_solvent_atom_fts = []

        for solute_ft, solvent_ft in zip(fts_solu, fts_solv):
            # Effect of the solvent on the solute
            solute_fts_att_w  = torch.matmul(self.solute_W_a(solute_ft), solute_ft.t()) 
            solute_fts_att_w = torch.nn.functional.softmax(solute_fts_att_w, dim=0)
            
            solvent_fts_att_w  = torch.matmul(self.solvent_W_a(solvent_ft), solvent_ft.t()) 
            solvent_fts_att_w = torch.nn.functional.softmax(solvent_fts_att_w, dim=0)

            solute_attn_hiddens = torch.matmul(solute_fts_att_w, solute_ft)
            solute_attn_hiddens = self.W_activation(self.solute_W_b(solute_attn_hiddens))

            solvent_attn_hiddens = torch.matmul(solvent_fts_att_w, solvent_ft) 
            solvent_attn_hiddens = self.W_activation(self.solvent_W_b(solvent_attn_hiddens))

            new_solute_feats = solute_ft + solute_attn_hiddens
            new_solvent_feats = solvent_ft + solvent_attn_hiddens

            updated_solute_atom_fts.append(new_solute_feats)
            updated_solvent_atom_fts.append(new_solvent_feats)

        new_solute_feats = torch.cat(updated_solute_atom_fts)
        new_solvent_feats = torch.cat(updated_solvent_atom_fts)

        solute_feats["atom"] = new_solute_feats
        solvent_feats["atom"] = new_solvent_feats

        # readout layer - set2set
        solute_feats = self.readout_layer(solute_graph, solute_feats) 
        solvent_feats = self.readout_layer(solvent_graph, solvent_feats) 

        # concatenate
        feats = torch.cat([solute_feats, solvent_feats], dim=1) 

        # fc
        for layer in self.fc_layers:
            feats = layer(feats)

        return feats


class InteractionMap(AttentionGCN):
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

        solute_feats = self.solute_embedding(solute_feats)
        solvent_feats = self.solvent_embedding(solvent_feats)

        # pass through gated layer
        for layer in self.gated_layers:
            solute_feats = layer(solute_graph, solute_feats, solute_norm_atom, solute_norm_bond)
            solvent_feats = layer(solvent_graph, solvent_feats, solvent_norm_atom, solvent_norm_bond)

       # interaction map - attention mechanism
       # adapted from https://github.com/tbwxmu/SAMPN/blob/7d8db6223e8f6f35f0953310da03fa842187fbcc/mpn.py

        fts_solu = _split_batched_output_atoms(solute_graph, solute_feats["atom"]) 
        fts_solv = _split_batched_output_atoms(solvent_graph, solvent_feats["atom"]) 
        updated_solute_atom_fts = []
        updated_solvent_atom_fts = []

        for solute_ft, solvent_ft in zip(fts_solu, fts_solv):
            pairwise_solute_feature = F.leaky_relu(self.solute_W_a(solute_ft), 0.1) 
            pairwise_solvent_feature = F.leaky_relu(self.solvent_W_a(solvent_ft), 0.1) 
            pairwise_pred = torch.sigmoid(torch.matmul(
                pairwise_solute_feature, pairwise_solvent_feature.t())) 

            new_solvent_feats = torch.matmul(pairwise_pred.t(), pairwise_solute_feature)
            new_solute_feats = torch.matmul(pairwise_pred, pairwise_solvent_feature) 

            # Add the old solute_ft to the new one to get a representation of both inter- and intra-molecular interactions.
            new_solute_feats += solute_ft
            new_solvent_feats += solvent_ft
            updated_solute_atom_fts.append(new_solute_feats)
            updated_solvent_atom_fts.append(new_solvent_feats)

        new_solute_feats = torch.cat(updated_solute_atom_fts)
        new_solvent_feats = torch.cat(updated_solvent_atom_fts)

        solute_feats["atom"] = new_solute_feats
        solvent_feats["atom"] = new_solvent_feats
        
        # readout layer - set2set
        solute_feats_prime = self.readout_layer(solute_graph, solute_feats) 
        solvent_feats_prime = self.readout_layer(solvent_graph, solvent_feats) 

        # concatenate
        feats = torch.cat([solute_feats_prime, solvent_feats_prime], dim=1) 

        # fc
        for layer in self.fc_layers:
            feats = layer(feats)

        return feats

        
    def visualise_attn_weights(self, solute_graph, solvent_graph, solute_feats, solvent_feats,
     solute_norm_atom=None, solute_norm_bond=None, solvent_norm_atom=None, solvent_norm_bond=None):

        solute_feats = self.solute_embedding(solute_feats)
        solvent_feats = self.solvent_embedding(solvent_feats)

        # pass through gated layer
        for layer in self.gated_layers:
            solute_feats = layer(solute_graph, solute_feats, solute_norm_atom, solute_norm_bond)
            solvent_feats = layer(solvent_graph, solvent_feats, solvent_norm_atom, solvent_norm_bond)

        solute_wts = []
        solvent_wts = []

        fts_solu = _split_batched_output_atoms(solute_graph, solute_feats["atom"])
        fts_solv = _split_batched_output_atoms(solvent_graph, solvent_feats["atom"]) 

        for solute_ft, solvent_ft in zip(fts_solu, fts_solv):
            pairwise_solute_feature = F.leaky_relu(self.solute_W_a(solute_ft), 0.1)
            pairwise_solvent_feature = F.leaky_relu(self.solvent_W_a(solvent_ft), 0.1) 
            
            pairwise_pred = torch.sigmoid(torch.matmul(
                pairwise_solute_feature, pairwise_solvent_feature.t()))

            solute_fts_att_w  = torch.matmul(pairwise_pred, pairwise_solvent_feature)       
            solvent_fts_att_w  = torch.matmul(pairwise_pred.t(), pairwise_solute_feature)

            solute_wts.append(solute_fts_att_w)
            solvent_wts.append(solvent_fts_att_w)

        return solute_wts, solvent_wts


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
        solute_feats = self.solute_embedding(solute_feats)
        solvent_feats = self.solvent_embedding(solvent_feats)

        # pass through gated layer
        for layer in self.gated_layers:
            solute_feats = layer(solute_graph, solute_feats, solute_norm_atom, solute_norm_bond)
            solvent_feats = layer(solvent_graph, solvent_feats, solvent_norm_atom, solvent_norm_bond)

        # readout layer - set2set
        solute_feats = self.readout_layer(solute_graph, solute_feats) 
        solvent_feats = self.readout_layer(solvent_graph, solvent_feats) 

        # concatenate
        feats = torch.cat([solute_feats, solvent_feats], dim=1) 

        # fc
        for layer in self.fc_layers:
            feats = layer(feats)

        return feats
    
    def feature_before_fc(self, solute_graph, solvent_graph, solute_feats, solvent_feats, 
                          solute_norm_atom=None, solute_norm_bond=None, solvent_norm_atom=None, solvent_norm_bond=None):
        """
        Get the features before the final fully-connected.
        This is used for feature visualization.
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
        feats = torch.cat((solute_feats, solvent_feats)) # 200 * hidden_dim

        return solute_feats, solvent_feats, feats

    def atom_features_at_each_layer(self, solute_graph, solvent_graph, solute_feats, solvent_feats,
                                    solute_norm_atom=None, solute_norm_bond=None, solvent_norm_atom=None, solvent_norm_bond=None):
        """
        Get the atom features at each layer before the final fully-connected layer
        This is used for feature visualisation to see how the model learns.

        Returns:
            dict (layer_idx, feats), each feats is a list of each atom's features.
        """

        layer_idx = 0
        all_feats = dict()

        # embedding
        solute_feats = self.embedding(solute_feats)
        solvent_feats = self.embedding(solvent_feats)

        # store atom features of each molecule
        solute_atom_fts = _split_batched_output_atoms(solute_graph, solute_feats["atom"])
        solvent_atom_fts = _split_batched_output_atoms(solvent_graph, solvent_feats["atom"])
        all_feats[layer_idx] = (solute_atom_fts, solvent_atom_fts)
        layer_idx += 1

        # gated layer
        for layer in self.gated_layers:
            solute_feats = layer(solute_graph, solute_feats, solute_norm_atom, solute_norm_bond)
            solvent_feats = layer(solvent_graph, solvent_feats, solvent_norm_atom, solvent_norm_bond)
            solute_atom_fts = _split_batched_output_atoms(solute_graph, solute_feats["atom"])
            solvent_atom_fts = _split_batched_output_atoms(solvent_graph, solvent_feats["atom"])
            all_feats[layer_idx] = (solute_atom_fts, solvent_atom_fts)
            layer_idx += 1

        return all_feats


def _split_batched_output_bonds(graph, value):
    """
    Split a tensor into `num_graphs` chunks, the size of each chunk equals the
    number of bonds in the graph.
    Returns:
        list of tensor.
    """
    nbonds = tuple(graph.batch_num_nodes("bond"))
    return torch.split(value, nbonds)

def _split_batched_output_atoms(graph, value):
    """
    Split a tensor into `num_graphs` chunks, the size of each chunk equals the
    number of bonds in the graph.
    Returns:
        list of tensor.
    """
    natoms = tuple(graph.batch_num_nodes("atom"))
    return torch.split(value, natoms)