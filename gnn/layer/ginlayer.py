import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from typing import Callable, Union, Dict
from gnn.layer.gatedconv import select_not_equal

logger = logging.getLogger(__name__)

class GINLayer(nn.Module):
    """
    Graph Isomorphism Network layer.
    Code from: https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/layers/gin_layer.py
    
    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula.
    aggr_type :
        Aggregator type to use (``sum``, ``max`` or ``mean``).
    dropout :
        Required for dropout of output features.
    batch_norm :
        boolean flag for batch_norm layer.
    residual :
        boolean flag for using residual connection.
    init_eps : optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter.
    
    """

    def __init__(
        self,
        apply_func: function,
        aggr_type: str,
        batch_norm: bool = True,
        residual: bool = False,
        init_eps: float = 0,
        learn_eps: bool = False,
        dropout: Union[float, None] = None,
    ):
        super().__init__()
        self.apply_func = apply_func

        if aggr_type == 'sum':
            self._reducer = fn.sum
        elif aggr_type == 'max':
            self._reducer = fn.max
        elif aggr_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggr_type))
        
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout
        
        in_dim = apply_func.mlp.input_dim
        out_dim = apply_func.mlp.output_dim

        if in_dim != out_dim:
            self.residual = False
        
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))
        
        self.bn_node_h = nn.BatchNorm1d(out_dim)
    
    @staticmethod
    def reduce_fn_a2b(nodes):
        """
        Reduce `Eh_j` from atom nodes to bond nodes.
        Expand dim 1 such that every bond has two atoms connecting to it.
        This is to deal with the special case of single atom graph (e.g. H+).
        For such graph, an artificial bond is created and connected to the atom in
        `grapher`. Here, we expand it to let each bond connecting to two atoms.
        This is necessary because, otherwise, the reduce_fn wil not work since
        dimension mismatch.
        """
        x = nodes.mailbox["Eh_j"]
        if x.shape[1] == 1:
            x = x.repeat_interleave(2, dim=1)

        return {"Eh_j": x}

    @staticmethod
    def message_fn(edges):
        return {"Eh_j": edges.src["Eh_j"], "e": edges.src["e"]}

    @staticmethod
    def reduce_fn(nodes):
        Eh_i = nodes.data["Eh"]
        e = nodes.mailbox["e"]
        Eh_j = nodes.mailbox["Eh_j"]

        # TODO select_not_equal is time consuming; it might be improved by passing node
        #  index along with Eh_j and compare the node index to select the different one
        Eh_j = select_not_equal(Eh_j, Eh_i)
        sigma_ij = torch.sigmoid(e)  # sigma_ij = sigmoid(e_ij)

        # (sum_j eta_ij * Ehj)/(sum_j' eta_ij') <= dense attention
        h = torch.sum(sigma_ij * Eh_j, dim=1) / (torch.sum(sigma_ij, dim=1) + 1e-6)

        return {"h": h}

    def forward(
        self,
        g: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        norm_atom: torch.Tensor = None,
        norm_bond: torch.Tensor = None,
    )   -> Dict[str, torch.Tensor]:
        """
        Args:
                g: the graph
                feats: node features. Allowed node types are `atom`, `bond` and `global`.
                norm_atom: values used to normalize atom features as proposed in graph norm.
                norm_bond: values used to normalize bond features as proposed in graph norm.
            Returns:
                updated node features.
        """
        g = g.to('cuda:0')
        g = g.local_var()

        h = feats["atom"]
        e = feats["bond"]
        u = feats["global"]

        # for residual connection
        h_in = h
        e_in = e
        u_in = u

        g.nodes["atom"].data.update({"h": h})

        g.nodes["atom"].data.update_all(fn.copy_u('h', 'm'), self._reducer('m', 'neigh'))
        h = (1 + self.eps) * h + g.nd


