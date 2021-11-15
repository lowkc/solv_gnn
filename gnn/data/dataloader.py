import torch
import dgl
import itertools


class DataLoader(torch.utils.data.DataLoader):
    """
    This dataloader works for the case where the labels of all data points are of the
    same shape. For example, regression on molecule energy.
    """

    def __init__(self, dataset, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'bondnet.data', you need not to "
                "provide one"
            )

        def collate(samples):
            graphs, labels = map(list, zip(*samples))
            batched_graphs = dgl.batch(graphs)
            batched_labels = torch.utils.data.dataloader.default_collate(labels)

            return batched_graphs, batched_labels

        super(DataLoader, self).__init__(dataset, collate_fn=collate, **kwargs)

class DataLoaderGraphNorm(torch.utils.data.DataLoader):
    """
    This dataloader works for the case where the label of each data point are of the
    same shape. For example, regression on molecule energy.
    """

    def __init__(self, dataset, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'bondnet.data', you need not to "
                "provide one"
            )

        def collate(samples):
            graphs, labels = map(list, zip(*samples))

            # g = graphs[0]
            # if isinstance(g, dgl.DGLGraph):
            #     batched_graphs = dgl.batch(graphs)
            #     sizes_atom = [g.number_of_nodes() for g in graphs]
            #     sizes_bond = [g.number_of_edges() for g in graphs]
            #
            # elif isinstance(g, dgl.DGLHeteroGraph):
            #     batched_graphs = dgl.batch_hetero(graphs)
            #     sizes_atom = [g.number_of_nodes("atom") for g in graphs]
            #     sizes_bond = [g.number_of_nodes("bond") for g in graphs]
            # else:
            # raise ValueError(
            #     f"graph type {g.__class__.__name__} not supported. Should be either "
            #     f"dgl.DGLGraph or dgl.DGLHeteroGraph."
            # )

            batched_graphs = dgl.batch(graphs)
            sizes_atom = [g.number_of_nodes("atom") for g in graphs]
            sizes_bond = [g.number_of_nodes("bond") for g in graphs]

            batched_labels = torch.utils.data.dataloader.default_collate(labels)

            norm_atom = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_atom]
            norm_bond = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_bond]
            batched_labels["norm_atom"] = 1.0 / torch.cat(norm_atom).sqrt()
            batched_labels["norm_bond"] = 1.0 / torch.cat(norm_bond).sqrt()

            return batched_graphs, batched_labels

        super(DataLoaderGraphNorm, self).__init__(dataset, collate_fn=collate, **kwargs)

class DataLoaderReaction(torch.utils.data.DataLoader):
    """
    This dataloader works specifically for the reaction dataset where each reaction is
    represented by a list of the molecules (i.e. reactants and products).
    Also, the label value of each datapoint should be of the same shape.
    """

    def __init__(self, dataset, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'bondnet.data', you need not to "
                "provide one"
            )

        def collate(samples):
            graphs, labels = map(list, zip(*samples))

            # note each element of graph is a list of mol graphs that constitute a rxn
            # flatten double list
            graphs = list(itertools.chain.from_iterable(graphs))
            batched_graphs = dgl.batch(graphs)

            target_class = torch.stack([la["value"] for la in labels])
            atom_mapping = [la["atom_mapping"] for la in labels]
            bond_mapping = [la["bond_mapping"] for la in labels]
            global_mapping = [la["global_mapping"] for la in labels]
            num_mols = [la["num_mols"] for la in labels]
            identifier = [la["id"] for la in labels]

            batched_labels = {
                "value": target_class,
                "atom_mapping": atom_mapping,
                "bond_mapping": bond_mapping,
                "global_mapping": global_mapping,
                "num_mols": num_mols,
                "id": identifier,
            }

            # add label scaler if it is used
            try:
                mean = [la["scaler_mean"] for la in labels]
                stdev = [la["scaler_stdev"] for la in labels]
                batched_labels["scaler_mean"] = torch.stack(mean)
                batched_labels["scaler_stdev"] = torch.stack(stdev)
            except KeyError:
                pass

            return batched_graphs, batched_labels

        super(DataLoaderReaction, self).__init__(dataset, collate_fn=collate, **kwargs)

class DataLoaderSolvation(torch.utils.data.DataLoader):
    """
    This dataloader works specifically for the solvation dataset where each solute-solvent pair is
    represented by a list of the molecules (i.e., solute and solvent.)
    The label value of each data point should be of the same shape.
    """

    def __init__(self, dataset, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'bondnet.data', you need not to "
                "provide one"
            )
        
        def collate(samples):
            graphs, labels = map(list, zip(*samples))
            # note each element of graphs is a list of mol graphs that constitute a solute-solvent pair

            graphs = list(itertools.chain.from_iterable(graphs))
            solute_graphs = list(itertools.islice(graphs, 0, None, 2)) # Every first element = solute
            solvent_graphs = list(itertools.islice(graphs, 1, None, 2)) # Every second element = solvent

            solute_batched_graphs = dgl.batch(solute_graphs)
            solvent_batched_graphs = dgl.batch(solvent_graphs)

            solute_sizes_atom = [g.number_of_nodes("atom") for g in solute_graphs]
            solute_sizes_bond = [g.number_of_nodes("bond") for g in solute_graphs]
            solvent_sizes_atom = [g.number_of_nodes("atom") for g in solvent_graphs]
            solvent_sizes_bond = [g.number_of_nodes("bond") for g in solvent_graphs]

            target_class = torch.stack([la["value"] for la in labels])
            identifier = [la["id"] for la in labels]

            batched_labels = {
                "value": target_class,
                "id": identifier
            }

            # add label scaler if it is used
            try:
                mean = [la["scaler_mean"] for la in labels]
                stdev = [la["scaler_stdev"] for la in labels]
                batched_labels["scaler_mean"] = torch.stack(mean)
                batched_labels["scaler_stdev"] = torch.stack(stdev)
            except KeyError:
                pass

            # graph norm
            solute_norm_atom = [torch.FloatTensor(s, 1).fill_(s) for s in solute_sizes_atom]
            solute_norm_bond = [torch.FloatTensor(s, 1).fill_(s) for s in solute_sizes_bond]
            solvent_norm_atom = [torch.FloatTensor(s, 1).fill_(s) for s in solvent_sizes_atom]
            solvent_norm_bond = [torch.FloatTensor(s, 1).fill_(s) for s in solvent_sizes_bond]

            batched_labels["solute_norm_atom"] = 1.0 / torch.cat(solute_norm_atom).sqrt()
            batched_labels["solute_norm_bond"] = 1.0 / torch.cat(solute_norm_bond).sqrt()
            batched_labels["solvent_norm_atom"] = 1.0 / torch.cat(solvent_norm_atom).sqrt()
            batched_labels["solvent_norm_bond"] = 1.0 / torch.cat(solvent_norm_bond).sqrt()

            return solute_batched_graphs, solvent_batched_graphs, batched_labels
        
        super(DataLoaderSolvation, self).__init__(dataset, collate_fn=collate, **kwargs)
