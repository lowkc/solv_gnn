import logging
import sys, os
import time
import warnings
import torch
import optuna
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss, L1Loss
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.serialization import save
from gnn.model.metric import EarlyStopping
from gnn.model.gated_solv_network import GatedGCNSolvationNetwork, InteractionMap
from gnn.data.dataset import SolvationDataset, train_validation_test_split
from gnn.data.dataloader import DataLoaderSolvation
from gnn.data.grapher import HeteroMoleculeGraph
from gnn.data.featurizer import (
    SolventAtomFeaturizer,
    BondAsNodeFeaturizerFull,
    SolventGlobalFeaturizer,
)
from gnn.data.dataset import load_mols_labels
from gnn.utils import (
    load_checkpoints,
    save_checkpoints,
    seed_torch,
    pickle_dump,
    yaml_dump,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GatedSolvationNetwork")
    parser.add_argument('--dataset-file', type=str, default=None)
    #parser.add_argument('--dielectric-constants', type=str, default=None)
    #parser.add_argument('--molecular-volume', type=bool, default=False)

    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--feature-scaling', type=bool, default=True)
    parser.add_argument('--attention-map', type=bool, default=False)
    
    args = parser.parse_args()
    return args

def grapher(dielectric_constant=None, mol_volume=False):
    atom_featurizer = SolventAtomFeaturizer()
    bond_featurizer = BondAsNodeFeaturizerFull(length_featurizer=None, dative=False)
    global_featurizer = SolventGlobalFeaturizer(dielectric_constant=None, mol_volume=None)

    grapher = HeteroMoleculeGraph(atom_featurizer, bond_featurizer, global_featurizer, self_loop=True)

    return grapher

def train(optimizer, model, nodes, data_loader, loss_fn, metric_fn, device=None):
    """
    Args:
        metric_fn (function): the function should be using a `sum` reduction method.
    """

    model.train()

    epoch_loss = 0.0
    accuracy = 0.0
    count = 0.0

    for it, (solute_batched_graph, solvent_batched_graph, label) in enumerate(data_loader):
        solute_feats = {nt: solute_batched_graph.nodes[nt].data["feat"] for nt in nodes}
        solvent_feats = {nt: solvent_batched_graph.nodes[nt].data["feat"] for nt in nodes}
        target = torch.squeeze(label["value"])
        solute_norm_atom = label["solute_norm_atom"]
        solute_norm_bond = label["solute_norm_bond"]
        solvent_norm_atom = label["solvent_norm_atom"]
        solvent_norm_bond = label["solvent_norm_bond"]
        #stdev = label["scaler_stdev"]

        if device is not None:
            solute_feats = {k: v.to(device) for k, v in solute_feats.items()}
            solvent_feats = {k: v.to(device) for k, v in solvent_feats.items()}
            target = target.to(device)
            solute_norm_atom = solute_norm_atom.to(device)
            solute_norm_bond = solute_norm_bond.to(device)
            solvent_norm_atom = solvent_norm_atom.to(device)
            solvent_norm_bond = solvent_norm_bond.to(device)
            #stdev = stdev.to(device)
        
        pred = model(solute_batched_graph, solvent_batched_graph, solute_feats, 
                     solvent_feats, solute_norm_atom, solute_norm_bond, 
                     solvent_norm_atom, solvent_norm_bond)
        pred = pred.view(-1)

        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()
        accuracy += metric_fn(pred, target).detach().item()
        count += len(target)
    
    epoch_loss /= it + 1
    accuracy /= count

    return epoch_loss, accuracy

def evaluate(model, nodes, data_loader, metric_fn, scaler = None, device=None, return_preds=False):
    """
    Evaluate the accuracy of an validation set of test set.
    Args:
        metric_fn (function): the function should be using a `sum` reduction method.
    """

    model.eval()

    with torch.no_grad():
        accuracy = 0.0
        count = 0.0

        preds = []
        y_true = []

        for solute_batched_graph, solvent_batched_graph, label in data_loader:
            solute_feats = {nt: solute_batched_graph.nodes[nt].data["feat"] for nt in nodes}
            solvent_feats = {nt: solvent_batched_graph.nodes[nt].data["feat"] for nt in nodes}
            target = torch.squeeze(label["value"])
            #stdev = label["scaler_stdev"]
            solvent_norm_atom = label["solvent_norm_atom"]
            solvent_norm_bond = label["solvent_norm_bond"]
            solute_norm_atom = label["solute_norm_atom"]
            solute_norm_bond = label["solute_norm_bond"]
            
            if device is not None:
                solute_feats = {k: v.to(device) for k, v in solute_feats.items()}
                solvent_feats = {k: v.to(device) for k, v in solvent_feats.items()}
                target = target.to(device)
                solute_norm_atom = solute_norm_atom.to(device)
                solute_norm_bond = solute_norm_bond.to(device)
                solvent_norm_atom = solvent_norm_atom.to(device)
                solvent_norm_bond = solvent_norm_bond.to(device)
                #stdev = stdev.to(device)

            pred = model(solute_batched_graph, solvent_batched_graph, solute_feats, 
                     solvent_feats, solute_norm_atom, solute_norm_bond, 
                     solvent_norm_atom, solvent_norm_bond)
            pred = pred.view(-1)

            # Inverse scale 
            if scaler is not None:
                pred = scaler.inverse_transform(pred.cpu())
                pred = pred.to(device)

            accuracy += metric_fn(pred, target).detach().item()
            count += len(target)
            
            batch_pred = pred.tolist()
            batch_target = target.tolist()
            preds.extend(batch_pred)
            y_true.extend(batch_target)

    if return_preds:
        return y_true, preds

    else:
        return accuracy / count


def objective(trial, dataset, random_seed, save_dir):
    embedding_size = trial.suggest_int("embedding_size", 24, 72, step=24)
    gated_graph_norm = trial.suggest_int("gated_graph_norm", 0, 1)
    gated_residual = trial.suggest_int("gated_residual", 0, 1)
    gated_batch_norm = trial.suggest_int("gated_batch_norm", 0, 1)
    gated_dropout = trial.suggest_discrete_uniform("gated_dropout", 0.0, 0.5, 0.1)
    fc_batch_norm = trial.suggest_int("fc_batch_norm", 0, 1)
    fc_dropout = trial.suggest_discrete_uniform("fc_dropout", 0.0, 0.5, 0.1)
    gated_hidden_size = trial.suggest_int("gated_hidden_size", 200, 800, step=200)
    fc_hidden_size = trial.suggest_int("fc_hidden_size", 800, 1400, step=200)
    batch_size = trial.suggest_int("batch_size", 50, 100, step=50)
    fc_num_layers = trial.suggest_int("fc_num_layers", 2, 4)
    gated_num_layers = trial.suggest_int("gated_num_layers", 2, 4)
    gated_num_fc_layers = trial.suggest_int("gated_num_fc_layers", 2, 4)
    num_lstm_layers = trial.suggest_int("num_lstm_layers", 2, 4)
    num_lstm_iters = trial.suggest_int("num_lstm_iters", 5, 8)
    lr = 10**trial.suggest_int('lr', -4, -2)

    feature_names = ["atom", "bond", "global"]
    set2set_ntypes_direct = ["global"]
    solute_feature_size = dataset.feature_sizes[0]
    solvent_feature_size = dataset.feature_sizes[1]    

    gated_hidden_size = [gated_hidden_size] * gated_num_layers
    valx = 2 * gated_hidden_size[-1]
    fc_hidden_size = [max(valx // 2 ** i, 8) for i in range(fc_num_layers)]

    best = np.finfo(np.float32).max
    model = GatedGCNSolvationNetwork(
        solute_in_feats=solute_feature_size,
        solvent_in_feats=solvent_feature_size,
        embedding_size=embedding_size,
        gated_num_layers=gated_num_layers,
        gated_hidden_size=gated_hidden_size,
        gated_num_fc_layers=gated_num_fc_layers,
        gated_graph_norm=gated_graph_norm,
        gated_batch_norm=gated_batch_norm,
        gated_activation="LeakyReLU",
        gated_residual=gated_residual,
        gated_dropout=gated_dropout,
        num_lstm_iters=num_lstm_iters,
        num_lstm_layers=num_lstm_layers,
        set2set_ntypes_direct=set2set_ntypes_direct,
        fc_num_layers=fc_num_layers,
        fc_hidden_size=fc_hidden_size,
        fc_batch_norm=fc_batch_norm,
        fc_activation="LeakyReLU",
        fc_dropout=fc_dropout,
        outdim=1,
        conv='GatedGCNConv',
    )

    if torch.cuda.is_available():
        device = "cuda:0"
        model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=0
    )
    loss_func = MSELoss(reduction="mean")
    metric = L1Loss(reduction="sum")
    ### learning rate scheduler and stopper
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.4, patience=50, verbose=True
    )
    stopper = EarlyStopping(patience=150)
    # load checkpoint
    state_dict_objs = {"model": model, "optimizer": optimizer, "scheduler": scheduler}

    print(f'Building dataset, splitting with random seed {random_seed}')

    trainset, valset, testset = train_validation_test_split(
        dataset, validation=0.1, test=0.1, random_seed=random_seed)

    solute_features_scaler, solvent_features_scaler = trainset.normalize_features()
    valset.normalize_features(solute_features_scaler, solvent_features_scaler)
    testset.normalize_features(solute_features_scaler, solvent_features_scaler)
    label_scaler = trainset.normalize_labels()
    
    print("Trainset size: {}, valset size: {}: testset size: {}.".format(
            len(trainset), len(valset), len(testset)))

    train_loader = DataLoaderSolvation(
            trainset,
            batch_size = batch_size,
            shuffle = True,
            sampler = None
        )

    bs = max(len(valset) // 10, 1)
    val_loader = DataLoaderSolvation(valset, batch_size=bs, shuffle=False)
    bs = max(len(testset) // 10, 1)
    #test_loader = DataLoaderSolvation(testset, batch_size=bs, shuffle=False)

    print("\n\n# Epoch     Loss         TrainAcc        ValAcc     Time (s)")
    sys.stdout.flush()

    for epoch in range(300):
        ti = time.time()
        loss, train_acc = train(
                optimizer, model, feature_names, train_loader, loss_func, metric, device)
        
        if np.isnan(loss):
            print("\n\nBad, we get nan for loss. Exiting")
            sys.stdout.flush()
            sys.exit(1)

        val_acc = evaluate(model, feature_names, val_loader, metric, label_scaler, device)
        scheduler.step(val_acc)
        
        is_best = val_acc < best
        if is_best:
            best = val_acc
        
        misc_objs = {"best": best, "epoch": epoch}
        scaler_objs = {'label_scaler': {
            'means': label_scaler.mean,
            'stds': label_scaler.std
            } if label_scaler is not None else None,
            'solute_features_scaler': {
            'means': solute_features_scaler.mean,
            'stds': solute_features_scaler.std
            } if solute_features_scaler is not None else None,
            'solvent_features_scaler': {
            'means': solvent_features_scaler.mean,
            'stds': solvent_features_scaler.std
            } if solvent_features_scaler is not None else None}
        save_checkpoints(
            state_dict_objs,
            misc_objs,
            scaler_objs,
            is_best,
            msg=f"epoch: {epoch}, score {val_acc}",
            save_dir=save_dir)

        tt = time.time() - ti
        print("{:5d}   {:12.6e}   {:12.6e}   {:12.6e}   {:.2f}".format(
            epoch, loss, train_acc, val_acc, tt) )
        if epoch % 10 == 0:
            sys.stdout.flush()

        trial.report(val_acc, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    
    return val_acc

if __name__ == "__main__":
    args = parse_args()
    
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    logging.basicConfig(
    filename=os.path.join(args.save_dir, '{}.log'.format(
        datetime.now().strftime("gnn_%Y_%m_%d-%I_%M_%p"))),
    format="%(asctime)s:%(name)s:%(levelname)s: %(message)s",
    level=logging.INFO,
    )

    random_seed = args.random_seed
    seed_torch(random_seed)

    print("\n\nStart training at: ", datetime.now())


    if args.save_dir is None:
        save_dir = os.getcwd()
    else:
        save_dir = args.save_dir

    mols, labels = load_mols_labels(args.dataset_file)

    dataset = SolvationDataset(
        solute_grapher = grapher(mol_volume=False),
        solvent_grapher = grapher(mol_volume=False),
        molecules = mols,
        labels = labels,
        solute_extra_features = None,
        solvent_extra_features = None,
        feature_transformer = False,
        label_transformer= False,
        state_dict_filename=None
    )
    os.makedirs(args.save_dir, exist_ok=True)

    func = lambda trial: objective(trial, dataset, random_seed, save_dir)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction='minimize', study_name='HyperparamOpt', pruner=pruner)
    study.optimize(func, n_trials=100)

    pruned_trials = study.get_trials(states=(optuna.trial.TrialState.PRUNED,))
    complete_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
