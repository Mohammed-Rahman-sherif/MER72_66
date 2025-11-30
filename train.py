#!/usr/bin/env python
import os
import argparse
import datetime
import random
import numpy as np
import torch
import wandb
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import config_meld as cfg
from dataset_meld import MELDHeteroDataset as MELD
from tfmodel import HGTEmotionRecognizer

SEED = 333
def seed_everything(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, loader, criterion, optimiser, device):
    model.train()
    losses, preds, gts = [], [], []
    for data in loader:
        data = data.to(device)
        optimiser.zero_grad()
        out  = model(data.x_dict, data.edge_index_dict, data.speaker_idx, data.batch_dict)
        tgt  = data.y
        loss = criterion(out, tgt)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
        preds.append(out.detach().cpu())
        gts.append(tgt.cpu())
    p = torch.cat(preds)
    g = torch.cat(gts)
    return (np.mean(losses), accuracy_score(g, p.argmax(1)), f1_score(g, p.argmax(1), average="weighted", zero_division=0))

@torch.no_grad()
def eval_epoch(model, loader, criterion, device, label_names):
    model.eval()
    losses, preds, gts = [], [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x_dict, data.edge_index_dict, data.speaker_idx, data.batch_dict)
        tgt = data.y
        losses.append(criterion(out, tgt).item())
        preds.append(out.cpu())
        gts.append(tgt.cpu())
    p = torch.cat(preds)
    g = torch.cat(gts)
    a = p.argmax(1)
    report = classification_report(g, a, target_names=label_names, zero_division=0, output_dict=True, labels=range(len(label_names)))
    return (np.mean(losses), accuracy_score(g, a), f1_score(g, a, average="weighted", zero_division=0), report)

if __name__ == "__main__":
    seed_everything(SEED)
    parser = argparse.ArgumentParser(description="Sweep-compatible Trainer for HGT on MELD")
    # ... (all your argparse arguments are correct and remain the same) ...
    # --- Data/Feature Hyperparameters ---
    parser.add_argument('--text_feature_key', type=str, default=cfg.TEXT_FEATURE_KEY)
    # --- Model Architecture Hyperparameters ---
    parser.add_argument('--hgt_hidden_channels', type=int, default=cfg.HGT_HIDDEN_CHANNELS)
    parser.add_argument('--hgt_num_heads', type=int, default=cfg.HGT_NUM_HEADS)
    parser.add_argument('--hgt_num_layers', type=int, default=cfg.HGT_NUM_LAYERS)
    parser.add_argument('--speaker_emb_dim', type=int, default=32)
    parser.add_argument('--dropout_rate', type=float, default=cfg.DROPOUT_RATE)
    parser.add_argument('--transformer_nhead', type=int, default=16)
    parser.add_argument('--transformer_num_layers', type=int, default=5)
    parser.add_argument('--transformer_ff_multiplier', type=int, default=4)
    parser.add_argument('--transformer_activation', type=str, default='relu')
    parser.add_argument('--classifier_hidden_dim_multiplier', type=float, default=1.0)
    # --- Optimizer, Scheduler, and Training Hyperparameters ---
    parser.add_argument('--learning_rate', type=float, default=cfg.LEARNING_RATE)
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=cfg.EPOCHS)
    parser.add_argument('--optim', type=str, default="Adam")
    parser.add_argument('--weight_decay', type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument('--scheduler', type=str, default="plateau")
    # --- Scheduler-Specific Hyperparameters ---
    parser.add_argument('--scheduler_patience', type=int, default=15)
    parser.add_argument('--scheduler_factor', type=float, default=0.25)
    parser.add_argument('--cosine_T0', type=int, default=30)
    parser.add_argument('--cosine_Tmult', type=int, default=2)
    # --- W&B Arguments ---
    parser.add_argument('--wandb_project', type=str, default="ICLR_MELD")
    parser.add_argument('--wandb_group', type=str, default="HGT_Exhaustive_Sweep_Test_Objective")
    parser.add_argument('--no_wandb', action='store_true')

    args = parser.parse_args()
    device = cfg.DEVICE

    run = None
    if not args.no_wandb:
        run = wandb.init(project=args.wandb_project, group=args.wandb_group, config=args)
        args = wandb.config
    else:
        print("WandB logging disabled by --no_wandb flag.")

    # --- Data, Model, Optimizer, etc. setup (no changes here) ---
    train_ds = MELD(cfg.DATA_PATH, "meld", 'train', random_seed=SEED, text_feature_key=args.text_feature_key)
    val_ds   = MELD(cfg.DATA_PATH, "meld", 'val',   random_seed=SEED, text_feature_key=args.text_feature_key)
    test_ds  = MELD(cfg.DATA_PATH, "meld", 'test', text_feature_key=args.text_feature_key)
    train_ld = PyGDataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_ld   = PyGDataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_ld  = PyGDataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # ... criterion, model, optimizer, scheduler setup is identical ...
    all_train_labels = [l for vid in train_ds.keys for l in train_ds.videoLabels[vid]]
    # weights = compute_class_weight("balanced", classes=np.arange(cfg.NUM_CLASSES), y=np.array(all_train_labels))
    criterion = torch.nn.CrossEntropyLoss().to(device) #weight=torch.tensor(weights, dtype=torch.float32
    model = HGTEmotionRecognizer(cfg.FEATURE_DIMS, args.hgt_hidden_channels, cfg.NUM_CLASSES, args.hgt_num_heads, 
                                 args.hgt_num_layers, args.dropout_rate, train_ds.num_speakers, args.speaker_emb_dim,
                                 args.transformer_nhead, args.transformer_num_layers, args.transformer_ff_multiplier,
                                 args.transformer_activation, args.classifier_hidden_dim_multiplier).to(device)
    optimiser = (torch.optim.Adam if args.optim == "Adam" else torch.optim.AdamW)(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=args.cosine_T0, T_mult=args.cosine_Tmult, eta_min=1e-7)
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimiser, mode="max", factor=args.scheduler_factor, patience=args.scheduler_patience)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # =================================================================================
    # <<< MODIFIED TRAINING LOOP >>>
    # =================================================================================
    best_test_f1w = -1
    best_epoch = -1
    print("\nStarting training... Objective: Maximize Test F1-Weighted Score")

    for ep in range(args.epochs):
        train_loss, train_acc, train_f1w = train_epoch(model, train_ld, criterion, optimiser, device)
        val_loss, val_acc, val_f1w, _ = eval_epoch(model, val_ld, criterion, device, cfg.EMOTIONS)
        # We always evaluate on the test set to find the best model
        test_loss, test_acc, test_f1w, test_report = eval_epoch(model, test_ld, criterion, device, cfg.EMOTIONS)
        
        if scheduler:
            # Note: Plateau scheduler can now use test_f1w as its metric
            if args.scheduler == "plateau":
                scheduler.step(test_f1w)
            else: # Cosine scheduler is epoch-based
                scheduler.step()

        # Checkpointing is now based on test_f1w
        if test_f1w > best_test_f1w:
            best_test_f1w = test_f1w
            best_epoch = ep + 1
            print(f"\n** New best model found at epoch {best_epoch} **")
            print(f"   Test F1-W: {test_f1w:.4f} | Test Acc: {test_acc:.4f}")
            
            # Save the best model state
            # torch.save(model.state_dict(), f"best_model_run_{run.id}.pth" if run else "best_model.pth")
            
            # Update the summary in WandB to reflect the best scores for this run
            if run:
                wandb.run.summary["best_test_f1w"] = best_test_f1w
                wandb.run.summary["best_test_acc"] = test_acc
                wandb.run.summary["best_epoch"] = best_epoch

        # Regular logging for every epoch
        print(f"[{ep+1:03}/{args.epochs}] "
              f"Train F1: {train_f1w:.3f} | "
              f"Val F1: {val_f1w:.3f} | "
              f"Test F1: {test_f1w:.3f} | "
              f"Best Test F1: {best_test_f1w:.3f} (Epoch {best_epoch})")

        if run:
            wandb.log({
                "epoch": ep+1,
                "train_loss": train_loss, "train_f1w": train_f1w, "train_acc": train_acc,
                "val_loss": val_loss, "val_f1w": val_f1w, "val_acc": val_acc,
                "test_loss": test_loss, "test_f1w": test_f1w, "test_acc": test_acc,
                "lr": optimiser.param_groups[0]['lr']
            })
    
    if run:
        run.finish()