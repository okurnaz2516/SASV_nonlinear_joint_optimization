#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel Model for ASV and CM Tasks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from dataset import SASV_Dataset_redimnet_sslaasist
from metrics import get_all_EERs
import matplotlib.pyplot as plt
import seaborn as sns
from adcf_utils import aDCF_loss, calculate_adcf_hard_act, calculate_adcf_hard_min, calculate_adcf_soft_act

# === Argument Parser ===
parser = argparse.ArgumentParser(description="Parallel Model for ASV and CM tasks")
parser.add_argument('--seed', type=int, default=688, help="Random number seed")
parser.add_argument("-o", "--output_dir", type=str, default="./results_5/", help="Output directory for results")
parser.add_argument("--batch_size", type=int, default=192, help="Mini batch size for training")
parser.add_argument('--lr', type=float, default=0.000861, help="Learning rate")
parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
parser.add_argument("--embedding_dir", type=str, default="./embeddings_5/", help="Folder for embeddings")
parser.add_argument("--spk_meta_dir", type=str, default="./spk_meta_5/", help="Folder for speaker meta info")
parser.add_argument("--sasv_dev_trial", type=str, default="./protocols_5/ASVspoof5.dev.trial.txt")
parser.add_argument("--sasv_eval_trial", type=str, default="./protocols_5/ASVspoof5.track_2.eval.trial.txt")

# ASV Task Arguments
parser.add_argument('--asv_num_layers', type=int, default=2, help="Number of layers for ASV task")
parser.add_argument('--asv_node_sizes', nargs='+', type=int, default=[384, 160], help="Node sizes per layer for ASV task")

# CM Task Arguments
parser.add_argument('--cm_num_layers', type=int, default=2, help="Number of layers for CM task")
parser.add_argument('--cm_node_sizes', nargs='+', type=int, default=[384, 160], help="Node sizes per layer for CM task")

args = parser.parse_args()

DEFAULT_ASV_INPUT_DIM = 192 + 192
DEFAULT_CM_INPUT_DIM = 192 + 160

# === Model Definitions ===
class FlexibleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(FlexibleClassifier, self).__init__()
        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ParallelModel(nn.Module):
    def __init__(self, asv_input_dim, asv_hidden_layers, cm_input_dim, cm_hidden_layers):
        super(ParallelModel, self).__init__()
        self.asv_classifier = FlexibleClassifier(asv_input_dim, asv_hidden_layers)
        self.cm_classifier = FlexibleClassifier(cm_input_dim, cm_hidden_layers)
        self.rho = 0.5
        self.asv_layer = nn.Linear(1, 1, bias=True)
        self.cm_layer = nn.Linear(1, 1, bias=True)

    def forward(self, asv_input, cm_input):
        asv_output = self.asv_classifier(asv_input)  # ASV score
        cm_output = self.cm_classifier(cm_input)    # CM score
        asv_out = self.asv_layer(asv_output)
        cm_out = self.cm_layer(cm_output)
        
        shift = (asv_out + cm_out) / 2.0
        # Compute s_sasv
        s_sasv = (
            0.5 * torch.exp(-asv_out + shift)
            + (1 - 0.5) * torch.exp(-cm_out + shift)
        )
        epsilon = 1e-10
        s_sasv = - (torch.log(s_sasv + epsilon) - shift)
        return s_sasv, asv_out, cm_out

# === Training Function ===
def train(model, dataloader, optimizer, loss_fn, loss_adcf, device):
    model.train()
    total_loss = 0
    l_adcf = 0
    l_bce = 0
    l_asv = 0
    l_cm = 0
    all_predictions = []
    all_targets = []
    all_keys = []

    for batch in dataloader:
        asv1, asv2, cm1, labels, key, asv_label, cm_label = [
            x.to(device) if isinstance(x, torch.Tensor) else x for x in batch
        ]
        asv_inputs = torch.cat((asv1, asv2), dim=1)
        cm_inputs = torch.cat((asv2, cm1), dim=1)
        targets = labels.to(device).float()

        with torch.enable_grad():
            optimizer.zero_grad()  # Reset gradients
            predictions, asv_out, cm_out = model(asv_inputs, cm_inputs)  # Forward pass
            loss1 = loss_fn(predictions, labels.unsqueeze(1).float())
            loss2 = loss_adcf.calculate_a_dcf(predictions, np.array(key), 0.5)
            loss2 = loss2.to(device)
            loss_asv =  loss_fn(asv_out, asv_label.unsqueeze(1).float())
            loss_cm =  loss_fn(cm_out, cm_label.unsqueeze(1).float())
            loss = (loss1 + loss2 + loss_asv + loss_cm) / 4
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        l_adcf += loss2.item()
        l_bce += loss1.item()
        l_asv += loss_asv.item()
        l_cm += loss_cm.item()
        all_predictions.append(predictions.detach())
        all_targets.extend(targets.cpu().numpy())
        all_keys.extend(list(key))

    all_predictions = torch.cat(all_predictions, dim=0).detach().cpu().numpy()
    all_keys = np.array(all_keys)

    sasv_eer, sv_eer, spf_eer = get_all_EERs(all_predictions, all_keys)
    
    hard_act_far_asv, hard_act_far_cm, hard_act_frr, hard_act_adcf = calculate_adcf_hard_act(all_predictions, all_keys, 0.5)
    
    soft_act_far_asv, soft_act_far_cm, soft_act_frr, soft_act_adcf = calculate_adcf_soft_act(all_predictions, all_keys, 0.5)
    
    hard_far_asvs, hard_far_cms, hard_frrs, hard_fars, hard_adcfs, hard_min_adcf, hard_adcf_thresholds, hard_min_adcf_threshold, \
    hard_min_far_asv, hard_min_far_cm, hard_min_frr = calculate_adcf_hard_min(all_predictions, all_keys)
    
    # soft_far_asvs, soft_far_cms, soft_frrs, soft_fars, soft_adcfs, soft_min_adcf, soft_adcf_thresholds, soft_min_adcf_threshold, \
    # soft_min_far_asv, soft_min_far_cm, soft_min_frr = calculate_adcf_soft_min(all_predictions, all_keys)
    
    return total_loss / len(dataloader), l_adcf / len(dataloader), l_bce / len(dataloader), l_asv / len(dataloader), \
        l_cm / len(dataloader), sasv_eer, sv_eer, spf_eer, hard_act_far_asv, hard_act_far_cm, hard_act_frr, hard_act_adcf, \
        soft_act_far_asv, soft_act_far_cm, soft_act_frr, soft_act_adcf, hard_min_adcf, hard_min_adcf_threshold

# === Evaluation Function ===
def evaluate(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    all_keys = []

    with torch.no_grad():
        for batch in dataloader:
            asv1, asv2, cm1, labels, key, asv_label, cm_label = [
                x.to(device) if isinstance(x, torch.Tensor) else x for x in batch
            ]
            asv_inputs = torch.cat((asv1, asv2), dim=1)
            cm_inputs = torch.cat((asv2, cm1), dim=1)
            targets = labels.to(device).float()
            predictions, asv_out, cm_out = model(asv_inputs, cm_inputs)
            all_predictions.append(predictions)
            all_targets.extend(targets.cpu().numpy())
            all_keys.extend(list(key))

    all_predictions = torch.cat(all_predictions, dim=0).detach().cpu().numpy()
    all_keys = np.array(all_keys)

    sasv_eer, sv_eer, spf_eer = get_all_EERs(all_predictions, all_keys)

    hard_act_far_asv, hard_act_far_cm, hard_act_frr, hard_act_adcf = calculate_adcf_hard_act(all_predictions, all_keys, 0.5)
    
    soft_act_far_asv, soft_act_far_cm, soft_act_frr, soft_act_adcf = calculate_adcf_soft_act(all_predictions, all_keys, 0.5)
    
    hard_far_asvs, hard_far_cms, hard_frrs, hard_fars, hard_adcfs, hard_min_adcf, hard_adcf_thresholds, hard_min_adcf_threshold, \
    hard_min_far_asv, hard_min_far_cm, hard_min_frr = calculate_adcf_hard_min(all_predictions, all_keys)
    
    # soft_far_asvs, soft_far_cms, soft_frrs, soft_fars, soft_adcfs, soft_min_adcf, soft_adcf_thresholds, soft_min_adcf_threshold, \
    # soft_min_far_asv, soft_min_far_cm, soft_min_frr = calculate_adcf_soft_min(all_predictions, all_keys)
    
    return sasv_eer, sv_eer, spf_eer, hard_act_far_asv, hard_act_far_cm, hard_act_frr, hard_act_adcf, soft_act_far_asv, \
        soft_act_far_cm, soft_act_frr, soft_act_adcf, hard_min_adcf, hard_min_adcf_threshold, all_predictions, all_keys



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_seed(seed=688):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def build_model():
        return ParallelModel(
            asv_input_dim=DEFAULT_ASV_INPUT_DIM,
            asv_hidden_layers=args.asv_node_sizes,
            cm_input_dim=DEFAULT_CM_INPUT_DIM,
            cm_hidden_layers=args.cm_node_sizes
        ).to(device)

    def build_dataloaders():
        train_dataset = SASV_Dataset_redimnet_sslaasist(args, partition="trn")
        dev_dataset = SASV_Dataset_redimnet_sslaasist(args, partition="dev")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
        return train_loader, dev_loader

    def format_log_message(epoch, trn_metrics, dev_metrics):
        trn_loss, trn_loss_adcf, trn_loss_bce, trn_loss_asv, trn_loss_cm, trn_sasv_eer, trn_sv_eer, trn_spf_eer, \
        trn_hard_act_far_asv, trn_hard_act_far_cm, trn_hard_act_frr, trn_hard_act_adcf, trn_soft_act_far_asv, \
        trn_soft_act_far_cm, trn_soft_act_frr, trn_soft_act_adcf, trn_hard_min_adcf, trn_hard_min_adcf_threshold = trn_metrics

        dev_sasv_eer, dev_sv_eer, dev_spf_eer, dev_hard_act_far_asv, dev_hard_act_far_cm, \
        dev_hard_act_frr, dev_hard_act_adcf, dev_soft_act_far_asv, dev_soft_act_far_cm, \
        dev_soft_act_frr, dev_soft_act_adcf, dev_hard_min_adcf, dev_hard_min_adcf_threshold, _, _ = dev_metrics

        return (
            f"Epoch {epoch + 1}/{args.num_epochs}, Train Loss: {trn_loss:.4f}, "
            f"Train Loss BCE: {trn_loss_bce:.4f}, Train Loss a-DCF: {trn_loss_adcf:.4f}, "
            f"Train Loss ASV: {trn_loss_asv:.4f}, Train Loss CM: {trn_loss_cm:.4f}, "
            f"Train SASV EER: {trn_sasv_eer:.4f}, Train SV EER: {trn_sv_eer:.4f}, "
            f"Train SPF EER: {trn_spf_eer:.4f}, Train HardActFARASV: {trn_hard_act_far_asv:.4f}, "
            f"Train HardActFARCM: {trn_hard_act_far_cm:.4f}, Train HardActFRR: {trn_hard_act_frr:.4f}, "
            f"Train HardActADCF: {trn_hard_act_adcf[0]:.4f}, Train SoftActFARASV: {trn_soft_act_far_asv[0]:.4f}, "
            f"Train SoftActFARCM: {trn_soft_act_far_cm[0]:.4f}, Train SoftActFRR: {trn_soft_act_frr[0]:.4f}, "
            f"Train SoftActADCF: {trn_soft_act_adcf[0]:.4f}, Train HardMinADCF: {trn_hard_min_adcf:.4f}, "
            f"Train HardMinADCFThreshold: {trn_hard_min_adcf_threshold[0]:.4f},"
            f"Dev SASV EER: {dev_sasv_eer:.4f}, Dev SV EER: {dev_sv_eer:.4f}, Dev SPF EER: {dev_spf_eer:.4f}, "
            f"Dev HardActFARASV: {dev_hard_act_far_asv:.4f}, "
            f"Dev HardActFARCM: {dev_hard_act_far_cm:.4f}, Dev HardActFRR: {dev_hard_act_frr:.4f}, "
            f"Dev HardActADCF: {dev_hard_act_adcf[0]:.4f}, Dev SoftActFARASV: {dev_soft_act_far_asv[0]:.4f}, "
            f"Dev SoftActFARCM: {dev_soft_act_far_cm[0]:.4f}, Dev SoftActFRR: {dev_soft_act_frr[0]:.4f}, "
            f"Dev SoftActADCF: {dev_soft_act_adcf[0]:.4f}, Dev HardMinADCF: {dev_hard_min_adcf:.4f}, "
            f"Dev HardMinADCFThreshold: {dev_hard_min_adcf_threshold[0]:.4f}\n"
        )

    def train_single_run():
        set_seed(args.seed)
        os.makedirs(args.output_dir, exist_ok=True)

        model = build_model()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        adcf_loss = aDCF_loss()
        train_loader, dev_loader = build_dataloaders()

        log_file = os.path.join(
            args.output_dir,
            "training_log.txt"
        )
        best_model_path = os.path.join(
            args.output_dir,
            "best_model.pth"
        )

        best_adcf = float("inf")
        best_model_state_dict = None

        with open(log_file, "w") as f:
            for epoch in range(args.num_epochs):
                trn_metrics = train(model, train_loader, optimizer, loss_fn, adcf_loss, device)
                dev_metrics = evaluate(model, dev_loader, device)
                log_msg = format_log_message(epoch, trn_metrics, dev_metrics)
                print(log_msg.strip())
                f.write(log_msg)

                dev_hard_min_adcf = dev_metrics[11]
                if dev_hard_min_adcf < best_adcf:
                    best_adcf = dev_hard_min_adcf
                    best_model_state_dict = model.state_dict()
                    print(f"New best model found at epoch {epoch + 1} with hard min a-DCF: {best_adcf:.4f}\n")
                    torch.save(best_model_state_dict, best_model_path)

        if best_model_state_dict is None:
            raise RuntimeError("Best model could not be selected during training.")

        print(f"Best model saved with hard min a-DCF: {best_adcf:.4f}")
        return best_model_path, dev_loader

    def print_final_results(eval_metrics):
        best_dev_sasv_eer, best_dev_sv_eer, best_dev_spf_eer, best_dev_hard_act_far_asv, best_dev_hard_act_far_cm, \
        best_dev_hard_act_frr, best_dev_hard_act_adcf, best_dev_soft_act_far_asv, best_dev_soft_act_far_cm, \
        best_dev_soft_act_frr, best_dev_soft_act_adcf, best_dev_hard_min_adcf, best_dev_hard_min_adcf_threshold, best_preds, best_keys = eval_metrics

        print("\n=== Final Results (Single Run) ===")
        print("\n=== Random Non-Linear Fusion L2 ===")
        print(f"SASV EER: {best_dev_sasv_eer:.4f}")
        print(f"SV EER: {best_dev_sv_eer:.4f}")
        print(f"SPF EER: {best_dev_spf_eer:.4f}")
        print(f"Hard Act FAR ASV: {best_dev_hard_act_far_asv:.4f}")
        print(f"Hard Act FAR CM: {best_dev_hard_act_far_cm:.4f}")
        print(f"Hard Act FRR: {best_dev_hard_act_frr:.4f}")
        print(f"Hard Act a-DCF: {best_dev_hard_act_adcf[0]:.4f}")
        print(f"Soft Act FAR ASV: {best_dev_soft_act_far_asv[0]:.4f}")
        print(f"Soft Act FAR CM: {best_dev_soft_act_far_cm[0]:.4f}")
        print(f"Soft Act FRR: {best_dev_soft_act_frr[0]:.4f}")
        print(f"Soft Act a-DCF: {best_dev_soft_act_adcf[0]:.4f}")
        print(f"Hard Min a-DCF: {best_dev_hard_min_adcf:.4f}")
        print(f"Hard Min a-DCF Threshold: {best_dev_hard_min_adcf_threshold[0]:.4f}")
        return best_preds, best_keys

    def save_kde_plot(best_preds, best_keys):
        best_preds = np.array(best_preds)
        best_keys = np.array(best_keys)
        target_preds = best_preds[best_keys == 'target'].ravel()
        nontarget_preds = best_preds[best_keys == 'nontarget'].ravel()
        spoof_preds = best_preds[best_keys == 'spoof'].ravel()

        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=target_preds, color="green", label="Target", fill=True, alpha=0.5, bw_adjust=1.5)
        sns.kdeplot(data=nontarget_preds, color="red", label="Non-Target", fill=True, alpha=0.5, bw_adjust=1.5)
        sns.kdeplot(data=spoof_preds, color="black", label="Spoof", fill=True, alpha=0.5, bw_adjust=1.5)
        plt.title("SASV Dev Set Predictions KDE Distribution")
        plt.xlabel("Prediction Score")
        plt.ylabel("Density")
        plt.legend()
        final_kde_path = os.path.join(
            args.output_dir,
            "dev_scores_kde.png"
        )
        plt.savefig(final_kde_path)
        plt.show()
        plt.close()
        print(f"Final KDE saved to {final_kde_path}.")

    best_model_path, dev_loader = train_single_run()
    best_model = build_model()
    best_model.load_state_dict(torch.load(best_model_path))
    eval_metrics = evaluate(best_model, dev_loader, device)
    best_preds, best_keys = print_final_results(eval_metrics)
    save_kde_plot(best_preds, best_keys)