#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel Model for ASV and CM Tasks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from dataset import SASV_Dataset_redimnet_sslaasist
from metrics import get_all_EERs
from adcf_utils import C_FA_ASV, C_FA_CM, C_MISS, aDCF_loss, calculate_adcf_hard_act, calculate_adcf_hard_min, calculate_adcf_soft_act


def compute_cllr_loss(scores, positive_mask, negative_mask):
    """Compute C_llr with numerically stable softplus terms."""
    pos_scores = scores[positive_mask]
    neg_scores = scores[negative_mask]

    if pos_scores.numel() == 0:
        pos_term = torch.zeros((), device=scores.device, dtype=scores.dtype)
    else:
        pos_term = F.softplus(-pos_scores).mean()

    if neg_scores.numel() == 0:
        neg_term = torch.zeros((), device=scores.device, dtype=scores.dtype)
    else:
        neg_term = F.softplus(neg_scores).mean()

    return (pos_term + neg_term) / (2.0 * np.log(2.0))


# === Argument Parser ===
parser = argparse.ArgumentParser(description="Parallel Model for ASV and CM tasks")
parser.add_argument("-o", "--output_dir", type=str, default="./results_5/", help="Output directory for results")
parser.add_argument("--batch_size", type=int, default=192, help="Mini batch size for training")
parser.add_argument('--lr', type=float, default=0.000861, help="Learning rate")
parser.add_argument('--num_epochs', type=int, default=250, help="Number of epochs for training")
parser.add_argument("--embedding_dir", type=str, default="./embeddings_5/", help="Folder for embeddings")
parser.add_argument("--spk_meta_dir", type=str, default="./spk_meta_5/", help="Folder for speaker meta info")
parser.add_argument("--sasv_dev_trial", type=str, default="./protocols_5/ASVspoof5.dev.trial.txt")
parser.add_argument("--sasv_eval_trial", type=str, default="./protocols_5/ASVspoof5.eval.track_2.trial.txt")

args = parser.parse_args()

# Cosine similarity function
def cosine_similarity(a, b):
    return F.cosine_similarity(a, b, dim=1).unsqueeze(1)  # shape: (batch_size, 1)

# Siamese layer for ASV embeddings
class ASVSiameseLayer(nn.Module):
    def __init__(self, input_dim=192):
        super(ASVSiameseLayer, self).__init__()
        self.projection = nn.Parameter(torch.randn(input_dim))

    def forward(self, asv_enr, asv_tst):
        proj_enr = asv_enr * self.projection  # (batch_size,)
        proj_tst = asv_tst * self.projection  # (batch_size,)
        return proj_enr, proj_tst

# Flexible CM classifier branch
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

# Main SASV model
class SASVModel(nn.Module):
    def __init__(self):
        super(SASVModel, self).__init__()
        self.siamese_layer = ASVSiameseLayer()
        self.cm_branch = FlexibleClassifier(input_dim=1216, hidden_layers=[384, 160])
        self.asv_calibration_layer = nn.Linear(1, 1, bias=True)
        self.cm_calibration_layer = nn.Linear(1, 1, bias=True)

    def forward(self, asv_enr, asv_tst, cm_input):

        # ASV branch
        emb_enr, emb_tst = self.siamese_layer(asv_enr, asv_tst)
        asv_score = cosine_similarity(emb_enr, emb_tst)  # shape: (batch_size, 1)

        # CM branch
        cm_score = self.cm_branch(cm_input)  # shape: (batch_size, 1)

        # Calibration
        calibrated_asv = self.asv_calibration_layer(asv_score)
        calibrated_cm = self.cm_calibration_layer(cm_score)

        shift = (calibrated_asv + calibrated_cm) / 2.0
        # Compute s_sasv
        s_sasv = 0.5 * (C_FA_ASV / C_MISS) * torch.exp(-calibrated_asv + shift) + \
                    (1 - 0.5) * (C_FA_CM / C_MISS) * torch.exp(-calibrated_cm + shift)
        epsilon = 1e-10
        s_sasv = - (torch.log(s_sasv + epsilon) - shift)
        return s_sasv, calibrated_asv, calibrated_cm
    
# === Training Function ===
def train(model, dataloader, optimizer, loss_fn, loss_adcf, device):
    model.train()
    total_loss = 0
    l_adcf = 0
    l_bce = 0
    all_predictions = []
    all_targets = []
    all_keys = []


    for batch in dataloader:
        asv1, asv2, cm1, labels, key, asv_label, cm_label = [
            x.to(device) if isinstance(x, torch.Tensor) else x for x in batch
        ]
        targets = labels.to(device).float()
        optimizer.zero_grad()
        cm_input = torch.cat((asv2, cm1), dim=1)
        predictions, calibrated_asv, calibrated_cm = model(asv1, asv2, cm_input)
        loss1 = loss_fn(predictions, labels.unsqueeze(1).float())
        loss2 = loss_adcf.calculate_a_dcf(predictions, np.array(key), 0.5)
        loss2 = loss2.to(device)

        loss = (loss1 + loss2) / 2
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        l_adcf += loss2.item()
        l_bce += loss1.item()
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
    
    return total_loss / len(dataloader), l_adcf / len(dataloader), l_bce / len(dataloader), sasv_eer, sv_eer, spf_eer, \
        hard_act_far_asv, hard_act_far_cm, hard_act_frr, hard_act_adcf, soft_act_far_asv, soft_act_far_cm, soft_act_frr, \
        soft_act_adcf, hard_min_adcf, hard_min_adcf_threshold

# === Evaluation Function ===
def evaluate(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    all_keys = []

    for batch in dataloader:
        asv1, asv2, cm1, labels, key, asv_label, cm_label = [
            x.to(device) if isinstance(x, torch.Tensor) else x for x in batch
        ]
        targets = labels.to(device).float()
        cm_input = torch.cat((asv2.squeeze(), cm1), dim=1)
        predictions, _, _ = model(asv1, asv2, cm_input)
        predictions = predictions.squeeze()
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

    # Model, Optimizer, Loss Function
    model = SASVModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    adcf_loss = aDCF_loss()

    train_dataset = SASV_Dataset_redimnet_sslaasist(args, partition="trn")
    dev_dataset = SASV_Dataset_redimnet_sslaasist(args, partition="dev")
    eval_dataset = SASV_Dataset_redimnet_sslaasist(args, partition="eval")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    log_file = os.path.join(args.output_dir, "training_log.txt")
    os.makedirs(args.output_dir, exist_ok=True)

    best_adcf = float('inf')
    best_model_state_dict = None

    with open(log_file, "w") as f:
        for epoch in range(args.num_epochs):
            trn_loss, trn_loss_adcf, trn_loss_bce, trn_sasv_eer, trn_sv_eer, trn_spf_eer, \
            trn_hard_act_far_asv, trn_hard_act_far_cm, trn_hard_act_frr, trn_hard_act_adcf, trn_soft_act_far_asv, \
            trn_soft_act_far_cm, trn_soft_act_frr, trn_soft_act_adcf, trn_hard_min_adcf, trn_hard_min_adcf_threshold = train(model, train_loader, optimizer, loss_fn, adcf_loss, device)
            
            dev_sasv_eer, dev_sv_eer, dev_spf_eer, dev_hard_act_far_asv, dev_hard_act_far_cm, \
            dev_hard_act_frr, dev_hard_act_adcf, dev_soft_act_far_asv, dev_soft_act_far_cm, \
            dev_soft_act_frr, dev_soft_act_adcf, dev_hard_min_adcf, dev_hard_min_adcf_threshold, _, _ = evaluate(model, dev_loader, device)

            log_msg = (
                f"Epoch {epoch + 1}/{args.num_epochs}, Train Loss: {trn_loss:.4f}, "
                f"Train Loss BCE: {trn_loss_bce:.4f}, Train Loss a-DCF: {trn_loss_adcf:.4f}, "
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
                f"Dev HardActADCF: {dev_hard_act_adcf[0]:.4f}, Dev SoftActFARASV: {dev_soft_act_far_asv:.4f}, "
                f"Dev SoftActFARCM: {dev_soft_act_far_cm:.4f}, Dev SoftActFRR: {dev_soft_act_frr:.4f}, "
                f"Dev SoftActADCF: {dev_soft_act_adcf[0]:.4f}, Dev HardMinADCF: {dev_hard_min_adcf:.4f}, "
                f"Dev HardMinADCFThreshold: {dev_hard_min_adcf_threshold:.4f}\n\n "
            )
            print(log_msg.strip())
            f.write(log_msg)

            if dev_hard_min_adcf < best_adcf:
                best_adcf = dev_hard_min_adcf
                best_model_state_dict = model.state_dict()
                print(f"New best model found at epoch {epoch + 1} with hard min a-DCF: {best_adcf:.4f}\n")
                torch.save(
                    best_model_state_dict, 
                    os.path.join(args.output_dir, "best_model.pth")
                )

    print(f"Best model saved with hard min a-DCF: {best_adcf:.4f}")

    best_model_path = os.path.join(args.output_dir, "best_model.pth")

    best_model = SASVModel().to(device)
    best_model.load_state_dict(torch.load(best_model_path))

    best_dev_sasv_eer, best_dev_sv_eer, best_dev_spf_eer, best_dev_hard_act_far_asv, best_dev_hard_act_far_cm, \
    best_dev_hard_act_frr, best_dev_hard_act_adcf, best_dev_soft_act_far_asv, best_dev_soft_act_far_cm, \
    best_dev_soft_act_frr, best_dev_soft_act_adcf, best_dev_hard_min_adcf, best_dev_hard_min_adcf_threshold, best_preds, best_keys = evaluate(best_model, dev_loader, device)

    print("\n=== FINAL DEV RESULTS ===")
    print(f"SASV EER: {best_dev_sasv_eer:.4f}")
    print(f"SV EER: {best_dev_sv_eer:.4f}")
    print(f"SPF EER: {best_dev_spf_eer:.4f}")
    print(f"Hard Act FAR ASV: {best_dev_hard_act_far_asv:.4f}")
    print(f"Hard Act FAR CM: {best_dev_hard_act_far_cm:.4f}")
    print(f"Hard Act FRR: {best_dev_hard_act_frr:.4f}")
    print(f"Hard Act a-DCF: {best_dev_hard_act_adcf[0]:.4f}")
    print(f"Soft Act FAR ASV: {best_dev_soft_act_far_asv:.4f}")
    print(f"Soft Act FAR CM: {best_dev_soft_act_far_cm:.4f}")
    print(f"Soft Act FRR: {best_dev_soft_act_frr:.4f}")
    print(f"Soft Act a-DCF: {best_dev_soft_act_adcf[0]:.4f}")
    print(f"Hard Min a-DCF: {best_dev_hard_min_adcf:.4f}")
    print(f"Hard Min a-DCF Threshold: {best_dev_hard_min_adcf_threshold:.4f}")

    best_eval_sasv_eer, best_eval_sv_eer, best_eval_spf_eer, best_eval_hard_act_far_asv, best_eval_hard_act_far_cm, \
    best_eval_hard_act_frr, best_eval_hard_act_adcf, best_eval_soft_act_far_asv, best_eval_soft_act_far_cm, \
    best_eval_soft_act_frr, best_eval_soft_act_adcf, best_eval_hard_min_adcf, best_eval_hard_min_adcf_threshold, best_eval_preds, best_eval_keys = evaluate(best_model, eval_loader, device)

    print("\n=== FINAL EVAL RESULTS ===")
    print(f"SASV EER: {best_eval_sasv_eer:.4f}")
    print(f"SV EER: {best_eval_sv_eer:.4f}")
    print(f"SPF EER: {best_eval_spf_eer:.4f}")
    print(f"Hard Act FAR ASV: {best_eval_hard_act_far_asv:.4f}")
    print(f"Hard Act FAR CM: {best_eval_hard_act_far_cm:.4f}")
    print(f"Hard Act FRR: {best_eval_hard_act_frr:.4f}")
    print(f"Hard Act a-DCF: {best_eval_hard_act_adcf[0]:.4f}")
    print(f"Soft Act FAR ASV: {best_eval_soft_act_far_asv:.4f}")
    print(f"Soft Act FAR CM: {best_eval_soft_act_far_cm:.4f}")
    print(f"Soft Act FRR: {best_eval_soft_act_frr:.4f}")
    print(f"Soft Act a-DCF: {best_eval_soft_act_adcf[0]:.4f}")
    print(f"Hard Min a-DCF: {best_eval_hard_min_adcf:.4f}")
    print(f"Hard Min a-DCF Threshold: {best_eval_hard_min_adcf_threshold:.4f}")

    with open(log_file, "a") as f:
        f.write("\n=== FINAL DEV RESULTS ===\n")
        f.write(f"SASV EER: {best_dev_sasv_eer:.4f}\n")
        f.write(f"SV EER: {best_dev_sv_eer:.4f}\n")
        f.write(f"SPF EER: {best_dev_spf_eer:.4f}\n")
        f.write(f"Hard Act FAR ASV: {best_dev_hard_act_far_asv:.4f}\n")
        f.write(f"Hard Act FAR CM: {best_dev_hard_act_far_cm:.4f}\n")
        f.write(f"Hard Act FRR: {best_dev_hard_act_frr:.4f}\n")
        f.write(f"Hard Act a-DCF: {best_dev_hard_act_adcf[0]:.4f}\n")
        f.write(f"Soft Act FAR ASV: {best_dev_soft_act_far_asv:.4f}\n")
        f.write(f"Soft Act FAR CM: {best_dev_soft_act_far_cm:.4f}\n")
        f.write(f"Soft Act FRR: {best_dev_soft_act_frr:.4f}\n")
        f.write(f"Soft Act a-DCF: {best_dev_soft_act_adcf[0]:.4f}\n")
        f.write(f"Hard Min a-DCF: {best_dev_hard_min_adcf:.4f}\n")
        f.write(f"Hard Min a-DCF Threshold: {best_dev_hard_min_adcf_threshold:.4f}\n")

        f.write("\n=== FINAL EVAL RESULTS ===\n")
        f.write(f"SASV EER: {best_eval_sasv_eer:.4f}\n")
        f.write(f"SV EER: {best_eval_sv_eer:.4f}\n")
        f.write(f"SPF EER: {best_eval_spf_eer:.4f}\n")
        f.write(f"Hard Act FAR ASV: {best_eval_hard_act_far_asv:.4f}\n")
        f.write(f"Hard Act FAR CM: {best_eval_hard_act_far_cm:.4f}\n")
        f.write(f"Hard Act FRR: {best_eval_hard_act_frr:.4f}\n")
        f.write(f"Hard Act a-DCF: {best_eval_hard_act_adcf[0]:.4f}\n")
        f.write(f"Soft Act FAR ASV: {best_eval_soft_act_far_asv:.4f}\n")
        f.write(f"Soft Act FAR CM: {best_eval_soft_act_far_cm:.4f}\n")
        f.write(f"Soft Act FRR: {best_eval_soft_act_frr:.4f}\n")
        f.write(f"Soft Act a-DCF: {best_eval_soft_act_adcf[0]:.4f}\n")
        f.write(f"Hard Min a-DCF: {best_eval_hard_min_adcf:.4f}\n")
        f.write(f"Hard Min a-DCF Threshold: {best_eval_hard_min_adcf_threshold:.4f}\n")
 
