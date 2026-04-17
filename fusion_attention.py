# =====================================================
# IMPORTS
# =====================================================

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset


# =====================================================
# MSRS DATASET CLASS
# =====================================================

class MSRSDataset(Dataset):

    def __init__(self, root_dir):

        self.ir_dir = os.path.join(root_dir, "ir")
        self.vi_dir = os.path.join(root_dir, "vi")
        self.seg_dir = os.path.join(root_dir, "Segmentation_labels")

        self.files = sorted(os.listdir(self.ir_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        fname = self.files[idx]

        ir = cv2.imread(os.path.join(self.ir_dir, fname), 0)
        vi = cv2.imread(os.path.join(self.vi_dir, fname))
        seg = cv2.imread(os.path.join(self.seg_dir, fname), 0)

        vi = cv2.cvtColor(vi, cv2.COLOR_BGR2RGB)

        # Preprocessing
        ir = cv2.resize(ir, (256,256))
        vi = cv2.resize(vi, (256,256))
        seg = cv2.resize(seg, (256,256))

        ir = ir.astype(np.float32)/255.0
        vi = vi.astype(np.float32)/255.0
        seg = seg.astype(np.float32)/255.0

        ir = torch.tensor(ir).unsqueeze(0)
        vi = torch.tensor(vi).permute(2,0,1)
        seg = torch.tensor(seg).unsqueeze(0)

        return ir, vi, seg


# =====================================================
# MULTI-SCALE CNN
# =====================================================

class MultiScaleCNN(nn.Module):

    def __init__(self, in_channels):

        super().__init__()

        self.conv3 = nn.Conv2d(in_channels,32,3,padding=1)
        self.conv5 = nn.Conv2d(in_channels,32,5,padding=2)
        self.conv7 = nn.Conv2d(in_channels,32,7,padding=3)

        self.relu = nn.ReLU()

    def forward(self,x):

        f1 = self.relu(self.conv3(x))
        f2 = self.relu(self.conv5(x))
        f3 = self.relu(self.conv7(x))

        return torch.cat([f1,f2,f3], dim=1)


# =====================================================
# GENERATOR (DUAL BRANCH)
# =====================================================

class Generator(nn.Module):

    def __init__(self, in_channels):

        super().__init__()

        self.feature = MultiScaleCNN(in_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(96,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,32,3,padding=1),
            nn.ReLU()
        )

    def forward(self,x):

        f = self.feature(x)
        return self.conv(f)


# =====================================================
# STEP 6 — FEATURE-WISE ATTENTION FUSION
# =====================================================

class AttentionFusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.att = nn.Sequential(
            nn.Conv2d(64,64,1),
            nn.Sigmoid()
        )

    def forward(self, f_ir, f_vi, seg):

        fused = torch.cat([f_ir,f_vi], dim=1)

        att = self.att(fused)

        # segmentation guided attention
        att = att * seg

        return fused * att


# =====================================================
# LOAD DATASET + MODELS
# =====================================================

root_dir = "MSRS"

dataset = MSRSDataset(root_dir)

gen_ir = Generator(in_channels=1)
gen_vi = Generator(in_channels=3)

fusion_model = AttentionFusion()


# =====================================================
# TEST PIPELINE (DISPLAY OUTPUT)
# =====================================================

for i in range(5):

    ir, vi, seg = dataset[i]

    ir = ir.unsqueeze(0)
    vi = vi.unsqueeze(0)
    seg = seg.unsqueeze(0)

    # Generator outputs
    f_ir = gen_ir(ir)
    f_vi = gen_vi(vi)

    # Attention fusion
    fused_features = fusion_model(f_ir, f_vi, seg)

    print(f"\n===== SAMPLE {i} =====")
    print("IR feature shape:", f_ir.shape)
    print("VI feature shape:", f_vi.shape)
    print("Fused feature shape:", fused_features.shape)

    # Visualize first channel
    ir_feat = f_ir[0,0].detach().numpy()
    vi_feat = f_vi[0,0].detach().numpy()
    fused_feat = fused_features[0,0].detach().numpy()

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(ir_feat, cmap='gray')
    plt.title("IR Generator Feature")

    plt.subplot(1,3,2)
    plt.imshow(vi_feat, cmap='gray')
    plt.title("VI Generator Feature")

    plt.subplot(1,3,3)
    plt.imshow(fused_feat, cmap='gray')
    plt.title("Attention Fused Feature")

    plt.show()
