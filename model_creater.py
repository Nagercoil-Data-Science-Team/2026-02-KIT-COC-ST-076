# =========================================================
# IMPORTS
# =========================================================

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# DATASET
# =========================================================

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

        ir = cv2.resize(ir, (256, 256)) / 255.0
        vi = cv2.resize(vi, (256, 256)) / 255.0
        seg = cv2.resize(seg, (256, 256)) / 255.0

        ir = torch.tensor(ir).unsqueeze(0).float()
        vi = torch.tensor(vi).permute(2, 0, 1).float()
        seg = torch.tensor(seg).unsqueeze(0).float()

        return ir, vi, seg


# =========================================================
# MODEL BLOCKS
# =========================================================

class MultiScaleCNN(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.conv3 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, 32, 5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, 32, 7, padding=3)

    def forward(self, x):
        f1 = self.conv3(x)
        f2 = self.conv5(x)
        f3 = self.conv7(x)

        return torch.cat([f1, f2, f3], dim=1)


class Generator(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.feature = MultiScaleCNN(in_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(96, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        f = self.feature(x)
        return self.conv(f)


class AttentionFusion(nn.Module):

    def __init__(self):
        super().__init__()

        self.att = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.Sigmoid()
        )

    def forward(self, f_ir, f_vi, seg):
        fused = torch.cat([f_ir, f_vi], dim=1)

        att = self.att(fused)

        seg = seg.repeat(1, fused.shape[1], 1, 1)

        att = att * seg

        output = fused * att + fused

        return output


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, 4, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# =========================================================
# METRICS CALCULATION FUNCTIONS
# =========================================================

def calculate_entropy(img):
    """Calculate entropy of an image"""
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 1])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def calculate_mutual_information(img1, img2, fused):
    """Calculate mutual information"""
    h1 = calculate_entropy(img1)
    h2 = calculate_entropy(img2)
    hf = calculate_entropy(fused)
    return h1 + h2 - hf


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    return ssim(img1, img2, data_range=1.0)


def calculate_psnr(img1, img2):
    """Calculate PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))


def calculate_spatial_frequency(img):
    """Calculate spatial frequency"""
    RF = np.sqrt(np.mean(np.diff(img, axis=0) ** 2))
    CF = np.sqrt(np.mean(np.diff(img, axis=1) ** 2))
    return np.sqrt(RF ** 2 + CF ** 2)


def calculate_std(img):
    """Calculate standard deviation"""
    return np.std(img)


def calculate_all_metrics(ir_img, vi_img, fused_img):
    """Calculate all metrics for fusion evaluation"""

    metrics = {}

    # Entropy
    metrics['EN'] = calculate_entropy(fused_img)

    # Mutual Information
    metrics['MI'] = calculate_mutual_information(ir_img, vi_img, fused_img)

    # SSIM (average with both source images)
    metrics['SSIM'] = (calculate_ssim(fused_img, ir_img) +
                       calculate_ssim(fused_img, vi_img)) / 2

    # PSNR (average with both source images)
    metrics['PSNR'] = (calculate_psnr(fused_img, ir_img) +
                       calculate_psnr(fused_img, vi_img)) / 2

    # Spatial Frequency
    metrics['SF'] = calculate_spatial_frequency(fused_img)

    # Standard Deviation
    metrics['SD'] = calculate_std(fused_img)

    return metrics


# =========================================================
# INITIALIZE
# =========================================================

dataset = MSRSDataset("MSRS")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

gen_ir = Generator(1).to(device)
gen_vi = Generator(3).to(device)
fusion = AttentionFusion().to(device)
decoder = Decoder().to(device)
disc = Discriminator().to(device)

L1_loss = nn.L1Loss()
BCE = nn.BCELoss()

opt_G = torch.optim.Adam(
    list(gen_ir.parameters()) +
    list(gen_vi.parameters()) +
    list(fusion.parameters()) +
    list(decoder.parameters()), lr=1e-4)

opt_D = torch.optim.Adam(disc.parameters(), lr=1e-4)

# =========================================================
# TRAINING LOOP WITH LOSS TRACKING
# =========================================================

epochs = 5

# Loss tracking
g_losses = []
d_losses = []
recon_losses = []
adv_losses = []

for epoch in range(epochs):

    epoch_g_loss = 0
    epoch_d_loss = 0
    epoch_recon_loss = 0
    epoch_adv_loss = 0

    for ir, vi, seg in dataloader:
        ir, vi, seg = ir.to(device), vi.to(device), seg.to(device)

        # Generator forward
        f_ir = gen_ir(ir)
        f_vi = gen_vi(vi)

        fused_feat = fusion(f_ir, f_vi, seg)
        fused_img = decoder(fused_feat)

        # =====================
        # Train Discriminator
        # =====================

        real_out = disc(ir)
        fake_out = disc(fused_img.detach())

        real_label = torch.ones_like(real_out)
        fake_label = torch.zeros_like(fake_out)

        loss_real = BCE(real_out, real_label)
        loss_fake = BCE(fake_out, fake_label)

        loss_D = (loss_real + loss_fake) / 2

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # =====================
        # Train Generator
        # =====================

        fake_out = disc(fused_img)

        recon_loss = L1_loss(fused_img, ir) + \
                     L1_loss(fused_img, vi.mean(1, keepdim=True))

        adv_loss = BCE(fake_out, torch.ones_like(fake_out))

        total_loss = recon_loss + 0.01 * adv_loss

        opt_G.zero_grad()
        total_loss.backward()
        opt_G.step()

        # Track losses
        epoch_g_loss += total_loss.item()
        epoch_d_loss += loss_D.item()
        epoch_recon_loss += recon_loss.item()
        epoch_adv_loss += adv_loss.item()

    # Average losses for epoch
    num_batches = len(dataloader)
    g_losses.append(epoch_g_loss / num_batches)
    d_losses.append(epoch_d_loss / num_batches)
    recon_losses.append(epoch_recon_loss / num_batches)
    adv_losses.append(epoch_adv_loss / num_batches)

    print(f"Epoch {epoch + 1}/{epochs} | G Loss {g_losses[-1]:.4f} | D Loss {d_losses[-1]:.4f}")

# =========================================================
# VISUALIZE 5 FUSED IMAGES
# =========================================================

print("\n" + "=" * 60)
print("GENERATING FUSED IMAGES")
print("=" * 60)

num_samples = min(5, len(dataset))

fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
if num_samples == 1:
    axes = axes.reshape(1, -1)

for i in range(num_samples):
    ir, vi, seg = dataset[i]

    ir_input = ir.unsqueeze(0).to(device)
    vi_input = vi.unsqueeze(0).to(device)
    seg_input = seg.unsqueeze(0).to(device)

    with torch.no_grad():
        fused_img = decoder(fusion(gen_ir(ir_input), gen_vi(vi_input), seg_input))

    # Convert to numpy for display
    ir_np = ir[0].cpu().numpy()
    vi_np = vi.cpu().numpy().transpose(1, 2, 0)
    seg_np = seg[0].cpu().numpy()
    fused_np = fused_img[0, 0].cpu().numpy()

    # Display images
    axes[i, 0].imshow(ir_np, cmap='gray')
    axes[i, 0].set_title(f'IR Image {i + 1}')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(vi_np)
    axes[i, 1].set_title(f'Visible Image {i + 1}')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(seg_np, cmap='gray')
    axes[i, 2].set_title(f'Segmentation {i + 1}')
    axes[i, 2].axis('off')

    axes[i, 3].imshow(fused_np, cmap='gray')
    axes[i, 3].set_title(f'Fused Image {i + 1}')
    axes[i, 3].axis('off')

plt.tight_layout()
plt.savefig('fused_images.png', dpi=150, bbox_inches='tight')
plt.show()

# =========================================================
# CALCULATE METRICS FOR ALL 5 IMAGES
# =========================================================

print("\n" + "=" * 60)
print("CALCULATING METRICS")
print("=" * 60)

all_metrics = {
    'EN': [],
    'MI': [],
    'SSIM': [],
    'PSNR': [],
    'SF': [],
    'SD': []
}

for i in range(num_samples):
    ir, vi, seg = dataset[i]

    ir_input = ir.unsqueeze(0).to(device)
    vi_input = vi.unsqueeze(0).to(device)
    seg_input = seg.unsqueeze(0).to(device)

    with torch.no_grad():
        fused_img = decoder(fusion(gen_ir(ir_input), gen_vi(vi_input), seg_input))

    # Convert to numpy
    ir_np = ir[0].cpu().numpy()
    vi_np = vi.mean(0).cpu().numpy()  # Convert RGB to grayscale
    fused_np = fused_img[0, 0].cpu().numpy()

    # Calculate metrics
    metrics = calculate_all_metrics(ir_np, vi_np, fused_np)

    for key in all_metrics.keys():
        all_metrics[key].append(metrics[key])

    print(f"\nImage {i + 1} Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

# =========================================================
# PLOT METRICS AS WAVE PLOTS
# =========================================================

print("\n" + "=" * 60)
print("GENERATING METRIC WAVE PLOTS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Image Fusion Quality Metrics', fontsize=16, fontweight='bold')

metrics_info = [
    ('EN', 'Entropy', 'blue', 'Information Content'),
    ('MI', 'Mutual Information', 'green', 'Information Preservation'),
    ('SSIM', 'Structural Similarity', 'red', 'Structural Quality'),
    ('PSNR', 'Peak Signal-to-Noise Ratio', 'purple', 'Signal Quality (dB)'),
    ('SF', 'Spatial Frequency', 'orange', 'Edge Preservation'),
    ('SD', 'Standard Deviation', 'brown', 'Contrast')
]

for idx, (metric_key, metric_name, color, ylabel) in enumerate(metrics_info):
    row = idx // 3
    col = idx % 3

    ax = axes[row, col]

    x = np.arange(1, num_samples + 1)
    y = all_metrics[metric_key]

    # Create wave plot with markers
    ax.plot(x, y, marker='o', linewidth=2.5, markersize=8,
            color=color, label=metric_key, linestyle='-', markerfacecolor='white',
            markeredgewidth=2, markeredgecolor=color)

    # Fill area under curve
    ax.fill_between(x, y, alpha=0.3, color=color)

    # Add value labels on points
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.annotate(f'{yi:.3f}', (xi, yi), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

    # Styling
    ax.set_xlabel('Image Number', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(f'{metric_key}: {metric_name}', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(x)

    # Add average line
    avg_val = np.mean(y)
    ax.axhline(y=avg_val, color='red', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'Avg: {avg_val:.3f}')
    ax.legend(loc='best', fontsize=9)

    # Set nice limits
    y_margin = (max(y) - min(y)) * 0.1
    ax.set_ylim([min(y) - y_margin, max(y) + y_margin])

plt.tight_layout()
plt.savefig('metrics_wave_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# =========================================================
# PLOT TRAINING LOSSES
# =========================================================

print("\n" + "=" * 60)
print("GENERATING LOSS PLOTS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Training Loss Analysis', fontsize=16, fontweight='bold')

epochs_range = np.arange(1, epochs + 1)

# Generator Loss
ax1 = axes[0, 0]
ax1.plot(epochs_range, g_losses, marker='o', linewidth=2.5, markersize=8,
         color='blue', markerfacecolor='white', markeredgewidth=2)
ax1.fill_between(epochs_range, g_losses, alpha=0.3, color='blue')
ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax1.set_title('Generator Loss', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
for i, (x, y) in enumerate(zip(epochs_range, g_losses)):
    ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=9)

# Discriminator Loss
ax2 = axes[0, 1]
ax2.plot(epochs_range, d_losses, marker='s', linewidth=2.5, markersize=8,
         color='red', markerfacecolor='white', markeredgewidth=2)
ax2.fill_between(epochs_range, d_losses, alpha=0.3, color='red')
ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax2.set_title('Discriminator Loss', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
for i, (x, y) in enumerate(zip(epochs_range, d_losses)):
    ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=9)

# Reconstruction Loss
ax3 = axes[1, 0]
ax3.plot(epochs_range, recon_losses, marker='^', linewidth=2.5, markersize=8,
         color='green', markerfacecolor='white', markeredgewidth=2)
ax3.fill_between(epochs_range, recon_losses, alpha=0.3, color='green')
ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax3.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax3.set_title('Reconstruction Loss', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
for i, (x, y) in enumerate(zip(epochs_range, recon_losses)):
    ax3.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=9)

# Adversarial Loss
ax4 = axes[1, 1]
ax4.plot(epochs_range, adv_losses, marker='D', linewidth=2.5, markersize=8,
         color='purple', markerfacecolor='white', markeredgewidth=2)
ax4.fill_between(epochs_range, adv_losses, alpha=0.3, color='purple')
ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax4.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax4.set_title('Adversarial Loss', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle='--')
for i, (x, y) in enumerate(zip(epochs_range, adv_losses)):
    ax4.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('training_losses.png', dpi=150, bbox_inches='tight')
plt.show()

# =========================================================
# ADDITIONAL ANALYSIS PLOTS
# =========================================================

print("\n" + "=" * 60)
print("GENERATING ADDITIONAL ANALYSIS PLOTS")
print("=" * 60)

# Plot 1: Combined G and D Loss Comparison
fig1, ax = plt.subplots(figsize=(12, 6))
ax.plot(epochs_range, g_losses, marker='o', linewidth=2.5, markersize=8,
        color='blue', label='Generator Loss', markerfacecolor='white', markeredgewidth=2)
ax.plot(epochs_range, d_losses, marker='s', linewidth=2.5, markersize=8,
        color='red', label='Discriminator Loss', markerfacecolor='white', markeredgewidth=2)
ax.fill_between(epochs_range, g_losses, alpha=0.2, color='blue')
ax.fill_between(epochs_range, d_losses, alpha=0.2, color='red')
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.set_title('Generator vs Discriminator Loss', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('g_vs_d_loss.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot 2: Metrics Radar Chart (for last image)
fig2, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Normalize metrics to 0-1 range for radar chart
metrics_names = list(all_metrics.keys())
last_image_metrics = [all_metrics[key][-1] for key in metrics_names]

# Normalize
max_vals = [max(all_metrics[key]) for key in metrics_names]
normalized_metrics = [val / max_val if max_val > 0 else 0
                      for val, max_val in zip(last_image_metrics, max_vals)]

angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
normalized_metrics += normalized_metrics[:1]
angles += angles[:1]

ax.plot(angles, normalized_metrics, 'o-', linewidth=2.5, color='darkblue', markersize=10)
ax.fill(angles, normalized_metrics, alpha=0.25, color='skyblue')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_names, fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_title(f'Fusion Quality Radar (Image {num_samples})',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True)
plt.tight_layout()
plt.savefig('metrics_radar.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot 3: Metrics Heatmap
fig3, ax = plt.subplots(figsize=(12, 8))

# Create heatmap data
heatmap_data = np.array([all_metrics[key] for key in metrics_names])

im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

# Set ticks and labels
ax.set_xticks(np.arange(num_samples))
ax.set_yticks(np.arange(len(metrics_names)))
ax.set_xticklabels([f'Image {i + 1}' for i in range(num_samples)], fontsize=11)
ax.set_yticklabels(metrics_names, fontsize=11, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Metric Value', fontsize=11, fontweight='bold')

# Add text annotations
for i in range(len(metrics_names)):
    for j in range(num_samples):
        text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=10, fontweight='bold')

ax.set_title('Fusion Metrics Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('metrics_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# =========================================================
# SUMMARY STATISTICS
# =========================================================

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

for metric_key in all_metrics.keys():
    values = all_metrics[metric_key]
    print(f"\n{metric_key}:")
    print(f"  Mean: {np.mean(values):.4f}")
    print(f"  Std:  {np.std(values):.4f}")
    print(f"  Min:  {np.min(values):.4f}")
    print(f"  Max:  {np.max(values):.4f}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE - ALL PLOTS SAVED")
print("=" * 60)