import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_spatial_attention(feats, layer_idx, save_dir):
    layer_dir = os.path.join(save_dir, f'layer_{layer_idx}')
    os.makedirs(layer_dir, exist_ok=True)

    # 统一设置图像大小和DPI
    FIGURE_SIZE = (12, 12)  # 统一图像大小
    DPI = 600  # 提高分辨率

    for batch_idx in range(feats.shape[1]):
        curr_feats = feats[:, batch_idx]

        # 1. Channel attention
        feature_indices = [0, 256, 512, 768]
        fig = plt.figure(figsize=FIGURE_SIZE)  # 统一大小
        for i, feat_idx in enumerate(feature_indices):
            ax = fig.add_subplot(1, 4, i + 1)
            spatial_feat = curr_feats[:, feat_idx].reshape(64, 64)
            ax.imshow(spatial_feat.cpu().numpy(), cmap='viridis')
            ax.axis('off')
        plt.subplots_adjust(wspace=0.1)
        plt.savefig(os.path.join(layer_dir, f'channel_attention_batch{batch_idx}.png'),
                    bbox_inches='tight', dpi=DPI, pad_inches=0)
        plt.close()

        # 2. Spatial attention
        feat_activation = torch.norm(curr_feats, dim=1)
        spatial_activation = feat_activation.reshape(64, 64)

        fig = plt.figure(figsize=FIGURE_SIZE)
        ax = plt.gca()
        im = ax.imshow(spatial_activation.cpu().numpy(), cmap='viridis')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(im, cax=cax)
        ax.axis('off')
        plt.savefig(os.path.join(layer_dir, f'spatial_attention_batch{batch_idx}.png'),
                    bbox_inches='tight', dpi=DPI, pad_inches=0)
        plt.close()

        # 3. Attention flow
        fig = plt.figure(figsize=FIGURE_SIZE)
        ax = plt.gca()
        attention_flow = torch.mean(curr_feats, dim=1).reshape(64, 64)
        X, Y = np.meshgrid(np.arange(64), np.arange(64))
        U = attention_flow.cpu().numpy()
        V = attention_flow.cpu().numpy().T

        magnitude = np.sqrt(U ** 2 + V ** 2)
        strm = ax.streamplot(X, Y, U, V,
                             color=magnitude,
                             cmap='coolwarm',
                             density=1.5,
                             linewidth=1.5,
                             arrowsize=1.2)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(strm.lines, cax=cax)
        ax.axis('off')

        # 设置图像范围以确保一致性
        ax.set_xlim(0, 63)
        ax.set_ylim(0, 63)

        plt.savefig(os.path.join(layer_dir, f'attention_flow_batch{batch_idx}.png'),
                    bbox_inches='tight', dpi=DPI, pad_inches=0)
        plt.close()