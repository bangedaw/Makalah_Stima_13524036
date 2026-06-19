#!/usr/bin/env python3
"""
Visualisasi Makalah: Branch and Bound untuk Mixed-Precision Quantization
=========================================================================

Program ini memvisualisasikan konsep-konsep utama dari makalah tentang
penerapan algoritma Branch and Bound (B&B) dalam optimasi mixed-precision
quantization pada Neural Network dengan batasan kapasitas memori.

Visualisasi yang disediakan:
1. Distribusi parameter LeNet-5
2. Sensitivitas Hessian trace per layer
3. Perbandingan efisiensi: Brute Force vs Branch and Bound
4. Trade-off antara ukuran model dan akurasi
5. Pohon pencarian B&B dengan mekanisme pruning

Penulis: Edward David Rumahorbo
Tanggal: Juni 2026
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Data dari makalah
LENET_LAYERS = ['C1', 'S2', 'C3', 'S4', 'C5', 'F6']
PARAMS = [156, 12, 1516, 32, 48000, 10164]
HESSIAN_SENSITIVITY = [0.15, 0.08, 0.45, 0.12, 0.05, 0.85]  # Normalized
HESSIAN_ORDER = ['F6', 'C3', 'C1', 'S4', 'S2', 'C5']

MEMORY_LIMITS = ['10 MB', '5 MB', '2 MB']
BRUTE_FORCE_NODES = [59049, 59049, 59049]
BB_NODES = [1240, 856, 413]
PRUNING_EFFICIENCY = [97.89, 98.55, 99.30]

CONFIG_SIZE = [15.0, 10.0, 5.0, 2.0]
ACCURACY = [99.1, 98.9, 97.5, 92.4]
LOSS_INCREASE = [0.00, 0.05, 0.21, 1.85]


def plot_parameter_distribution():
    """Visualisasi 1: Distribusi parameter LeNet-5"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    bars = ax1.bar(LENET_LAYERS, PARAMS, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Jumlah Parameter', fontsize=12, fontweight='bold')
    ax1.set_title('Distribusi Parameter LeNet-5', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, param in zip(bars, PARAMS):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Pie chart
    ax2.pie(PARAMS, labels=LENET_LAYERS, autopct='%1.1f%%', colors=colors,
            startangle=90, explode=[0.05]*6, shadow=True)
    ax2.set_title('Proporsi Parameter per Layer', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_hessian_sensitivity():
    """Visualisasi 2: Sensitivitas Hessian trace per layer"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by sensitivity
    sorted_indices = np.argsort(HESSIAN_SENSITIVITY)[::-1]
    sorted_layers = [HESSIAN_ORDER[i] for i in range(len(HESSIAN_ORDER))]
    sorted_sensitivity = [HESSIAN_SENSITIVITY[LENET_LAYERS.index(layer)] for layer in sorted_layers]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(sorted_layers)))
    bars = ax.barh(sorted_layers, sorted_sensitivity, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Sensitivitas Hessian Trace (Normalized)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Layer', fontsize=12, fontweight='bold')
    ax.set_title('Urutan Prioritas Layer Berdasarkan Sensitivitas Hessian', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, val in zip(bars, sorted_sensitivity):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Add annotation
    ax.annotate('Layer paling sensitif\n(diproses pertama di B&B)', 
                xy=(sorted_sensitivity[0], 0), xytext=(0.6, 4.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_pruning_efficiency():
    """Visualisasi 3: Perbandingan efisiensi Brute Force vs B&B"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(MEMORY_LIMITS))
    width = 0.35
    
    # Bar chart comparison
    bars1 = ax1.bar(x - width/2, BRUTE_FORCE_NODES, width, label='Brute Force', 
                    color='#FF6B6B', edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, BB_NODES, width, label='Branch & Bound', 
                    color='#4ECDC4', edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Batasan Memori', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Jumlah Node yang Dieksplorasi', fontsize=12, fontweight='bold')
    ax1.set_title('Perbandingan Node: Brute Force vs Branch & Bound', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(MEMORY_LIMITS)
    ax1.legend(fontsize=11)
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x(), height, f'{int(height):,}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x(), height, f'{int(height):,}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Pruning efficiency
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(PRUNING_EFFICIENCY)))
    bars = ax2.bar(MEMORY_LIMITS, PRUNING_EFFICIENCY, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Batasan Memori', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Efisiensi Pemangkasan (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Efisiensi Pemangkasan Branch & Bound', fontsize=14, fontweight='bold')
    ax2.set_ylim(95, 100)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, eff in zip(bars, PRUNING_EFFICIENCY):
        height = bar.get_height()
        ax2.text(bar.get_x(), height + 0.1, f'{eff:.2f}%', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_accuracy_tradeoff():
    """Visualisasi 4: Trade-off ukuran model vs akurasi"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy vs Size
    ax1.plot(CONFIG_SIZE, ACCURACY, 'o-', color='#4ECDC4', linewidth=2.5, 
             markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    ax1.set_xlabel('Ukuran Model (MB)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Akurasi (% Baseline)', fontsize=12, fontweight='bold')
    ax1.set_title('Trade-off: Ukuran Model vs Akurasi', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 16)
    ax1.set_ylim(90, 100)
    
    # Add value labels
    for x, y in zip(CONFIG_SIZE, ACCURACY):
        ax1.annotate(f'{y}%', xy=(x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    # Loss increase vs Size
    ax2.bar([f'{s} MB' for s in CONFIG_SIZE], LOSS_INCREASE, 
            color=['#2ECC71', '#F39C12', '#E67E22', '#E74C3C'], 
            edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Konfigurasi (Ukuran)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Estimasi Kenaikan Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Dampak Kompresi terhadap Loss', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, loss) in enumerate(zip(ax2.patches, LOSS_INCREASE)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.05, 
                f'+{loss:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_bb_tree():
    """Visualisasi 5: Pohon pencarian B&B dengan pruning"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    ax.set_title('Pohon Pencarian Branch and Bound dengan Mekanisme Pruning', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Node styling
    def draw_node(x, y, text, color='#4ECDC4', pruned=False):
        if pruned:
            color = '#FF6B6B'
            alpha = 0.4
        else:
            alpha = 1.0
        
        bbox = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=color, edgecolor='black', 
                              linewidth=2, alpha=alpha)
        ax.add_patch(bbox)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
        
        if pruned:
            ax.plot([x-0.6, x+0.6], [y-0.3, y+0.3], 'r-', linewidth=3, alpha=0.7)
            ax.plot([x-0.6, x+0.6], [y+0.3, y-0.3], 'r-', linewidth=3, alpha=0.7)
    
    def draw_arrow(x1, y1, x2, y2, pruned=False):
        color = '#FF6B6B' if pruned else '#2C3E50'
        style = 'dashed' if pruned else 'solid'
        ax.annotate('', xy=(x2, y2+0.4), xycoords='data',
                   xytext=(x1, y1-0.4), textcoords='data',
                   arrowprops=dict(arrowstyle='->', color=color, lw=2, linestyle=style))
    
    # Root
    draw_node(8, 11, 'Root\n(Memori: 0)')
    
    # Level 1: F6 (most sensitive)
    draw_node(4, 9, 'F6: 8-bit\nMem: 81K')
    draw_node(8, 9, 'F6: 4-bit\nMem: 40K')
    draw_node(12, 9, 'F6: 2-bit\nMem: 20K', pruned=True)
    
    draw_arrow(8, 11, 4, 9)
    draw_arrow(8, 11, 8, 9)
    draw_arrow(8, 11, 12, 9, pruned=True)
    ax.text(13.5, 9.5, 'PRUNED\n(LB > UB)', fontsize=8, color='red', fontweight='bold')
    
    # Level 2: C3
    draw_node(2, 7, 'C3: 8-bit\nMem: 93K')
    draw_node(4, 7, 'C3: 4-bit\nMem: 87K')
    draw_node(6, 7, 'C3: 2-bit\nMem: 84K')
    
    draw_node(7, 7, 'C3: 8-bit\nMem: 52K')
    draw_node(9, 7, 'C3: 4-bit\nMem: 46K')
    draw_node(11, 7, 'C3: 2-bit\nMem: 43K')
    
    for i, x in enumerate([2, 4, 6]):
        draw_arrow(4, 9, x, 7)
    for i, x in enumerate([7, 9, 11]):
        draw_arrow(8, 9, x, 7)
    
    # Level 3: Show some pruned nodes
    draw_node(1.5, 5, 'C1: 8-bit\nMem: 105K', pruned=True)
    draw_node(3.5, 5, 'C1: 4-bit\nMem: 99K')
    draw_node(5.5, 5, 'C1: 2-bit\nMem: 96K')
    
    draw_arrow(2, 7, 1.5, 5, pruned=True)
    draw_arrow(4, 7, 3.5, 5)
    draw_arrow(6, 7, 5.5, 5)
    
    ax.text(0.5, 4.5, 'PRUNED\n(Mem > Max)', fontsize=8, color='red', fontweight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#4ECDC4', edgecolor='black', label='Node Aktif'),
        mpatches.Patch(facecolor='#FF6B6B', edgecolor='black', alpha=0.4, label='Node Pruned'),
        plt.Line2D([0], [0], color='black', lw=2, label='Branch Aktif'),
        plt.Line2D([0], [0], color='red', lw=2, linestyle='dashed', label='Branch Pruned')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)
    
    # Annotations
    ax.text(0.5, 1.5, 'Keterangan:\n'
            '• F6 memiliki sensitivitas Hessian tertinggi\n'
            '• Diproses pertama di level atas pohon\n'
            '• Pruning terjadi jika:\n'
            '  - Memori melebihi batas (M_max)\n'
            '  - Lower Bound > Upper Bound', 
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    return fig


def main():
    """Fungsi utama untuk menampilkan semua visualisasi"""
    print("=" * 80)
    print("VISUALISASI MAKALAH: Branch and Bound untuk Mixed-Precision Quantization")
    print("=" * 80)
    print("\nMembuat visualisasi...")
    
    # Create all plots
    fig1 = plot_parameter_distribution()
    fig2 = plot_hessian_sensitivity()
    fig3 = plot_pruning_efficiency()
    fig4 = plot_accuracy_tradeoff()
    fig5 = plot_bb_tree()
    
    # Save all figures
    fig1.savefig('01_parameter_distribution.png', dpi=150, bbox_inches='tight')
    fig2.savefig('02_hessian_sensitivity.png', dpi=150, bbox_inches='tight')
    fig3.savefig('03_pruning_efficiency.png', dpi=150, bbox_inches='tight')
    fig4.savefig('04_accuracy_tradeoff.png', dpi=150, bbox_inches='tight')
    fig5.savefig('05_bb_tree.png', dpi=150, bbox_inches='tight')
    
    print("✓ 01_parameter_distribution.png - Distribusi parameter LeNet-5")
    print("✓ 02_hessian_sensitivity.png - Sensitivitas Hessian trace per layer")
    print("✓ 03_pruning_efficiency.png - Perbandingan efisiensi Brute Force vs B&B")
    print("✓ 04_accuracy_tradeoff.png - Trade-off ukuran model vs akurasi")
    print("✓ 05_bb_tree.png - Pohon pencarian B&B dengan pruning")
    
    print("\n" + "=" * 80)
    print("Semua visualisasi berhasil disimpan!")
    print("=" * 80)
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()
