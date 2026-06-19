#!/usr/bin/env python3
"""
Visualisasi GUI Makalah: Branch and Bound untuk Mixed-Precision Quantization
=============================================================================

Versi GUI dari program visualisasi makalah. Menggunakan tkinter untuk
membuat antarmuka interaktif dengan tab untuk setiap visualisasi.

Fitur:
- Tab interaktif untuk setiap jenis visualisasi
- Navigasi mudah antara grafik
- Tooltip dan informasi tambahan

Penulis: Edward David Rumahorbo
Tanggal: Juni 2026
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import numpy as np

# Data dari makalah
LENET_LAYERS = ['C1', 'S2', 'C3', 'S4', 'C5', 'F6']
PARAMS = [156, 12, 1516, 32, 48000, 10164]
HESSIAN_SENSITIVITY = [0.15, 0.08, 0.45, 0.12, 0.05, 0.85]
HESSIAN_ORDER = ['F6', 'C3', 'C1', 'S4', 'S2', 'C5']

MEMORY_LIMITS = ['10 MB', '5 MB', '2 MB']
BRUTE_FORCE_NODES = [59049, 59049, 59049]
BB_NODES = [1240, 856, 413]
PRUNING_EFFICIENCY = [97.89, 98.55, 99.30]

CONFIG_SIZE = [15.0, 10.0, 5.0, 2.0]
ACCURACY = [99.1, 98.9, 97.5, 92.4]
LOSS_INCREASE = [0.00, 0.05, 0.21, 1.85]


class PaperVisualizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualisasi Makalah: Branch and Bound untuk Mixed-Precision Quantization")
        self.root.geometry("1200x800")
        
        self.create_widgets()
    
    def create_widgets(self):
        # Header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        
        title_label = ttk.Label(header_frame, 
                                text="Visualisasi Makalah: Branch and Bound untuk Mixed-Precision Quantization",
                                font=('Arial', 14, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame,
                                   text="Edward David Rumahorbo | Teknik Informatika ITB | 2026",
                                   font=('Arial', 10))
        subtitle_label.pack()
        
        # Notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tab 1: Parameter Distribution
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="  Distribusi Parameter  ")
        self.create_param_tab()
        
        # Tab 2: Hessian Sensitivity
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="  Sensitivitas Hessian  ")
        self.create_hessian_tab()
        
        # Tab 3: Pruning Efficiency
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="  Efisiensi Pruning  ")
        self.create_pruning_tab()
        
        # Tab 4: Accuracy Trade-off
        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text="  Trade-off Akurasi  ")
        self.create_accuracy_tab()
        
        # Tab 5: B&B Tree
        self.tab5 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab5, text="  Pohon B&B  ")
        self.create_tree_tab()
        
        # Info panel
        info_frame = ttk.Frame(self.root)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        info_text = ("Konten: Distribusi parameter LeNet-5 | Sensitivitas Hessian per layer | "
                    "Perbandingan BF vs B&B | Trade-off akurasi | Pohon pencarian B&B")
        info_label = ttk.Label(info_frame, text=info_text, font=('Arial', 9), foreground='gray')
        info_label.pack()
    
    def create_param_tab(self):
        fig = Figure(figsize=(12, 5), dpi=100)
        
        ax1 = fig.add_subplot(121)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        bars = ax1.bar(LENET_LAYERS, PARAMS, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Layer', fontweight='bold')
        ax1.set_ylabel('Jumlah Parameter', fontweight='bold')
        ax1.set_title('Distribusi Parameter LeNet-5', fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        for bar, param in zip(bars, PARAMS):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{param:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2 = fig.add_subplot(122)
        ax2.pie(PARAMS, labels=LENET_LAYERS, autopct='%1.1f%%', colors=colors,
                startangle=90, explode=[0.05]*6, shadow=True)
        ax2.set_title('Proporsi Parameter per Layer', fontweight='bold')
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tab1)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, self.tab1)
        toolbar.update()
    
    def create_hessian_tab(self):
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        sorted_layers = HESSIAN_ORDER
        sorted_sensitivity = [HESSIAN_SENSITIVITY[LENET_LAYERS.index(layer)] for layer in sorted_layers]
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(sorted_layers)))
        bars = ax.barh(sorted_layers, sorted_sensitivity, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Sensitivitas Hessian Trace (Normalized)', fontweight='bold')
        ax.set_ylabel('Layer', fontweight='bold')
        ax.set_title('Urutan Prioritas Layer Berdasarkan Sensitivitas Hessian', fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        for bar, val in zip(bars, sorted_sensitivity):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.annotate('Layer paling sensitif\n(diproses pertama di B&B)', 
                    xy=(sorted_sensitivity[0], 0), xytext=(0.6, 4.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, color='red', fontweight='bold')
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tab2)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, self.tab2)
        toolbar.update()
    
    def create_pruning_tab(self):
        fig = Figure(figsize=(12, 5), dpi=100)
        
        ax1 = fig.add_subplot(121)
        x = np.arange(len(MEMORY_LIMITS))
        width = 0.35
        bars1 = ax1.bar(x - width/2, BRUTE_FORCE_NODES, width, label='Brute Force', 
                        color='#FF6B6B', edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, BB_NODES, width, label='Branch & Bound', 
                        color='#4ECDC4', edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Batasan Memori', fontweight='bold')
        ax1.set_ylabel('Jumlah Node', fontweight='bold')
        ax1.set_title('Brute Force vs Branch & Bound', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(MEMORY_LIMITS)
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        for bar in bars1:
            ax1.text(bar.get_x(), bar.get_height(), f'{int(bar.get_height()):,}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        for bar in bars2:
            ax1.text(bar.get_x(), bar.get_height(), f'{int(bar.get_height()):,}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax2 = fig.add_subplot(122)
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(PRUNING_EFFICIENCY)))
        bars = ax2.bar(MEMORY_LIMITS, PRUNING_EFFICIENCY, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Batasan Memori', fontweight='bold')
        ax2.set_ylabel('Efisiensi Pemangkasan (%)', fontweight='bold')
        ax2.set_title('Efisiensi Pemangkasan B&B', fontweight='bold')
        ax2.set_ylim(95, 100)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        for bar, eff in zip(bars, PRUNING_EFFICIENCY):
            ax2.text(bar.get_x(), bar.get_height() + 0.1, f'{eff:.2f}%', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tab3)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, self.tab3)
        toolbar.update()
    
    def create_accuracy_tab(self):
        fig = Figure(figsize=(12, 5), dpi=100)
        
        ax1 = fig.add_subplot(121)
        ax1.plot(CONFIG_SIZE, ACCURACY, 'o-', color='#4ECDC4', linewidth=2.5, 
                 markersize=10, markeredgecolor='black', markeredgewidth=1.5)
        ax1.set_xlabel('Ukuran Model (MB)', fontweight='bold')
        ax1.set_ylabel('Akurasi (% Baseline)', fontweight='bold')
        ax1.set_title('Ukuran Model vs Akurasi', fontweight='bold')
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.set_xlim(0, 16)
        ax1.set_ylim(90, 100)
        for x, y in zip(CONFIG_SIZE, ACCURACY):
            ax1.annotate(f'{y}%', xy=(x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        ax2 = fig.add_subplot(122)
        ax2.bar([f'{s} MB' for s in CONFIG_SIZE], LOSS_INCREASE, 
                color=['#2ECC71', '#F39C12', '#E67E22', '#E74C3C'], 
                edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Konfigurasi (Ukuran)', fontweight='bold')
        ax2.set_ylabel('Estimasi Kenaikan Loss', fontweight='bold')
        ax2.set_title('Dampak Kompresi terhadap Loss', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        for bar, loss in zip(ax2.patches, LOSS_INCREASE):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'+{loss:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tab4)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, self.tab4)
        toolbar.update()
    
    def create_tree_tab(self):
        fig = Figure(figsize=(14, 8), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.axis('off')
        ax.set_title('Pohon Pencarian Branch and Bound dengan Mekanisme Pruning', 
                     fontsize=14, fontweight='bold', pad=20)
        
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
            ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
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
        draw_node(8, 11, 'Root\n(Mem: 0)')
        
        # Level 1: F6
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
        
        for x in [2, 4, 6]:
            draw_arrow(4, 9, x, 7)
        for x in [7, 9, 11]:
            draw_arrow(8, 9, x, 7)
        
        # Level 3
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
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)
        
        ax.text(0.5, 1.5, 'Keterangan:\n'
                'F6: sensitivitas tertinggi, diproses pertama\n'
                'Pruning: Memori > M_max atau LB > UB', 
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top')
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tab5)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, self.tab5)
        toolbar.update()


def main():
    root = tk.Tk()
    app = PaperVisualizationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
