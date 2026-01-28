#!/usr/bin/env python3
"""
Publication-Quality Benchmark Visualization for MAD-Lab
========================================================
Designed for NeurIPS-style scientific papers with modern aesthetics.

Features:
- Clean, minimalist design following NeurIPS guidelines
- Colorblind-friendly palettes
- Multi-panel comparison plots
- Heatmaps, radar charts, and scaling analysis
- LaTeX-ready figure exports

Author: MAD-Lab Team
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Optional imports for enhanced visualizations
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ============================================================================
# NEURIPS-STYLE CONFIGURATION
# ============================================================================

# NeurIPS paper dimensions (in inches)
NEURIPS_TEXTWIDTH = 5.5
NEURIPS_FULL_WIDTH = 6.75
NEURIPS_COLUMN_WIDTH = 3.25

# Color palettes - colorblind friendly (based on Paul Tol's palette)
COLORS = {
    'primary': {
        'blue': '#4477AA',
        'cyan': '#66CCEE', 
        'green': '#228833',
        'yellow': '#CCBB44',
        'red': '#EE6677',
        'purple': '#AA3377',
        'grey': '#BBBBBB'
    },
    'model': {
        'BSGDN': '#4477AA',    # Blue - BS-Gated Delta Net
        'GDN': '#228833',       # Green - Gated Delta Net
        'H': '#EE6677',         # Red - Hyena
        'mA': '#CCBB44',        # Yellow - Multi-head Attention
        'Mb': '#AA3377',        # Purple - Mamba
        'R5t': '#66CCEE',       # Cyan - RWKV5
        'R6t': '#CC6699',       # Pink - RWKV6
    },
    'task': {
        'CR': '#4477AA',        # In-Context Recall
        'NR': '#228833',        # Noisy In-Context Recall
        'FR': '#EE6677',        # Fuzzy In-Context Recall
        'M': '#CCBB44',         # Memorization
        'SC': '#AA3377',        # Selective Copying
    }
}

# Model display names
MODEL_NAMES = {
    'BSGDN': 'BS-GDN',
    'GDN': 'Gated-DeltaNet',
    'H': 'Hyena',
    'mA': 'MH-Attention',
    'Mb': 'Mamba',
    'R5t': 'RWKV-5',
    'R6t': 'RWKV-6',
    'Sg': 'SwiGLU',
}

# Task display names
TASK_NAMES = {
    'CR': 'In-Context Recall',
    'NR': 'Noisy Recall',
    'FR': 'Fuzzy Recall',
    'M': 'Memorization',
    'SC': 'Selective Copy',
}


def setup_neurips_style():
    """Configure matplotlib for NeurIPS-quality figures."""
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        
        # Figure settings
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        
        # Axes settings
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        
        # Grid settings
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
        
        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        
        # LaTeX rendering (if available)
        'text.usetex': False,  # Set to True if LaTeX is installed
        
        # Tight layout
        'figure.constrained_layout.use': True,
    })
    
    if HAS_SEABORN:
        sns.set_palette([COLORS['primary'][c] for c in ['blue', 'green', 'red', 'yellow', 'purple', 'cyan']])


# ============================================================================
# DATA LOADING & PARSING
# ============================================================================

def parse_experiment_path(path: str) -> Dict:
    """Parse experiment directory name into configuration dict."""
    path_name = Path(path).name if os.path.isdir(path) else Path(path).parent.name
    
    config = {}
    parts = path_name.split('_')
    
    for part in parts:
        if '-' in part:
            key, *values = part.split('-')
            value = '-'.join(values)
            
            # Convert numeric values
            if '#' in value:
                value = float(value.replace('#', '.'))
            elif value.isdigit():
                value = int(value)
            
            config[key] = value
    
    return config


def extract_model_name(config: Dict) -> str:
    """Extract model architecture name from config."""
    model = config.get('model', '')
    # Get primary sequence mixer (first component)
    parts = model.split('-')
    if parts:
        # Filter out channel mixers (Sg, M)
        seq_mixers = [p for p in parts if p not in ['Sg', 'M']]
        if seq_mixers:
            return seq_mixers[0]
    return model


def load_benchmark_results(log_dir: str) -> pd.DataFrame:
    """Load all benchmark results from directory into DataFrame."""
    results = []
    
    for exp_dir in glob.glob(os.path.join(log_dir, 't-*')):
        results_file = os.path.join(exp_dir, 'results.csv')
        if os.path.exists(results_file):
            try:
                df = pd.read_csv(results_file)
                if len(df) > 0:
                    config = parse_experiment_path(exp_dir)
                    for col in df.columns:
                        config[col] = df[col].iloc[0]
                    config['model_name'] = extract_model_name(config)
                    results.append(config)
            except Exception as e:
                print(f"Warning: Could not load {results_file}: {e}")
    
    return pd.DataFrame(results)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_task_comparison_bars(df: pd.DataFrame, 
                               metric: str = 'test_acc',
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a grouped bar chart comparing models across tasks.
    
    Style: Clean bars with subtle shadows and task grouping.
    """
    fig, ax = plt.subplots(figsize=(NEURIPS_FULL_WIDTH, 3.5))
    
    # Get unique tasks and models
    tasks = sorted(df['t'].unique())
    models = sorted(df['model_name'].unique())
    
    # Filter to best performing hyperparams for each task-model combo
    best_results = df.loc[df.groupby(['t', 'model_name'])[metric].idxmax()]
    
    x = np.arange(len(tasks))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        model_data = best_results[best_results['model_name'] == model]
        values = []
        for task in tasks:
            task_data = model_data[model_data['t'] == task]
            if len(task_data) > 0:
                values.append(task_data[metric].iloc[0])
            else:
                values.append(0)
        
        color = COLORS['model'].get(model, COLORS['primary']['grey'])
        label = MODEL_NAMES.get(model, model)
        
        bars = ax.bar(x + i * width - (len(models) - 1) * width / 2, 
                     values, width * 0.9, label=label, color=color,
                     edgecolor='white', linewidth=0.5)
        
        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            if val > 0.05:
                ax.annotate(f'{val:.2f}', 
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=6, rotation=90)
    
    ax.set_xlabel('Task')
    ax.set_ylabel(f'{"Accuracy" if "acc" in metric else metric.replace("_", " ").title()}')
    ax.set_title('Model Performance Across Synthetic Tasks', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_NAMES.get(t, t) for t in tasks], rotation=15, ha='right')
    ax.legend(loc='upper right', ncol=2)
    ax.set_ylim(0, 1.1)
    
    # Add horizontal reference lines
    ax.axhline(y=1.0, color='#228833', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0.5, color='#BBBBBB', linestyle=':', alpha=0.5, linewidth=1)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_scaling_curves(df: pd.DataFrame,
                        task: str = 'CR',
                        x_var: str = 'sl',
                        metric: str = 'test_acc',
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Create scaling curves showing performance vs sequence length or vocab size.
    
    Style: Line plot with confidence bands and log-scale x-axis.
    """
    fig, ax = plt.subplots(figsize=(NEURIPS_COLUMN_WIDTH, 2.5))
    
    # Filter to task
    task_df = df[df['t'] == task].copy()
    
    if len(task_df) == 0:
        ax.text(0.5, 0.5, f'No data for task {task}', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    models = sorted(task_df['model_name'].unique())
    
    for model in models:
        model_df = task_df[task_df['model_name'] == model]
        
        # Group by x variable and get mean/std
        grouped = model_df.groupby(x_var)[metric].agg(['mean', 'std', 'max'])
        
        if len(grouped) > 1:
            x_vals = grouped.index.values
            y_mean = grouped['max'].values  # Use best result
            
            color = COLORS['model'].get(model, COLORS['primary']['grey'])
            label = MODEL_NAMES.get(model, model)
            
            ax.plot(x_vals, y_mean, 'o-', color=color, label=label, 
                   markersize=6, markeredgecolor='white', markeredgewidth=0.5)
    
    ax.set_xlabel({'sl': 'Sequence Length', 'vs': 'Vocabulary Size', 'ntr': 'Training Examples'}.get(x_var, x_var))
    ax.set_ylabel('Test Accuracy')
    ax.set_title(f'{TASK_NAMES.get(task, task)}: Scaling Analysis', fontweight='bold')
    
    # Log scale for sequence length
    if x_var == 'sl' and task_df[x_var].max() / task_df[x_var].min() > 4:
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    ax.legend(loc='best', fontsize=7)
    ax.set_ylim(0, 1.05)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_heatmap_performance(df: pd.DataFrame,
                              metric: str = 'test_acc',
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a heatmap showing model performance across tasks and configurations.
    
    Style: Clean heatmap with annotations and diverging colormap.
    """
    fig, ax = plt.subplots(figsize=(NEURIPS_TEXTWIDTH, 4))
    
    # Pivot to get model x task matrix
    pivot_df = df.groupby(['model_name', 't'])[metric].max().unstack(fill_value=0)
    
    # Rename columns and index
    pivot_df.columns = [TASK_NAMES.get(c, c) for c in pivot_df.columns]
    pivot_df.index = [MODEL_NAMES.get(m, m) for m in pivot_df.index]
    
    # Create custom colormap (white to blue)
    colors_list = ['#FFFFFF', '#E8F4F8', '#B3D9E8', '#4477AA', '#1A3A5C']
    cmap = LinearSegmentedColormap.from_list('neurips', colors_list, N=256)
    
    # Plot heatmap
    im = ax.imshow(pivot_df.values, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Test Accuracy', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.values[i, j]
            text_color = 'white' if val > 0.6 else 'black'
            ax.annotate(f'{val:.2f}', xy=(j, i), ha='center', va='center',
                       fontsize=9, color=text_color, fontweight='bold' if val > 0.9 else 'normal')
    
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=30, ha='right')
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)
    
    ax.set_title('Model Performance Heatmap (Best Test Accuracy)', fontweight='bold', pad=10)
    
    # Add grid lines
    ax.set_xticks(np.arange(len(pivot_df.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(pivot_df.index)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_radar_comparison(df: pd.DataFrame,
                          models: Optional[List[str]] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a radar/spider chart comparing models across multiple dimensions.
    
    Style: Semi-transparent filled polygons with clear labels.
    """
    fig, ax = plt.subplots(figsize=(NEURIPS_COLUMN_WIDTH + 1, NEURIPS_COLUMN_WIDTH + 1), 
                           subplot_kw=dict(projection='polar'))
    
    # Get tasks as dimensions
    tasks = sorted(df['t'].unique())
    if models is None:
        models = sorted(df['model_name'].unique())
    
    # Number of dimensions
    N = len(tasks)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Plot each model
    for model in models:
        model_df = df[df['model_name'] == model]
        values = []
        for task in tasks:
            task_data = model_df[model_df['t'] == task]
            if len(task_data) > 0:
                values.append(task_data['test_acc'].max())
            else:
                values.append(0)
        values += values[:1]  # Close the polygon
        
        color = COLORS['model'].get(model, COLORS['primary']['grey'])
        label = MODEL_NAMES.get(model, model)
        
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color, markersize=4)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([TASK_NAMES.get(t, t) for t in tasks], size=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], size=7, color='grey')
    ax.grid(True, alpha=0.3)
    
    # Move legend outside
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title('Multi-Task Capability Comparison', fontweight='bold', pad=20)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_multi_panel_summary(df: pd.DataFrame,
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comprehensive multi-panel figure for the main paper results.
    
    Layout:
    +---------------+---------------+
    |  (a) Heatmap  | (b) Radar    |
    +---------------+---------------+
    |        (c) Scaling Curves     |
    +-------------------------------+
    """
    fig = plt.figure(figsize=(NEURIPS_FULL_WIDTH, 7))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 0.8], wspace=0.25, hspace=0.35)
    
    # --- Panel (a): Heatmap ---
    ax_heat = fig.add_subplot(gs[0, 0])
    
    pivot_df = df.groupby(['model_name', 't'])['test_acc'].max().unstack(fill_value=0)
    pivot_df.columns = [TASK_NAMES.get(c, c) for c in pivot_df.columns]
    pivot_df.index = [MODEL_NAMES.get(m, m) for m in pivot_df.index]
    
    colors_list = ['#FFFFFF', '#E8F4F8', '#B3D9E8', '#4477AA', '#1A3A5C']
    cmap = LinearSegmentedColormap.from_list('neurips', colors_list, N=256)
    
    im = ax_heat.imshow(pivot_df.values, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.values[i, j]
            text_color = 'white' if val > 0.6 else 'black'
            ax_heat.annotate(f'{val:.2f}', xy=(j, i), ha='center', va='center',
                           fontsize=7, color=text_color)
    
    ax_heat.set_xticks(range(len(pivot_df.columns)))
    ax_heat.set_xticklabels(pivot_df.columns, rotation=35, ha='right', fontsize=7)
    ax_heat.set_yticks(range(len(pivot_df.index)))
    ax_heat.set_yticklabels(pivot_df.index, fontsize=7)
    ax_heat.set_title('(a) Performance Matrix', fontweight='bold', fontsize=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label('Accuracy', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    
    # --- Panel (b): Radar ---
    ax_radar = fig.add_subplot(gs[0, 1], projection='polar')
    
    tasks = sorted(df['t'].unique())
    models = sorted(df['model_name'].unique())
    N = len(tasks)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    for model in models[:4]:  # Limit to top 4 models for clarity
        model_df = df[df['model_name'] == model]
        values = [model_df[model_df['t'] == t]['test_acc'].max() if len(model_df[model_df['t'] == t]) > 0 else 0 for t in tasks]
        values += values[:1]
        
        color = COLORS['model'].get(model, COLORS['primary']['grey'])
        label = MODEL_NAMES.get(model, model)
        
        ax_radar.plot(angles, values, 'o-', linewidth=1.5, label=label, color=color, markersize=3)
        ax_radar.fill(angles, values, alpha=0.1, color=color)
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels([TASK_NAMES.get(t, t) for t in tasks], size=6)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.5, 1.0])
    ax_radar.set_yticklabels(['0.5', '1.0'], size=6, color='grey')
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=6)
    ax_radar.set_title('(b) Multi-Task Profile', fontweight='bold', fontsize=10, pad=15)
    
    # --- Panel (c): Scaling Curves ---
    ax_scale = fig.add_subplot(gs[1, :])
    
    # Find a task with sequence length variation
    task_for_scaling = 'CR'  # In-context recall typically has scaling data
    task_df = df[df['t'] == task_for_scaling].copy()
    
    if len(task_df) > 0 and 'sl' in task_df.columns:
        for model in models:
            model_df = task_df[task_df['model_name'] == model]
            grouped = model_df.groupby('sl')['test_acc'].max()
            
            if len(grouped) > 1:
                x_vals = grouped.index.values
                y_vals = grouped.values
                
                color = COLORS['model'].get(model, COLORS['primary']['grey'])
                label = MODEL_NAMES.get(model, model)
                
                ax_scale.plot(x_vals, y_vals, 'o-', color=color, label=label,
                            markersize=5, markeredgecolor='white', markeredgewidth=0.3)
        
        ax_scale.set_xlabel('Sequence Length')
        ax_scale.set_ylabel('Test Accuracy')
        ax_scale.set_title(f'(c) Scaling on {TASK_NAMES.get(task_for_scaling, task_for_scaling)}', 
                          fontweight='bold', fontsize=10)
        
        if task_df['sl'].max() / task_df['sl'].min() > 4:
            ax_scale.set_xscale('log', base=2)
            ax_scale.xaxis.set_major_formatter(ticker.ScalarFormatter())
        
        ax_scale.legend(loc='lower left', ncol=3, fontsize=7)
        ax_scale.set_ylim(0, 1.05)
    else:
        ax_scale.text(0.5, 0.5, 'Scaling data not available', ha='center', va='center', transform=ax_scale.transAxes)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_hyperparameter_sensitivity(df: pd.DataFrame,
                                    task: str = 'CR',
                                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a visualization of hyperparameter sensitivity analysis.
    
    Shows how learning rate and weight decay affect performance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(NEURIPS_FULL_WIDTH, 2.5))
    
    task_df = df[df['t'] == task].copy()
    
    if len(task_df) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, f'No data for task {task}', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    models = sorted(task_df['model_name'].unique())
    
    # Panel (a): Learning rate sensitivity
    ax = axes[0]
    for model in models:
        model_df = task_df[task_df['model_name'] == model]
        if 'lr' in model_df.columns:
            grouped = model_df.groupby('lr')['test_acc'].agg(['mean', 'std', 'max'])
            if len(grouped) > 1:
                x = grouped.index.values
                y = grouped['max'].values
                
                color = COLORS['model'].get(model, COLORS['primary']['grey'])
                label = MODEL_NAMES.get(model, model)
                
                ax.semilogx(x, y, 'o-', color=color, label=label, markersize=5)
    
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Best Test Accuracy')
    ax.set_title('(a) Learning Rate Sensitivity', fontweight='bold')
    ax.legend(fontsize=6, loc='best')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Panel (b): Weight decay sensitivity
    ax = axes[1]
    for model in models:
        model_df = task_df[task_df['model_name'] == model]
        if 'wd' in model_df.columns:
            grouped = model_df.groupby('wd')['test_acc'].agg(['mean', 'std', 'max'])
            if len(grouped) > 1:
                x = grouped.index.values
                y = grouped['max'].values
                
                color = COLORS['model'].get(model, COLORS['primary']['grey'])
                label = MODEL_NAMES.get(model, model)
                
                ax.plot(x, y, 'o-', color=color, label=label, markersize=5)
    
    ax.set_xlabel('Weight Decay')
    ax.set_ylabel('Best Test Accuracy')
    ax.set_title('(b) Weight Decay Sensitivity', fontweight='bold')
    ax.legend(fontsize=6, loc='best')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_efficiency(df: pd.DataFrame,
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize sample efficiency: performance vs training examples.
    """
    fig, ax = plt.subplots(figsize=(NEURIPS_COLUMN_WIDTH, 2.5))
    
    # Filter to in-context recall task
    task_df = df[df['t'] == 'CR'].copy()
    
    if len(task_df) == 0 or 'ntr' not in task_df.columns:
        ax.text(0.5, 0.5, 'Training efficiency data not available', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    models = sorted(task_df['model_name'].unique())
    
    for model in models:
        model_df = task_df[task_df['model_name'] == model]
        grouped = model_df.groupby('ntr')['test_acc'].max()
        
        if len(grouped) > 1:
            x = grouped.index.values
            y = grouped.values
            
            color = COLORS['model'].get(model, COLORS['primary']['grey'])
            label = MODEL_NAMES.get(model, model)
            
            ax.semilogx(x, y, 'o-', color=color, label=label, 
                       markersize=5, markeredgecolor='white', markeredgewidth=0.3)
    
    ax.set_xlabel('Training Examples')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Sample Efficiency Comparison', fontweight='bold')
    ax.legend(loc='lower right', fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_latex_table(df: pd.DataFrame, metric: str = 'test_acc') -> str:
    """
    Generate a LaTeX table of results for the paper.
    """
    pivot_df = df.groupby(['model_name', 't'])[metric].max().unstack(fill_value=float('nan'))
    
    # Rename for display
    pivot_df.columns = [TASK_NAMES.get(c, c) for c in pivot_df.columns]
    pivot_df.index = [MODEL_NAMES.get(m, m) for m in pivot_df.index]
    
    # Add mean column
    pivot_df['Mean'] = pivot_df.mean(axis=1)
    
    # Format as LaTeX
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Model performance across synthetic tasks (test accuracy).}\n"
    latex += "\\label{tab:results}\n"
    latex += "\\begin{tabular}{l" + "c" * len(pivot_df.columns) + "}\n"
    latex += "\\toprule\n"
    latex += "Model & " + " & ".join(pivot_df.columns) + " \\\\\n"
    latex += "\\midrule\n"
    
    for idx, row in pivot_df.iterrows():
        values = []
        for val in row:
            if pd.isna(val):
                values.append("-")
            elif val == row.max():
                values.append(f"\\textbf{{{val:.3f}}}")
            else:
                values.append(f"{val:.3f}")
        latex += f"{idx} & " + " & ".join(values) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}"
    
    return latex


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point for visualization script."""
    setup_neurips_style()
    
    # Find the benchmark logs directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    log_dir = project_root / 'benchmark' / 'logs'
    
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        return
    
    print(f"Loading benchmark results from: {log_dir}")
    df = load_benchmark_results(str(log_dir))
    
    if len(df) == 0:
        print("No results found!")
        return
    
    print(f"Loaded {len(df)} experiment results")
    print(f"Tasks: {sorted(df['t'].unique())}")
    print(f"Models: {sorted(df['model_name'].unique())}")
    
    # Create output directory
    output_dir = project_root / 'assets' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating figures in: {output_dir}")
    
    # Generate all figures (PNG format for better compatibility)
    print("  → Creating multi-panel summary...")
    plot_multi_panel_summary(df, save_path=output_dir / 'benchmark_summary.png')
    
    print("  → Creating heatmap...")
    plot_heatmap_performance(df, save_path=output_dir / 'performance_heatmap.png')
    
    print("  → Creating radar chart...")
    plot_radar_comparison(df, save_path=output_dir / 'radar_comparison.png')
    
    print("  → Creating scaling curves...")
    for task in sorted(df['t'].unique()):
        plot_scaling_curves(df, task=task, x_var='sl', 
                           save_path=output_dir / f'scaling_{task}_seqlen.png')
    
    print("  → Creating hyperparameter sensitivity plots...")
    for task in sorted(df['t'].unique())[:2]:  # Top 2 tasks
        plot_hyperparameter_sensitivity(df, task=task,
                                       save_path=output_dir / f'hyperparam_{task}.png')
    
    print("  → Creating training efficiency plot...")
    plot_training_efficiency(df, save_path=output_dir / 'training_efficiency.png')
    
    print("  → Creating task comparison bars...")
    plot_task_comparison_bars(df, save_path=output_dir / 'task_comparison.png')
    
    # Generate LaTeX table
    latex_table = generate_latex_table(df)
    table_file = output_dir / 'results_table.tex'
    with open(table_file, 'w') as f:
        f.write(latex_table)
    print(f"  → LaTeX table saved to: {table_file}")
    
    print("\n✓ All figures generated successfully!")
    print(f"\nGenerated files:")
    for f in sorted(output_dir.glob('*')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
