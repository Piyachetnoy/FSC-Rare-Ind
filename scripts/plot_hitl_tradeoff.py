#!/usr/bin/env python3
"""
HITL Tradeoff Curve Plotter
Generates accuracy vs. efficiency trade-off curves from workflow comparison results.
"""

import matplotlib
matplotlib.use('Agg')  # GUIなしバックエンド

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# データ読み込み
csv_path = "/Users/piyachetnoy/Desktop/repositories/thesis/rare-ind-counting/hitl_results_modify_thresholds/workflow_comparison_20260117_153014.csv"
df = pd.read_csv(csv_path)

# スタイル設定（論文向け）
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Yu Gothic',
    'font.weight': 'bold',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
})

fig, ax = plt.subplots()

# データ分類
random_sampling = df[df['workflow_name'].str.contains('Random Sampling')]
adaptive_hitl = df[df['workflow_name'].str.contains('Adaptive HITL')]
fully_automated = df[df['workflow_name'] == 'Fully Automated']
full_manual = df[df['workflow_name'] == 'Full Manual Verification']

# Random Sampling の曲線（始点・終点を含む統合曲線）
rs_hir = [0] + random_sampling['human_intervention_rate'].tolist() + [100]
rs_mae = [fully_automated['mae'].values[0]] + random_sampling['mae'].tolist() + [full_manual['mae'].values[0]]
ax.plot(rs_hir, rs_mae, 'b--o', linewidth=2, markersize=8, label='Random Sampling', alpha=0.8)

# Adaptive HITL の曲線（始点・終点を含む統合曲線）
ah_hir = [0] + adaptive_hitl['human_intervention_rate'].tolist() + [100]
ah_mae = [fully_automated['mae'].values[0]] + adaptive_hitl['mae'].tolist() + [full_manual['mae'].values[0]]
ax.plot(ah_hir, ah_mae, 'r-s', linewidth=2, markersize=8, label='Adaptive HITL (Proposed)', alpha=0.9)

# 始点・終点のポイントをマーク
ax.scatter([0], [fully_automated['mae'].values[0]], 
           c='black', s=64, marker='o', zorder=5)
ax.scatter([100], [full_manual['mae'].values[0]], 
           c='black', s=64, marker='o', zorder=5)

# 始点・終点のラベル
ax.annotate('Fully\nAutomated', 
            xy=(0, fully_automated['mae'].values[0]),
            xytext=(10, 20), textcoords='offset points',
            fontsize=10, ha='left', color='black', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1))
ax.annotate('Full Manual\nVerification', 
            xy=(100, full_manual['mae'].values[0]),
            xytext=(-80, 30), textcoords='offset points',
            fontsize=10, ha='left', color='black', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

# Adaptive HITLポイントにθラベル追加（位置調整）
theta_offsets = {0.6: (8, 12), 0.7: (-35, -15), 0.8: (-35, 10), 0.9: (-35, -5)}
for _, row in adaptive_hitl.iterrows():
    theta = row['threshold']
    offset = theta_offsets.get(theta, (5, 10))
    ax.annotate(f'θ={theta}', 
                xy=(row['human_intervention_rate'], row['mae']),
                xytext=offset, textcoords='offset points',
                fontsize=9, color='darkred', fontweight='bold')

# 軸設定
ax.set_xlabel('Human Intervention Rate (%)', fontweight='bold')
ax.set_ylabel('Mean Absolute Error (MAE)', fontweight='bold')
ax.set_title('Accuracy-Efficiency Trade-off: Adaptive HITL vs Random Sampling', fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=12)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
ax.set_xlim(-5, 105)
ax.set_ylim(0, max(rs_mae) * 1.15)

# グリッドと凡例
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', prop={'weight': 'bold'})

# 領域のハイライト（薄く）
common_hir = np.linspace(0, 100, 100)
rs_mae_interp = np.interp(common_hir, rs_hir, rs_mae)
ah_mae_interp = np.interp(common_hir, ah_hir, ah_mae)
ax.fill_between(common_hir, rs_mae_interp, ah_mae_interp, 
                where=(rs_mae_interp > ah_mae_interp), 
                alpha=0.1, color='green')

plt.tight_layout()

# 保存
output_dir = "/Users/piyachetnoy/Desktop/repositories/thesis/rare-ind-counting/images"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "hitl_tradeoff_curve.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}")

# PDF版も保存（論文用）
plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved: {output_path.replace('.png', '.pdf')}")

# plt.show()
