"""Script to generate all custom technical diagrams and evaluation plots for the final report."""

import os
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
out_dir = Path("report_figures")
out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Figure 1.1: Gantt Chart / Project Timeline
# -------------------------------------------------------------
def generate_gantt_chart():
    tasks = [
        "Problem Formulation & Dataset Audit (PTB-XL)",
        "Synthetic Rendering & Degradation Engine",
        "Layout Detection CNN & Heuristic Canvas Bounding",
        "Lead Bounding & Grid Line Suppression Pipeline",
        "U-Net Waveform Segmentation & Coordinate Regression",
        "Calibration, Time-Dilation & Rhythm Chunking Remediation",
        "FastAPI Service & Interactive Workstation UI",
        "Held-Out Fold 10 & Real-Image Verification Audit",
        "Final Documentation & NBA Major Project Report",
    ]
    start_weeks = [1, 3, 5, 7, 9, 12, 14, 16, 18]
    durations =   [3, 3, 3, 3, 4,  3,  3,  3,  2]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    y_pos = np.arange(len(tasks))
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(tasks)))

    for i, (task, start, dur, col) in enumerate(zip(tasks, start_weeks, durations, colors)):
        ax.barh(i, dur, left=start, height=0.55, align='center', color=col, edgecolor='black', linewidth=0.8, alpha=0.9)
        ax.text(start + dur/2.0, i, f"W{start}-W{start+dur-1}", ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tasks, fontsize=9, fontweight='medium')
    ax.invert_yaxis()
    ax.set_xlabel("Project Timeline (Academic Calendar Weeks)", fontsize=10, fontweight='bold', labelpad=8)
    ax.set_title("ECG DigiAI Work Breakdown Structure & Execution Timeline", fontsize=12, fontweight='bold', pad=12)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 21)
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_1_gantt_chart.png", dpi=300)
    plt.close()
    print("Generated fig1_1_gantt_chart.png")

# -------------------------------------------------------------
# Figure 3.1: System Architecture Block Diagram
# -------------------------------------------------------------
def generate_system_architecture():
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def draw_box(x, y, w, h, title, subtitle, color, text_color='black'):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                      facecolor=color, edgecolor='#2C3E50', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=text_color)
        ax.text(x + w/2, y + h*0.35, subtitle, ha='center', va='center', fontsize=7.5, color=text_color, style='italic')

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#2C3E50', edgecolor='#2C3E50', arrowstyle='->', lw=1.5))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.12, label, ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#1A5276')

    # Main Blocks
    draw_box(0.3, 2.2, 1.8, 1.6, "Input ECG Source", "Paper Scan / Mobile Photo\nPNG, JPG, JPEG", "#D4E6F1")
    draw_arrow(2.1, 3.0, 2.6, 3.0, "Raw BGR")

    draw_box(2.6, 2.2, 2.0, 1.6, "1. Preprocessing", "Hough Deskewing\nLocal Illum Close\nColor De-mixing", "#FCF3CF")
    draw_arrow(4.6, 3.0, 5.1, 3.0, "Deskewed/Trace")

    draw_box(5.1, 2.2, 2.2, 1.6, "2. Layout & Bounding", "Canvas Bounding Box\nECGLayoutCNN (4 Class)\nLead Box Partitioning", "#D5F5E3")
    draw_arrow(7.3, 3.0, 7.8, 3.0, "Lead Crops")

    draw_box(7.8, 2.2, 2.5, 1.6, "3. Deep Digitizer", "ECGDigitizerUNet\nBaseline Centered Pad\nSoft-Argmax Regression", "#E8DAEF")

    # Lower branches
    draw_arrow(9.05, 2.2, 9.05, 1.4)
    draw_box(7.8, 0.2, 2.5, 1.2, "4. Signal Assembly", "Affine Scaling (px->mV)\n100Hz Temporal Resample\n12-Lead Synchronization", "#FADBD8")

    draw_arrow(7.8, 0.8, 4.6, 0.8, "Structured Signal")
    draw_box(2.6, 0.2, 2.0, 1.2, "5. Export & UI", "Multi-Lead CSV / WFDB\nFastAPI REST Engine\nVisual Overlay HTML", "#D6DBDF")

    # Sub-system boundary labels
    rect_outer = patches.Rectangle((0.1, 0.05), 10.5, 5.7, fill=False, edgecolor='#7F8C8D', linestyle=':', linewidth=1.2)
    ax.add_patch(rect_outer)
    ax.text(5.5, 5.4, "ECG DigiAI Hierarchical End-to-End Digitisation Architecture", ha='center', fontsize=12, fontweight='bold', color='#2C3E50')
    ax.text(5.5, 5.1, "Non-Diagnostic Signal Reconstruction Framework (PTB-XL Quarantined Compliance)", ha='center', fontsize=8.5, style='italic', color='#566573')

    plt.tight_layout()
    plt.savefig(out_dir / "fig3_1_system_architecture.png", dpi=300)
    plt.close()
    print("Generated fig3_1_system_architecture.png")

# -------------------------------------------------------------
# Figure 3.3: U-Net Model Architecture
# -------------------------------------------------------------
def generate_unet_architecture():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    def block(x, y, w, h, name, desc, color):
        rect = patches.Rectangle((x, y), w, h, facecolor=color, edgecolor='#1B2631', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + 0.12, name, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1B2631')
        ax.text(x + w/2, y + h/2 - 0.18, desc, ha='center', va='center', fontsize=7, color='#2C3E50')

    # Title
    ax.text(5.0, 5.1, "ECGDigitizerUNet with Soft-Argmax Waveform Regression", ha='center', fontsize=12, fontweight='bold', color='#1B2631')
    ax.text(5.0, 4.75, "Standardized Receptive Input (128x256x3) -> Continuous 1D Coordinates (256,)", ha='center', fontsize=8.5, style='italic', color='#566573')

    # Input
    block(0.4, 3.0, 1.2, 1.2, "Input Crop", "128x256x3\nRGB Lead", "#D4E6F1")

    # Encoder
    block(1.9, 2.6, 1.1, 1.6, "Enc 1", "32ch | 64x128", "#AED6F1")
    block(3.2, 2.1, 1.1, 2.0, "Enc 2", "64ch | 32x64", "#85C1E9")
    block(4.5, 1.5, 1.1, 2.4, "Bottleneck", "128ch | 16x32", "#5DADE2")

    # Decoder
    block(5.8, 2.1, 1.1, 2.0, "Dec 2", "64ch | 32x64", "#A3E4D7")
    block(7.1, 2.6, 1.1, 1.6, "Dec 1", "32ch | 64x128", "#A2D9CE")

    # Heads
    block(8.4, 3.4, 1.3, 1.0, "Mask Head", "128x256x1\nSigmoid", "#FCF3CF")
    block(8.4, 1.8, 1.3, 1.2, "Soft-Argmax", "Spatial Softmax\nExpectation Y(x)", "#FADBD8")

    # Arrows
    def arrow(x1, y1, x2, y2, col='#1B2631'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor=col, edgecolor=col, arrowstyle='->', lw=1.2))

    arrow(1.6, 3.6, 1.9, 3.6)
    arrow(3.0, 3.2, 3.2, 3.2)
    arrow(4.3, 2.6, 4.5, 2.6)
    arrow(5.6, 2.6, 5.8, 2.6)
    arrow(6.9, 3.2, 7.1, 3.2)

    arrow(8.2, 3.8, 8.4, 3.8)
    arrow(8.2, 2.8, 8.4, 2.4)

    # Skip connections
    ax.annotate('', xy=(7.1, 3.8), xytext=(3.0, 3.8),
                arrowprops=dict(facecolor='#E74C3C', edgecolor='#E74C3C', arrowstyle='->', linestyle='--', lw=1.2))
    ax.text(5.05, 4.0, "Skip Connection 1", color='#C0392B', fontsize=7.5, ha='center', fontweight='bold')

    ax.annotate('', xy=(5.8, 3.6), xytext=(4.3, 3.6),
                arrowprops=dict(facecolor='#E74C3C', edgecolor='#E74C3C', arrowstyle='->', linestyle='--', lw=1.2))
    ax.text(5.05, 3.4, "Skip 2", color='#C0392B', fontsize=7.5, ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_dir / "fig3_3_unet_architecture.png", dpi=300)
    plt.close()
    print("Generated fig3_3_unet_architecture.png")

# -------------------------------------------------------------
# Figure 5.2: Per-Lead Metrics Plot
# -------------------------------------------------------------
def generate_per_lead_metrics():
    with open('models/fold10_heldout_evaluation_report.json') as f:
        data = json.load(f)

    lead_stats = data['per_lead_performance']
    # Filter only standard 12 leads + rhythm II
    leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'II_RHYTHM']
    disp_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'II (Rhy)']

    med_pearsons = [lead_stats[l]['median_pearson'] for l in leads]
    mean_pearsons = [lead_stats[l]['mean_pearson'] for l in leads]
    rmses = [lead_stats[l]['mean_rmse_mv'] for l in leads]

    fig, ax1 = plt.subplots(figsize=(10, 4.8), dpi=300)

    x = np.arange(len(leads))
    width = 0.35

    rects1 = ax1.bar(x - width/2, med_pearsons, width, label='Median Pearson r', color='#2E86C1', edgecolor='black', linewidth=0.6)
    rects2 = ax1.bar(x + width/2, mean_pearsons, width, label='Mean Pearson r', color='#85C1E9', edgecolor='black', linewidth=0.6)

    ax1.set_ylabel('Pearson Correlation Coefficient (r)', fontsize=10, fontweight='bold', color='#1B4F72')
    ax1.set_ylim(0, 1.08)
    ax1.set_xticks(x)
    ax1.set_xticklabels(disp_names, fontsize=9, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # Secondary axis for RMSE
    ax2 = ax1.twinx()
    line1 = ax2.plot(x, rmses, color='#E74C3C', marker='o', linewidth=1.8, markersize=5, label='Voltage RMSE (mV)')
    ax2.set_ylabel('Mean Voltage RMSE (mV)', fontsize=10, fontweight='bold', color='#922B21')
    ax2.set_ylim(0, 0.5)

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right', framealpha=0.95, fontsize=8.5)

    plt.title('Held-Out PTB-XL Fold 10 (N=40): Per-Lead Pearson Correlation & Voltage RMSE', fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(out_dir / "fig5_2_per_lead_metrics.png", dpi=300)
    plt.close()
    print("Generated fig5_2_per_lead_metrics.png")

if __name__ == '__main__':
    generate_gantt_chart()
    generate_system_architecture()
    generate_unet_architecture()
    generate_per_lead_metrics()
