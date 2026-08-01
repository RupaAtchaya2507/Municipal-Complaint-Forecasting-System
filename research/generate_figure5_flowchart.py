import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import numpy as np

# Initialize figure with professional horizontal two-column journal dimensions
fig, ax = plt.subplots(figsize=(19.0, 10.5), dpi=300)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Strict Academic Color Palette (Dark Blue, Gray, and Teal)
C_PRIMARY = '#0F2C59'    # Deep Journal Blue (Primary headers & borders)
C_ACCENT = '#1A5F7A'     # Teal (Subheaders, Accents, and Highlight text)
C_BG_DARK = '#F8F9FA'    # Light Architectural Gray (Main block backgrounds)
C_BG_LIGHT = '#FFFFFF'   # Pure White (Inner block backgrounds)
C_TEXT_DARK = '#212529'  # Off-Black (High-contrast body text)
C_TEXT_MUTED = '#495057' # Slate Gray (Annotations & arrows)
C_LINE = '#868E96'       # Clean Gray (Connectors & loops)

# Typography Scaling
FS_TITLE = 13
FS_HEADER = 10.0
FS_BODY = 8.5
FS_MATH = 9.5

# Column Centers
X_COL1 = 4.8
X_COL2 = 14.8

# ----------------------------------------------------
# Custom Shape Drawing Helpers
# ----------------------------------------------------
def draw_oval(ax, cx, cy, w, h, text, bg_color=C_BG_LIGHT, border_color=C_PRIMARY):
    """Draws a clean, publication-grade oval for START/END nodes."""
    rect = patches.FancyBboxPatch((cx - w/2, cy - h/2), w, h, boxstyle="round,pad=0.15,rounding_size=0.3", facecolor=bg_color, edgecolor=border_color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(cx, cy, text, color=C_TEXT_DARK, weight='bold', fontsize=FS_HEADER, ha='center', va='center')

def draw_rect(ax, cx, cy, w, h, title, items=None, bg_color=C_BG_DARK, border_color=C_PRIMARY):
    """Draws a clean standard rectangular flowchart block with centered header."""
    x = cx - w/2
    y = cy - h/2
    # Base Box
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", facecolor=bg_color, edgecolor=border_color, linewidth=1.3)
    ax.add_patch(rect)
    
    # Header Banner Accent
    header_h = 0.38
    header_rect = patches.FancyBboxPatch((x, y + h - header_h), w, header_h, boxstyle="round,pad=0.01", facecolor=border_color, edgecolor=border_color)
    ax.add_patch(header_rect)
    
    # Header Title Text
    ax.text(cx, cy + h/2 - header_h/2, title, color='white', weight='bold', fontsize=FS_HEADER, ha='center', va='center')
    
    # Body Items Text
    if items:
        start_y = cy + h/2 - header_h - 0.18
        for item in items:
            ax.text(x + 0.18, start_y, f"•  {item}", color=C_TEXT_DARK, fontsize=FS_BODY, ha='left', va='center')
            start_y -= 0.22

def draw_diamond(ax, cx, cy, w, h, text, bg_color=C_BG_LIGHT, border_color=C_PRIMARY):
    """Draws a clean flowchart decision diamond with centered text."""
    vertices = [
        (cx, cy - h/2),     # Bottom
        (cx + w/2, cy),     # Right
        (cx, cy + h/2),     # Top
        (cx - w/2, cy)      # Left
    ]
    poly = Polygon(vertices, facecolor=bg_color, edgecolor=border_color, linewidth=1.3, closed=True)
    ax.add_patch(poly)
    ax.text(cx, cy, text, color=C_TEXT_DARK, weight='bold', fontsize=FS_HEADER - 0.5, ha='center', va='center')

def draw_arrow(ax, x1, y1, x2, y2, text=None, side_text_offset=0.15):
    """Draws a straight vector flow arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_LINE, lw=1.6, mutation_scale=12))
    if text:
        # Check orientation
        if abs(x1 - x2) < 0.1:  # Vertical arrow
            ax.text(x1 + side_text_offset, (y1 + y2)/2, text, color=C_TEXT_MUTED, fontsize=FS_BODY, ha='left', va='center', weight='bold')
        else:  # Horizontal arrow
            ax.text((x1 + x2)/2, y1 + side_text_offset, text, color=C_TEXT_MUTED, fontsize=FS_BODY, ha='center', va='bottom', weight='bold')

def draw_path(ax, points, arrow_at_end=True):
    """Draws a multi-segmented pipeline path (useful for loopback paths)."""
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=C_LINE, lw=1.4, linestyle='-')
    if arrow_at_end:
        # Draw arrow at last segment
        x1, y1 = points[-2], points[-1]
        ax.annotate('', xy=(points[-1][0], points[-1][1]), xytext=(points[-2][0], points[-2][1]),
                    arrowprops=dict(arrowstyle="-|>", color=C_LINE, lw=1.6, mutation_scale=12))

# ====================================================
# TWO-COLUMN FLOWCHART LAYOUT
# ====================================================

# ────────────────────────────────────────────────────
# COLUMN 1: PRIOR MODELING & ENVIRONMENTAL SHOCKS
# ────────────────────────────────────────────────────

# 1. START [y = 9.8]
draw_oval(ax, X_COL1, 9.8, 1.8, 0.45, "START")

# 2. Initialize Time Window (t) [y = 8.6]
draw_rect(ax, X_COL1, 8.6, 4.2, 1.1, "Initialize Time Window (t)", 
          ["Input: Daily aggregation window", "Temporal sequence index"])

# 3. Select Municipal Zone (z) [y = 7.2]
draw_rect(ax, X_COL1, 7.2, 4.2, 1.1, "Select Municipal Zone (z)", 
          ["Input: Current spatial zone node", "Spatial GNN alignment"])

# 4. Compute Baseline Complaint Rate [y = 5.7]
draw_rect(ax, X_COL1, 5.7, 4.6, 1.3, r"Compute Baseline Rate $\lambda_{\text{base}}$", 
          ["Derived from: Historical zone probability $P(z)$", "Diurnal, weekly, & seasonal temporal PMFs", "Complaint category prior distributions"])

# 5. Apply Weather Conditioning [y = 4.2]
draw_rect(ax, X_COL1, 4.2, 4.6, 1.3, "Apply Weather Conditioning", 
          ["Inputs: Temperature, Rainfall, Humidity", "Multipliers: Binned rain + Temp/Hum slopes", "Output: Weather-conditioned rate $\lambda_{\text{weather}}$"])

# 6. Apply Festival Conditioning [y = 2.7]
draw_rect(ax, X_COL1, 2.7, 4.6, 1.3, "Apply Festival Conditioning", 
          ["Inputs: Festival Day Flag, Festival Eve Indicator", "Multipliers: Holiday surge ($1.30\\times$) / Eve ($1.15\\times$)", "Output: Festival-conditioned rate $\lambda_{\text{festival}}$"])

# 7. Backlog Decision Diamond [y = 1.6]
draw_diamond(ax, X_COL1, 1.6, 3.2, 1.0, "Unresolved Backlog\nPresent?")

# 8. Apply Backlog Amplification [y = 0.5]
draw_rect(ax, X_COL1, 0.5, 4.2, 0.8, "Apply Backlog Amplification", 
          ["Boost multiplier: $1.0 + \\min(0.3, \\text{Queue}_z \\cdot 0.05)$", "Equation: $\lambda_{\text{backlog}} = \lambda_{\\text{env}} \\times M_{\\text{backlog}}$"])


# ────────────────────────────────────────────────────
# COLUMN 2: SAMPLING, ATTRIBUTES & ITERATION
# ────────────────────────────────────────────────────

# 9. Apply Graph Spillover [y = 9.2]
draw_rect(ax, X_COL2, 9.2, 4.8, 1.4, "Apply Graph Spillover", 
          ["Inputs: KNN Zone Graph ($k=3$), Neighbor Influence", "Equation: $\\Lambda = (1-\\eta)\\Lambda_{\\text{raw}} + \\eta(A_{\\text{norm}}\\Lambda_{\\text{raw}})$", "Diffusion Parameter: $\\eta = 0.15$ (15% spillover)"])

# 10. Poisson Complaint Sampling [y = 7.7]
draw_rect(ax, X_COL2, 7.7, 4.2, 1.1, "Poisson Complaint Sampling", 
          ["Sampling: $N \\sim \\text{Poisson}(\\Lambda)$", "Output: Raw volumetric number of incidents"])

# 11. Generate Complaint Attributes [y = 6.2]
draw_rect(ax, X_COL2, 6.2, 4.2, 1.3, "Generate Complaint Attributes", 
          ["Loop: For each complaint $i \\in [0, N-1]$", "•  Timestamp: Uniform window offsets", "•  Coordinates: Gaussian spatial spread + Jitter", "•  Priors: Category ID ($P(\\text{Cat}\\mid z)$) & Backlog ($P(\\text{Open}\\mid\\text{Cat})$)"])

# 12. Store Synthetic Complaint Record [y = 4.8]
draw_rect(ax, X_COL2, 4.8, 4.2, 0.9, "Store Synthetic Complaint Record", 
          ["Hydrate descriptions & metadata from historical lookup", "Append record to complaint database"])

# 13. Decision Loop: More Complaints? [y = 3.5]
draw_diamond(ax, X_COL2, 3.5, 3.2, 1.1, "More Complaints\nNeeded?")

# 14. Decision Loop: Next Zone? [y = 2.2]
draw_diamond(ax, X_COL2, 2.2, 3.2, 1.1, "Next spatial\nzone node?")

# 15. Decision Loop: Next Time Window? [y = 0.9]
draw_diamond(ax, X_COL2, 0.9, 3.2, 1.1, "Next temporal\ntime step?")

# 16. END [y = -0.05]
draw_oval(ax, X_COL2, -0.05, 1.8, 0.45, "END")


# ====================================================
# CONNECTORS & SEGMENT PATHWAY ROUTINGS
# ====================================================

# --- COLUMN 1 CONNECTORS ---
draw_arrow(ax, X_COL1, 9.55, X_COL1, 9.15)
draw_arrow(ax, X_COL1, 8.05, X_COL1, 7.75)
draw_arrow(ax, X_COL1, 6.65, X_COL1, 6.35)
draw_arrow(ax, X_COL1, 5.05, X_COL1, 4.85)
draw_arrow(ax, X_COL1, 3.55, X_COL1, 3.35)
draw_arrow(ax, X_COL1, 2.05, X_COL1, 2.10)

# Backlog Diamond YES Split
draw_arrow(ax, X_COL1, 1.10, X_COL1, 0.90, "YES")

# Backlog Diamond NO Split (Bypasses Backlog Amplification)
c1_bypass = [
    (X_COL1 + 1.6, 1.6),      # Exit right of diamond
    (X_COL1 + 2.5, 1.6),      # Go further right
    (X_COL1 + 2.5, -0.15),    # Drop below backlog block
    (X_COL1, -0.15)           # Merge into main output line
]
draw_path(ax, c1_bypass, arrow_at_end=False)
ax.text(X_COL1 + 1.8, 1.6, "NO", color=C_TEXT_MUTED, fontsize=FS_BODY, ha='left', weight='bold')

# --- CROSS-COLUMN MAIN FLOW CONNECTOR (Col 1 to Col 2) ---
col_cross_flow = [
    (X_COL1, -0.05),          # Bottom of Column 1 main line
    (X_COL1, -0.35),          # Go down slightly
    (9.8, -0.35),             # Cross to gap center
    (9.8, 10.2),              # Rise to top of Column 2
    (X_COL2, 10.2),           # Align above Graph Spillover
    (X_COL2, 9.9)             # Arrow down into Graph Spillover
]
draw_path(ax, col_cross_flow)

# --- COLUMN 2 CONNECTORS ---
draw_arrow(ax, X_COL2, 8.5, X_COL2, 8.25)
draw_arrow(ax, X_COL2, 7.15, X_COL2, 6.85)
draw_arrow(ax, X_COL2, 5.55, X_COL2, 5.25)
draw_arrow(ax, X_COL2, 4.35, X_COL2, 4.05)
draw_arrow(ax, X_COL2, 2.95, X_COL2, 2.75, "NO")
draw_arrow(ax, X_COL2, 1.65, X_COL2, 1.45, "NO")
draw_arrow(ax, X_COL2, 0.35, X_COL2, 0.175, "NO")

# More Complaints? YES Loopback (Inside Column 2)
loop1_pts = [
    (X_COL2 + 1.6, 3.5),      # Exit right of diamond
    (X_COL2 + 2.6, 3.5),      # Go right
    (X_COL2 + 2.6, 6.2),      # Rise up level with attributes block
    (X_COL2 + 2.1, 6.2)       # Arrow into right of attributes block
]
draw_path(ax, loop1_pts)
ax.text(X_COL2 + 1.8, 3.65, "YES", color=C_TEXT_MUTED, fontsize=FS_BODY, ha='left', weight='bold')

# Next Zone? YES Loopback (Col 2 back to Col 1 Select Zone)
loop2_pts = [
    (X_COL2 - 1.6, 2.2),      # Exit left of diamond
    (11.8, 2.2),              # Go left in gap
    (11.8, 7.2),              # Rise up level with select zone block
    (X_COL1 + 2.1, 7.2)       # Arrow into right of select zone block
]
draw_path(ax, loop2_pts)
ax.text(X_COL2 - 1.8, 2.35, "YES", color=C_TEXT_MUTED, fontsize=FS_BODY, ha='right', weight='bold')

# Next Time Window? YES Loopback (Col 2 back to Col 1 Initialize Window)
loop3_pts = [
    (X_COL2 - 1.6, 0.9),      # Exit left of diamond
    (11.0, 0.9),              # Go further left in gap
    (11.0, 8.6),              # Rise up level with initialize window block
    (X_COL1 + 2.1, 8.6)       # Arrow into right of initialize window block
]
draw_path(ax, loop3_pts)
ax.text(X_COL2 - 1.8, 1.05, "YES", color=C_TEXT_MUTED, fontsize=FS_BODY, ha='right', weight='bold')


# ====================================================
# CANVAS BOUNDARIES & EXPORT
# ====================================================
ax.set_xlim(0.0, 20.0)
ax.set_ylim(-0.6, 10.4)
ax.axis('off')
plt.tight_layout()

# Save paths defined in paper/figures/ folder
project_root = r"c:\Users\utham\Desktop\final year project\project"
figures_dir = os.path.join(project_root, "paper", "figures")
os.makedirs(figures_dir, exist_ok=True)

svg_path = os.path.join(figures_dir, "Figure5_Complaint_Expansion_Workflow.svg")
png_path = os.path.join(figures_dir, "Figure5_Complaint_Expansion_Workflow.png")

# Save as vector SVG and high-resolution PNG, overwriting the old files
plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"Workflow flowchart successfully saved to:\nSVG: {svg_path}\nPNG: {png_path}")
