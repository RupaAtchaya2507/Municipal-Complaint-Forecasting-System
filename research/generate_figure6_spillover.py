import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon

# Initialize figure with professional IEEE double-column width and publication spacing
fig, ax = plt.subplots(figsize=(20.0, 9.0), dpi=300)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Strict Academic Color Palette (Dark Blue, Gray, and Teal)
C_PRIMARY = '#0F2C59'      # Deep Journal Blue (Borders, Main nodes)
C_ACCENT = '#1A5F7A'       # Teal (Subheaders, Arrows, Neighbors)
C_BG_DARK = '#F8F9FA'      # Light Architectural Gray (Main block backgrounds)
C_BG_LIGHT = '#FFFFFF'     # Pure White (Inner block backgrounds)
C_TEXT_DARK = '#212529'    # Off-Black (High-contrast text)
C_TEXT_MUTED = '#495057'   # Slate Gray (Annotations & labels)
C_LINE = '#868E96'         # Clean Gray (Connectors & grids)
C_HOTSPOT = '#D9534F'      # Deep Red (Hotspot High Stress Indicator)
C_MEDIUM_STRESS = '#F0AD4E'# Soft Orange (Medium Stress Indicator)
C_LOW_STRESS = '#E9ECEF'   # Very Light Gray (Low Stress Indicator)

# Typography Scaling
FS_TITLE = 13.5
FS_HEADER = 11.0
FS_BODY = 9.0
FS_MATH = 10.0

def draw_panel_border(ax, x, y, w, h, title):
    """Draws a clean boundary enclosing each of the three scientific panels."""
    # Border
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", facecolor='none', edgecolor=C_LINE, linewidth=1.2, linestyle='-')
    ax.add_patch(rect)
    
    # Title Block Header Banner
    header_h = 0.45
    header_rect = patches.FancyBboxPatch((x, y + h - header_h), w, header_h, boxstyle="round,pad=0.01", facecolor=C_PRIMARY, edgecolor=C_PRIMARY)
    ax.add_patch(header_rect)
    
    # Title text
    ax.text(x + w/2, y + h - (header_h/2), title, color='white', weight='bold', fontsize=FS_HEADER, ha='center', va='center')

def draw_arrow(ax, x1, y1, x2, y2, text=None):
    """Draws a clean, standard vector arrow between processing stages."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_LINE, lw=1.6, mutation_scale=12))
    if text:
        ax.text((x1 + x2)/2, ((y1 + y2)/2) + 0.15, text, color=C_TEXT_MUTED, fontsize=FS_BODY - 1, ha='center', va='center', weight='bold')

# ====================================================
# PANEL A: MUNICIPAL SPATIAL GRAPH
# ====================================================
pa_x, pa_y, pa_w, pa_h = 0.5, 0.8, 6.0, 7.2
draw_panel_border(ax, pa_x, pa_y, pa_w, pa_h, "A. Municipal Spatial Graph")

# Deterministic spatial coordinates for 20 zones
np.random.seed(42)
num_nodes = 20
xs = np.random.uniform(pa_x + 0.6, pa_x + pa_w - 0.6, num_nodes)
ys = np.random.uniform(pa_y + 1.2, pa_y + pa_h - 1.2, num_nodes)

# Force Node 0 to be the central hotspot "Zone i" and place it centered
xs[0] = pa_x + pa_w / 2
ys[0] = pa_y + pa_h / 2 + 0.2

# Connect edges using K-Nearest-Neighbors (k=3)
dists = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    for j in range(num_nodes):
        dists[i, j] = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)

# Find 3 nearest neighbors for each node (excluding self)
neighbors = {}
for i in range(num_nodes):
    idx = np.argsort(dists[i])[1:4]
    neighbors[i] = idx

# Draw all standard graph edges first
for i in range(num_nodes):
    for neigh in neighbors[i]:
        if i != 0 and neigh != 0:
            ax.plot([xs[i], xs[neigh]], [ys[i], ys[neigh]], color=C_LINE, lw=0.7, alpha=0.6, zorder=1)

# Highlight KNN Edges for Central Node 0 ("Zone i")
for neigh in neighbors[0]:
    ax.plot([xs[0], xs[neigh]], [ys[0], ys[neigh]], color=C_ACCENT, lw=1.6, zorder=2)

# Draw all standard nodes
for i in range(1, num_nodes):
    if i in neighbors[0]:  # Highlight neighbors
        ax.scatter(xs[i], ys[i], color=C_ACCENT, s=80, edgecolor=C_PRIMARY, linewidth=1.0, zorder=4)
        ax.text(xs[i] + 0.15, ys[i] - 0.05, f"Node {i}", color=C_TEXT_DARK, fontsize=FS_BODY - 1, weight='bold')
    else:
        ax.scatter(xs[i], ys[i], color=C_LOW_STRESS, edgecolor=C_LINE, s=60, linewidth=1.0, zorder=3)

# Highlight Central Node 0 ("Zone i")
ax.scatter(xs[0], ys[0], color=C_HOTSPOT, s=150, edgecolor=C_PRIMARY, linewidth=1.5, zorder=5)
ax.text(xs[0] + 0.22, ys[0] + 0.1, "Central Hotspot\n(Zone i)", color=C_HOTSPOT, fontsize=FS_BODY, weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

# Legend inside Panel A
leg_y = pa_y + 0.3
rect_leg = patches.Rectangle((pa_x + 0.3, leg_y), 3.2, 0.9, facecolor='white', edgecolor=C_LINE, linewidth=0.8, zorder=4)
ax.add_patch(rect_leg)
ax.scatter(pa_x + 0.5, leg_y + 0.65, color=C_HOTSPOT, s=50, edgecolor=C_PRIMARY, linewidth=1.0, zorder=5)
ax.text(pa_x + 0.8, leg_y + 0.6, "Node = Central Zone i", color=C_TEXT_DARK, fontsize=FS_BODY - 1)
ax.scatter(pa_x + 0.5, leg_y + 0.45, color=C_ACCENT, s=40, edgecolor=C_PRIMARY, linewidth=1.0, zorder=5)
ax.text(pa_x + 0.8, leg_y + 0.4, "Teal Node = Neighbor Zones", color=C_TEXT_DARK, fontsize=FS_BODY - 1)
ax.plot([pa_x + 0.4, pa_x + 0.6], [leg_y + 0.2, leg_y + 0.2], color=C_ACCENT, lw=1.5, zorder=5)
ax.text(pa_x + 0.8, leg_y + 0.15, "Teal Edge = Spatial KNN Connection", color=C_TEXT_DARK, fontsize=FS_BODY - 1)


# ====================================================
# PANEL B: SPILLOVER PROPAGATION PROCESS
# ====================================================
pb_x, pb_y, pb_w, pb_h = 6.8, 0.8, 6.4, 7.2
draw_panel_border(ax, pb_x, pb_y, pb_w, pb_h, "B. Spillover Diffusion Process")

# Diagram: Centered Zone i diffusing to 3 Neighbors
cx, cy = pb_x + pb_w / 2, pb_y + 4.9

# Neighbor circles
neigh_pts = [
    (cx - 1.8, cy - 1.6, "Neighbor j1"),
    (cx,       cy - 2.0, "Neighbor j2"),
    (cx + 1.8, cy - 1.6, "Neighbor j3")
]

# Outward arrows proportional to spillover influence
for idx, (nx, ny, name) in enumerate(neigh_pts):
    # Proportional line thickness and alpha
    lw = 4.0 if idx == 0 else (2.5 if idx == 1 else 1.2)
    ax.annotate('', xy=(nx, ny + 0.35), xytext=(cx, cy - 0.4),
                arrowprops=dict(arrowstyle="-|>", color=C_ACCENT, lw=lw, mutation_scale=14, shrinkA=5, shrinkB=5))
    
    # Draw neighbor nodes
    rect_n = patches.Circle((nx, ny), 0.3, facecolor=C_BG_DARK, edgecolor=C_ACCENT, linewidth=1.5, zorder=4)
    ax.add_patch(rect_n)
    ax.text(nx, ny, name, color=C_TEXT_DARK, weight='bold', fontsize=FS_BODY - 1, ha='center', va='center')
    # Indicate diffused output rate callout
    ax.text(nx, ny - 0.5, r"$\eta \cdot A_{ij} \lambda_i$", color=C_ACCENT, fontsize=FS_BODY - 1, ha='center', va='center', weight='bold')

# Draw Central Node i (Large circle with thick border and hot red core)
rect_c = patches.Circle((cx, cy), 0.45, facecolor=C_BG_LIGHT, edgecolor=C_PRIMARY, linewidth=2.0, zorder=5)
ax.add_patch(rect_c)
inner_c = patches.Circle((cx, cy), 0.32, facecolor=C_HOTSPOT, alpha=0.9, zorder=6)
ax.add_patch(inner_c)
ax.text(cx, cy, "Zone i", color='white', weight='bold', fontsize=FS_HEADER - 1, ha='center', va='center', zorder=7)

# Equations Box inside Panel B
eq_w, eq_h = 5.8, 1.8
eq_x, eq_y = pb_x + 0.3, pb_y + 0.3
rect_eq = patches.FancyBboxPatch((eq_x, eq_y), eq_w, eq_h, boxstyle="round,pad=0.01", facecolor=C_BG_DARK, edgecolor=C_PRIMARY, linewidth=1.2, zorder=4)
ax.add_patch(rect_eq)

ax.text(eq_x + eq_w/2, eq_y + eq_h - 0.35, r"Governing Spatial Diffusion Formula", color=C_PRIMARY, weight='bold', fontsize=FS_BODY + 0.5, ha='center')
ax.text(eq_x + eq_w/2, eq_y + eq_h - 0.9, r"$\Lambda = (1 - \eta)\Lambda_{\mathrm{raw}} + \eta(A_{\mathrm{norm}} \Lambda_{\mathrm{raw}})$", color=C_PRIMARY, weight='bold', fontsize=FS_MATH + 1.0, ha='center')

param_txt = r"$\eta = 0.15$ (Spillover Coefficient)  |  $A_{\mathrm{norm}}$ = Row-Normalized KNN Matrix"
ax.text(eq_x + eq_w/2, eq_y + 0.5, param_txt, color=C_TEXT_MUTED, fontsize=FS_BODY - 1.5, ha='center')
ax.text(eq_x + eq_w/2, eq_y + 0.22, "15% of local complaint intensity is redistributed to neighbors.", color=C_ACCENT, weight='bold', fontsize=FS_BODY - 1.5, ha='center')


# ====================================================
# PANEL C: OPERATIONAL EFFECT OF SPILLOVERS
# ====================================================
pc_x, pc_y, pc_w, pc_h = 13.5, 0.8, 6.0, 7.2
draw_panel_border(ax, pc_x, pc_y, pc_w, pc_h, "C. Operational Effect of Spillovers")

# Draw three grids side-by-side representing stress transition maps
def draw_micro_grid(ax, gx, gy, w, stress_type):
    """Draws a 3x3 spatial grid representing localized stress."""
    cell_w = w / 3
    for r in range(3):
        for c in range(3):
            cx = gx - w/2 + c*cell_w + cell_w/2
            cy = gy - w/2 + r*cell_w + cell_w/2
            
            # Determine color based on stress type and cell position
            if r == 1 and c == 1:  # Center node
                color = C_HOTSPOT
            else:  # surrounding neighbors
                if stress_type == "without":
                    color = C_LOW_STRESS
                elif stress_type == "diffusion":
                    color = C_BG_LIGHT
                else:  # with spillover
                    color = C_MEDIUM_STRESS
                    
            rect = patches.Rectangle((cx - cell_w/2 + 0.02, cy - cell_w/2 + 0.02), cell_w - 0.04, cell_w - 0.04, 
                                     facecolor=color, edgecolor=C_LINE, linewidth=0.6)
            ax.add_patch(rect)
            
            # Add outward diffusion arrows on middle grid
            if stress_type == "diffusion" and (r != 1 or c != 1):
                # vector from center cell
                dx, dy = (c - 1) * 0.16, (r - 1) * 0.16
                ax.annotate('', xy=(cx, cy), xytext=(gx, gy),
                            arrowprops=dict(arrowstyle="-|>", color=C_ACCENT, lw=0.8, mutation_scale=8))

# Sub-grids locations
gy_offset = pb_y + 3.8
draw_micro_grid(ax, pc_x + 1.2, gy_offset, 1.4, "without")
draw_micro_grid(ax, pc_x + 3.0, gy_offset, 1.4, "diffusion")
draw_micro_grid(ax, pc_x + 4.8, gy_offset, 1.4, "with")

# Panel C Column Subheaders & Bullet Texts
# Left: Without Spillover
ax.text(pc_x + 1.2, gy_offset - 1.0, "Without Spillover", color=C_PRIMARY, weight='bold', fontsize=FS_BODY, ha='center')
wout_items = ["• Isolated Hotspot", "• Abrupt Boundaries", "• Poor Realism"]
for idx, item in enumerate(wout_items):
    ax.text(pc_x + 0.4, gy_offset - 1.4 - idx*0.3, item, color=C_TEXT_DARK, fontsize=FS_BODY - 1.5, ha='left')

# Middle: Diffusion Process
ax.text(pc_x + 3.0, gy_offset - 1.0, "Diffusion Process", color=C_ACCENT, weight='bold', fontsize=FS_BODY, ha='center')
diff_items = ["• Outward Flow", "• Adjacent Decay", "• KNN Propagation"]
for idx, item in enumerate(diff_items):
    ax.text(pc_x + 2.2, gy_offset - 1.4 - idx*0.3, item, color=C_TEXT_DARK, fontsize=FS_BODY - 1.5, ha='left')

# Right: With Spillover
ax.text(pc_x + 4.8, gy_offset - 1.0, "With Spillover", color=C_HOTSPOT, weight='bold', fontsize=FS_BODY, ha='center')
with_items = ["• Multi-Zone Stress", "• Smooth Transitions", "• Spatial Correlation"]
for idx, item in enumerate(with_items):
    ax.text(pc_x + 4.0, gy_offset - 1.4 - idx*0.3, item, color=C_TEXT_DARK, fontsize=FS_BODY - 1.5, ha='left')

# Linear connecting arrows between micro-grids
draw_arrow(ax, pc_x + 1.95, gy_offset, pc_x + 2.25, gy_offset)
draw_arrow(ax, pc_x + 3.75, gy_offset, pc_x + 4.05, gy_offset)

# Simple Stress Scale Colorbar Legend at bottom of Panel C
cb_y = pc_y + 0.3
cb_w = 4.8
cb_x = pc_x + 0.6
rect_cb = patches.Rectangle((cb_x, cb_y), cb_w, 0.25, facecolor='none', edgecolor=C_LINE, linewidth=0.8)
ax.add_patch(rect_cb)

# Draw colorbar segments
for i, color in enumerate([C_LOW_STRESS, C_MEDIUM_STRESS, C_HOTSPOT]):
    seg = patches.Rectangle((cb_x + i*(cb_w/3), cb_y), cb_w/3, 0.25, facecolor=color, edgecolor='none')
    ax.add_patch(seg)

ax.text(cb_x, cb_y - 0.22, "Low Stress", color=C_TEXT_MUTED, fontsize=FS_BODY - 2.0, ha='left', weight='bold')
ax.text(cb_x + cb_w/2, cb_y - 0.22, "Medium", color=C_TEXT_MUTED, fontsize=FS_BODY - 2.0, ha='center', weight='bold')
ax.text(cb_x + cb_w, cb_y - 0.22, "High Stress", color=C_TEXT_MUTED, fontsize=FS_BODY - 2.0, ha='right', weight='bold')


# ====================================================
# TITLE & CANVAS BOUNDARY EXPORTS
# ====================================================
plt.title("Figure 6. Graph-Based Spatial Spillover Mechanism", 
          fontsize=FS_TITLE, weight='bold', color=C_PRIMARY, y=0.98, family='sans-serif')

ax.set_xlim(0.0, 20.0)
ax.set_ylim(0.0, 8.5)
ax.axis('off')
plt.tight_layout()

# Save paths defined in paper/figures/ folder
project_root = r"c:\Users\utham\Desktop\final year project\project"
figures_dir = os.path.join(project_root, "paper", "figures")
os.makedirs(figures_dir, exist_ok=True)

svg_path = os.path.join(figures_dir, "Figure6_Spatial_Spillover_Mechanism.svg")
png_path = os.path.join(figures_dir, "Figure6_Spatial_Spillover_Mechanism.png")

# Save as vector SVG and high-resolution PNG
plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"Spillover Mechanism successfully saved to:\nSVG: {svg_path}\nPNG: {png_path}")
