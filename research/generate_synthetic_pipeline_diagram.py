import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Initialize figure with professional vertical journal dimensions
fig, ax = plt.subplots(figsize=(11.5, 20.5), dpi=300)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Strict Academic Color Palette (Dark Blue, Gray, and Teal)
C_PRIMARY = '#0F2C59'    # Deep Journal Blue (Primary headers & borders)
C_ACCENT = '#1A5F7A'     # Teal (Sub-block headers & graphical elements)
C_BG_DARK = '#F8F9FA'    # Light Architectural Gray (Main block backgrounds)
C_BG_LIGHT = '#FFFFFF'   # Pure White (Inner block backgrounds)
C_TEXT_DARK = '#212529'  # Off-Black (High-contrast body text)
C_TEXT_MUTED = '#495057' # Slate Gray (Annotations & arrows)
C_LINE = '#868E96'       # Clean Gray (Connectors & brackets)

# Typography Scaling
FS_TITLE = 13
FS_HEADER = 10.5
FS_BODY = 9.0

# Central Alignment Axis
X_CENTER = 6.0

def draw_block(ax, x, y, w, h, title, items=None, bg_color=C_BG_DARK, border_color=C_PRIMARY, is_subblock=False):
    """Draws a clean, publication-grade block with robust internal margins."""
    # Base block
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", facecolor=bg_color, edgecolor=border_color, linewidth=1.3)
    ax.add_patch(rect)
    
    # Header banner
    header_h = 0.42
    header_bg = C_ACCENT if is_subblock else border_color
    header_rect = patches.FancyBboxPatch((x, y + h - header_h), w, header_h, boxstyle="round,pad=0.01", facecolor=header_bg, edgecolor=header_bg)
    ax.add_patch(header_rect)
    
    # Header text
    ax.text(x + w/2, y + h - (header_h/2), title, color='white', weight='bold', fontsize=FS_HEADER, ha='center', va='center')
    
    # Bullet text items with safe padding margins
    if items:
        start_y = y + h - header_h - 0.22
        for item in items:
            ax.text(x + 0.18, start_y, f"•  {item}", color=C_TEXT_DARK, fontsize=FS_BODY, ha='left', va='center')
            start_y -= 0.25

def draw_arrow(ax, x1, y1, x2, y2, text=None):
    """Draws a clean, standard vector arrow between processing stages."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_LINE, lw=1.6, mutation_scale=12))
    if text:
        ax.text((x1 + x2)/2 + 0.15, ((y1 + y2)/2), text, color=C_TEXT_MUTED, fontsize=FS_BODY - 1, ha='left', va='center', weight='bold')

# ====================================================
# VERTICAL PIPELINE LAYOUT TRACKING (STAGE 1 TO 7)
# ====================================================

# --- STAGE 1: HISTORICAL MUNICIPAL DATASET ---
s1_w = 5.6
s1_x = X_CENTER - s1_w / 2
draw_block(ax, s1_x, 19.3, s1_w, 1.6, "Historical Municipal Dataset", 
           ["16,071 Complaints", "Spatial Coordinates (Lat/Lon Bounds)", "Complaint Categories & Backlog Statuses"])

# --- STAGE 2: PRIOR LEARNING ENGINE (Stateful Container) ---
s2_w = 9.2
s2_x = X_CENTER - s2_w / 2
s2_box = patches.FancyBboxPatch((s2_x, 13.0), s2_w, 5.7, boxstyle="round,pad=0.01", facecolor=C_BG_DARK, edgecolor=C_PRIMARY, linewidth=1.5)
ax.add_patch(s2_box)
s2_header = patches.FancyBboxPatch((s2_x, 18.28), s2_w, 0.42, boxstyle="round,pad=0.01", facecolor=C_PRIMARY, edgecolor=C_PRIMARY)
ax.add_patch(s2_header)
ax.text(s2_x + s2_w/2, 18.49, "Prior Learning Engine", color='white', weight='bold', fontsize=FS_HEADER, ha='center', va='center')

# Sub-blocks inside Prior Learning Engine (Layed out side-by-side horizontally inside the container)
draw_block(ax, s2_x+0.15, 13.2, 2.7, 4.7, "A. Spatial Prior Learning", ["Zone Hotspots", "Coordinate Centroids", "Zone Probabilities", "Gaussian Spread (std)"], bg_color=C_BG_LIGHT, border_color=C_ACCENT, is_subblock=True)
draw_block(ax, s2_x+3.00, 13.2, 2.7, 4.7, "B. Temporal Prior Learning", ["Diurnal Patterns (24h)", "Weekly Patterns (7d)", "Seasonal Patterns (12m)", "Base Arrival Rates"], bg_color=C_BG_LIGHT, border_color=C_ACCENT, is_subblock=True)
draw_block(ax, s2_x+5.85, 13.2, 2.7, 4.7, "C. Category Prior Learning", ["Complaint Categories", "P(Category | Zone)", "Resolution Probability", "P(Open | Category)"], bg_color=C_BG_LIGHT, border_color=C_ACCENT, is_subblock=True)


# --- STAGE 3: ENVIRONMENTAL CONDITIONING ---
s3_w = 6.2
s3_x = X_CENTER - s3_w / 2
s3_box = patches.FancyBboxPatch((s3_x, 8.6), s3_w, 3.8, boxstyle="round,pad=0.01", facecolor=C_BG_DARK, edgecolor=C_PRIMARY, linewidth=1.5)
ax.add_patch(s3_box)
s3_header = patches.FancyBboxPatch((s3_x, 11.98), s3_w, 0.42, boxstyle="round,pad=0.01", facecolor=C_PRIMARY, edgecolor=C_PRIMARY)
ax.add_patch(s3_header)
ax.text(s3_x + s3_w/2, 12.19, "Environmental Conditioning", color='white', weight='bold', fontsize=FS_HEADER, ha='center', va='center')

# Sub-blocks inside Environmental Conditioning
draw_block(ax, s3_x+0.15, 8.8, 2.8, 2.8, "Weather Conditioning", ["Temperature Slopes", "Rainfall Multipliers", "Relative Humidity", "Monsoon Backlog Decays"], bg_color=C_BG_LIGHT, border_color=C_ACCENT, is_subblock=True)
draw_block(ax, s3_x+3.15, 8.8, 2.8, 2.8, "Festival Conditioning", ["Festival Holiday Surges", "Eve Amplifications", "Calendar Synchronizer", "Surge = 1.30x / 1.15x"], bg_color=C_BG_LIGHT, border_color=C_ACCENT, is_subblock=True)


# --- STAGE 4: PROBABILISTIC INCIDENT GENERATOR ---
s4_w = 5.6
s4_x = X_CENTER - s4_w / 2
draw_block(ax, s4_x, 5.6, s4_w, 2.4, r"Probabilistic Incident Generator $\lambda(t, z)$", 
           ["Poisson Rates $\lambda_{t,z} = \\text{Base} \\times P(z) \\times M_{\\text{seas}} \\times M_{\\text{wea}} \\times M_{\\text{fest}}$", 
            "Backlog Recurrence Boost (Duplicate reporting rate scaling)",
            "Poisson Event Sampling: $\\text{Counts} \\sim \\text{Poisson}(\\lambda)$"])


# --- STAGE 5: GRAPH SPILLOVER LAYER ---
s5_w = 5.6
s5_x = X_CENTER - s5_w / 2
draw_block(ax, s5_x, 3.7, s5_w, 1.3, "Graph Spillover Layer", 
           ["Inverse Centroid Distance Graph ($k$-NN, $k=3$)", "Spillover Propagation Formula: $(1-\\eta)\\Lambda + \\eta(A_{\\text{norm}}\\Lambda)$", "Diffusion Coefficient $\\eta = 15\\%$"])


# --- STAGE 6: SYNTHETIC COMPLAINT GENERATOR ---
s6_w = 5.6
s6_x = X_CENTER - s6_w / 2
draw_block(ax, s6_x, 2.0, s6_w, 1.3, "Synthetic Complaint Generator", 
           ["Gaussian Coordinates Jittering", "Uniform Window Offsets", "Metadata lookup Hydration"])


# --- FINAL STAGE: SYNTHETIC URBAN DATASET ---
s7_w = 5.6
s7_x = X_CENTER - s7_w / 2
draw_block(ax, s7_x, 0.5, s7_w, 1.1, "Synthetic Urban Dataset", 
           ["611,879 Complaint Records (2019-2026)", "Wasserstein Realism Correlation ($>98\\%$)"])

# ====================================================
# LINEAR PIPELINE ROUTING CONNECTIONS
# ====================================================
# Vertical flow links directly centered on X_CENTER
draw_arrow(ax, X_CENTER, 19.3, X_CENTER, 18.7, "Extract")
draw_arrow(ax, X_CENTER, 13.0, X_CENTER, 12.4, "Learn")
draw_arrow(ax, X_CENTER, 8.6, X_CENTER, 8.0, "Condition")
draw_arrow(ax, X_CENTER, 5.6, X_CENTER, 5.0, "Generate")
draw_arrow(ax, X_CENTER, 3.7, X_CENTER, 3.3, "Diffuse")
draw_arrow(ax, X_CENTER, 2.0, X_CENTER, 1.6, "Synthesize")

# ====================================================
# CANVAS FORMATTING & CAPTIONING
# ====================================================
plt.title("Figure 4. Synthetic Urban Complaint Generation Pipeline", 
          fontsize=FS_TITLE, weight='bold', color=C_PRIMARY, y=0.98, family='sans-serif')

ax.set_xlim(0, 12.0)
ax.set_ylim(0.0, 22.0)
ax.axis('off')
plt.tight_layout()

# Save publication vector file formats directly
project_root = r"c:\Users\utham\Desktop\final year project\project"
images_dir = os.path.join(project_root, "images")
os.makedirs(images_dir, exist_ok=True)

pdf_path = os.path.join(images_dir, "synthetic_generation_pipeline.pdf")
png_path = os.path.join(images_dir, "synthetic_generation_pipeline.png")

plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"Diagram successfully saved to:\nPDF: {pdf_path}\nPNG: {png_path}")
