# ============================================================
# laplaceplane_gui.py
# ------------------------------------------------------------
# Interactive Z-plane visualizer for poles and zeros of discrete-time systems.
#
# Features:
#   - Interactive placement of poles and zeros on the Z-plane.
#   - Manual insertion via text input.
#   - Automatic computation of magnitude, phase, and group delay responses.
#   - Support for conjugate points, undo, save/load (JSON), and region of convergence (ROC).
#   - 3D visualization of |H(z)| over the complex Z-plane.
#   - Locking movement along real/imag axes, ROC shading, and multiple insertions per click.
#
# ============================================================

import matplotlib

matplotlib.use('TkAgg')  # Use TkAgg backend for interactive GUI support
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from matplotlib.widgets import Button, TextBox, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import colors

# ============================================================
# Global Variables
# ============================================================

# Lists to store the complex zeros and poles of the system
poles = []
zeros = []

# A stack (list) used to record user actions for undo functionality
history = []

# Value entered manually in the input box (as a list to keep reference)
manual_value = [0]

# Flag to determine if conjugate pairs should be added automatically
add_conjugate = [False]

# Number of poles/zeros to add per mouse click
count_value = [1]

# Locking modes for dragging points:
#   'none'  - move freely
#   'imag'  - lock real part (vertical movement)
#   'real'  - lock imaginary part (horizontal movement)
drag_lock_mode = ['none']

# Keeps track of the original position when a drag starts
drag_start_fixed = [None, None]

# Stores currently dragged point information (tuple: ('zero'/'pole', index))
dragging_point = None

# ROC mode - determines which region of convergence is highlighted:
#   'none', 'causal', 'anticausal', or 'noncausal'
roc_mode = ['none']

# References for ROC overlay objects (so they can be updated/removed later)
roc_im = [None]
roc_label = [None]

# ============================================================
# Figure and Axes Layout
# ============================================================

# Main figure for the GUI, widescreen layout
fig = plt.figure(figsize=(16, 9))

# ------------------------------------------------------------
# Define button and input box areas (absolute positions)
# ------------------------------------------------------------
# All coordinates are given as fractions of the figure size: [left, bottom, width, height]

# Upper row of controls
ax_undo = plt.axes([0.02, 0.95, 0.06, 0.04])
ax_save = plt.axes([0.1, 0.95, 0.06, 0.04])
ax_conj = plt.axes([0.18, 0.95, 0.09, 0.04])
ax_lock_imag = plt.axes([0.29, 0.95, 0.12, 0.04])
ax_count = plt.axes([0.77, 0.95, 0.06, 0.04])
ax_add_zero = plt.axes([0.84, 0.95, 0.14, 0.04])

# Second row of controls
ax_reset = plt.axes([0.02, 0.87, 0.06, 0.04])
ax_load = plt.axes([0.1, 0.87, 0.06, 0.04])
ax_3d = plt.axes([0.18, 0.87, 0.09, 0.04])
ax_lock_real = plt.axes([0.29, 0.87, 0.12, 0.04])
ax_input = plt.axes([0.77, 0.87, 0.06, 0.04])
ax_add_pole = plt.axes([0.84, 0.87, 0.14, 0.04])

# Radio buttons area for ROC (Region of Convergence)
ax_roc = plt.axes([0.43, 0.87, 0.09, 0.12])

# Z-plane diagram (left center)
ax_plane = plt.axes([0.05, 0.15, 0.35, 0.60])

# Right-side plots: magnitude, phase, and group delay responses
ax_mag = plt.axes([0.45, 0.62, 0.50, 0.15])
ax_phase = plt.axes([0.45, 0.40, 0.50, 0.15])
ax_gdelay = plt.axes([0.45, 0.16, 0.50, 0.15])

# ============================================================
# Setup Laplace-plane axes appearance
# ============================================================

# Precompute unit circle points for visualization of |z|=1 boundary
# theta = np.linspace(0, 2*np.pi, 300)
# unit_circle, = ax_plane.plot(np.cos(theta), np.sin(theta), 'b:', linewidth=1.5)

# Configure axis limits, titles, and labels for Z-plane
ax_plane.set_title("Διάγραμμα Πόλων-Μηδενικών", pad=8)
ax_plane.set_xlim(-1.5, 1.5)
ax_plane.set_ylim(-1.5, 1.5)
ax_plane.set_aspect('equal')
ax_plane.set_xlabel("Real Part")
ax_plane.set_ylabel("Imaginary Part")
ax_plane.grid(True)
ax_plane.axhline(0, color='dodgerblue', linestyle=':', linewidth=1.5)
ax_plane.axvline(0, color='dodgerblue', linestyle=':', linewidth=1.5)


# ============================================================
# Helper function: consistent styling for frequency plots
# ============================================================

def style_mag_phase(ax, title, ylabel):
    """
    Apply consistent formatting and tick labels for all frequency-domain plots.
    """
    ax.set_title(title, pad=6)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylabel(ylabel)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.grid(True)


# Apply formatting to magnitude, phase, and group delay subplots
style_mag_phase(ax_mag, "Απόκριση Πλάτους", r"$|H(e^{j\omega})|$")
style_mag_phase(ax_phase, "Απόκριση Φάσης", r"$\angle H(e^{j\omega})$")
ax_phase.set_xlabel(r"$\omega$")
style_mag_phase(ax_gdelay, "Καθυστέρηση Ομάδας", r"$\tau_g(\omega)$")
ax_gdelay.set_xlabel(r"$\omega$")


# ============================================================
# Function: compute_frequency_response()
# ============================================================
def compute_frequency_response():
    """
    Compute and display the frequency response H(e^{jω}) of the discrete-time system
    defined by the current poles and zeros.

    Steps:
        1. Evaluate H(z) = Π(z - zero) / Π(z - pole) over points on the unit circle.
        2. Compute magnitude, unwrapped phase, and numerical group delay.
        3. Clear and redraw the corresponding plots.
    """
    # Sample ω from -π to π for evaluation on the unit circle
    omega = np.logspace(-2, 3, 1000)
    s = 1j * omega  # s = jω frequency of continious time system's response

    # Initialize numerator and denominator of H(s)
    numerator = np.ones_like(s, dtype=complex)
    denominator = np.ones_like(s, dtype=complex)

    # Multiply terms for zeros and poles
    for zero in zeros:
        numerator *= (s - zero)
    for pole in poles:
        denominator *= (s - pole)

    # Compute the frequency response
    H = numerator / denominator
    H_mag = np.abs(H)
    H_phase = np.unwrap(np.angle(H))  # unwrap avoids phase discontinuities
    gdelay = -np.gradient(H_phase, omega)  # group delay = -dφ/dω

    # ------------------------------------------------------------
    # Magnitude plot
    # ------------------------------------------------------------
    ax_mag.clear()
    ax_mag.plot(omega, H_mag, color='red', linewidth=2)
    style_mag_phase(ax_mag, "Απόκριση Πλάτους", r"$|H(e^{j\omega})|$")

    # ------------------------------------------------------------
    # Phase plot
    # ------------------------------------------------------------
    ax_phase.clear()
    ax_phase.plot(omega, H_phase, color='red', linewidth=2)
    style_mag_phase(ax_phase, "Απόκριση Φάσης", r"$\angle H(e^{j\omega})$")
    ax_phase.set_xlabel(r"$\omega$")

    # ------------------------------------------------------------
    # Group delay plot
    # ------------------------------------------------------------
    ax_gdelay.clear()
    ax_gdelay.plot(omega, gdelay, color='red', linewidth=2)
    style_mag_phase(ax_gdelay, "Καθυστέρηση Ομάδας", r"$\tau_g(\omega)$")
    ax_gdelay.set_xlabel(r"$\omega$")


# ============================================================
# Helper function: _compute_noncausal_band()
# ============================================================
def _compute_noncausal_band(radii):
    """
    Compute an appropriate ROC band for the noncausal case, given pole radii.

    A "band" ROC exists between two circles |z| = r_in and |z| = r_out.
    Ideally, the band straddles the unit circle (|z|=1). If not possible,
    the function returns the largest available gap between pole magnitudes.

    Args:
        radii (list[float]): List of absolute values (radii) of poles.

    Returns:
        (r_in, r_out): Inner and outer radii of the ROC band, or None if invalid.
    """
    if not radii or len(radii) < 2:
        return None

    # Sort and deduplicate the pole radii
    radii = sorted(set(radii))

    # Separate poles inside and outside the unit circle
    inside = [r for r in radii if r < 1.0]
    outside = [r for r in radii if r > 1.0]

    # If both sides exist, pick the pair that straddles |z|=1
    if inside and outside:
        return max(inside), min(outside)

    # Otherwise, find the largest gap between consecutive poles
    gaps = [(radii[i], radii[i + 1], radii[i + 1] - radii[i]) for i in range(len(radii) - 1)]
    gaps.sort(key=lambda t: t[2], reverse=True)
    r_in, r_out, _ = gaps[0]
    return r_in, r_out


# ============================================================
# Function: draw_roc()
# ============================================================
def draw_roc(lim):
    """
    Visualize the Region of Convergence (ROC) in the Z-plane as a shaded area.

    Depending on `roc_mode`, it highlights:
        - Causal: |z| > max(|p_i|)
        - Anticausal: |z| < min(|p_i|)
        - Noncausal: r_in < |z| < r_out (band)
        - None: no shading

    Args:
        lim (float): Axis limit for consistent extent of the shading.
    """

    # Remove previous ROC visualization (if any)
    if roc_im[0] is not None:
        try:
            roc_im[0].remove()
        except Exception:
            pass
        roc_im[0] = None
    if roc_label[0] is not None:
        try:
            roc_label[0].remove()
        except Exception:
            pass
        roc_label[0] = None

    # No ROC mode selected
    if roc_mode[0] == 'none':
        return

    # No poles → ROC undefined
    if len(poles) == 0:
        return

    # Compute radii of all poles
    radii = [abs(p) for p in poles]
    r_max = max(radii)
    r_min = min(radii)

    # Prepare a grid of (x, y) points over the Z-plane
    N = 400
    x = np.linspace(-lim, lim, N)
    y = np.linspace(-lim, lim, N)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)  # |z| magnitude grid

    # Determine the shading mask and text label based on ROC mode
    if roc_mode[0] == 'causal':
        mask = R > (r_max + 1e-6)
        roc_text = fr"ROC: |z| > {r_max:.3f}"
    elif roc_mode[0] == 'anticausal':
        mask = R < (r_min - 1e-6)
        roc_text = fr"ROC: |z| < {r_min:.3f}"
    else:
        # Noncausal band case
        band = _compute_noncausal_band(radii)
        if band is None:
            return
        r_in, r_out = band
        if r_out - r_in < 1e-4:
            return
        mask = (R > (r_in + 1e-6)) & (R < (r_out - 1e-6))
        roc_text = fr"ROC: {r_in:.3f} < |z| < {r_out:.3f}"

    # Draw semi-transparent ROC region overlay
    roc_im[0] = ax_plane.imshow(
        mask.astype(float),
        extent=[-lim, lim, -lim, lim],
        origin='lower',
        alpha=0.15,
        cmap=cm.Greens,
        zorder=0
    )

    # Add ROC label text on top of Z-plane
    roc_label[0] = ax_plane.text(
        0.02, 0.02, roc_text,
        transform=ax_plane.transAxes,
        fontsize=10,
        color='green',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'),
        zorder=5
    )


# ============================================================
# Function: update_plot()
# ============================================================
def update_plot():
    """
    Redraw the entire Z-plane diagram, including:
        - Poles and zeros
        - Unit circle and grid
        - ROC shading
        - Frequency response (magnitude, phase, group delay)

    This function is called whenever user actions modify the system
    (adding/removing points, dragging, loading, etc.).
    """
    # Clear the Z-plane axes
    ax_plane.cla()
    ax_plane.set_title("Διάγραμμα Πόλων-Μηδενικών στο χώρο Laplace", pad=8)

    # Determine axis limits dynamically based on all points
    all_points = zeros + poles
    if all_points:
        max_r = max(max(abs(p.real), abs(p.imag)) for p in all_points)
        lim = max(1.5, max_r + 0.5)
    else:
        lim = 1.5

    # Reapply grid, labels, and unit circle
    ax_plane.set_xlim(-lim, lim)
    ax_plane.set_ylim(-lim, lim)
    ax_plane.set_aspect('equal')
    ax_plane.set_xlabel("Re(s)")
    ax_plane.set_ylabel("Im(s)")
    ax_plane.grid(True)
    ax_plane.axhline(0, color='dodgerblue', linestyle=':', linewidth=1.5)
    ax_plane.axvline(0, color='dodgerblue', linestyle=':', linewidth=1.5)

    # Plot zeros as blue circles
    for z in zeros:
        ax_plane.plot(z.real, z.imag, 'bo')
        ax_plane.text(z.real + 0.05, z.imag + 0.05,
                      f"Zero: {z.real:.2f}{z.imag:+.2f}j",
                      color='blue', fontsize=9)

    # Plot poles as red X marks
    for p in poles:
        ax_plane.plot(p.real, p.imag, 'rx')
        ax_plane.text(p.real + 0.05, p.imag + 0.05,
                      f"Pole: {p.real:.2f}{p.imag:+.2f}j",
                      color='red', fontsize=9)

    # Update the three frequency-domain plots
    compute_frequency_response()

    # Draw ROC region shading
    draw_roc(lim)

    # Trigger a GUI refresh
    fig.canvas.draw_idle()


# ============================================================
# Function: plot_final_3d()
# ============================================================
def plot_final_3d(event):
    """
    Render a 3D surface visualization of the magnitude response |H(z)|
    over the complex Z-plane. This helps users understand how poles and zeros
    shape the magnitude of H(z) outside the unit circle.

    Triggered when the "Τελική 3D Προβολή" button is clicked.
    """
    LINES_STEP = 2  # Grid sampling rate for drawn surface lines
    POINT_STEP = 2

    # Define the grid for the complex Z-plane
    N = 160
    x = np.linspace(-2, 2, N)
    y = np.linspace(-2, 2, N)
    X, Y = np.meshgrid(x, y)
    Zc = X + 1j * Y  # Complex coordinates for each grid point

    # Compute H(z) = numerator / denominator on this grid
    num = np.ones_like(Zc, dtype=complex)
    den = np.ones_like(Zc, dtype=complex)
    for z in zeros:
        num *= (Zc - z)
    for p in poles:
        den *= (Zc - p)
    H = num / den
    Hmag = np.clip(np.abs(H), 0, 6)  # Limit magnitude for clearer visualization

    # Create a new 3D figure for the surface plot
    fig3d = plt.figure(figsize=(10, 8))
    ax = fig3d.add_subplot(111, projection='3d')
    ax.set_title("Μέτρο του Μετασχ. Ζ |H(z)|", fontsize=12)
    ax.set_xlabel("Πραγματικό μέρος (Re[z])")
    ax.set_ylabel("Φανταστικό μέρος (Im[z])")
    ax.set_zlabel("|H(z)|")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 6)
    ax.view_init(elev=35, azim=130)
    ax.set_box_aspect([1, 1, 0.8])

    # Define color mapping for magnitude values
    norm = colors.Normalize(vmin=0, vmax=6)
    cmap = matplotlib.colormaps['jet']

    # Helper function to create colored 3D line segments (for mesh visualization)
    def collection_from_grid_lines(xs, ys, zs, lw=0.8, alpha=0.95):
        segs, vals = [], []
        n_lines, n_pts = xs.shape
        for i in range(n_lines):
            for j in range(n_pts - 1):
                p0 = (xs[i, j], ys[i, j], zs[i, j])
                p1 = (xs[i, j + 1], ys[i, j + 1], zs[i, j + 1])
                segs.append([p0, p1])
                vals.append(0.5 * (zs[i, j] + zs[i, j + 1]))
        lc = Line3DCollection(segs, linewidth=lw, alpha=alpha)
        lc.set_cmap(cmap)
        lc.set_norm(norm)
        lc.set_array(np.asarray(vals))
        return lc

    # Draw the grid lines for both directions (rows and columns)
    xs_row = X[::LINES_STEP, ::POINT_STEP]
    ys_row = Y[::LINES_STEP, ::POINT_STEP]
    zs_row = Hmag[::LINES_STEP, ::POINT_STEP]
    ax.add_collection3d(collection_from_grid_lines(xs_row, ys_row, zs_row))

    xs_col = X[::POINT_STEP, ::LINES_STEP].T
    ys_col = Y[::POINT_STEP, ::LINES_STEP].T
    zs_col = Hmag[::POINT_STEP, ::LINES_STEP].T
    ax.add_collection3d(collection_from_grid_lines(xs_col, ys_col, zs_col))

    plt.tight_layout()
    plt.show()


# ============================================================
# Mouse Event: onclick()
# ============================================================
def onclick(event):
    """
    Handles mouse clicks on the Z-plane.

    Left-click  → add zeros
    Right-click → add poles

    Supports:
        - Multiple points per click (count_value)
        - Automatic conjugate addition (if enabled)
        - Avoids overlapping points
    """
    if event.inaxes != ax_plane or event.xdata is None or event.ydata is None:
        return

    # Prevent overlapping identical points if adding single ones
    if (not add_conjugate[0] and count_value[0] == 1):
        click = complex(event.xdata, event.ydata)
        thresh = 0.12
        for z in zeros:
            if abs(z - click) < thresh:
                return
        for p in poles:
            if abs(p - click) < thresh:
                return

    # Convert click position to complex number
    z_point = complex(event.xdata, event.ydata)
    conj_point = z_point.conjugate()

    # Left button: add zeros
    if event.button == 1:
        for _ in range(count_value[0]):
            zeros.append(z_point)
            history.append(('zero', z_point))
            if add_conjugate[0] and abs(z_point.imag) > 1e-12:
                zeros.append(conj_point)
                history.append(('zero', conj_point))

    # Right button: add poles
    elif event.button == 3:
        for _ in range(count_value[0]):
            poles.append(z_point)
            history.append(('pole', z_point))
            if add_conjugate[0] and abs(z_point.imag) > 1e-12:
                poles.append(conj_point)
                history.append(('pole', conj_point))

    update_plot()


# ============================================================
# Undo last action
# ============================================================
def undo(event):
    """
    Undo the most recent addition of a pole or zero.
    Removes it from both the plot and the history stack.
    """
    if not history:
        return
    last = history.pop()
    if last[0] == 'zero' and last[1] in zeros:
        zeros.remove(last[1])
    elif last[0] == 'pole' and last[1] in poles:
        poles.remove(last[1])
    update_plot()


# ============================================================
# Reset everything
# ============================================================
def reset(event):
    """
    Clear all poles, zeros, and history — resets the diagram completely.
    """
    zeros.clear()
    poles.clear()
    history.clear()
    update_plot()


# ============================================================
# Save current poles and zeros to a JSON file
# ============================================================
def save(event):
    """
    Save the current poles and zeros into 'zplane_data.json'.
    Each is stored as a pair [real, imag].
    """
    data = {
        "zeros": [[z.real, z.imag] for z in zeros],
        "poles": [[p.real, p.imag] for p in poles]
    }
    with open("zplane_data.json", "w") as f:
        json.dump(data, f)
    print("Αποθηκεύτηκε σε zplane_data.json")


# ============================================================
# Load poles and zeros from JSON
# ============================================================
def load(event):
    """
    Load poles and zeros from 'zplane_data.json', replacing the current diagram.
    Automatically refreshes the plot.
    """
    if not os.path.exists("zplane_data.json"):
        print("Το αρχείο δεν υπάρχει.")
        return
    with open("zplane_data.json", "r") as f:
        data = json.load(f)
    zeros.clear()
    poles.clear()
    history.clear()
    for z in data.get("zeros", []):
        zeros.append(complex(*z))
        history.append(('zero', complex(*z)))
    for p in data.get("poles", []):
        poles.append(complex(*p))
        history.append(('pole', complex(*p)))
    update_plot()
    print("Φορτώθηκαν δεδομένα από zplane_data.json")


# ============================================================
# Manual input submission handler
# ============================================================
def manual_submit(text):
    """
    Parse and store a manually entered complex number from the input box.
    Example: '0.5+0.5j'
    """
    try:
        manual_value[0] = complex(text.strip())
        print(f"Εισήχθη: {manual_value[0]}")
    except Exception:
        print("Μη έγκυρη μορφή. Π.χ. 0.5+0.5j")


# ============================================================
# Add manually defined zero
# ============================================================
def add_manual_zero(event):
    """
    Add the manually entered zero (and its conjugate if enabled) to the diagram.
    """
    z = manual_value[0]
    for _ in range(count_value[0]):
        zeros.append(z)
        history.append(('zero', z))
        if add_conjugate[0] and abs(z.imag) > 1e-12:
            zeros.append(z.conjugate())
            history.append(('zero', z.conjugate()))
    update_plot()


# ============================================================
# Add manually defined pole
# ============================================================
def add_manual_pole(event):
    """
    Add the manually entered pole (and its conjugate if enabled) to the diagram.
    """
    p = manual_value[0]
    for _ in range(count_value[0]):
        poles.append(p)
        history.append(('pole', p))
        if add_conjugate[0] and abs(p.imag) > 1e-12:
            poles.append(p.conjugate())
            history.append(('pole', p.conjugate()))
    update_plot()


# ============================================================
# Mouse Event: on_press()
# ============================================================
def on_press(event):
    """
    Detect when the user clicks on a zero or pole to start dragging it.

    Notes:
        - Dragging is disabled when 'add_conjugate' is ON or multiple points
          (count_value > 1) are active, to prevent inconsistencies.
        - Records which point is being dragged and its starting position.
    """
    global dragging_point
    if add_conjugate[0] or count_value[0] > 1:
        print('Η μετακίνηση απενεργοποιημένη όταν Συζυγή:ON ή Πλήθος>1 (τα κλειδώματα επιτρέπονται)')
        return
    if event.inaxes != ax_plane or event.xdata is None or event.ydata is None:
        return
    click = complex(event.xdata, event.ydata)

    # Check if the click is near a zero
    for i, z in enumerate(zeros):
        if abs(z - click) < 0.1:
            dragging_point = ('zero', i)
            drag_start_fixed[0], drag_start_fixed[1] = z.real, z.imag
            return

    # Check if the click is near a pole
    for i, p in enumerate(poles):
        if abs(p - click) < 0.1:
            dragging_point = ('pole', i)
            drag_start_fixed[0], drag_start_fixed[1] = p.real, p.imag
            return


# ============================================================
# Mouse Event: on_motion()
# ============================================================
def on_motion(event):
    """
    Update the position of a zero or pole during mouse dragging.

    Supports lock modes:
        - 'imag': fix real part (vertical drag)
        - 'real': fix imaginary part (horizontal drag)
        - 'none': free movement

    Updates both the diagram and the frequency response in real time.
    """
    global dragging_point
    if dragging_point and event.inaxes == ax_plane and event.xdata is not None and event.ydata is not None:
        typ, i = dragging_point
        if drag_lock_mode[0] == 'imag':
            new_real = 0.0
            new_imag = event.ydata
        elif drag_lock_mode[0] == 'real':
            new_real = event.xdata
            new_imag = 0.0
        else:
            new_real = event.xdata
            new_imag = event.ydata

        # Update the corresponding element
        if typ == 'zero' and i < len(zeros):
            zeros[i] = complex(new_real, new_imag)
        elif typ == 'pole' and i < len(poles):
            poles[i] = complex(new_real, new_imag)
        update_plot()


# ============================================================
# Mouse Event: on_release()
# ============================================================
def on_release(event):
    """
    Called when the mouse button is released after dragging a point.
    Finalizes the move and adds it to the history stack.
    """
    global dragging_point
    if dragging_point is not None:
        typ, i = dragging_point
        if typ == 'zero' and i < len(zeros):
            history.append(('zero', zeros[i]))
        elif typ == 'pole' and i < len(poles):
            history.append(('pole', poles[i]))
    dragging_point = None


# ============================================================
# GUI Buttons Setup
# ============================================================

# Undo button
btn_undo = Button(ax_undo, "Undo")
btn_undo.on_clicked(undo)

# Save button
btn_save = Button(ax_save, "Save")
btn_save.on_clicked(save)

# Conjugate toggle button
btn_conj = Button(ax_conj, "Συζυγή: OFF")


def toggle_conj(event):
    """
    Toggle automatic addition of conjugate points.
    """
    add_conjugate[0] = not add_conjugate[0]
    btn_conj.label.set_text("Συζυγή: ON" if add_conjugate[0] else "Συζυγή: OFF")


btn_conj.on_clicked(toggle_conj)

# ============================================================
# Lock movement along Real or Imaginary axes
# ============================================================

btn_lock_imag = Button(ax_lock_imag, "Φανταστικό: OFF")
btn_lock_real = Button(ax_lock_real, "Πραγματικό: OFF")


def _refresh_lock_labels():
    """
    Update the lock button labels to reflect current mode.
    """
    btn_lock_imag.label.set_text("Φανταστικό: ON" if drag_lock_mode[0] == 'imag' else "Φανταστικό: OFF")
    btn_lock_real.label.set_text("Πραγματικό: ON" if drag_lock_mode[0] == 'real' else "Πραγματικό: OFF")


def toggle_lock_imag(event):
    """
    Toggle locking along the imaginary axis (vertical motion only).
    """
    drag_lock_mode[0] = 'none' if drag_lock_mode[0] == 'imag' else 'imag'
    _refresh_lock_labels()


def toggle_lock_real(event):
    """
    Toggle locking along the real axis (horizontal motion only).
    """
    drag_lock_mode[0] = 'none' if drag_lock_mode[0] == 'real' else 'real'
    _refresh_lock_labels()


btn_lock_imag.on_clicked(toggle_lock_imag)
btn_lock_real.on_clicked(toggle_lock_real)

# ============================================================
# Reset, Load, and 3D buttons
# ============================================================

btn_reset = Button(ax_reset, "Reset")
btn_reset.on_clicked(reset)

btn_load = Button(ax_load, "Load")
btn_load.on_clicked(load)

btn_3d = Button(ax_3d, "Τελική 3D Προβολή")
btn_3d.on_clicked(plot_final_3d)

# ============================================================
# Text boxes for manual input and count value
# ============================================================

# Text box: number of points to add per click
text_count = TextBox(ax_count, "Πλήθος= ", initial="1")


def submit_count(text):
    """
    Change how many poles/zeros are added with a single click.
    """
    try:
        val = int(text.strip())
        if val < 1:
            val = 1
        count_value[0] = val
        print(f"Θα προστεθούν {val} σημεία ανά κλικ.")
    except Exception:
        print("Δώσε ακέραιο ≥1")


text_count.on_submit(submit_count)

# Text box: manual complex number input (e.g., 0.5+0.5j)
text_box = TextBox(ax_input, "z= ", initial="0.0+0.0j")
text_box.on_submit(manual_submit)

# ============================================================
# Buttons for manually adding poles and zeros
# ============================================================
btn_add_zero = Button(ax_add_zero, "Προσθήκη Μηδενικού")
btn_add_zero.on_clicked(add_manual_zero)

btn_add_pole = Button(ax_add_pole, "Προσθήκη Πόλου")
btn_add_pole.on_clicked(add_manual_pole)

# ============================================================
# Radio Buttons for ROC mode selection
# ============================================================

radio_roc = RadioButtons(ax_roc, ('Κενό', 'Αιτιατό', 'Μη αιτιατό', 'Αντιαιτιατό'))


def on_roc_change(label):
    """
    Handle changes in ROC mode selection.
    Updates ROC visualization immediately.
    """
    if 'Κενό' in label:
        roc_mode[0] = 'none'
    elif 'Αντιαιτιατό' in label:
        roc_mode[0] = 'anticausal'
    elif 'Μη' in label:
        roc_mode[0] = 'noncausal'
    else:
        roc_mode[0] = 'causal'
    update_plot()


radio_roc.on_clicked(on_roc_change)

# ============================================================
# Initialize Lock Labels
# ============================================================
_refresh_lock_labels()

# ============================================================
# Connect Mouse Event Handlers
# ============================================================

# Button press → add or start dragging
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('button_press_event', on_press)

# Mouse move → update dragging
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# Button release → finish dragging
fig.canvas.mpl_connect('button_release_event', on_release)

# ============================================================
# Initial Rendering
# ============================================================
update_plot()
plt.show()
