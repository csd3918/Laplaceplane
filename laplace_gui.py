# ============================================================
# laplace_gui_clean.py
# ------------------------------------------------------------
# Interactive S-plane visualizer for poles and zeros of
# continuous-time LTI systems.
#
# Features:
#   - Interactive placement of poles and zeros in the S-plane.
#   - Real-time Bode magnitude and phase responses.
#   - Automatic logarithmic frequency range selection.
#   - Manual insertion of complex poles and zeros.
#   - Drag-and-drop movement of existing points.
#   - Optional automatic addition of conjugate points.
#   - Laplace Region of Convergence visualization.
#   - BIBO stability and Fourier-axis existence checks.
#   - Information panel with H(s) and system properties.
#   - Safe handling of poles on the imaginary axis.
#   - Three-dimensional visualization of |H(s)|.
#   - Undo, reset, save, and load functionality.
# ============================================================

import matplotlib

matplotlib.use('TkAgg')  # Use TkAgg backend for interactive GUI support
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from matplotlib.widgets import Button, TextBox, RadioButtons
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

status_message = [None]

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

# S-plane diagram (left center)
ax_plane = plt.axes([0.05, 0.15, 0.35, 0.60])

# Right-side panels: Bode magnitude, Bode phase, and system information
ax_mag = plt.axes([0.45, 0.62, 0.50, 0.15])
ax_phase = plt.axes([0.45, 0.40, 0.50, 0.15])
ax_info = plt.axes([0.45, 0.16, 0.50, 0.15])

# ============================================================
# Setup S-plane axes appearance
# ============================================================

# Configure axis limits, titles, and labels for S-plane
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
    ax.set_ylabel(ylabel)
    ax.grid(True)


# Apply consistent formatting to the Bode plots
style_mag_phase(ax_mag, "Bode Διάγραμμα Πλάτους", r"$20\log_{10}|H(j\omega)|$ [dB]")
style_mag_phase(ax_phase, "Bode Διάγραμμα Φάσης", r"$\angle H(j\omega)$ [deg]")
ax_phase.set_xlabel(r"$\omega$ [rad/s]")

# System information panel
ax_info.axis('off')
ax_info.set_title("Πληροφορίες Συστήματος", pad=6)

def get_imaginary_axis_poles(tolerance=1e-9):
    """
    Return poles located on or numerically very close
    to the imaginary axis Re(s) = 0.
    """
    return [
        pole
        for pole in poles
        if abs(pole.real) <= tolerance
    ]

def compute_frequency_range(num_points=2000):
    """
    Create an automatic logarithmic frequency range based on
    the current pole and zero locations.

    Two decades are displayed below the smallest characteristic
    frequency and above the largest characteristic frequency.
    """
    all_roots = poles + zeros

    characteristic_frequencies = [
        abs(root)
        for root in all_roots
        if abs(root) > 1e-9
    ]

    if not characteristic_frequencies:
        omega_min = 1e-2
        omega_max = 1e3
    else:
        smallest_frequency = min(characteristic_frequencies)
        largest_frequency = max(characteristic_frequencies)

        omega_min = smallest_frequency / 100.0
        omega_max = largest_frequency * 100.0

        # Avoid impractically small or large values
        omega_min = max(omega_min, 1e-6)
        omega_max = min(omega_max, 1e6)

        if omega_max <= omega_min:
            omega_min = 1e-2
            omega_max = 1e3

    return np.logspace(
        np.log10(omega_min),
        np.log10(omega_max),
        num_points
    )

def get_frequency_range_text():
    """
    Return the current automatic Bode frequency range.
    """
    omega = compute_frequency_range(num_points=2)

    return (
        f"Bode range: {omega[0]:.3g} to "
        f"{omega[-1]:.3g} rad/s"
    )

# ============================================================
# Function: compute_frequency_response()
# ============================================================
def compute_frequency_response():
    """
    Compute and display the frequency response H(jω) of the continuous-time system
    defined by the current poles and zeros.

    Steps:
        1. Build an automatic logarithmic frequency grid.
        2. Evaluate H(s) along the imaginary axis, where s = jω.
        3. Convert magnitude to decibels and phase to degrees.
        4. Redraw the Bode magnitude and phase plots.
    """
    # Evaluate the transfer function along the positive imaginary axis.
    omega = compute_frequency_range(num_points=2000)
    s = 1j * omega

    # Initialize numerator and denominator of H(s)
    numerator = np.ones_like(s, dtype=complex)
    denominator = np.ones_like(s, dtype=complex)

    # Multiply terms for zeros and poles
    for zero in zeros:
        numerator *= (s - zero)
    for pole in poles:
        denominator *= (s - pole)

    # Compute the frequency response
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        H = numerator / denominator
    H_abs = np.abs(H)

    with np.errstate(divide='ignore', invalid='ignore'):
        H_mag_db = 20 * np.log10(H_abs)

    H_phase_rad = np.unwrap(np.angle(H))
    H_phase_deg = np.degrees(H_phase_rad)

    # Create copies used only for safe plotting
    H_mag_plot = H_mag_db.copy()
    H_phase_plot = H_phase_deg.copy()

    # Replace infinite or invalid values with NaN
    # Matplotlib will leave a gap at singularities
    H_mag_plot[~np.isfinite(H_mag_plot)] = np.nan
    H_phase_plot[~np.isfinite(H_phase_plot)] = np.nan

    # Visual limits for the magnitude diagram
    MAG_MIN_DB = -120
    MAG_MAX_DB = 120

    H_mag_plot = np.clip(
        H_mag_plot,
        MAG_MIN_DB,
        MAG_MAX_DB
    )

    # ------------------------------------------------------------
    # Magnitude plot
    # ------------------------------------------------------------
    ax_mag.clear()
    ax_mag.semilogx(omega, H_mag_plot, color='red', linewidth=2)
    style_mag_phase(ax_mag, "Bode Διάγραμμα Πλάτους", r"$20\log_{10}|H(j\omega)|$ [dB]")

    # ------------------------------------------------------------
    # Phase plot
    # ------------------------------------------------------------
    ax_phase.clear()
    ax_phase.semilogx(omega, H_phase_plot, color='red', linewidth=2)
    style_mag_phase(ax_phase, "Bode Διάγραμμα Φάσης", r"$\angle H(j\omega)$ [deg]")
    ax_phase.set_xlabel(r"$\omega$ [rad/s]")

    #--------------------------------------------------------
    # Mark frequencies corresponding to poles on the jω axis
    #--------------------------------------------------------
    axis_frequencies = sorted({
        abs(pole.imag)
        for pole in get_imaginary_axis_poles()
        if omega[0] <= abs(pole.imag) <= omega[-1]
    })

    for pole_frequency in axis_frequencies:
        ax_mag.axvline(
            pole_frequency,
            color='darkorange',
            linestyle='--',
            linewidth=1.5,
            alpha=0.9
        )

        ax_phase.axvline(
            pole_frequency,
            color='darkorange',
            linestyle='--',
            linewidth=1.5,
            alpha=0.9
        )

# ============================================================
# Helper function: _compute_noncausal_strip()
# ============================================================
def _compute_noncausal_strip(sigmas):
    """
    Compute an appropriate ROC strip for the noncausal Laplace case.

    In the s-plane, a noncausal ROC is a vertical strip:
        sigma_in < Re(s) < sigma_out

    The strip must lie between two pole real parts.
    """
    if not sigmas or len(sigmas) < 2:
        return None

    sigmas = sorted(set(sigmas))

    if len(sigmas) < 2:
        return None

    # Find the largest gap between consecutive pole real parts
    gaps = [
        (sigmas[i], sigmas[i + 1], sigmas[i + 1] - sigmas[i])
        for i in range(len(sigmas) - 1)
    ]

    gaps.sort(key=lambda t: t[2], reverse=True)

    sigma_left, sigma_right, _ = gaps[0]

    return sigma_left, sigma_right

# ============================================================
# Function: draw_roc()
# ============================================================
def draw_roc(lim):
    """
    Visualize the Region of Convergence (ROC) in the Laplace S-plane.

    Depending on roc_mode, it highlights:
        - Causal:      Re(s) > max(Re(p_i))
        - Anticausal:  Re(s) < min(Re(p_i))
        - Noncausal:   sigma_left < Re(s) < sigma_right
        - None:        no shading

    Args:
        lim (float): Axis limit for consistent extent of the shading.
    """

    # Remove previous ROC visualization if it exists
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

    # No ROC selected
    if roc_mode[0] == 'none':
        return

    # ROC is defined based on poles
    if len(poles) == 0:
        return

    # Use the real parts of the poles, not their magnitudes
    sigmas = [p.real for p in poles]

    sigma_max = max(sigmas)
    sigma_min = min(sigmas)

    # Grid over the S-plane
    N = 400
    x = np.linspace(-lim, lim, N)
    y = np.linspace(-lim, lim, N)
    X, Y = np.meshgrid(x, y)

    # In the S-plane, X corresponds to Re(s)
    if roc_mode[0] == 'causal':
        mask = X > (sigma_max + 1e-6)
        roc_text = fr"ROC: Re(s) > {sigma_max:.3f}"

    elif roc_mode[0] == 'anticausal':
        mask = X < (sigma_min - 1e-6)
        roc_text = fr"ROC: Re(s) < {sigma_min:.3f}"

    else:
        # Noncausal ROC: vertical strip between two pole real parts
        strip = _compute_noncausal_strip(sigmas)

        if strip is None:
            return

        sigma_left, sigma_right = strip

        if sigma_right - sigma_left < 1e-6:
            return

        mask = (X > (sigma_left + 1e-6)) & (X < (sigma_right - 1e-6))
        roc_text = fr"ROC: {sigma_left:.3f} < Re(s) < {sigma_right:.3f}"

    # Draw semi-transparent ROC region
    roc_im[0] = ax_plane.imshow(
        mask.astype(float),
        extent=[-lim, lim, -lim, lim],
        origin='lower',
        alpha=0.15,
        cmap=cm.Greens,
        zorder=0
    )

    # Add ROC label
    roc_label[0] = ax_plane.text(
        0.02,
        0.02,
        roc_text,
        transform=ax_plane.transAxes,
        fontsize=10,
        color='green',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'),
        zorder=5
    )


# ============================================================
# Helper functions
# ============================================================

def format_complex(c):
    """
    Format a complex number for general display.
    """
    real = 0.0 if abs(c.real) < 1e-10 else c.real
    imag = 0.0 if abs(c.imag) < 1e-10 else c.imag

    return f"{real:.2f}{imag:+.2f}j"

def format_factor(root):
    """
    Format the factor (s - root) in a readable form.

    Examples:
        root = -1       -> (s + 1.00)
        root = 2        -> (s - 2.00)
        root = -1+2j    -> (s + 1.00 - 2.00j)
        root = 1-2j     -> (s - 1.00 + 2.00j)
    """
    real = 0.0 if abs(root.real) < 1e-10 else root.real
    imag = 0.0 if abs(root.imag) < 1e-10 else root.imag

    factor = "s"

    if real < 0:
        factor += f" + {abs(real):.2f}"
    elif real > 0:
        factor += f" - {real:.2f}"

    if imag < 0:
        factor += f" + {abs(imag):.2f}j"
    elif imag > 0:
        factor += f" - {imag:.2f}j"

    return f"({factor})"


def build_transfer_function_text():
    """
    Build a readable factored representation of H(s)
    from the current zeros and poles.
    """
    if len(zeros) == 0:
        numerator = "1"
    else:
        numerator = "".join(format_factor(z) for z in zeros)

    if len(poles) == 0:
        denominator = "1"
    else:
        denominator = "".join(format_factor(p) for p in poles)

    return f"H(s) = {numerator} / {denominator}"

def get_stability_text():
    """
    Determine BIBO stability using pole locations
    and the selected Laplace ROC.
    """
    tolerance = 1e-10

    # This conclusion is valid regardless of ROC selection
    if any(abs(p.real) < tolerance for p in poles):
        return "Stability: Unstable, pole on jω axis"

    if roc_mode[0] == 'none':
        return "Stability: Unknown, select ROC"

    if len(zeros) > len(poles):
        return "Stability: Unstable, improper H(s)"

    if len(poles) == 0:
        return "Stability: Stable"

    sigmas = [p.real for p in poles]

    if roc_mode[0] == 'causal':
        if max(sigmas) < 0:
            return "Stability: Stable"

        return "Stability: Unstable"

    if roc_mode[0] == 'anticausal':
        if min(sigmas) > 0:
            return "Stability: Stable"

        return "Stability: Unstable"

    if roc_mode[0] == 'noncausal':
        strip = _compute_noncausal_strip(sigmas)

        if strip is None:
            return "Stability: Undefined ROC"

        sigma_left, sigma_right = strip

        if sigma_left < 0 < sigma_right:
            return "Stability: Stable"

        return "Stability: Unstable"

    return "Stability: Unknown"


def get_roc_text():
    """
    Return a text description of the current Laplace ROC.
    """
    if roc_mode[0] == 'none':
        return "ROC: Not selected"

    if len(poles) == 0:
        return "ROC: Undefined, no poles"

    sigmas = [p.real for p in poles]

    if roc_mode[0] == 'causal':
        return f"ROC: Re(s) > {max(sigmas):.3f}"

    if roc_mode[0] == 'anticausal':
        return f"ROC: Re(s) < {min(sigmas):.3f}"

    strip = _compute_noncausal_strip(sigmas)

    if strip is None:
        return "ROC: Noncausal requires at least two distinct pole real parts"

    sigma_left, sigma_right = strip
    return f"ROC: {sigma_left:.3f} < Re(s) < {sigma_right:.3f}"


def get_fourier_axis_text():
    """
    Determine whether H(jω) exists as an ordinary
    frequency response.
    """
    tolerance = 1e-10

    # A pole on the imaginary axis makes H(jω)
    # undefined at the corresponding frequency
    if any(abs(p.real) < tolerance for p in poles):
        return "H(jω): Undefined, pole on jω axis"

    if roc_mode[0] == 'none':
        return "H(jω): Unknown, select ROC"

    if len(poles) == 0:
        return "H(jω): No finite poles on jω axis"

    sigmas = [p.real for p in poles]

    if roc_mode[0] == 'causal':
        if max(sigmas) < 0:
            return "H(jω): Exists, jω axis is in ROC"

        return "H(jω): jω axis is outside ROC"

    if roc_mode[0] == 'anti-causal':
        if min(sigmas) > 0:
            return "H(jω): Exists, jω axis is in ROC"

        return "H(jω): jω axis is outside ROC"

    if roc_mode[0] == 'noncausal':
        strip = _compute_noncausal_strip(sigmas)

        if strip is None:
            return "H(jω): Undefined ROC"

        sigma_left, sigma_right = strip

        if sigma_left < 0 < sigma_right:
            return "H(jω): Exists, jω axis is in ROC"

        return "H(jω): jω axis is outside ROC"

    return "H(jω): Unknown"


def draw_info_panel():
    """
    Display the transfer function and key system properties.
    """
    ax_info.clear()
    ax_info.axis('off')
    ax_info.set_title("Πληροφορίες Συστήματος", pad=6)

    if len(poles) > 0:
        rightmost_pole = max(poles, key=lambda p: p.real)
        rightmost_text = (
            f"Rightmost pole: {format_complex(rightmost_pole)}"
        )
    else:
        rightmost_text = "Rightmost pole: None"

    stability_text = get_stability_text()

    axis_poles = get_imaginary_axis_poles()

    if axis_poles:
        axis_poles_text = (
                "Warning: pole(s) on jω axis: "
                + ", ".join(
            format_complex(pole)
            for pole in axis_poles
        )
        )
    else:
        axis_poles_text = "No poles on jω axis"

    # Select indicator color according to stability result
    if stability_text == "Stability: Stable":
        stability_color = "green"
    elif "Unstable" in stability_text:
        stability_color = "red"
    else:
        stability_color = "gray"

    info_lines = [
        build_transfer_function_text(),
        "",
        f"Number of zeros: {len(zeros)}",
        f"Number of poles: {len(poles)}",
        rightmost_text,
        "",
        stability_text,
        get_roc_text(),
        get_fourier_axis_text(),
        axis_poles_text
    ]

    info_text = "\n".join(info_lines)

    ax_info.text(
        0.02,
        0.95,
        info_text,
        transform=ax_info.transAxes,
        fontsize=10,
        verticalalignment='top',
        family='monospace',
        color='black',
        bbox=dict(
            facecolor='white',
            alpha=0.85,
            edgecolor='lightgray'
        )
    )

    # Colored stability indicator
    ax_info.text(
        0.98,
        0.95,
        "●",
        transform=ax_info.transAxes,
        fontsize=16,
        verticalalignment='top',
        horizontalalignment='right',
        color=stability_color
    )



# ============================================================
# Function: update_plot()
# ============================================================
def update_plot():
    """
    Redraw the entire S-plane diagram, including:
        - Poles and zeros
        - ROC shading
        - Bode magnitude and phase responses
        - System information panel

    This function is called whenever user actions modify the system
    (adding/removing points, dragging, loading, etc.).
    """
    # Clear the S-plane axes
    ax_plane.cla()
    ax_plane.set_title("Διάγραμμα Πόλων-Μηδενικών στο χώρο Laplace", pad=8)

    # Determine axis limits dynamically based on all points
    all_points = zeros + poles
    if all_points:
        max_r = max(max(abs(p.real), abs(p.imag)) for p in all_points)
        lim = max(1.5, max_r + 0.5)
    else:
        lim = 1.5

    # Reapply grid, labels
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

    #Draw system information panel
    draw_info_panel()

    # Trigger a GUI refresh
    fig.canvas.draw_idle()



# ============================================================
# Function: plot_final_3d()
# ============================================================
def plot_final_3d(event):
    """
    Render a 3D surface visualization of the magnitude response |H(s)|
    over the complex Laplace S-plane.

    Poles appear as peaks and zeros appear as valleys on the magnitude surface.
    This helps visualize the geometric influence of poles and zeros in the
    Laplace domain.
    """
    LINES_STEP = 2
    POINT_STEP = 2

    # Define the grid for the complex S-plane
    N = 160
    sigma = np.linspace(-5, 5, N)
    omega = np.linspace(-5, 5, N)

    Sigma, Omega = np.meshgrid(sigma, omega)

    # Complex coordinates: s = sigma + j omega
    Sc = Sigma + 1j * Omega

    # Compute H(s) = numerator / denominator over the S-plane
    num = np.ones_like(Sc, dtype=complex)
    den = np.ones_like(Sc, dtype=complex)

    for zero in zeros:
        num *= (Sc - zero)

    for pole in poles:
        den *= (Sc - pole)

    # Avoid division warnings near poles
    with np.errstate(divide='ignore', invalid='ignore'):
        H = num / den
        Hmag = np.abs(H)

    # Clip magnitude for clearer visualization
    Hmag = np.nan_to_num(Hmag, nan=0.0, posinf=6.0, neginf=0.0)
    Hmag = np.clip(Hmag, 0, 6)

    # Create a new 3D figure
    fig3d = plt.figure(figsize=(10, 8))
    ax = fig3d.add_subplot(111, projection='3d')

    ax.set_title("Μέτρο της Συνάρτησης Μεταφοράς |H(s)|", fontsize=12)
    ax.set_xlabel("Πραγματικό μέρος σ = Re(s)")
    ax.set_ylabel("Φανταστικό μέρος ω = Im(s)")
    ax.set_zlabel("|H(s)|")

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 6)

    ax.view_init(elev=35, azim=130)
    ax.set_box_aspect([1, 1, 0.8])

    # Define color mapping for magnitude values
    norm = colors.Normalize(vmin=0, vmax=6)
    cmap = matplotlib.colormaps['jet']

    # Helper function to create colored 3D line segments
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

    # Draw grid lines along sigma direction
    xs_row = Sigma[::LINES_STEP, ::POINT_STEP]
    ys_row = Omega[::LINES_STEP, ::POINT_STEP]
    zs_row = Hmag[::LINES_STEP, ::POINT_STEP]
    ax.add_collection3d(collection_from_grid_lines(xs_row, ys_row, zs_row))

    # Draw grid lines along omega direction
    xs_col = Sigma[::POINT_STEP, ::LINES_STEP].T
    ys_col = Omega[::POINT_STEP, ::LINES_STEP].T
    zs_col = Hmag[::POINT_STEP, ::LINES_STEP].T
    ax.add_collection3d(collection_from_grid_lines(xs_col, ys_col, zs_col))

    plt.tight_layout()
    plt.show()


# ============================================================
# Mouse interaction helpers
# ============================================================

def show_status_message(message, color='darkorange'):
    """
    Display a temporary status message inside the S-plane.
    """
    if status_message[0] is not None:
        try:
            status_message[0].remove()
        except Exception:
            pass

    status_message[0] = ax_plane.text(
        0.5,
        0.98,
        message,
        transform=ax_plane.transAxes,
        horizontalalignment='center',
        verticalalignment='top',
        fontsize=10,
        color=color,
        bbox=dict(
            facecolor='white',
            edgecolor=color,
            alpha=0.9
        ),
        zorder=20
    )

    fig.canvas.draw_idle()

def on_mouse_press(event):
    """
    Handle mouse button presses on the S-plane.

    Priority:
        1. If the click is close to an existing pole or zero,
           start dragging that point.
        2. Otherwise, add a new point:
           - Left click: add zero
           - Right click: add pole

    Dragging is disabled when conjugate mode is active
    or when count_value is greater than 1.
    """
    global dragging_point

    # Ignore clicks outside the S-plane
    if (
        event.inaxes != ax_plane
        or event.xdata is None
        or event.ydata is None
    ):
        return

    click_position = complex(event.xdata, event.ydata)
    selection_threshold = 0.1

    # --------------------------------------------------------
    # First priority: detect an existing zero
    # --------------------------------------------------------
    for index, zero in enumerate(zeros):
        if abs(zero - click_position) < selection_threshold:

            # Prevent accidental insertion on an existing point
            if add_conjugate[0] or count_value[0] > 1:
                message = (
                    "Η μετακίνηση είναι απενεργοποιημένη όταν "
                    "Συζυγή: ON ή Πλήθος > 1."
                )

                print(message)
                show_status_message(message)

                return

            dragging_point = ('zero', index)

            drag_start_fixed[0] = zero.real
            drag_start_fixed[1] = zero.imag

            return

    # --------------------------------------------------------
    # Second priority: detect an existing pole
    # --------------------------------------------------------
    for index, pole in enumerate(poles):
        if abs(pole - click_position) < selection_threshold:

            # Prevent accidental insertion on an existing point
            if add_conjugate[0] or count_value[0] > 1:
                message = (
                    "Η μετακίνηση είναι απενεργοποιημένη όταν "
                    "Συζυγή: ON ή Πλήθος > 1"
                )

                print(message)
                show_status_message(message)

                return

            dragging_point = ('pole', index)

            drag_start_fixed[0] = pole.real
            drag_start_fixed[1] = pole.imag

            return

    # --------------------------------------------------------
    # No existing point was selected.
    # Add a new zero or pole as one grouped action.
    # --------------------------------------------------------
    new_point = click_position
    conjugate_point = new_point.conjugate()

    added_points = []

    # Left mouse button: add zeros
    if event.button == 1:
        for _ in range(count_value[0]):
            zeros.append(new_point)
            added_points.append(('zero', new_point))

            if (
                    add_conjugate[0]
                    and abs(new_point.imag) > 1e-12
            ):
                zeros.append(conjugate_point)
                added_points.append(
                    ('zero', conjugate_point)
                )

    # Right mouse button: add poles
    elif event.button == 3:
        for _ in range(count_value[0]):
            poles.append(new_point)
            added_points.append(('pole', new_point))

            if (
                    add_conjugate[0]
                    and abs(new_point.imag) > 1e-12
            ):
                poles.append(conjugate_point)
                added_points.append(
                    ('pole', conjugate_point)
                )

    # Ignore middle button and other buttons
    else:
        return

    # Store the entire click as one history action
    if added_points:
        history.append(
            ('add_group', added_points)
        )

    update_plot()

# ============================================================
# Undo last action
# ============================================================
def undo(event):
    """
    Undo the most recent user action.

    Supported actions:
        - Grouped addition of poles or zeros
        - Movement of a pole or zero
        - Legacy individual additions
    """
    if not history:
        print("Δεν υπάρχει ενέργεια για αναίρεση.")
        return

    last_action = history.pop()
    action_type = last_action[0]

    # --------------------------------------------------------
    # Undo grouped addition
    # Format:
    # ('add_group', [('zero', point), ('zero', point), ...])
    # --------------------------------------------------------
    if action_type == 'add_group':
        added_points = last_action[1]

        # Reverse order is safer when duplicate points exist
        for point_type, position in reversed(added_points):

            if point_type == 'zero':
                for index in range(len(zeros) - 1, -1, -1):
                    if zeros[index] == position:
                        zeros.pop(index)
                        break

            elif point_type == 'pole':
                for index in range(len(poles) - 1, -1, -1):
                    if poles[index] == position:
                        poles.pop(index)
                        break

    # --------------------------------------------------------
    # Undo movement
    # Format:
    # ('move', point_type, index, old_position, new_position)
    # --------------------------------------------------------
    elif action_type == 'move':
        _, point_type, index, old_position, new_position = last_action

        if point_type == 'zero' and index < len(zeros):
            zeros[index] = old_position

        elif point_type == 'pole' and index < len(poles):
            poles[index] = old_position

    # --------------------------------------------------------
    # Compatibility with old individual zero additions
    # --------------------------------------------------------
    elif action_type == 'zero':
        position = last_action[1]

        for index in range(len(zeros) - 1, -1, -1):
            if zeros[index] == position:
                zeros.pop(index)
                break

    # --------------------------------------------------------
    # Compatibility with old individual pole additions
    # --------------------------------------------------------
    elif action_type == 'pole':
        position = last_action[1]

        for index in range(len(poles) - 1, -1, -1):
            if poles[index] == position:
                poles.pop(index)
                break

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
    Save the current poles and zeros into 'splane_data.json'.
    Each is stored as a pair [real, imag].
    """
    data = {
        "zeros": [[z.real, z.imag] for z in zeros],
        "poles": [[p.real, p.imag] for p in poles]
    }
    with open("splane_data.json", "w") as f:
        json.dump(data, f)
    print("Αποθηκεύτηκε σε splane_data.json")


# ============================================================
# Load poles and zeros from JSON
# ============================================================
def load(event):
    """
    Load poles and zeros from 'splane_data.json'.

    The loaded configuration becomes the new initial state,
    so the undo history is cleared.
    """
    filename = "splane_data.json"

    if not os.path.exists(filename):
        print("Το αρχείο δεν υπάρχει.")
        return

    try:
        with open(filename, "r") as file:
            data = json.load(file)

        loaded_zeros = [
            complex(*value)
            for value in data.get("zeros", [])
        ]

        loaded_poles = [
            complex(*value)
            for value in data.get("poles", [])
        ]

    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(f"Αποτυχία φόρτωσης δεδομένων: {error}")
        return

    zeros.clear()
    poles.clear()
    history.clear()

    zeros.extend(loaded_zeros)
    poles.extend(loaded_poles)

    update_plot()

    print("Φορτώθηκαν δεδομένα από splane_data.json")
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
        print("Μη έγκυρη μορφή για σημείο s. Π.χ. -1+2j")


# ============================================================
# Add manually defined zero
# ============================================================
def add_manual_zero(event):
    """
    Add manually entered zeros as one grouped action.
    """
    zero = manual_value[0]
    added_points = []

    for _ in range(count_value[0]):
        zeros.append(zero)
        added_points.append(('zero', zero))

        if (
            add_conjugate[0]
            and abs(zero.imag) > 1e-12
        ):
            conjugate_zero = zero.conjugate()
            zeros.append(conjugate_zero)
            added_points.append(
                ('zero', conjugate_zero)
            )

    if added_points:
        history.append(
            ('add_group', added_points)
        )

    update_plot()

# ============================================================
# Add manually defined pole
# ============================================================
def add_manual_pole(event):
    """
    Add manually entered poles as one grouped action.
    """
    pole = manual_value[0]
    added_points = []

    for _ in range(count_value[0]):
        poles.append(pole)
        added_points.append(('pole', pole))

        if (
            add_conjugate[0]
            and abs(pole.imag) > 1e-12
        ):
            conjugate_pole = pole.conjugate()
            poles.append(conjugate_pole)
            added_points.append(
                ('pole', conjugate_pole)
            )

    if added_points:
        history.append(
            ('add_group', added_points)
        )

    update_plot()
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
    Finish dragging a pole or zero and store the movement
    as a separate undoable action.
    """
    global dragging_point

    if dragging_point is None:
        return

    typ, index = dragging_point

    old_position = complex(
        drag_start_fixed[0],
        drag_start_fixed[1]
    )

    if typ == 'zero' and index < len(zeros):
        new_position = zeros[index]

    elif typ == 'pole' and index < len(poles):
        new_position = poles[index]

    else:
        dragging_point = None
        drag_start_fixed[0] = None
        drag_start_fixed[1] = None
        return

    # Store the movement only if the position actually changed
    if abs(new_position - old_position) > 1e-12:
        history.append(
            (
                'move',
                typ,
                index,
                old_position,
                new_position
            )
        )

    dragging_point = None
    drag_start_fixed[0] = None
    drag_start_fixed[1] = None

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

btn_3d = Button(ax_3d, "3D Προβολή |H(s)|")
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
text_box = TextBox(ax_input, "s= ", initial="0.0+0.0j")
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

fig.canvas.mpl_connect(
    'button_press_event',
    on_mouse_press
)

fig.canvas.mpl_connect(
    'motion_notify_event',
    on_motion
)

fig.canvas.mpl_connect(
    'button_release_event',
    on_release
)


# ============================================================
# Initial Rendering
# ============================================================
update_plot()
plt.show()
