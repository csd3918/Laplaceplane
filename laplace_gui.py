# ============================================================
# laplace_gui.py
# ------------------------------------------------------------
# Interactive S-plane visualizer for poles and zeros of
# continuous-time LTI systems.
#
# Features:
#   - Interactive placement of poles and zeros in the S-plane.
#   - Real-time magnitude and phase spectra:
#       linear scale over negative and positive frequencies,
#       or Bode diagram (selected with a check box).
#   - Automatic frequency range selection.
#   - Manual insertion of complex poles and zeros.
#   - Drag-and-drop movement of existing points.
#   - Deletion of a point (middle click or Shift + left click).
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
import sys
from matplotlib.widgets import Button, TextBox, RadioButtons, CheckButtons
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

# Check box area for the spectrum scale (linear / Bode)
ax_spectrum = plt.axes([0.54, 0.87, 0.13, 0.12])

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
ax_plane.set_xlabel("Re(s)")
ax_plane.set_ylabel("Im(s)")
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


# ============================================================
# Spectrum display mode
# ------------------------------------------------------------
#   'linear': |H(jω)| and phase over negative AND positive ω,
#             linear frequency axis (default)
#   'bode'  : 20log10|H(jω)| and phase over positive ω,
#             logarithmic frequency axis
# ============================================================
spectrum_mode = ['linear']


def apply_spectrum_axes():
    """
    Set axis scale, titles and labels of the two spectrum plots
    according to the selected spectrum mode.
    """
    if spectrum_mode[0] == 'bode':
        ax_mag.set_xscale('log')
        ax_phase.set_xscale('log')
        style_mag_phase(ax_mag, "Bode Διάγραμμα Πλάτους",
                        r"$20\log_{10}|H(j\omega)|$ [dB]")
        style_mag_phase(ax_phase, "Bode Διάγραμμα Φάσης",
                        r"$\angle H(j\omega)$ [deg]")
    else:
        ax_mag.set_xscale('linear')
        ax_phase.set_xscale('linear')
        style_mag_phase(ax_mag, "Φάσμα Πλάτους", r"$|H(j\omega)|$")
        style_mag_phase(ax_phase, "Φάσμα Φάσης",
                        r"$\angle H(j\omega)$ [deg]")

    ax_phase.set_xlabel(r"$\omega$ [rad/s]")


# System information panel
ax_info.axis('off')
ax_info.set_title("Πληροφορίες Συστήματος", pad=6)

# ============================================================
# Persistent plot objects
# ------------------------------------------------------------
# These objects are created once. On every change only their
# data is updated (set_data / set_text), instead of clearing and
# rebuilding the axes. Rebuilding the axes on every mouse
# movement made dragging lag behind the cursor.
# ============================================================

# Pole and zero markers in the S-plane
zeros_line, = ax_plane.plot([], [], 'bo', linestyle='none', zorder=4)
poles_line, = ax_plane.plot([], [], 'rx', linestyle='none', zorder=4)

# Text labels next to every pole and zero
point_labels = []

# Magnitude and phase curves
mag_line, = ax_mag.plot([], [], color='red', linewidth=2)
phase_line, = ax_phase.plot([], [], color='red', linewidth=2)

# Vertical markers at the frequencies of poles on the jω axis
axis_pole_markers = []

# Information panel text and colored stability indicator
info_text_artist = ax_info.text(
    0.02,
    0.95,
    "",
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

stability_indicator = ax_info.text(
    0.98,
    0.95,
    "●",
    transform=ax_info.transAxes,
    fontsize=16,
    verticalalignment='top',
    horizontalalignment='right',
    color='gray'
)

apply_spectrum_axes()


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

def compute_frequency_range(num_points=2000, bounds=None):
    """
    Create an automatic logarithmic frequency range (Bode mode)
    based on the current pole and zero locations.

    Two decades are displayed below the smallest characteristic
    frequency and above the largest characteristic frequency.

    bounds=(omega_min, omega_max) overrides the automatic range
    (used while dragging, when the axis shows a given range).
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

    if bounds is not None:
        omega_min, omega_max = bounds

    omega = np.logspace(
        np.log10(omega_min),
        np.log10(omega_max),
        num_points
    )

    # Add the exact frequencies |Im(root)| to the grid, so that sharp
    # resonance peaks (poles very close to the jω axis) or notches are
    # not missed between two logarithmic samples.
    if num_points > 2:
        root_frequencies = [
            abs(root.imag)
            for root in all_roots
            if omega_min <= abs(root.imag) <= omega_max
        ]
        omega = np.union1d(omega, root_frequencies)

    return omega


def compute_linear_limit():
    """
    Return W for the linear spectrum range [-W, W].

    W is twice the largest root magnitude, so that every
    resonance and notch is inside the plot with some margin.
    """
    all_roots = poles + zeros

    if not all_roots:
        return 10.0

    W = 2.0 * max(abs(root) for root in all_roots)

    return min(max(W, 1.0), 1e6)


def compute_linear_range(num_points=4001, W=None):
    """
    Create a symmetric linear frequency grid [-W, W] (linear mode),
    which includes negative frequencies.

    ω = 0 and the exact imaginary parts of all roots are added to
    the grid, for the same reason as in the Bode range.

    W overrides the automatic limit (used while dragging).
    """
    if W is None:
        W = compute_linear_limit()

    omega = np.linspace(-W, W, num_points)

    extra = [0.0] + [
        root.imag
        for root in poles + zeros
        if -W <= root.imag <= W
    ]

    return np.union1d(omega, extra)


def get_frequency_range_text():
    """
    Return the current automatic frequency range of the spectrum plots.
    """
    if spectrum_mode[0] == 'bode':
        omega = compute_frequency_range(num_points=2)

        return (
            f"Bode range: {omega[0]:.3g} to "
            f"{omega[-1]:.3g} rad/s"
        )

    W = compute_linear_limit()

    return f"Spectrum range: -{W:.3g} to {W:.3g} rad/s (linear)"


def evaluate_H(s):
    """
    Evaluate H(s) = Π(s - z_k) / Π(s - p_l) at the points s.
    Division by zero (s exactly on a pole) gives inf / nan
    without warnings.
    """
    numerator = np.ones_like(s, dtype=complex)
    denominator = np.ones_like(s, dtype=complex)

    for zero in zeros:
        numerator *= (s - zero)
    for pole in poles:
        denominator *= (s - pole)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return numerator / denominator


# ============================================================
# Function: compute_frequency_response()
# ============================================================
def compute_frequency_response(freeze_limits=False):
    """
    Compute and display the frequency response H(jω) of the continuous-time system
    defined by the current poles and zeros.

    Steps:
        1. Build the frequency grid of the selected mode:
             linear: symmetric linear grid [-W, W]
             bode:   positive logarithmic grid
        2. Evaluate H(s) along the imaginary axis, where s = jω.
        3. Magnitude: |H(jω)| (linear) or 20log10|H(jω)| in dB (Bode).
           Phase: unwrapped, in degrees.
        4. Update the data of the existing magnitude and phase curves.
    """
    is_bode = spectrum_mode[0] == 'bode'

    if freeze_limits:
        # While dragging, the grid covers both the automatic range and
        # the range currently shown, so the curve always fills the axis
        view_min, view_max = ax_mag.get_xlim()

        if is_bode:
            auto = compute_frequency_range(num_points=2)
            bounds = (min(auto[0], view_min), max(auto[-1], view_max))
            omega = compute_frequency_range(num_points=2000, bounds=bounds)
        else:
            W = max(compute_linear_limit(), view_max)
            omega = compute_linear_range(num_points=4001, W=W)
    elif is_bode:
        omega = compute_frequency_range(num_points=2000)
    else:
        omega = compute_linear_range(num_points=4001)

    H = evaluate_H(1j * omega)
    H_abs = np.abs(H)

    # Unwrap only the finite samples: a single NaN (e.g. a pole exactly
    # on a grid frequency) would otherwise propagate through np.unwrap
    # and erase the rest of the phase curve.
    H_phase_rad = np.full(H.shape, np.nan)
    finite = np.isfinite(H)
    if np.any(finite):
        unwrapped = np.unwrap(np.angle(H[finite]))

        # np.unwrap starts from the first sample, so the whole curve may be
        # shifted by a multiple of 360°. Anchor it at the sample closest to
        # ω = 0, where the phase must lie in (-180°, 180°].
        # (In Bode mode this is simply the first, lowest frequency.)
        k = np.argmin(np.abs(omega[finite]))
        wrapped = np.angle(H[finite][k])
        unwrapped -= 2 * np.pi * np.round((unwrapped[k] - wrapped) / (2 * np.pi))

        H_phase_rad[finite] = unwrapped
    H_phase_plot = np.degrees(H_phase_rad)

    if is_bode:
        with np.errstate(divide='ignore', invalid='ignore'):
            H_mag_plot = 20 * np.log10(H_abs)

        # Infinite / invalid values -> NaN (gap in the curve),
        # then visual limits for the dB axis
        H_mag_plot[~np.isfinite(H_mag_plot)] = np.nan
        H_mag_plot = np.clip(H_mag_plot, -120, 120)
    else:
        H_mag_plot = H_abs.copy()
        H_mag_plot[~np.isfinite(H_mag_plot)] = np.nan

    H_phase_plot[~np.isfinite(H_phase_plot)] = np.nan

    # ------------------------------------------------------------
    # Update the existing curves
    # ------------------------------------------------------------
    mag_line.set_data(omega, H_mag_plot)
    phase_line.set_data(omega, H_phase_plot)

    # While dragging, the limits are adapted separately, only when
    # needed (see adapt_limits_during_drag below)
    if not freeze_limits:
        for ax in (ax_mag, ax_phase):
            ax.relim()
            ax.set_autoscaley_on(True)
            ax.autoscale_view()
            # auto=None keeps the autoscale setting unchanged
            ax.set_xlim(omega[0], omega[-1], auto=None)

        if not is_bode:
            # A magnitude is never negative
            ax_mag.set_ylim(bottom=0, auto=None)

    # ------------------------------------------------------------
    # Mark frequencies corresponding to poles on the jω axis
    # ------------------------------------------------------------
    for marker in axis_pole_markers:
        marker.remove()
    axis_pole_markers.clear()

    if is_bode:
        # Only positive frequencies exist on a logarithmic axis
        axis_frequencies = {
            abs(pole.imag)
            for pole in get_imaginary_axis_poles()
            if omega[0] <= abs(pole.imag) <= omega[-1]
        }
    else:
        # Signed frequency: a pole at +j2 alone affects only ω = +2
        axis_frequencies = {
            pole.imag
            for pole in get_imaginary_axis_poles()
            if omega[0] <= pole.imag <= omega[-1]
        }

    for pole_frequency in sorted(axis_frequencies):
        for ax in (ax_mag, ax_phase):
            axis_pole_markers.append(
                ax.axvline(
                    pole_frequency,
                    color='darkorange',
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.9
                )
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

    The ROC is always a vertical half-plane or strip, so it is drawn
    as a shaded vertical band (axvspan) between two Re(s) values.

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

    if roc_mode[0] == 'causal':
        left, right = sigma_max, lim
        roc_text = fr"ROC: Re(s) > {sigma_max:.3f}"

    elif roc_mode[0] == 'anticausal':
        left, right = -lim, sigma_min
        roc_text = fr"ROC: Re(s) < {sigma_min:.3f}"

    else:
        # Noncausal ROC: vertical strip between two pole real parts
        strip = _compute_noncausal_strip(sigmas)

        if strip is None:
            return

        sigma_left, sigma_right = strip

        if sigma_right - sigma_left < 1e-6:
            return

        left, right = sigma_left, sigma_right
        roc_text = fr"ROC: {sigma_left:.3f} < Re(s) < {sigma_right:.3f}"

    # Draw semi-transparent ROC region
    roc_im[0] = ax_plane.axvspan(
        left,
        right,
        facecolor='#00441b',
        edgecolor='none',
        alpha=0.15,
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

    # Brackets avoid ambiguity, e.g. (s + 2) / [(s + 1)(s + 5)]
    if len(zeros) > 1:
        numerator = f"[{numerator}]"
    if len(poles) > 1:
        denominator = f"[{denominator}]"

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

    if roc_mode[0] == 'anticausal':
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
    Only the text and color of the existing objects are updated.
    """
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
        axis_poles_text,
        get_frequency_range_text()
    ]

    info_text_artist.set_text("\n".join(info_lines))

    # Colored stability indicator
    stability_indicator.set_color(stability_color)


# ============================================================
# Function: update_plot()
# ============================================================
def refresh_display(freeze_limits=False):
    """
    Update the data of all plot objects after any change of the system:
        - Poles and zeros in the S-plane
        - ROC shading
        - Magnitude and phase spectra
        - System information panel

    Only the data of the existing plot objects is updated; the axes are
    never rebuilt. With freeze_limits=True (used while dragging) the axis
    limits stay unchanged, so that the static background of the figure
    can be reused (blitting).
    """
    # Remove a temporary status message, if any
    if status_message[0] is not None:
        try:
            status_message[0].remove()
        except Exception:
            pass
        status_message[0] = None

    if freeze_limits:
        lim = ax_plane.get_xlim()[1]
    else:
        # Determine axis limits dynamically based on all points
        all_points = zeros + poles
        if all_points:
            max_r = max(max(abs(p.real), abs(p.imag)) for p in all_points)
            lim = max(1.5, max_r + 0.5)
        else:
            lim = 1.5

        ax_plane.set_xlim(-lim, lim)
        ax_plane.set_ylim(-lim, lim)

    # Zeros as blue circles, poles as red X marks
    zeros_line.set_data([z.real for z in zeros], [z.imag for z in zeros])
    poles_line.set_data([p.real for p in poles], [p.imag for p in poles])

    # Rebuild the text labels (only a few small text objects)
    for label in point_labels:
        label.remove()
    point_labels.clear()

    for z in zeros:
        point_labels.append(
            ax_plane.text(z.real + 0.05, z.imag + 0.05,
                          f"Zero: {z.real:.2f}{z.imag:+.2f}j",
                          color='blue', fontsize=9)
        )

    for p in poles:
        point_labels.append(
            ax_plane.text(p.real + 0.05, p.imag + 0.05,
                          f"Pole: {p.real:.2f}{p.imag:+.2f}j",
                          color='red', fontsize=9)
        )

    # Update the frequency-domain plots
    compute_frequency_response(freeze_limits)

    # Draw ROC region shading
    draw_roc(lim)

    # Update system information panel
    draw_info_panel()

    # Objects created above must follow the current drawing mode
    for artist in dynamic_artists():
        artist.set_animated(drag_render['active'])


def update_plot():
    """
    Refresh the whole display after a change of the system
    (adding/removing points, undo, load, ROC change, etc.).
    """
    refresh_display(freeze_limits=False)

    # Ask for a redraw; repeated requests are merged into one
    fig.canvas.draw_idle()


# ============================================================
# Fast redrawing while dragging (blitting)
# ------------------------------------------------------------
# A full redraw of the figure (axes, ticks, buttons, text boxes)
# is too slow to follow the mouse. When a drag starts, the static
# part of the figure is rendered once and stored. On every mouse
# movement only the objects that change are drawn on top of the
# stored background. When the moving point needs different axis
# limits, the limits are changed and the background is rendered
# again (see adapt_limits_during_drag). When the mouse button is
# released, a normal full update gives the exact final limits.
# ============================================================
drag_render = {
    'active': False,      # blitting mode on/off
    'background': None,   # stored static part of the figure
    'pending': False,     # a redraw is already scheduled
    'timer': None,        # single-shot timer that merges mouse events
    'full_redraws': 0     # number of limit changes during drags
}


def dynamic_artists():
    """
    Return every plot object whose content changes while dragging,
    in drawing order.
    """
    artists = [
        roc_im[0],
        zeros_line,
        poles_line,
        mag_line,
        phase_line,
        roc_label[0],
        info_text_artist,
        stability_indicator,
    ] + point_labels + axis_pole_markers

    artists = [a for a in artists if a is not None]

    return sorted(artists, key=lambda a: a.get_zorder())


def start_drag_rendering():
    """
    Enter blitting mode: render the static background once.
    """
    drag_render['active'] = True

    for artist in dynamic_artists():
        artist.set_animated(True)

    # Full draw without the animated objects -> static background
    fig.canvas.draw()
    drag_render['background'] = fig.canvas.copy_from_bbox(fig.bbox)

    blit_dynamic_artists()


def _visible_values(line, ax):
    """
    Finite y-values of a curve inside the visible x-range of its axes.
    """
    x, y = line.get_data()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_min, x_max = ax.get_xlim()
    keep = np.isfinite(y) & (x >= x_min) & (x <= x_max)
    return y[keep]


def _adapted_ylim(ax, values, magnitude_from_zero=False):
    """
    New y-limits for ax, or None if the current ones are still good.

    Hysteresis: the limits change only when the curve leaves the axis
    or uses less than about half of it. New limits get a small margin,
    so that small movements do not trigger a new change at once.
    """
    if values.size == 0:
        return None

    low, high = float(values.min()), float(values.max())
    current_low, current_high = ax.get_ylim()

    if magnitude_from_zero:
        if high > current_high or high < 0.6 * current_high:
            return (0.0, high * 1.4 if high > 0 else 1.0)
        return None

    span = max(high - low, 1.0)

    if (
        low < current_low
        or high > current_high
        or (high - low) < 0.5 * (current_high - current_low)
    ):
        return (low - 0.15 * span, high + 0.15 * span)

    return None


def adapt_limits_during_drag():
    """
    Adapt the axis limits while a point is being dragged.

    The axes grow or shrink with the moving pole / zero, like in a normal
    update, but only when it is really needed (hysteresis), because each
    change requires one full redraw of the figure. Between changes the
    fast blitting is used.

    Returns True if any limit changed.
    """
    changed = False
    is_bode = spectrum_mode[0] == 'bode'

    # ------------------------------------------------------------
    # S-plane: same rule as update_plot(), with extra room when growing
    # ------------------------------------------------------------
    all_points = zeros + poles
    if all_points:
        max_r = max(max(abs(p.real), abs(p.imag)) for p in all_points)
        target = max(1.5, max_r + 0.5)
    else:
        target = 1.5

    current = ax_plane.get_xlim()[1]

    if target > current or target < 0.75 * current:
        new_lim = target * 1.3 if target > current else target
        ax_plane.set_xlim(-new_lim, new_lim)
        ax_plane.set_ylim(-new_lim, new_lim)
        changed = True

    # ------------------------------------------------------------
    # Frequency axis of the spectra
    # ------------------------------------------------------------
    view_min, view_max = ax_mag.get_xlim()
    new_x = None

    if is_bode:
        auto = compute_frequency_range(num_points=2)
        target_min, target_max = auto[0], auto[-1]
        log = np.log10

        if (
            target_min < view_min
            or target_max > view_max
            or log(target_min) - log(view_min) > 0.5
            or log(view_max) - log(target_max) > 0.5
        ):
            # A quarter of a decade of extra room when growing
            new_min = target_min / 10 ** 0.25 if target_min < view_min else target_min
            new_max = target_max * 10 ** 0.25 if target_max > view_max else target_max
            new_x = (new_min, new_max)
    else:
        W = compute_linear_limit()

        if W > view_max or W < 0.75 * view_max:
            W_new = W * 1.3 if W > view_max else W
            new_x = (-W_new, W_new)

    if new_x is not None:
        for ax in (ax_mag, ax_phase):
            ax.set_xlim(*new_x, auto=None)
        changed = True

        # The frequency grid depends on the shown range
        compute_frequency_response(freeze_limits=True)

    # ------------------------------------------------------------
    # Value axes of the spectra
    # ------------------------------------------------------------
    new_y = _adapted_ylim(
        ax_mag,
        _visible_values(mag_line, ax_mag),
        magnitude_from_zero=not is_bode
    )
    if new_y is not None:
        ax_mag.set_ylim(*new_y, auto=None)
        changed = True

    new_y = _adapted_ylim(ax_phase, _visible_values(phase_line, ax_phase))
    if new_y is not None:
        ax_phase.set_ylim(*new_y, auto=None)
        changed = True

    return changed


def blit_dynamic_artists():
    """
    Draw only the changing objects on top of the stored background.
    If the axis limits had to change, the background is rendered again
    first (one full redraw), so the new limits and ticks appear.
    """
    drag_render['pending'] = False

    if not drag_render['active'] or drag_render['background'] is None:
        return

    refresh_display(freeze_limits=True)

    canvas = fig.canvas

    if adapt_limits_during_drag():
        # The ROC band and labels depend on the S-plane limits
        refresh_display(freeze_limits=True)

        # Full draw without the animated objects -> new background
        canvas.draw()
        drag_render['background'] = canvas.copy_from_bbox(fig.bbox)
        drag_render['full_redraws'] += 1

    canvas.restore_region(drag_render['background'])

    for artist in dynamic_artists():
        artist.axes.draw_artist(artist)

    canvas.blit(fig.bbox)


def request_drag_redraw():
    """
    Schedule one redraw for the latest mouse position.

    Mouse movements that arrive while a redraw is pending are merged
    into that single redraw, so the work never piles up behind the
    cursor.
    """
    if drag_render['pending']:
        return

    if drag_render['timer'] is None:
        drag_render['timer'] = fig.canvas.new_timer(interval=1)
        drag_render['timer'].single_shot = True
        drag_render['timer'].add_callback(blit_dynamic_artists)

    drag_render['pending'] = True
    drag_render['timer'].start()


def end_drag_rendering():
    """
    Leave blitting mode and do a normal full update, which also
    adapts the axis limits to the new positions.
    """
    if drag_render['timer'] is not None:
        drag_render['timer'].stop()

    drag_render['active'] = False
    drag_render['pending'] = False
    drag_render['background'] = None

    for artist in dynamic_artists():
        artist.set_animated(False)

    update_plot()


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

    # Define the grid for the complex S-plane.
    # At least [-5, 5], extended so that every root is visible.
    all_points = zeros + poles
    R = 5.0
    if all_points:
        R = max(R, max(max(abs(p.real), abs(p.imag)) for p in all_points) + 1.0)

    N = 160
    sigma = np.linspace(-R, R, N)
    omega = np.linspace(-R, R, N)

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

    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
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
        fontsize=11,
        color=color,
        bbox=dict(
            facecolor='white',
            edgecolor=color,
            alpha=0.9
        ),
        zorder=20
    )

    fig.canvas.draw_idle()

def delete_nearest_point(position, threshold):
    """
    Remove the zero or pole closest to `position`, if it lies
    within `threshold`. The deletion is stored in the history
    together with the original index, so Undo restores the point
    at exactly the same place in its list.
    """
    candidates = (
        [('zero', i, z) for i, z in enumerate(zeros)]
        + [('pole', i, p) for i, p in enumerate(poles)]
    )

    if not candidates:
        return

    point_type, index, value = min(
        candidates,
        key=lambda c: abs(c[2] - position)
    )

    if abs(value - position) >= threshold:
        return

    if point_type == 'zero':
        zeros.pop(index)
    else:
        poles.pop(index)

    history.append(('delete', point_type, index, value))
    update_plot()


def on_mouse_press(event):
    """
    Handle mouse button presses on the S-plane.

    Priority:
        0. Middle click or Shift + left click on an existing
           point deletes it.
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
    # Deletion: middle click, or Shift + left click,
    # on an existing point removes it (undoable).
    # --------------------------------------------------------
    is_delete_click = (
        event.button == 2
        or (event.button == 1 and event.key == 'shift')
    )

    if is_delete_click:
        delete_nearest_point(click_position, selection_threshold)
        return

    # --------------------------------------------------------
    # First priority: detect an existing zero
    # --------------------------------------------------------
    for index, zero in enumerate(zeros):
        if abs(zero - click_position) < selection_threshold:

            # Prevent accidental insertion on an existing point
            if add_conjugate[0] or read_count() > 1:
                message = (
                    "Η μετακίνηση είναι απενεργοποιημένη όταν "
                    "Συζυγή: ON ή Πλήθος > 1"
                )

                print(message)
                show_status_message(message)

                return

            dragging_point = ('zero', index)

            drag_start_fixed[0] = zero.real
            drag_start_fixed[1] = zero.imag

            start_drag_rendering()

            return

    # --------------------------------------------------------
    # Second priority: detect an existing pole
    # --------------------------------------------------------
    for index, pole in enumerate(poles):
        if abs(pole - click_position) < selection_threshold:

            # Prevent accidental insertion on an existing point
            if add_conjugate[0] or read_count() > 1:
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

            start_drag_rendering()

            return

    # --------------------------------------------------------
    # No existing point was selected.
    # Add a new zero or pole as one grouped action.
    # --------------------------------------------------------
    new_point = click_position
    conjugate_point = new_point.conjugate()

    added_points = []

    count = read_count()

    # Left mouse button: add zeros
    if event.button == 1:
        for _ in range(count):
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
        for _ in range(count):
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
        - Deletion of a pole or zero
        - Legacy individual additions
    """
    if not history:
        notify("Δεν υπάρχει ενέργεια για αναίρεση.")
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
    # Undo deletion
    # Format:
    # ('delete', point_type, index, position)
    # --------------------------------------------------------
    elif action_type == 'delete':
        _, point_type, index, position = last_action

        if point_type == 'zero':
            zeros.insert(index, position)
        else:
            poles.insert(index, position)

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
def data_file_path():
    """
    Location of 'splane_data.json'.

    When the program runs as a Windows executable (built with
    PyInstaller, sys.frozen is set), the file is kept next to the
    .exe. When it runs as a Python script, the current working
    directory is used, as before.
    """
    if getattr(sys, 'frozen', False):
        folder = os.path.dirname(sys.executable)
    else:
        folder = os.getcwd()

    return os.path.join(folder, "splane_data.json")


def notify(message, color='darkorange'):
    """
    Show a message both in the console and inside the window.
    The executable has no console, so the in-window message is the
    only feedback the user sees there.
    """
    print(message)
    show_status_message(message, color)


def save(event):
    """
    Save the current poles and zeros into 'splane_data.json'.
    Each is stored as a pair [real, imag].
    """
    data = {
        "zeros": [[z.real, z.imag] for z in zeros],
        "poles": [[p.real, p.imag] for p in poles]
    }
    filename = data_file_path()

    try:
        with open(filename, "w") as f:
            json.dump(data, f)
    except OSError as error:
        notify(f"Αποτυχία αποθήκευσης: {error}", color='red')
        return

    notify("Αποθηκεύτηκε σε splane_data.json", color='green')


# ============================================================
# Load poles and zeros from JSON
# ============================================================
def load(event):
    """
    Load poles and zeros from 'splane_data.json'.

    The loaded configuration becomes the new initial state,
    so the undo history is cleared.
    """
    filename = data_file_path()

    if not os.path.exists(filename):
        notify("Το αρχείο splane_data.json δεν υπάρχει.", color='red')
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
        notify(f"Αποτυχία φόρτωσης δεδομένων: {error}", color='red')
        return

    zeros.clear()
    poles.clear()
    history.clear()

    zeros.extend(loaded_zeros)
    poles.extend(loaded_poles)

    update_plot()

    notify("Φορτώθηκαν δεδομένα από splane_data.json", color='green')
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
        notify("Μη έγκυρη μορφή για σημείο s. Π.χ. -1+2j", color='red')


def read_manual_value():
    """
    Read the complex value currently typed in the s= box.

    The box is read at the moment a button is pressed, so the value
    is used even if the user did not press Enter first.
    Returns None and shows a message if the text is not valid.
    """
    try:
        manual_value[0] = complex(text_box.text.strip().replace(' ', ''))
        return manual_value[0]
    except ValueError:
        message = "Μη έγκυρη μορφή για σημείο s. Π.χ. -1+2j"
        print(message)
        show_status_message(message, color='red')
        return None


def read_count():
    """
    Read the number typed in the Πλήθος= box (integer >= 1).
    Falls back to the last valid value if the text is not valid.
    """
    try:
        count_value[0] = max(1, int(text_count.text.strip()))
    except ValueError:
        text_count.set_val(str(count_value[0]))
    return count_value[0]


# ============================================================
# Add manually defined zero
# ============================================================
def add_manual_zero(event):
    """
    Add manually entered zeros as one grouped action.
    """
    zero = read_manual_value()
    if zero is None:
        return
    added_points = []

    for _ in range(read_count()):
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
    pole = read_manual_value()
    if pole is None:
        return
    added_points = []

    for _ in range(read_count()):
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
    The handler itself only stores the new position; drawing is done
    by request_drag_redraw(), which merges fast mouse movements.
    """
    global dragging_point
    if dragging_point and event.inaxes == ax_plane and event.xdata is not None and event.ydata is not None:
        typ, i = dragging_point
        # Locked coordinates keep the value they had when the drag started
        if drag_lock_mode[0] == 'imag':
            new_real = drag_start_fixed[0]
            new_imag = event.ydata
        elif drag_lock_mode[0] == 'real':
            new_real = event.xdata
            new_imag = drag_start_fixed[1]
        else:
            new_real = event.xdata
            new_imag = event.ydata

        # Update the corresponding element
        if typ == 'zero' and i < len(zeros):
            zeros[i] = complex(new_real, new_imag)
        elif typ == 'pole' and i < len(poles):
            poles[i] = complex(new_real, new_imag)

        request_drag_redraw()


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

    # Back to normal drawing; also adapts the axis limits
    end_drag_rendering()

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
        notify("Πλήθος: δώσε ακέραιο ≥ 1", color='red')


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
# Check box for the spectrum scale
# ------------------------------------------------------------
#   unchecked: linear spectrum over negative and positive ω
#   checked:   Bode diagram (dB, logarithmic ω, positive ω only)
# ============================================================

check_spectrum = CheckButtons(ax_spectrum, ['Bode\n(λογαριθμική κλίμακα)'], [False])

# Note: at exactly 10 pt the bundled DejaVu Sans font draws 'ί'
# almost like '(' on screen, so Greek labels avoid that size.
for check_label in check_spectrum.labels:
    check_label.set_fontsize(9)


def on_spectrum_toggle(label):
    """
    Switch between the linear spectrum and the Bode diagram.
    """
    spectrum_mode[0] = 'bode' if check_spectrum.get_status()[0] else 'linear'

    # Empty the curves first, so that switching to a logarithmic
    # axis never sees the negative frequencies of the linear grid
    mag_line.set_data([], [])
    phase_line.set_data([], [])

    apply_spectrum_axes()
    update_plot()


check_spectrum.on_clicked(on_spectrum_toggle)

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