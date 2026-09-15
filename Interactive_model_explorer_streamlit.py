# =============================================================================
# LU-177 PSMA — INTERACTIVE MODEL EXPLORER
# =============================================================================
#
# Exploratory radiobiological model for Lu-177 PSMA therapy.
#
# Main outputs:
#   1. Physical vs effective dose rate
#   2. Total / sensitive / resistant tumour burden
#   3. Residual resistant tumour burden
#   4. Tumour control probability (TCP)
#
# The model uses:
#   - Critical dose rate
#   - Dose-rate dependent effectiveness
#   - Sensitive and resistant tumour populations
#   - Radiation killing using an LQ formulation
#   - Tumour repopulation
#   - Lu-177 physical decay
#   - Total tumour uptake
#
# NOTE:
# This is an exploratory mechanistic model and is NOT a validated
# patient-specific dosimetry or clinical treatment-planning model.
# =============================================================================

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import MaxNLocator


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Lu-177 PSMA Interactive Model Explorer",
    page_icon="☢️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# MODEL CONSTANTS
# =============================================================================

LU177_HALF_LIFE_DAYS = 6.647
LU177_LAMBDA_PER_DAY = np.log(2.0) / LU177_HALF_LIFE_DAYS

# Reference physical dose conversion.
#
# This is the reference dose-rate conversion used by the existing explorer.
# It is scaled using:
#   - total tumour uptake
#   - total metastatic tumour burden
#
DOSE_PER_GBQ_GY = 0.50

# Reference burden corresponding to the original model.
REFERENCE_METASTATIC_BURDEN_ML = 234.0

# Reference total tumour uptake.
DEFAULT_TUMOUR_UPTAKE_PERCENT = 1.0

TUMOUR_UPTAKE_MIN_PERCENT = 0.1
TUMOUR_UPTAKE_MAX_PERCENT = 10.0

# Numerical integration / simulation
DT_DAYS = 0.05
FOLLOW_UP_DAYS = 60.0

# Default tumour burden
DEFAULT_BURDEN = 234.0

# Radiobiological defaults
DEFAULT_ALPHA = 0.10
DEFAULT_BETA_ALPHA = 0.10
DEFAULT_TREP = 30.0

DEFAULT_SENSITIVE_FRACTION = 0.75

DEFAULT_RESISTANT_TK_MULTIPLIER = 1.50
DEFAULT_RESISTANT_RADIO_FACTOR = 0.25

DEFAULT_REPOPULATION_KICKOFF = 5.0

# Treatment defaults
DEFAULT_ACTIVITY = 7.4
DEFAULT_CYCLES = 4
DEFAULT_INTERVAL = 7

# Initial TCP assumption
INITIAL_TCP = 0.10

# Normalised initial clonogenic burden.
INITIAL_CLONOGENIC_BURDEN = -np.log(INITIAL_TCP)


# =============================================================================
# PLOT COLOURS
# =============================================================================

PLOT_BACKGROUND = "#000000"
PLOT_TEXT = "#F2F2F2"
PLOT_GRID = "#666666"
PLOT_TREATMENT = "#A0A0A0"

COLOR_BLUE = "#4C9BE8"
COLOR_ORANGE = "#FF8C42"
COLOR_GREEN = "#5CC85C"
COLOR_RED = "#E85C5C"
COLOR_PURPLE = "#B57EDC"
COLOR_CYAN = "#4DD0E1"
COLOR_YELLOW = "#F2C94C"


# =============================================================================
# MATPLOTLIB STYLE
# =============================================================================

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.facecolor": PLOT_BACKGROUND,
        "figure.facecolor": PLOT_BACKGROUND,
        "axes.edgecolor": PLOT_TEXT,
        "axes.labelcolor": PLOT_TEXT,
        "xtick.color": PLOT_TEXT,
        "ytick.color": PLOT_TEXT,
        "text.color": PLOT_TEXT,
        "legend.facecolor": PLOT_BACKGROUND,
        "legend.edgecolor": PLOT_GRID,
        "savefig.facecolor": PLOT_BACKGROUND,
        "savefig.edgecolor": PLOT_BACKGROUND,
    }
)


# =============================================================================
# CSS
# =============================================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .metric-card {
        padding: 0.8rem 0.9rem;
        border-radius: 10px;
        background: #F8FAFC;
        border: 1px solid #E5E7EB;
        min-height: 130px;
    }

    .metric-card-secondary {
        padding: 0.75rem 0.9rem;
        border-radius: 10px;
        background: #FFFBEB;
        border: 1px solid #F2C94C;
        border-left: 5px solid #F2C94C;
        min-height: 120px;
    }

    .metric-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #374151;
        margin-bottom: 0.25rem;
    }

    .metric-title-secondary {
        font-size: 0.78rem;
        font-weight: 700;
        color: #9A6700;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        font-size: 1.55rem;
        line-height: 1.15;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.15rem;
    }

    .metric-value-secondary {
        font-size: 1.45rem;
        line-height: 1.15;
        font-weight: 700;
        color: #7A5200;
        margin-bottom: 0.15rem;
    }

    .metric-caption {
        font-size: 0.70rem;
        color: #6B7280;
        line-height: 1.25;
    }

    .section-heading {
        margin-top: 0.7rem;
        margin-bottom: 0.35rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def integrate_trapezoid(y, x):
    """
    NumPy compatibility helper.

    NumPy >= 2.0:
        np.trapezoid()

    Older NumPy:
        np.trapz()
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)

    return np.trapz(y, x)


def calculate_critical_dose_rate(alpha, trep_days):
    """
    Critical dose rate:

        Rcrit = ln(2) / (alpha * Trep)

    where:
        alpha = linear LQ coefficient [Gy^-1]
        Trep  = tumour doubling time [days]

    Returns:
        Gy/day
    """
    trep_hours = trep_days * 24.0

    if alpha <= 0 or trep_hours <= 0:
        return np.inf

    return np.log(2.0) / (alpha * trep_hours)


def calculate_uptake_scale(tumour_uptake_percent):
    """
    Relative tumour uptake scale.

    1% uptake = scale of 1.
    """
    return tumour_uptake_percent / DEFAULT_TUMOUR_UPTAKE_PERCENT


def calculate_burden_scale(initial_burden_ml):
    """
    Relative tumour burden scale.

    234 mL = scale of 1.

    For a fixed total tumour uptake fraction, a larger total burden
    corresponds to a lower average dose rate per unit tumour burden.
    """
    return REFERENCE_METASTATIC_BURDEN_ML / initial_burden_ml


def calculate_dose_scale(initial_burden_ml, tumour_uptake_percent):
    """
    Combined dose scaling from tumour burden and uptake.
    """
    uptake_scale = calculate_uptake_scale(tumour_uptake_percent)
    burden_scale = calculate_burden_scale(initial_burden_ml)

    return uptake_scale * burden_scale


def generate_time_grid(simulation_days):
    """
    Generate simulation time grid.
    """
    return np.arange(
        0.0,
        simulation_days + DT_DAYS * 0.5,
        DT_DAYS,
    )


def generate_administration_times(n_cycles, interval_days):
    """
    Administration times for repeated cycles.
    """
    return np.array(
        [i * interval_days for i in range(n_cycles)],
        dtype=float,
    )


# =============================================================================
# DOSE-RATE MODEL
# =============================================================================

def calculate_physical_dose_rate(
    time_days,
    activity_gbq,
    n_cycles,
    interval_days,
    initial_burden_ml,
    tumour_uptake_percent,
):
    """
    Calculate physical Lu-177 dose rate.

    Activity from each cycle decays according to the Lu-177 physical
    half-life.

    The reference dose-rate conversion is scaled using tumour uptake
    and total metastatic tumour burden.
    """

    time_days = np.asarray(time_days)

    administration_times = generate_administration_times(
        n_cycles,
        interval_days,
    )

    dose_scale = calculate_dose_scale(
        initial_burden_ml,
        tumour_uptake_percent,
    )

    total_dose_rate = np.zeros_like(
        time_days,
        dtype=float,
    )

    for t_admin in administration_times:

        elapsed = time_days - t_admin

        active = elapsed >= 0.0

        activity_remaining = np.zeros_like(
            time_days,
            dtype=float,
        )

        activity_remaining[active] = (
            activity_gbq
            * np.exp(
                -LU177_LAMBDA_PER_DAY
                * elapsed[active]
            )
        )

        total_dose_rate += (
            activity_remaining
            * DOSE_PER_GBQ_GY
            * dose_scale
        )

    return total_dose_rate


# =============================================================================
# DOSE-RATE EFFECTIVENESS
# =============================================================================

def calculate_effective_dose_rate(
    physical_dose_rate_gy_day,
    critical_dose_rate_gy_day,
):
    """
    Dose-rate effectiveness model.

    The previous explorer exposed gamma as a user-adjustable parameter:

        E = min(1, ratio^gamma)

    with the default gamma = 1.

    Gamma is now removed from the user interface and from the model.

    The direct relationship is therefore:

        E = min(1, Rphysical / Rcritical)

    and:

        Reffective = Rphysical * E

    This preserves the behaviour of the previous default gamma = 1
    while avoiding an arbitrary user-adjustable gamma parameter.
    """

    physical_dose_rate_gy_day = np.asarray(
        physical_dose_rate_gy_day,
        dtype=float,
    )

    if critical_dose_rate_gy_day <= 0:
        effectiveness = np.ones_like(
            physical_dose_rate_gy_day
        )
    else:
        dose_rate_ratio = (
            physical_dose_rate_gy_day
            / critical_dose_rate_gy_day
        )

        effectiveness = np.minimum(
            1.0,
            np.maximum(
                0.0,
                dose_rate_ratio,
            ),
        )

    effective_dose_rate = (
        physical_dose_rate_gy_day
        * effectiveness
    )

    return effectiveness, effective_dose_rate


# =============================================================================
# TUMOUR DYNAMICS
# =============================================================================

def simulate_tumour_dynamics(
    time_days,
    physical_dose_rate_gy_day,
    critical_dose_rate_gy_day,
    initial_burden_ml,
    alpha,
    trep_days,
    sensitive_fraction,
    resistant_tk_multiplier,
    resistant_radio_factor,
    repopulation_kickoff_days,
):
    """
    Simulate sensitive and resistant tumour populations.

    Radiation killing:
        S *= exp(-alpha D - beta D^2)

    Resistant population:
        alpha_resistant = alpha * resistant_radio_factor

        beta_resistant =
            beta_sensitive * resistant_radio_factor

    Repopulation starts after the specified kickoff time.

    Returns:
        dictionary containing all trajectories.
    """

    n_points = len(time_days)

    sensitive = np.zeros(n_points)
    resistant = np.zeros(n_points)

    sensitive[0] = (
        initial_burden_ml
        * sensitive_fraction
    )

    resistant[0] = (
        initial_burden_ml
        * (1.0 - sensitive_fraction)
    )

    initial_resistant_burden = resistant[0]

    beta_sensitive = (
        alpha * DEFAULT_BETA_ALPHA
    )

    alpha_resistant = (
        alpha * resistant_radio_factor
    )

    beta_resistant = (
        beta_sensitive
        * resistant_radio_factor
    )

    sensitive_growth_rate = (
        np.log(2.0) / trep_days
    )

    resistant_growth_rate = (
        np.log(2.0)
        / (
            trep_days
            * resistant_tk_multiplier
        )
    )

    (
        effectiveness,
        effective_dose_rate_gy_day,
    ) = calculate_effective_dose_rate(
        physical_dose_rate_gy_day,
        critical_dose_rate_gy_day,
    )

    for i in range(1, n_points):

        dt = (
            time_days[i]
            - time_days[i - 1]
        )

        # ---------------------------------------------------------------------
        # Repopulation
        # ---------------------------------------------------------------------

        S = sensitive[i - 1]
        R = resistant[i - 1]

        if (
            time_days[i]
            >= repopulation_kickoff_days
        ):

            S *= np.exp(
                sensitive_growth_rate * dt
            )

            R *= np.exp(
                resistant_growth_rate * dt
            )

        # ---------------------------------------------------------------------
        # Radiation dose during interval
        # ---------------------------------------------------------------------

        dose_interval_gy = (
            physical_dose_rate_gy_day[i]
            * dt
        )

        # ---------------------------------------------------------------------
        # LQ radiation survival
        # ---------------------------------------------------------------------

        survival_sensitive = np.exp(
            -alpha * dose_interval_gy
            -beta_sensitive
            * dose_interval_gy ** 2
        )

        survival_resistant = np.exp(
            -alpha_resistant
            * dose_interval_gy
            -beta_resistant
            * dose_interval_gy ** 2
        )

        S *= survival_sensitive
        R *= survival_resistant

        sensitive[i] = max(S, 0.0)
        resistant[i] = max(R, 0.0)

    total_burden = (
        sensitive + resistant
    )

    # -------------------------------------------------------------------------
    # Resistant disease metrics
    # -------------------------------------------------------------------------

    if initial_resistant_burden > 0:

        residual_resistant_burden_percent = (
            resistant
            / initial_resistant_burden
            * 100.0
        )

    else:

        residual_resistant_burden_percent = (
            np.zeros_like(resistant)
        )

    # Composition of remaining tumour.
    #
    # This is NOT the same as residual resistant burden.
    #
    # It can reach 100% even while the absolute resistant population
    # is approaching zero.
    #
    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        resistant_composition_fraction = np.where(
            total_burden > 0,
            resistant / total_burden,
            0.0,
        )

    # -------------------------------------------------------------------------
    # TCP
    # -------------------------------------------------------------------------

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        relative_burden = np.where(
            initial_burden_ml > 0,
            total_burden / initial_burden_ml,
            0.0,
        )

    surviving_clonogens = (
        INITIAL_CLONOGENIC_BURDEN
        * relative_burden
    )

    tcp = np.exp(
        -surviving_clonogens
    )

    tcp = np.clip(
        tcp,
        0.0,
        1.0,
    )

    return {
        "sensitive": sensitive,
        "resistant": resistant,
        "total_burden": total_burden,
        "effectiveness": effectiveness,
        "effective_dose_rate_gy_day": (
            effective_dose_rate_gy_day
        ),
        "residual_resistant_burden_percent": (
            residual_resistant_burden_percent
        ),
        "resistant_composition_fraction": (
            resistant_composition_fraction
        ),
        "tcp": tcp,
        "initial_resistant_burden": (
            initial_resistant_burden
        ),
        "alpha_resistant": alpha_resistant,
        "beta_sensitive": beta_sensitive,
        "beta_resistant": beta_resistant,
        "sensitive_growth_rate": (
            sensitive_growth_rate
        ),
        "resistant_growth_rate": (
            resistant_growth_rate
        ),
    }


# =============================================================================
# SUMMARY CALCULATIONS
# =============================================================================

def calculate_time_above_threshold(
    dose_rate,
    critical_dose_rate,
    time_days,
):
    """
    Time during which dose rate is at or above critical dose rate.
    """

    above = (
        dose_rate
        >= critical_dose_rate
    )

    if len(time_days) < 2:
        return 0.0

    dt = np.diff(time_days)

    return float(
        np.sum(
            dt[above[:-1]]
        )
    )


def calculate_longest_continuous_above(
    dose_rate,
    critical_dose_rate,
    time_days,
):
    """
    Longest continuous interval during which dose rate is at or above
    critical dose rate.
    """

    above = (
        dose_rate
        >= critical_dose_rate
    )

    longest = 0.0
    current = 0.0

    for i in range(len(time_days) - 1):

        if above[i] and above[i + 1]:

            interval = (
                time_days[i + 1]
                - time_days[i]
            )

            current += interval

            longest = max(
                longest,
                current,
            )

        else:

            current = 0.0

    return float(longest)


def calculate_summary(
    time_days,
    physical_dose_rate_gy_day,
    effective_dose_rate_gy_day,
    critical_dose_rate_gy_day,
    tumour_results,
    activity_gbq,
    n_cycles,
    interval_days,
    initial_burden_ml,
    tumour_uptake_percent,
):
    """
    Calculate all summary outputs.
    """

    total_burden = (
        tumour_results["total_burden"]
    )

    sensitive = (
        tumour_results["sensitive"]
    )

    resistant = (
        tumour_results["resistant"]
    )

    tcp = tumour_results["tcp"]

    residual_resistant = (
        tumour_results[
            "residual_resistant_burden_percent"
        ]
    )

    resistant_composition = (
        tumour_results[
            "resistant_composition_fraction"
        ]
    )

    min_burden_index = int(
        np.argmin(total_burden)
    )

    max_tcp_index = int(
        np.argmax(tcp)
    )

    min_resistant_index = int(
        np.argmin(residual_resistant)
    )

    peak_rate_index = int(
        np.argmax(
            physical_dose_rate_gy_day
        )
    )

    cumulative_physical_dose = (
        integrate_trapezoid(
            physical_dose_rate_gy_day,
            time_days,
        )
    )

    cumulative_effective_dose = (
        integrate_trapezoid(
            effective_dose_rate_gy_day,
            time_days,
        )
    )

    dose_rate_ratio = np.divide(
        physical_dose_rate_gy_day,
        critical_dose_rate_gy_day,
        out=np.zeros_like(
            physical_dose_rate_gy_day
        ),
        where=critical_dose_rate_gy_day > 0,
    )

    time_above = (
        calculate_time_above_threshold(
            physical_dose_rate_gy_day,
            critical_dose_rate_gy_day,
            time_days,
        )
    )

    longest_above = (
        calculate_longest_continuous_above(
            physical_dose_rate_gy_day,
            critical_dose_rate_gy_day,
            time_days,
        )
    )

    final_burden = total_burden[-1]

    final_resistant_composition = (
        resistant_composition[-1] * 100.0
    )

    final_residual_resistant = (
        residual_resistant[-1]
    )

    return {
        "minimum_tumour_burden_ml": (
            float(total_burden[min_burden_index])
        ),
        "minimum_tumour_burden_day": (
            float(time_days[min_burden_index])
        ),
        "final_tumour_burden_ml": (
            float(final_burden)
        ),
        "maximum_tcp_percent": (
            float(np.max(tcp) * 100.0)
        ),
        "maximum_tcp_day": (
            float(time_days[max_tcp_index])
        ),
        "final_tcp_percent": (
            float(tcp[-1] * 100.0)
        ),
        "final_residual_resistant_percent": (
            float(final_residual_resistant)
        ),
        "minimum_residual_resistant_percent": (
            float(
                residual_resistant[
                    min_resistant_index
                ]
            )
        ),
        "minimum_residual_resistant_day": (
            float(
                time_days[
                    min_resistant_index
                ]
            )
        ),
        "final_resistant_composition_percent": (
            float(final_resistant_composition)
        ),
        "final_sensitive_burden_ml": (
            float(sensitive[-1])
        ),
        "final_resistant_burden_ml": (
            float(resistant[-1])
        ),
        "critical_dose_rate_gy_day": (
            float(critical_dose_rate_gy_day)
        ),
        "critical_dose_rate_gy_h": (
            float(
                critical_dose_rate_gy_day
                / 24.0
            )
        ),
        "peak_physical_rate_gy_day": (
            float(
                physical_dose_rate_gy_day[
                    peak_rate_index
                ]
            )
        ),
        "peak_physical_rate_gy_h": (
            float(
                physical_dose_rate_gy_day[
                    peak_rate_index
                ] / 24.0
            )
        ),
        "peak_rate_day": (
            float(time_days[peak_rate_index])
        ),
        "peak_to_critical_ratio": (
            float(np.max(dose_rate_ratio))
        ),
        "time_above_critical_days": (
            float(time_above)
        ),
        "longest_continuous_above_days": (
            float(longest_above)
        ),
        "cumulative_physical_dose_gy": (
            float(cumulative_physical_dose)
        ),
        "cumulative_effective_dose_gy": (
            float(cumulative_effective_dose)
        ),
        "final_effective_dose_rate_gy_day": (
            float(effective_dose_rate_gy_day[-1])
        ),
        "initial_resistant_burden_ml": (
            float(
                tumour_results[
                    "initial_resistant_burden"
                ]
            )
        ),
        "activity_gbq": float(activity_gbq),
        "n_cycles": int(n_cycles),
        "interval_days": float(interval_days),
        "initial_burden_ml": float(
            initial_burden_ml
        ),
        "tumour_uptake_percent": float(
            tumour_uptake_percent
        ),
        "dose_scale": float(
            calculate_dose_scale(
                initial_burden_ml,
                tumour_uptake_percent,
            )
        ),
    }


# =============================================================================
# PLOT FORMATTING
# =============================================================================

def style_axis(ax):
    """
    Apply consistent dark plot styling.
    """

    ax.set_facecolor(
        PLOT_BACKGROUND
    )

    ax.tick_params(
        which="both",
        colors=PLOT_TEXT,
    )

    ax.spines["bottom"].set_color(
        PLOT_TEXT
    )
    ax.spines["left"].set_color(
        PLOT_TEXT
    )
    ax.spines["top"].set_color(
        PLOT_TEXT
    )
    ax.spines["right"].set_color(
        PLOT_TEXT
    )

    ax.grid(
        which="major",
        color=PLOT_GRID,
        alpha=0.30,
        linewidth=0.7,
    )

    ax.grid(
        which="minor",
        color=PLOT_GRID,
        alpha=0.15,
        linewidth=0.5,
    )

    ax.minorticks_on()

    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=7)
    )

    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=6)
    )


def add_treatment_markers(
    ax,
    administration_times,
    ymax=None,
):
    """
    Add treatment cycle markers.

    Grey dotted lines distinguish treatment administrations from
    biological trajectories.
    """

    for t in administration_times:

        ax.axvline(
            t,
            color=COLOR_TREATMENT,
            linestyle=":",
            linewidth=0.8,
            alpha=0.65,
        )


def build_plots(
    time_days,
    physical_dose_rate_gy_day,
    effective_dose_rate_gy_day,
    critical_dose_rate_gy_day,
    tumour_results,
    administration_times,
):
    """
    Build the four main model plots.
    """

    total_burden = (
        tumour_results["total_burden"]
    )

    sensitive = (
        tumour_results["sensitive"]
    )

    resistant = (
        tumour_results["resistant"]
    )

    residual_resistant = (
        tumour_results[
            "residual_resistant_burden_percent"
        ]
    )

    tcp_percent = (
        tumour_results["tcp"]
        * 100.0
    )

    # -------------------------------------------------------------------------
    # Plot 1 — Dose rate
    # -------------------------------------------------------------------------

    fig1, ax1 = plt.subplots(
        figsize=(9.5, 5.5),
        dpi=160,
    )

    ax1.plot(
        time_days,
        physical_dose_rate_gy_day,
        color=COLOR_BLUE,
        linewidth=2.2,
        label="Physical dose rate",
    )

    ax1.plot(
        time_days,
        effective_dose_rate_gy_day,
        color=COLOR_ORANGE,
        linewidth=2.0,
        linestyle="--",
        label="Effective dose rate",
    )

    ax1.axhline(
        critical_dose_rate_gy_day,
        color=COLOR_RED,
        linewidth=1.7,
        linestyle=":",
        label="Critical dose rate",
    )

    add_treatment_markers(
        ax1,
        administration_times,
    )

    ax1.set_xlabel(
        "Time (days)"
    )

    ax1.set_ylabel(
        "Dose rate (Gy/day)"
    )

    ax1.set_title(
        "Physical vs effective dose rate",
        fontsize=13,
        fontweight="bold",
    )

    ax1.legend(
        loc="best",
        framealpha=0.9,
    )

    style_axis(ax1)

    fig1.tight_layout()

    # -------------------------------------------------------------------------
    # Plot 2 — Tumour burden
    # -------------------------------------------------------------------------

    fig2, ax2 = plt.subplots(
        figsize=(9.5, 5.5),
        dpi=160,
    )

    ax2.plot(
        time_days,
        total_burden,
        color=COLOR_BLUE,
        linewidth=2.2,
        label="Total tumour",
    )

    ax2.plot(
        time_days,
        sensitive,
        color=COLOR_GREEN,
        linewidth=1.9,
        linestyle="--",
        label="Sensitive tumour",
    )

    ax2.plot(
        time_days,
        resistant,
        color=COLOR_ORANGE,
        linewidth=1.9,
        linestyle=":",
        label="Resistant tumour",
    )

    add_treatment_markers(
        ax2,
        administration_times,
    )

    ax2.set_xlabel(
        "Time (days)"
    )

    ax2.set_ylabel(
        "Tumour burden (mL)"
    )

    ax2.set_title(
        "Tumour burden trajectory",
        fontsize=13,
        fontweight="bold",
    )

    ax2.legend(
        loc="best",
        framealpha=0.9,
    )

    style_axis(ax2)

    fig2.tight_layout()

    # -------------------------------------------------------------------------
    # Plot 3 — Sensitive / resistant populations
    # -------------------------------------------------------------------------

    fig3, ax3 = plt.subplots(
        figsize=(9.5, 5.5),
        dpi=160,
    )

    ax3.plot(
        time_days,
        sensitive,
        color=COLOR_GREEN,
        linewidth=2.0,
        label="Sensitive tumour (mL)",
    )

    ax3.plot(
        time_days,
        resistant,
        color=COLOR_ORANGE,
        linewidth=2.0,
        linestyle="--",
        label="Resistant tumour (mL)",
    )

    ax3.set_xlabel(
        "Time (days)"
    )

    ax3.set_ylabel(
        "Tumour burden (mL)"
    )

    ax3.set_title(
        "Sensitive and resistant tumour populations",
        fontsize=13,
        fontweight="bold",
    )

    ax3b = ax3.twinx()

    ax3b.plot(
        time_days,
        residual_resistant,
        color=COLOR_PURPLE,
        linewidth=1.9,
        linestyle=":",
        label=(
            "Residual resistant burden "
            "(% of initial)"
        ),
    )

    ax3b.set_ylabel(
        "Residual resistant burden "
        "(% of initial)",
        color=COLOR_PURPLE,
    )

    ax3b.tick_params(
        axis="y",
        colors=COLOR_PURPLE,
    )

    ax3b.spines["right"].set_color(
        COLOR_PURPLE
    )

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()

    ax3.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best",
        framealpha=0.9,
    )

    add_treatment_markers(
        ax3,
        administration_times,
    )

    style_axis(ax3)

    fig3.tight_layout()

    # -------------------------------------------------------------------------
    # Plot 4 — TCP
    # -------------------------------------------------------------------------

    fig4, ax4 = plt.subplots(
        figsize=(9.5, 5.5),
        dpi=160,
    )

    ax4.plot(
        time_days,
        tcp_percent,
        color=COLOR_BLUE,
        linewidth=2.2,
        label="TCP",
    )

    ax4.axhline(
        50.0,
        color=COLOR_ORANGE,
        linewidth=1.5,
        linestyle="--",
        label="50% TCP",
    )

    ax4.axhline(
        90.0,
        color=COLOR_GREEN,
        linewidth=1.5,
        linestyle=":",
        label="90% TCP",
    )

    add_treatment_markers(
        ax4,
        administration_times,
    )

    ax4.set_ylim(
        0.0,
        105.0,
    )

    ax4.set_xlabel(
        "Time (days)"
    )

    ax4.set_ylabel(
        "Tumour control probability (%)"
    )

    ax4.set_title(
        "Tumour control probability",
        fontsize=13,
        fontweight="bold",
    )

    ax4.legend(
        loc="best",
        framealpha=0.9,
    )

    style_axis(ax4)

    fig4.tight_layout()

    return (
        fig1,
        fig2,
        fig3,
        fig4,
    )


# =============================================================================
# UI HELPER FUNCTIONS
# =============================================================================

def key_result_status(
    text,
    status="neutral",
):
    """
    Green/red/neutral status badge used below key results.
    """

    if status == "positive":

        colour = "#166534"
        background = "#DCFCE7"

    elif status == "negative":

        colour = "#991B1B"
        background = "#FEE2E2"

    else:

        colour = "#4B5563"
        background = "#F3F4F6"

    return (
        f"""
        <span style="
            display:inline-block;
            padding:3px 8px;
            border-radius:999px;
            font-size:0.75rem;
            font-weight:600;
            color:{colour};
            background:{background};
            margin-top:3px;
        ">
            {text}
        </span>
        """
    )


def key_result_caption(text):
    """
    Small explanatory caption below the status badge.
    """

    return (
        f"""
        <div style="
            font-size:0.70rem;
            color:#6B7280;
            line-height:1.25;
            margin-top:5px;
        ">
            {text}
        </div>
        """
    )


def render_key_metric(
    title,
    value,
    status_text,
    status="neutral",
    caption="",
):
    """
    Render a primary key result card.
    """

    html = f"""
    <div class="metric-card">
        <div class="metric-title">
            {title}
        </div>

        <div class="metric-value">
            {value}
        </div>

        {key_result_status(
            status_text,
            status,
        )}

        {key_result_caption(
            caption
        )}
    </div>
    """

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def render_secondary_metric(
    title,
    value,
    caption="",
):
    """
    Render a yellow/gold secondary metric.
    """

    html = f"""
    <div class="metric-card-secondary">

        <div class="metric-title-secondary">
            {title}
        </div>

        <div class="metric-value-secondary">
            {value}
        </div>

        <div class="metric-caption">
            {caption}
        </div>

    </div>
    """

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title(
    "Model controls"
)

st.sidebar.caption(
    "Adjust the tumour and treatment parameters to explore "
    "dose-rate, tumour-response and TCP behaviour."
)


# =============================================================================
# TUMOUR PARAMETERS
# =============================================================================

st.sidebar.markdown(
    "### Tumour parameters"
)

initial_burden_ml = st.sidebar.slider(
    "Initial metastatic burden (mL)",
    min_value=10.0,
    max_value=1500.0,
    value=float(DEFAULT_BURDEN),
    step=1.0,
    help=(
        "Total metastatic tumour burden represented by the model. "
        "This is a phenomenological total tumour volume rather than "
        "a single solid tumour."
    ),
)

alpha = st.sidebar.slider(
    "Alpha",
    min_value=0.001,
    max_value=0.50,
    value=float(DEFAULT_ALPHA),
    step=0.001,
    format="%.3f",
    help=(
        "Linear radiation sensitivity parameter used in the "
        "exploratory LQ model."
    ),
)

trep_days = st.sidebar.slider(
    "Trep (days)",
    min_value=10.0,
    max_value=100.0,
    value=float(DEFAULT_TREP),
    step=1.0,
    help=(
        "Sensitive tumour population doubling time."
    ),
)

sensitive_fraction = st.sidebar.slider(
    "Sensitive fraction",
    min_value=0.50,
    max_value=0.95,
    value=float(DEFAULT_SENSITIVE_FRACTION),
    step=0.01,
    format="%.2f",
    help=(
        "Initial fraction of tumour burden assigned to "
        "the radiation-sensitive population."
    ),
)

resistant_tk_multiplier = st.sidebar.slider(
    "Tk/Trep",
    min_value=1.20,
    max_value=1.80,
    value=float(DEFAULT_RESISTANT_TK_MULTIPLIER),
    step=0.01,
    format="%.2f",
    help=(
        "Relative resistant-population growth timescale. "
        "A larger value means slower resistant repopulation."
    ),
)

resistant_radio_factor = st.sidebar.slider(
    "Resistant radio factor",
    min_value=0.05,
    max_value=1.00,
    value=float(DEFAULT_RESISTANT_RADIO_FACTOR),
    step=0.01,
    format="%.2f",
    help=(
        "Relative radiosensitivity of the resistant population. "
        "1.0 means the same radiation sensitivity as the sensitive "
        "population; lower values indicate greater radioresistance."
    ),
)

repopulation_kickoff_days = st.sidebar.slider(
    "Repopulation kickoff (days)",
    min_value=3.0,
    max_value=10.0,
    value=float(DEFAULT_REPOPULATION_KICKOFF),
    step=1.0,
    help=(
        "Delay before tumour repopulation begins."
    ),
)


# =============================================================================
# TREATMENT PARAMETERS
# =============================================================================

st.sidebar.markdown(
    "### Treatment parameters"
)

activity_gbq = st.sidebar.slider(
    "Activity per cycle (GBq)",
    min_value=1.0,
    max_value=15.0,
    value=float(DEFAULT_ACTIVITY),
    step=0.1,
    format="%.1f",
    help=(
        "Administered Lu-177 activity per treatment cycle."
    ),
)

n_cycles = st.sidebar.slider(
    "Number of cycles",
    min_value=1,
    max_value=8,
    value=int(DEFAULT_CYCLES),
    step=1,
    help=(
        "Number of treatment administrations."
    ),
)

interval_days = st.sidebar.slider(
    "Cycle interval (days)",
    min_value=1,
    max_value=42,
    value=int(DEFAULT_INTERVAL),
    step=1,
    help=(
        "Time between treatment administrations."
    ),
)

tumour_uptake_percent = st.sidebar.slider(
    "Total tumour uptake (%)",
    min_value=float(TUMOUR_UPTAKE_MIN_PERCENT),
    max_value=float(TUMOUR_UPTAKE_MAX_PERCENT),
    value=float(DEFAULT_TUMOUR_UPTAKE_PERCENT),
    step=0.1,
    format="%.1f",
    help=(
        "Fraction of administered activity attributed to the "
        "total metastatic tumour burden."
    ),
)


# =============================================================================
# SIMULATION DURATION
# =============================================================================

simulation_days = max(
    FOLLOW_UP_DAYS,
    (
        (n_cycles - 1)
        * interval_days
    )
    + FOLLOW_UP_DAYS,
)


# =============================================================================
# RUN MODEL
# =============================================================================

time_days = generate_time_grid(
    simulation_days
)

administration_times = (
    generate_administration_times(
        n_cycles,
        interval_days,
    )
)

critical_dose_rate_gy_day = (
    calculate_critical_dose_rate(
        alpha,
        trep_days,
    )
)

physical_dose_rate_gy_day = (
    calculate_physical_dose_rate(
        time_days=time_days,
        activity_gbq=activity_gbq,
        n_cycles=n_cycles,
        interval_days=interval_days,
        initial_burden_ml=initial_burden_ml,
        tumour_uptake_percent=tumour_uptake_percent,
    )
)

(
    effectiveness,
    effective_dose_rate_gy_day,
) = calculate_effective_dose_rate(
    physical_dose_rate_gy_day,
    critical_dose_rate_gy_day,
)

tumour_results = (
    simulate_tumour_dynamics(
        time_days=time_days,
        physical_dose_rate_gy_day=(
            physical_dose_rate_gy_day
        ),
        critical_dose_rate_gy_day=(
            critical_dose_rate_gy_day
        ),
        initial_burden_ml=initial_burden_ml,
        alpha=alpha,
        trep_days=trep_days,
        sensitive_fraction=sensitive_fraction,
        resistant_tk_multiplier=(
            resistant_tk_multiplier
        ),
        resistant_radio_factor=(
            resistant_radio_factor
        ),
        repopulation_kickoff_days=(
            repopulation_kickoff_days
        ),
    )
)

summary = calculate_summary(
    time_days=time_days,
    physical_dose_rate_gy_day=(
        physical_dose_rate_gy_day
    ),
    effective_dose_rate_gy_day=(
        effective_dose_rate_gy_day
    ),
    critical_dose_rate_gy_day=(
        critical_dose_rate_gy_day
    ),
    tumour_results=tumour_results,
    activity_gbq=activity_gbq,
    n_cycles=n_cycles,
    interval_days=interval_days,
    initial_burden_ml=initial_burden_ml,
    tumour_uptake_percent=tumour_uptake_percent,
)


# =============================================================================
# TITLE
# =============================================================================

st.title(
    "Lu-177 PSMA Interactive Model Explorer"
)

st.markdown(
    """
    Explore how tumour burden, radiobiological characteristics,
    Lu-177 activity, tumour uptake and treatment-cycle spacing
    influence dose rate, tumour response and tumour control probability.
    """
)


# =============================================================================
# KEY RESULTS
# =============================================================================

st.markdown(
    '<h3 class="section-heading">Key results</h3>',
    unsafe_allow_html=True,
)

key_col1, key_col2, key_col3, key_col4 = st.columns(4)


# -----------------------------------------------------------------------------
# Minimum tumour burden
# -----------------------------------------------------------------------------

minimum_burden = (
    summary["minimum_tumour_burden_ml"]
)

burden_reduction_percent = (
    1.0
    - minimum_burden
    / initial_burden_ml
) * 100.0

with key_col1:

    render_key_metric(
        title="Minimum tumour burden",
        value=(
            f"{minimum_burden:,.1f} mL"
        ),
        status_text=(
            f"↓ {burden_reduction_percent:.1f}% "
            f"from {initial_burden_ml:,.0f} mL baseline"
        ),
        status=(
            "positive"
            if burden_reduction_percent > 0
            else "negative"
        ),
        caption=(
            f"Minimum reached at day "
            f"{summary['minimum_tumour_burden_day']:.1f}"
        ),
    )


# -----------------------------------------------------------------------------
# Maximum TCP
# -----------------------------------------------------------------------------

maximum_tcp = (
    summary["maximum_tcp_percent"]
)

tcp_change = (
    maximum_tcp
    - INITIAL_TCP * 100.0
)

with key_col2:

    render_key_metric(
        title="Maximum TCP",
        value=(
            f"{maximum_tcp:.1f}%"
        ),
        status_text=(
            f"↑ {tcp_change:.1f} percentage points "
            f"from {INITIAL_TCP * 100.0:.0f}% baseline"
            if tcp_change >= 0
            else
            f"↓ {abs(tcp_change):.1f} percentage points "
            f"from {INITIAL_TCP * 100.0:.0f}% baseline"
        ),
        status=(
            "positive"
            if tcp_change > 0
            else "negative"
            if tcp_change < 0
            else "neutral"
        ),
        caption=(
            f"Maximum reached at day "
            f"{summary['maximum_tcp_day']:.1f}"
        ),
    )


# -----------------------------------------------------------------------------
# Residual resistant burden
# -----------------------------------------------------------------------------

final_resistant = (
    summary["final_residual_resistant_percent"]
)

initial_resistant_ml = (
    summary["initial_resistant_burden_ml"]
)

with key_col3:

    render_key_metric(
        title="Residual resistant burden",
        value=(
            f"{final_resistant:.1f}%"
        ),
        status_text=(
            f"↓ {max(0.0, 100.0 - final_resistant):.1f}% "
            f"from initial resistant burden"
            if final_resistant <= 100.0
            else
            f"↑ {final_resistant - 100.0:.1f}% "
            f"from initial resistant burden"
        ),
        status=(
            "positive"
            if final_resistant < 100.0
            else
            "negative"
            if final_resistant > 100.0
            else
            "neutral"
        ),
        caption=(
            "% of the initial resistant tumour burden remaining"
            f"<br>Initial resistant burden: "
            f"{initial_resistant_ml:.1f} mL "
            f"({100.0 - sensitive_fraction * 100.0:.1f}% of total)"
        ),
    )


# -----------------------------------------------------------------------------
# Cumulative physical dose
# -----------------------------------------------------------------------------

with key_col4:

    render_key_metric(
        title="Cumulative physical dose",
        value=(
            f"{summary['cumulative_physical_dose_gy']:.2f} Gy"
        ),
        status_text=(
            f"{activity_gbq:.1f} GBq × "
            f"{n_cycles} cycles"
        ),
        status="neutral",
        caption=(
            "Integrated physical dose over the simulation"
        ),
    )


# =============================================================================
# SECONDARY METRICS
# =============================================================================

st.markdown(
    '<h3 class="section-heading">Secondary metrics</h3>',
    unsafe_allow_html=True,
)

secondary_col1, secondary_col2, secondary_col3, secondary_col4 = (
    st.columns(4)
)


with secondary_col1:

    render_secondary_metric(
        title="Critical dose rate",
        value=(
            f"{summary['critical_dose_rate_gy_h']:.4f} Gy/h"
        ),
        caption=(
            f"{summary['critical_dose_rate_gy_day']:.4f} Gy/day"
        ),
    )


with secondary_col2:

    render_secondary_metric(
        title="Peak physical rate",
        value=(
            f"{summary['peak_physical_rate_gy_h']:.4f} Gy/h"
        ),
        caption=(
            f"Peak at day "
            f"{summary['peak_rate_day']:.1f}"
        ),
    )


with secondary_col3:

    render_secondary_metric(
        title="Time above critical rate",
        value=(
            f"{summary['time_above_critical_days']:.1f} days"
        ),
        caption=(
            f"Longest continuous interval: "
            f"{summary['longest_continuous_above_days']:.1f} days"
        ),
    )


with secondary_col4:

    render_secondary_metric(
        title="Effective cumulative dose",
        value=(
            f"{summary['cumulative_effective_dose_gy']:.2f} Gy"
        ),
        caption=(
            "Integrated dose after dose-rate effectiveness adjustment"
        ),
    )


# =============================================================================
# MODEL TRAJECTORIES
# =============================================================================

st.markdown(
    '<h3 class="section-heading">Model trajectories</h3>',
    unsafe_allow_html=True,
)

(
    fig1,
    fig2,
    fig3,
    fig4,
) = build_plots(
    time_days=time_days,
    physical_dose_rate_gy_day=(
        physical_dose_rate_gy_day
    ),
    effective_dose_rate_gy_day=(
        effective_dose_rate_gy_day
    ),
    critical_dose_rate_gy_day=(
        critical_dose_rate_gy_day
    ),
    tumour_results=tumour_results,
    administration_times=administration_times,
)


plot_col1, plot_col2 = st.columns(2)

with plot_col1:

    st.pyplot(
        fig1,
        use_container_width=True,
    )

with plot_col2:

    st.pyplot(
        fig2,
        use_container_width=True,
    )


plot_col3, plot_col4 = st.columns(2)

with plot_col3:

    st.pyplot(
        fig3,
        use_container_width=True,
    )

with plot_col4:

    st.pyplot(
        fig4,
        use_container_width=True,
    )


# =============================================================================
# DETAILED MODEL SUMMARY
# =============================================================================

with st.expander(
    "Detailed model summary"
):

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:

        st.markdown("#### Tumour response")

        st.write(
            f"Initial tumour burden: "
            f"{initial_burden_ml:.1f} mL"
        )

        st.write(
            f"Minimum tumour burden: "
            f"{summary['minimum_tumour_burden_ml']:.2f} mL"
        )

        st.write(
            f"Final tumour burden: "
            f"{summary['final_tumour_burden_ml']:.2f} mL"
        )

        st.write(
            f"Final sensitive burden: "
            f"{summary['final_sensitive_burden_ml']:.2f} mL"
        )

        st.write(
            f"Final resistant burden: "
            f"{summary['final_resistant_burden_ml']:.2f} mL"
        )

        st.write(
            f"Initial resistant burden: "
            f"{summary['initial_resistant_burden_ml']:.2f} mL"
        )

        st.write(
            f"Final residual resistant burden: "
            f"{summary['final_residual_resistant_percent']:.2f}%"
        )

        st.write(
            f"Final resistant composition: "
            f"{summary['final_resistant_composition_percent']:.2f}%"
        )

        st.write(
            f"Maximum TCP: "
            f"{summary['maximum_tcp_percent']:.2f}%"
        )

        st.write(
            f"Final TCP: "
            f"{summary['final_tcp_percent']:.2f}%"
        )

    with detail_col2:

        st.markdown("#### Dose-rate behaviour")

        st.write(
            f"Critical dose rate: "
            f"{summary['critical_dose_rate_gy_day']:.5f} Gy/day"
        )

        st.write(
            f"Critical dose rate: "
            f"{summary['critical_dose_rate_gy_h']:.5f} Gy/h"
        )

        st.write(
            f"Peak physical dose rate: "
            f"{summary['peak_physical_rate_gy_day']:.5f} Gy/day"
        )

        st.write(
            f"Peak physical dose rate: "
            f"{summary['peak_physical_rate_gy_h']:.5f} Gy/h"
        )

        st.write(
            f"Peak / critical dose-rate ratio: "
            f"{summary['peak_to_critical_ratio']:.2f} ×"
        )

        st.write(
            f"Time above critical dose rate: "
            f"{summary['time_above_critical_days']:.2f} days"
        )

        st.write(
            f"Longest continuous period above critical: "
            f"{summary['longest_continuous_above_days']:.2f} days"
        )

        st.write(
            f"Cumulative physical dose: "
            f"{summary['cumulative_physical_dose_gy']:.3f} Gy"
        )

        st.write(
            f"Cumulative effective dose: "
            f"{summary['cumulative_effective_dose_gy']:.3f} Gy"
        )

        st.write(
            f"Total tumour uptake: "
            f"{tumour_uptake_percent:.1f}%"
        )

        st.write(
            f"Combined dose scaling factor: "
            f"{summary['dose_scale']:.3f}"
        )

        st.write(
            f"Activity per cycle: "
            f"{activity_gbq:.1f} GBq"
        )

        st.write(
            f"Number of cycles: "
            f"{n_cycles}"
        )

        st.write(
            f"Cycle interval: "
            f"{interval_days} days"
        )


# =============================================================================
# MODEL ASSUMPTIONS AND LIMITATIONS
# =============================================================================

with st.expander(
    "Model assumptions and limitations"
):

    st.markdown(
        """
        **Tumour burden**

        Initial metastatic burden represents the total metastatic tumour
        volume in mL rather than a single solid tumour. The model therefore
        represents an average/aggregate tumour response.

        **Tumour uptake**

        Total tumour uptake represents the fraction of administered activity
        attributed to the complete metastatic tumour burden. The dose-rate
        calculation uses this as a phenomenological scaling parameter.

        **Critical dose rate**

        The critical dose rate is calculated from:

        $$
        R_{crit} =
        \\frac{\\ln(2)}
        {\\alpha T_{rep}}
        $$

        where alpha is the linear LQ parameter and Trep is the sensitive
        tumour doubling time.

        **Dose-rate effectiveness**

        The low-dose-rate effectiveness adjustment is now directly related
        to the calculated physical dose rate and critical dose rate:

        $$
        E =
        \\min\\left(
        1,
        \\frac{R_{physical}}
        {R_{critical}}
        \\right)
        $$

        Thus, dose rates at or above the critical rate receive no additional
        penalty, while dose rates below the critical rate are progressively
        reduced.

        **Sensitive and resistant populations**

        The tumour is divided into sensitive and resistant populations.
        The resistant population has reduced radiation sensitivity according
        to the resistant radio factor and a modified repopulation timescale
        according to Tk/Trep.

        **Repopulation**

        Tumour repopulation is delayed until the specified repopulation
        kickoff time. After this point, tumour growth and radiation killing
        occur simultaneously.

        **TCP**

        TCP is based on a normalized clonogenic burden model and an initial
        TCP of 10%. It is intended for comparative exploration of parameter
        changes rather than clinical TCP prediction.

        **Residual resistant burden**

        Residual resistant burden is expressed as the percentage of the
        initial resistant tumour burden remaining.

        This is intentionally different from resistant composition.

        A tumour can become 100% resistant in composition while the absolute
        resistant tumour burden is simultaneously becoming very small.

        **Clinical limitations**

        This model does not replace lesion-level dosimetry. Real Lu-177 PSMA
        dosimetry depends on lesion-specific uptake, time-activity curves,
        residence time, spatial dose heterogeneity, cross-dose, tumour
        geometry, absorbed fractions, biological repair, fractionation,
        patient-specific kinetics and other factors.
        """
    )


# =============================================================================
# DATA EXPORT
# =============================================================================

st.markdown(
    '<h3 class="section-heading">Export results</h3>',
    unsafe_allow_html=True,
)


# =============================================================================
# CSV
# =============================================================================

csv_df = pd.DataFrame(
    {
        "time_days": time_days,

        "physical_dose_rate_Gy_per_day": (
            physical_dose_rate_gy_day
        ),

        "physical_dose_rate_Gy_per_hour": (
            physical_dose_rate_gy_day / 24.0
        ),

        "effective_dose_rate_Gy_per_day": (
            effective_dose_rate_gy_day
        ),

        "effective_dose_rate_Gy_per_hour": (
            effective_dose_rate_gy_day / 24.0
        ),

        "critical_dose_rate_Gy_per_day": (
            np.full_like(
                time_days,
                critical_dose_rate_gy_day,
            )
        ),

        "dose_rate_ratio_to_critical": (
            physical_dose_rate_gy_day
            / critical_dose_rate_gy_day
        ),

        "dose_rate_effectiveness": (
            effectiveness
        ),

        "total_tumour_burden_ml": (
            tumour_results["total_burden"]
        ),

        "sensitive_tumour_burden_ml": (
            tumour_results["sensitive"]
        ),

        "resistant_tumour_burden_ml": (
            tumour_results["resistant"]
        ),

        "initial_resistant_burden_ml": (
            np.full_like(
                time_days,
                tumour_results[
                    "initial_resistant_burden"
                ],
            )
        ),

        "residual_resistant_burden_percent": (
            tumour_results[
                "residual_resistant_burden_percent"
            ]
        ),

        "resistant_composition_percent": (
            tumour_results[
                "resistant_composition_fraction"
            ]
            * 100.0
        ),

        "TCP_percent": (
            tumour_results["tcp"]
            * 100.0
        ),

        "initial_tumour_burden_ml": (
            np.full_like(
                time_days,
                initial_burden_ml,
            )
        ),

        "tumour_uptake_percent": (
            np.full_like(
                time_days,
                tumour_uptake_percent,
            )
        ),

        "activity_per_cycle_GBq": (
            np.full_like(
                time_days,
                activity_gbq,
            )
        ),

        "number_of_cycles": (
            np.full_like(
                time_days,
                n_cycles,
                dtype=int,
            )
        ),

        "cycle_interval_days": (
            np.full_like(
                time_days,
                interval_days,
            )
        ),

        "dose_scale": (
            np.full_like(
                time_days,
                summary["dose_scale"],
            )
        ),
    }
)

csv_data = csv_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download model results (CSV)",
    data=csv_data,
    file_name="Lu177_PSMA_interactive_model_results.csv",
    mime="text/csv",
)


# =============================================================================
# HIGH-RESOLUTION PNG EXPORT
# =============================================================================

png_buffer = io.BytesIO()

fig_export, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10),
    dpi=600,
)

# -------------------------------------------------------------------------
# Export plot 1
# -------------------------------------------------------------------------

ax = axes[0, 0]

ax.plot(
    time_days,
    physical_dose_rate_gy_day,
    color=COLOR_BLUE,
    linewidth=2.0,
    label="Physical dose rate",
)

ax.plot(
    time_days,
    effective_dose_rate_gy_day,
    color=COLOR_ORANGE,
    linewidth=1.8,
    linestyle="--",
    label="Effective dose rate",
)

ax.axhline(
    critical_dose_rate_gy_day,
    color=COLOR_RED,
    linewidth=1.5,
    linestyle=":",
    label="Critical dose rate",
)

add_treatment_markers(
    ax,
    administration_times,
)

ax.set_xlabel(
    "Time (days)"
)

ax.set_ylabel(
    "Dose rate (Gy/day)"
)

ax.set_title(
    "Physical vs effective dose rate",
    fontweight="bold",
)

ax.legend(
    loc="best",
    framealpha=0.9,
)

style_axis(ax)


# -------------------------------------------------------------------------
# Export plot 2
# -------------------------------------------------------------------------

ax = axes[0, 1]

ax.plot(
    time_days,
    tumour_results["total_burden"],
    color=COLOR_BLUE,
    linewidth=2.0,
    label="Total tumour",
)

ax.plot(
    time_days,
    tumour_results["sensitive"],
    color=COLOR_GREEN,
    linewidth=1.8,
    linestyle="--",
    label="Sensitive tumour",
)

ax.plot(
    time_days,
    tumour_results["resistant"],
    color=COLOR_ORANGE,
    linewidth=1.8,
    linestyle=":",
    label="Resistant tumour",
)

add_treatment_markers(
    ax,
    administration_times,
)

ax.set_xlabel(
    "Time (days)"
)

ax.set_ylabel(
    "Tumour burden (mL)"
)

ax.set_title(
    "Tumour burden trajectory",
    fontweight="bold",
)

ax.legend(
    loc="best",
    framealpha=0.9,
)

style_axis(ax)


# -------------------------------------------------------------------------
# Export plot 3
# -------------------------------------------------------------------------

ax = axes[1, 0]

ax.plot(
    time_days,
    tumour_results["sensitive"],
    color=COLOR_GREEN,
    linewidth=1.9,
    label="Sensitive tumour (mL)",
)

ax.plot(
    time_days,
    tumour_results["resistant"],
    color=COLOR_ORANGE,
    linewidth=1.9,
    linestyle="--",
    label="Resistant tumour (mL)",
)

ax.set_xlabel(
    "Time (days)"
)

ax.set_ylabel(
    "Tumour burden (mL)"
)

ax.set_title(
    "Sensitive and resistant tumour populations",
    fontweight="bold",
)

ax2 = ax.twinx()

ax2.plot(
    time_days,
    tumour_results[
        "residual_resistant_burden_percent"
    ],
    color=COLOR_PURPLE,
    linewidth=1.8,
    linestyle=":",
    label=(
        "Residual resistant burden "
        "(% of initial)"
    ),
)

ax2.set_ylabel(
    "Residual resistant burden (% of initial)",
    color=COLOR_PURPLE,
)

ax2.tick_params(
    axis="y",
    colors=COLOR_PURPLE,
)

ax2.spines["right"].set_color(
    COLOR_PURPLE
)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="best",
    framealpha=0.9,
)

add_treatment_markers(
    ax,
    administration_times,
)

style_axis(ax)


# -------------------------------------------------------------------------
# Export plot 4
# -------------------------------------------------------------------------

ax = axes[1, 1]

ax.plot(
    time_days,
    tumour_results["tcp"] * 100.0,
    color=COLOR_BLUE,
    linewidth=2.0,
    label="TCP",
)

ax.axhline(
    50.0,
    color=COLOR_ORANGE,
    linewidth=1.5,
    linestyle="--",
    label="50% TCP",
)

ax.axhline(
    90.0,
    color=COLOR_GREEN,
    linewidth=1.5,
    linestyle=":",
    label="90% TCP",
)

add_treatment_markers(
    ax,
    administration_times,
)

ax.set_ylim(
    0.0,
    105.0,
)

ax.set_xlabel(
    "Time (days)"
)

ax.set_ylabel(
    "Tumour control probability (%)"
)

ax.set_title(
    "Tumour control probability",
    fontweight="bold",
)

ax.legend(
    loc="best",
    framealpha=0.9,
)

style_axis(ax)


fig_export.tight_layout()

fig_export.savefig(
    png_buffer,
    format="png",
    dpi=600,
    bbox_inches="tight",
    facecolor=PLOT_BACKGROUND,
)

plt.close(
    fig_export
)

png_buffer.seek(0)

st.download_button(
    label="Download high-resolution plots (600 dpi PNG)",
    data=png_buffer,
    file_name="Lu177_PSMA_interactive_model_600dpi.png",
    mime="image/png",
)


# =============================================================================
# FOOTNOTE
# =============================================================================

st.caption(
    "Exploratory Lu-177 PSMA radiobiological model — "
    "not for clinical treatment planning or patient-specific dosing."
)
