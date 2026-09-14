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
)


# =============================================================================
# CUSTOM STREAMLIT STYLING
# =============================================================================

st.markdown(
    """
    <style>

    /* ------------------------------------------------------------------ */
    /* General spacing                                                     */
    /* ------------------------------------------------------------------ */

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* ------------------------------------------------------------------ */
    /* Key result metrics                                                  */
    /* ------------------------------------------------------------------ */

    .key-metric [data-testid="stMetricValue"] {
        font-size: 1.55rem;
        font-weight: 600;
    }

    .key-metric [data-testid="stMetricLabel"] {
        font-size: 0.88rem;
        font-weight: 500;
    }

    /* ------------------------------------------------------------------ */
    /* Secondary metrics                                                   */
    /* ------------------------------------------------------------------ */

    .secondary-metric [data-testid="stMetricValue"] {
        font-size: 1.05rem;
        font-weight: 600;
        color: #E6C229;
    }

    .secondary-metric [data-testid="stMetricLabel"] {
        font-size: 0.72rem;
        font-weight: 500;
    }

    .secondary-metric [data-testid="stMetricDelta"] {
        font-size: 0.70rem;
    }

    /* ------------------------------------------------------------------ */
    /* Key-result status pills                                             */
    /* ------------------------------------------------------------------ */

    .key-status {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        line-height: 1.25;
        margin-top: 2px;
        margin-bottom: 2px;
    }

    .key-status-positive {
        color: #166534;
        background: #DCFCE7;
    }

    .key-status-negative {
        color: #991B1B;
        background: #FEE2E2;
    }

    .key-status-neutral {
        color: #4B5563;
        background: #F3F4F6;
    }

    .key-caption {
        font-size: 0.70rem;
        color: #777777;
        line-height: 1.25;
    }

    /* ------------------------------------------------------------------ */
    /* Sidebar section headings                                           */
    /* ------------------------------------------------------------------ */

    .sidebar-section {
        font-size: 1.00rem;
        font-weight: 700;
        margin-top: 0.9rem;
        margin-bottom: 0.35rem;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.35);
    }

    /* ------------------------------------------------------------------ */
    /* Expander headings                                                   */
    /* ------------------------------------------------------------------ */

    .streamlit-expanderHeader {
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# MODEL CONSTANTS
# =============================================================================

LU177_HALF_LIFE_DAYS = 6.647
LU177_LAMBDA_PER_DAY = np.log(2.0) / LU177_HALF_LIFE_DAYS

# Reference physical dose-rate conversion.
#
# This is defined for:
#   234 mL total metastatic tumour burden
#   1% total tumour uptake
#
# The dose-rate scaling is then adjusted for tumour burden and uptake.
DOSE_PER_GBQ_GY = 0.50

REFERENCE_METASTATIC_BURDEN_ML = 234.0

DEFAULT_TUMOUR_UPTAKE_PERCENT = 1.0
TUMOUR_UPTAKE_MIN_PERCENT = 0.1
TUMOUR_UPTAKE_MAX_PERCENT = 10.0

# Numerical simulation
DT_DAYS = 0.05
FOLLOW_UP_DAYS = 60.0

# Default tumour parameters
DEFAULT_BURDEN = 234.0

# Radiobiology
DEFAULT_ALPHA = 0.10
DEFAULT_BETA_ALPHA = 0.10
DEFAULT_TREP = 30.0

DEFAULT_SENSITIVE_FRACTION = 0.75
DEFAULT_RESISTANT_TK_MULTIPLIER = 1.50
DEFAULT_RESISTANT_RADIO_FACTOR = 0.25
DEFAULT_REPOPULATION_KICKOFF = 5.0

# Treatment
DEFAULT_ACTIVITY = 7.4
DEFAULT_CYCLES = 4
DEFAULT_INTERVAL = 7

# Dose-effectiveness model
DEFAULT_GAMMA = 1.0

# TCP normalisation
#
# This is an exploratory normalisation rather than a patient-specific
# clinical TCP prediction.
INITIAL_TCP = 0.10
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

COLOR_SECONDARY = "#E6C229"


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
# NUMERICAL INTEGRATION
# =============================================================================

def integrate_trapezoid(y, x):
    """
    NumPy-version-compatible trapezoidal integration.
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)

    return np.trapz(y, x)


# =============================================================================
# DOSE SCALING
# =============================================================================

def calculate_dose_scaling(
    initial_burden_ml,
    tumour_uptake_percent,
):
    """
    Calculate the phenomenological dose scaling.

    Reference condition:
        234 mL tumour burden
        1% total tumour uptake

    At fixed total tumour uptake:
        - smaller tumour burden -> higher average dose rate
        - larger tumour burden -> lower average dose rate

    This is an exploratory model and is not a substitute for lesion-level
    Lu-177 PSMA dosimetry.
    """

    uptake_scale = (
        tumour_uptake_percent
        / DEFAULT_TUMOUR_UPTAKE_PERCENT
    )

    burden_scale = (
        REFERENCE_METASTATIC_BURDEN_ML
        / initial_burden_ml
    )

    dose_scale = uptake_scale * burden_scale

    return uptake_scale, burden_scale, dose_scale


# =============================================================================
# TREATMENT TIMES
# =============================================================================

def calculate_treatment_times(
    n_cycles,
    interval_days,
):
    """
    Return treatment administration times.
    """

    return np.arange(n_cycles, dtype=float) * interval_days


# =============================================================================
# PHYSICAL DOSE RATE
# =============================================================================

def calculate_physical_dose_rate(
    time_days,
    activity_gbq,
    treatment_times,
    initial_burden_ml,
    tumour_uptake_percent,
):
    """
    Calculate total physical dose rate from all Lu-177 administrations.

    The dose rate is scaled using:
        activity
        tumour uptake
        total metastatic tumour burden
    """

    (
        uptake_scale,
        burden_scale,
        dose_scale,
    ) = calculate_dose_scaling(
        initial_burden_ml,
        tumour_uptake_percent,
    )

    dose_rate_gy_day = np.zeros_like(
        time_days,
        dtype=float,
    )

    for treatment_time in treatment_times:

        mask = time_days >= treatment_time

        elapsed = (
            time_days[mask]
            - treatment_time
        )

        activity_remaining = (
            activity_gbq
            * np.exp(
                -LU177_LAMBDA_PER_DAY
                * elapsed
            )
        )

        dose_rate_gy_day[mask] += (
            activity_remaining
            * DOSE_PER_GBQ_GY
            * dose_scale
        )

    return dose_rate_gy_day


# =============================================================================
# CRITICAL DOSE RATE
# =============================================================================

def calculate_critical_dose_rate(
    alpha,
    trep_days,
):
    """
    Critical dose rate:

        Rcrit = ln(2) / (alpha * Trep_hours)

    Returned in Gy/h.
    """

    trep_hours = trep_days * 24.0

    return (
        np.log(2.0)
        / (alpha * trep_hours)
    )


# =============================================================================
# EFFECTIVE DOSE RATE
# =============================================================================

def calculate_effective_dose_rate(
    physical_dose_rate_gy_day,
    critical_dose_rate_gy_h,
    gamma,
):
    """
    Convert physical dose rate into an exploratory effective dose rate.

    R = physical dose rate / critical dose rate

    E = min(1, R^gamma)

    Reffective = Rphysical * E
    """

    critical_dose_rate_gy_day = (
        critical_dose_rate_gy_h * 24.0
    )

    ratio = np.divide(
        physical_dose_rate_gy_day,
        critical_dose_rate_gy_day,
        out=np.zeros_like(
            physical_dose_rate_gy_day
        ),
        where=critical_dose_rate_gy_day > 0,
    )

    effectiveness = np.minimum(
        1.0,
        ratio ** gamma,
    )

    effective_dose_rate_gy_day = (
        physical_dose_rate_gy_day
        * effectiveness
    )

    return (
        ratio,
        effectiveness,
        effective_dose_rate_gy_day,
    )


# =============================================================================
# TUMOUR DYNAMICS
# =============================================================================

def calculate_tumour_dynamics(
    time_days,
    physical_dose_rate_gy_day,
    initial_burden_ml,
    sensitive_fraction,
    alpha,
    beta_alpha,
    trep_days,
    resistant_tk_multiplier,
    resistant_radio_factor,
    repopulation_kickoff_days,
):
    """
    Two-compartment phenomenological tumour model:

        Sensitive tumour
        Resistant tumour

    Both compartments can repopulate after the specified kickoff time.

    Radiation response uses an LQ-style survival formulation.
    """

    n = len(time_days)

    sensitive = np.zeros(n)
    resistant = np.zeros(n)

    initial_sensitive = (
        initial_burden_ml
        * sensitive_fraction
    )

    initial_resistant = (
        initial_burden_ml
        * (1.0 - sensitive_fraction)
    )

    sensitive[0] = initial_sensitive
    resistant[0] = initial_resistant

    beta_sensitive = (
        alpha
        * beta_alpha
    )

    alpha_resistant = (
        alpha
        * resistant_radio_factor
    )

    beta_resistant = (
        beta_sensitive
        * resistant_radio_factor
    )

    sensitive_growth_rate = (
        np.log(2.0)
        / trep_days
    )

    resistant_growth_rate = (
        np.log(2.0)
        / (
            trep_days
            * resistant_tk_multiplier
        )
    )

    for i in range(1, n):

        dt = (
            time_days[i]
            - time_days[i - 1]
        )

        sensitive_current = sensitive[i - 1]
        resistant_current = resistant[i - 1]

        # --------------------------------------------------------------
        # Repopulation
        # --------------------------------------------------------------

        if (
            time_days[i]
            >= repopulation_kickoff_days
        ):

            sensitive_current *= np.exp(
                sensitive_growth_rate * dt
            )

            resistant_current *= np.exp(
                resistant_growth_rate * dt
            )

        # --------------------------------------------------------------
        # Radiation dose during interval
        # --------------------------------------------------------------

        dose_interval_gy = (
            physical_dose_rate_gy_day[i]
            * dt
        )

        # --------------------------------------------------------------
        # LQ survival
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Apply radiation killing
        # --------------------------------------------------------------

        sensitive[i] = (
            sensitive_current
            * survival_sensitive
        )

        resistant[i] = (
            resistant_current
            * survival_resistant
        )

    total_burden = (
        sensitive
        + resistant
    )

    # --------------------------------------------------------------
    # Resistant disease relative to its original burden
    # --------------------------------------------------------------

    if initial_resistant > 0:

        residual_resistant_burden_percent = (
            resistant
            / initial_resistant
        ) * 100.0

    else:

        residual_resistant_burden_percent = (
            np.zeros_like(resistant)
        )

    # --------------------------------------------------------------
    # Resistant composition
    # --------------------------------------------------------------

    resistant_composition_fraction = np.divide(
        resistant,
        total_burden,
        out=np.zeros_like(resistant),
        where=total_burden > 0,
    )

    return {
        "sensitive": sensitive,
        "resistant": resistant,
        "total": total_burden,
        "initial_sensitive": initial_sensitive,
        "initial_resistant": initial_resistant,
        "residual_resistant_burden_percent":
            residual_resistant_burden_percent,
        "resistant_composition_fraction":
            resistant_composition_fraction,
    }


# =============================================================================
# TCP
# =============================================================================

def calculate_tcp(
    total_burden_ml,
    initial_burden_ml,
):
    """
    Exploratory TCP model.

    The model normalises baseline TCP to 10%.

    Relative tumour burden is used as a surrogate for relative
    surviving clonogenic burden.
    """

    relative_burden = np.divide(
        total_burden_ml,
        initial_burden_ml,
        out=np.zeros_like(total_burden_ml),
        where=initial_burden_ml > 0,
    )

    surviving_clonogens = (
        INITIAL_CLONOGENIC_BURDEN
        * relative_burden
    )

    tcp = np.exp(
        -surviving_clonogens
    )

    return tcp


# =============================================================================
# SUMMARY
# =============================================================================

def calculate_summary(
    time_days,
    physical_dose_rate_gy_day,
    effective_dose_rate_gy_day,
    critical_dose_rate_gy_h,
    tumour,
    tcp,
    treatment_times,
    initial_burden_ml,
):
    """
    Generate summary metrics used by the dashboard.
    """

    total_burden = tumour["total"]

    residual_resistant = (
        tumour[
            "residual_resistant_burden_percent"
        ]
    )

    resistant_composition = (
        tumour[
            "resistant_composition_fraction"
        ]
    )

    # --------------------------------------------------------------
    # Dose-rate metrics
    # --------------------------------------------------------------

    physical_dose_rate_gy_h = (
        physical_dose_rate_gy_day / 24.0
    )

    effective_dose_rate_gy_h = (
        effective_dose_rate_gy_day / 24.0
    )

    peak_physical_rate_gy_h = (
        np.max(
            physical_dose_rate_gy_h
        )
    )

    # --------------------------------------------------------------
    # Time above critical dose rate
    # --------------------------------------------------------------

    above_critical = (
        physical_dose_rate_gy_h
        >= critical_dose_rate_gy_h
    )

    time_above_critical_days = (
        np.sum(above_critical)
        * DT_DAYS
    )

    # --------------------------------------------------------------
    # Longest continuous period above critical
    # --------------------------------------------------------------

    longest_continuous_above = 0.0
    current_duration = 0.0

    for value in above_critical:

        if value:

            current_duration += DT_DAYS

            longest_continuous_above = max(
                longest_continuous_above,
                current_duration,
            )

        else:

            current_duration = 0.0

    # --------------------------------------------------------------
    # Dose integration
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Tumour metrics
    # --------------------------------------------------------------

    min_burden_index = np.argmin(
        total_burden
    )

    minimum_tumour_burden = (
        total_burden[min_burden_index]
    )

    minimum_tumour_burden_day = (
        time_days[min_burden_index]
    )

    final_burden = total_burden[-1]

    final_residual_resistant = (
        residual_resistant[-1]
    )

    minimum_residual_resistant = (
        np.min(residual_resistant)
    )

    final_resistant_composition = (
        resistant_composition[-1]
        * 100.0
    )

    # --------------------------------------------------------------
    # TCP
    # --------------------------------------------------------------

    max_tcp_index = np.argmax(tcp)

    maximum_tcp = tcp[max_tcp_index]

    maximum_tcp_day = (
        time_days[max_tcp_index]
    )

    final_tcp = tcp[-1]

    return {
        "critical_dose_rate_gy_h":
            critical_dose_rate_gy_h,

        "peak_physical_rate_gy_h":
            peak_physical_rate_gy_h,

        "time_above_critical_days":
            time_above_critical_days,

        "longest_continuous_above_days":
            longest_continuous_above,

        "cumulative_physical_dose_gy":
            cumulative_physical_dose,

        "cumulative_effective_dose_gy":
            cumulative_effective_dose,

        "minimum_tumour_burden_ml":
            minimum_tumour_burden,

        "minimum_tumour_burden_day":
            minimum_tumour_burden_day,

        "final_tumour_burden_ml":
            final_burden,

        "maximum_tcp":
            maximum_tcp,

        "maximum_tcp_day":
            maximum_tcp_day,

        "final_tcp":
            final_tcp,

        "final_residual_resistant_burden_percent":
            final_residual_resistant,

        "minimum_residual_resistant_burden_percent":
            minimum_residual_resistant,

        "final_resistant_composition_percent":
            final_resistant_composition,

        "final_sensitive_burden_ml":
            tumour["sensitive"][-1],

        "final_resistant_burden_ml":
            tumour["resistant"][-1],

        "initial_resistant_burden_ml":
            tumour["initial_resistant"],
    }


# =============================================================================
# PLOT STYLING
# =============================================================================

def style_axis(
    ax,
    xlabel=None,
    ylabel=None,
):
    """
    Apply common black-background plot styling.
    """

    ax.set_facecolor(PLOT_BACKGROUND)

    if xlabel is not None:
        ax.set_xlabel(
            xlabel,
            color=PLOT_TEXT,
        )

    if ylabel is not None:
        ax.set_ylabel(
            ylabel,
            color=PLOT_TEXT,
        )

    ax.tick_params(
        colors=PLOT_TEXT,
        which="both",
    )

    for spine in ax.spines.values():
        spine.set_color(PLOT_TEXT)

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35,
        color=PLOT_GRID,
    )

    ax.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.4,
        alpha=0.20,
        color=PLOT_GRID,
    )

    ax.minorticks_on()

    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=8)
    )

    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=7)
    )


def style_legend(ax):
    """
    Style legend for dark plots.
    """

    legend = ax.legend(
        frameon=True,
        fontsize=8,
    )

    if legend is not None:

        legend.get_frame().set_facecolor(
            PLOT_BACKGROUND
        )

        legend.get_frame().set_edgecolor(
            PLOT_GRID
        )

        for text in legend.get_texts():
            text.set_color(PLOT_TEXT)


def add_treatment_markers(
    ax,
    treatment_times,
):
    """
    Add grey dotted treatment markers.
    """

    for treatment_time in treatment_times:

        ax.axvline(
            treatment_time,
            color=PLOT_TREATMENT,
            linestyle=":",
            linewidth=0.9,
            alpha=0.65,
        )


# =============================================================================
# KEY RESULT FORMATTING
# =============================================================================

def key_result_status(
    text,
    status="neutral",
):
    """
    Render a green/red/neutral status pill.
    """

    return (
        f'<div class="key-status '
        f'key-status-{status}">'
        f'{text}'
        f'</div>'
    )


def key_result_caption(text):
    """
    Render a small explanatory caption.
    """

    return (
        f'<div class="key-caption">'
        f'{text}'
        f'</div>'
    )


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("Model controls")


# =============================================================================
# TUMOUR PARAMETERS
# =============================================================================

st.sidebar.markdown(
    '<div class="sidebar-section">'
    'Tumour parameters'
    '</div>',
    unsafe_allow_html=True,
)

initial_burden_ml = st.sidebar.slider(
    "Initial metastatic burden (mL)",
    min_value=10.0,
    max_value=1500.0,
    value=234.0,
    step=1.0,
    help=(
        "Total metastatic tumour burden. "
        "This is treated as a phenomenological "
        "whole-body tumour burden rather than "
        "a single solid tumour."
    ),
)

tumour_uptake_percent = st.sidebar.slider(
    "Total tumour uptake (%)",
    min_value=TUMOUR_UPTAKE_MIN_PERCENT,
    max_value=TUMOUR_UPTAKE_MAX_PERCENT,
    value=DEFAULT_TUMOUR_UPTAKE_PERCENT,
    step=0.1,
    format="%.1f",
    help=(
        "Fraction of administered activity attributed "
        "to the total metastatic tumour burden."
    ),
)

sensitive_fraction = st.sidebar.slider(
    "Sensitive fraction",
    min_value=0.50,
    max_value=0.95,
    value=DEFAULT_SENSITIVE_FRACTION,
    step=0.01,
    format="%.2f",
    help=(
        "Initial fraction of tumour burden "
        "assigned to the radiation-sensitive compartment."
    ),
)


# =============================================================================
# TREATMENT PARAMETERS
# =============================================================================

st.sidebar.markdown(
    '<div class="sidebar-section">'
    'Treatment parameters'
    '</div>',
    unsafe_allow_html=True,
)

activity_gbq = st.sidebar.slider(
    "Activity / cycle (GBq)",
    min_value=1.0,
    max_value=15.0,
    value=DEFAULT_ACTIVITY,
    step=0.1,
    format="%.1f",
    help="Administered Lu-177 activity per treatment cycle.",
)

n_cycles = st.sidebar.slider(
    "Number of cycles",
    min_value=1,
    max_value=8,
    value=DEFAULT_CYCLES,
    step=1,
    help="Number of Lu-177 treatment administrations.",
)

interval_days = st.sidebar.slider(
    "Cycle interval (days)",
    min_value=1,
    max_value=42,
    value=DEFAULT_INTERVAL,
    step=1,
    help="Time between treatment administrations.",
)

gamma = st.sidebar.slider(
    "Effectiveness gamma",
    min_value=0.25,
    max_value=2.00,
    value=DEFAULT_GAMMA,
    step=0.05,
    format="%.2f",
    help=(
        "Controls the exploratory relationship between "
        "physical dose rate and effective dose rate."
    ),
)


# =============================================================================
# RADIOBIOLOGY / RESPONSE PARAMETERS
# =============================================================================

st.sidebar.markdown(
    '<div class="sidebar-section">'
    'Radiobiology / response'
    '</div>',
    unsafe_allow_html=True,
)

alpha = st.sidebar.slider(
    "Alpha",
    min_value=0.001,
    max_value=0.50,
    value=DEFAULT_ALPHA,
    step=0.001,
    format="%.3f",
    help=(
        "Linear radiation sensitivity parameter "
        "used in the exploratory LQ model."
    ),
)

trep_days = st.sidebar.slider(
    "Trep",
    min_value=10.0,
    max_value=100.0,
    value=DEFAULT_TREP,
    step=1.0,
    help=(
        "Tumour repopulation doubling time "
        "for the sensitive population."
    ),
)

resistant_tk_multiplier = st.sidebar.slider(
    "Tk / Trep",
    min_value=1.20,
    max_value=1.80,
    value=DEFAULT_RESISTANT_TK_MULTIPLIER,
    step=0.05,
    format="%.2f",
    help=(
        "Multiplier controlling the resistant "
        "population's effective repopulation time."
    ),
)

resistant_radio_factor = st.sidebar.slider(
    "Resistant radio factor",
    min_value=0.05,
    max_value=1.00,
    value=DEFAULT_RESISTANT_RADIO_FACTOR,
    step=0.05,
    format="%.2f",
    help=(
        "Relative radiation sensitivity of "
        "the resistant population."
    ),
)

repopulation_kickoff_days = st.sidebar.slider(
    "Repopulation kickoff",
    min_value=3.0,
    max_value=10.0,
    value=DEFAULT_REPOPULATION_KICKOFF,
    step=0.5,
    format="%.1f",
    help=(
        "Time after which tumour repopulation "
        "is allowed to occur."
    ),
)


# =============================================================================
# SIMULATION
# =============================================================================

treatment_times = calculate_treatment_times(
    n_cycles,
    interval_days,
)

simulation_days = max(
    FOLLOW_UP_DAYS,
    (
        (n_cycles - 1)
        * interval_days
        + FOLLOW_UP_DAYS
    ),
)

time_days = np.arange(
    0.0,
    simulation_days + DT_DAYS,
    DT_DAYS,
)


# =============================================================================
# DOSE MODEL
# =============================================================================

physical_dose_rate_gy_day = (
    calculate_physical_dose_rate(
        time_days=time_days,
        activity_gbq=activity_gbq,
        treatment_times=treatment_times,
        initial_burden_ml=initial_burden_ml,
        tumour_uptake_percent=tumour_uptake_percent,
    )
)

critical_dose_rate_gy_h = (
    calculate_critical_dose_rate(
        alpha=alpha,
        trep_days=trep_days,
    )
)

(
    dose_rate_ratio,
    dose_effectiveness,
    effective_dose_rate_gy_day,
) = calculate_effective_dose_rate(
    physical_dose_rate_gy_day=physical_dose_rate_gy_day,
    critical_dose_rate_gy_h=critical_dose_rate_gy_h,
    gamma=gamma,
)


# =============================================================================
# TUMOUR MODEL
# =============================================================================

tumour = calculate_tumour_dynamics(
    time_days=time_days,
    physical_dose_rate_gy_day=physical_dose_rate_gy_day,
    initial_burden_ml=initial_burden_ml,
    sensitive_fraction=sensitive_fraction,
    alpha=alpha,
    beta_alpha=DEFAULT_BETA_ALPHA,
    trep_days=trep_days,
    resistant_tk_multiplier=resistant_tk_multiplier,
    resistant_radio_factor=resistant_radio_factor,
    repopulation_kickoff_days=repopulation_kickoff_days,
)


# =============================================================================
# TCP
# =============================================================================

tcp = calculate_tcp(
    total_burden_ml=tumour["total"],
    initial_burden_ml=initial_burden_ml,
)


# =============================================================================
# SUMMARY
# =============================================================================

summary = calculate_summary(
    time_days=time_days,
    physical_dose_rate_gy_day=physical_dose_rate_gy_day,
    effective_dose_rate_gy_day=effective_dose_rate_gy_day,
    critical_dose_rate_gy_h=critical_dose_rate_gy_h,
    tumour=tumour,
    tcp=tcp,
    treatment_times=treatment_times,
    initial_burden_ml=initial_burden_ml,
)


# =============================================================================
# HEADER
# =============================================================================

st.title("Lu-177 PSMA Interactive Model Explorer")

st.caption(
    "Exploratory tumour-response, dose-rate, resistant-burden "
    "and TCP model for Lu-177 PSMA therapy."
)


# =============================================================================
# KEY RESULTS
# =============================================================================

st.subheader("Key results")

key_cols = st.columns(4)


# -----------------------------------------------------------------------------
# Minimum tumour burden
# -----------------------------------------------------------------------------

with key_cols[0]:

    st.markdown(
        '<div class="key-metric">',
        unsafe_allow_html=True,
    )

    st.metric(
        "Minimum tumour burden",
        f"{summary['minimum_tumour_burden_ml']:.1f} mL",
    )

    burden_change_percent = (
        (
            summary["minimum_tumour_burden_ml"]
            / initial_burden_ml
        )
        - 1.0
    ) * 100.0

    if burden_change_percent < 0:

        status_text = (
            f"↓ {abs(burden_change_percent):.1f}% "
            f"from {initial_burden_ml:.0f} mL baseline"
        )

        status = "positive"

    elif burden_change_percent > 0:

        status_text = (
            f"↑ {burden_change_percent:.1f}% "
            f"from {initial_burden_ml:.0f} mL baseline"
        )

        status = "negative"

    else:

        status_text = "No change from baseline"
        status = "neutral"

    st.markdown(
        key_result_status(
            status_text,
            status,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        key_result_caption(
            f"Minimum reached at day "
            f"{summary['minimum_tumour_burden_day']:.1f}"
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Maximum TCP
# -----------------------------------------------------------------------------

with key_cols[1]:

    st.markdown(
        '<div class="key-metric">',
        unsafe_allow_html=True,
    )

    st.metric(
        "Maximum TCP",
        f"{summary['maximum_tcp'] * 100.0:.1f}%",
    )

    tcp_change_points = (
        summary["maximum_tcp"] * 100.0
        - INITIAL_TCP * 100.0
    )

    if tcp_change_points > 0:

        status_text = (
            f"↑ {tcp_change_points:.1f} percentage points "
            f"from 10% baseline"
        )

        status = "positive"

    elif tcp_change_points < 0:

        status_text = (
            f"↓ {abs(tcp_change_points):.1f} percentage points "
            f"from 10% baseline"
        )

        status = "negative"

    else:

        status_text = (
            "No change from 10% baseline"
        )

        status = "neutral"

    st.markdown(
        key_result_status(
            status_text,
            status,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        key_result_caption(
            f"Maximum reached at day "
            f"{summary['maximum_tcp_day']:.1f}"
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Residual resistant burden
# -----------------------------------------------------------------------------

with key_cols[2]:

    st.markdown(
        '<div class="key-metric">',
        unsafe_allow_html=True,
    )

    st.metric(
        "Residual resistant burden",
        f"{summary['final_residual_resistant_burden_percent']:.1f}%",
    )

    resistant_change = (
        summary[
            "final_residual_resistant_burden_percent"
        ]
        - 100.0
    )

    if resistant_change < 0:

        status_text = (
            f"↓ {abs(resistant_change):.1f}% "
            "from initial resistant burden"
        )

        status = "positive"

    elif resistant_change > 0:

        status_text = (
            f"↑ {resistant_change:.1f}% "
            "from initial resistant burden"
        )

        status = "negative"

    else:

        status_text = (
            "No change from initial resistant burden"
        )

        status = "neutral"

    st.markdown(
        key_result_status(
            status_text,
            status,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        key_result_caption(
            "% of the initial resistant tumour burden remaining"
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        key_result_caption(
            f"Initial resistant burden: "
            f"{summary['initial_resistant_burden_ml']:.1f} mL "
            f"({(1.0 - sensitive_fraction) * 100.0:.0f}% of total)"
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Cumulative physical dose
# -----------------------------------------------------------------------------

with key_cols[3]:

    st.markdown(
        '<div class="key-metric">',
        unsafe_allow_html=True,
    )

    st.metric(
        "Cumulative physical dose",
        f"{summary['cumulative_physical_dose_gy']:.2f} Gy",
    )

    st.markdown(
        key_result_status(
            "Integrated physical dose",
            "neutral",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        key_result_caption(
            f"{activity_gbq:.1f} GBq × "
            f"{n_cycles} cycle"
            f"{'s' if n_cycles != 1 else ''}"
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# SECONDARY METRICS
# =============================================================================

st.subheader("Secondary metrics")

sec_cols = st.columns(4)


# -----------------------------------------------------------------------------
# Critical dose rate
# -----------------------------------------------------------------------------

with sec_cols[0]:

    st.markdown(
        '<div class="secondary-metric">',
        unsafe_allow_html=True,
    )

    st.metric(
        "Critical dose rate",
        f"{summary['critical_dose_rate_gy_h']:.4f} Gy/h",
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Peak physical rate
# -----------------------------------------------------------------------------

with sec_cols[1]:

    st.markdown(
        '<div class="secondary-metric">',
        unsafe_allow_html=True,
    )

    st.metric(
        "Peak physical rate",
        f"{summary['peak_physical_rate_gy_h']:.4f} Gy/h",
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Time above critical rate
# -----------------------------------------------------------------------------

with sec_cols[2]:

    st.markdown(
        '<div class="secondary-metric">',
        unsafe_allow_html=True,
    )

    st.metric(
        "Time above critical rate",
        f"{summary['time_above_critical_days']:.1f} days",
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Effective cumulative dose
# -----------------------------------------------------------------------------

with sec_cols[3]:

    st.markdown(
        '<div class="secondary-metric">',
        unsafe_allow_html=True,
    )

    st.metric(
        "Effective cumulative dose",
        f"{summary['cumulative_effective_dose_gy']:.2f} Gy",
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# DOSE SCALING INFORMATION
# =============================================================================

(
    uptake_scale,
    burden_scale,
    dose_scale,
) = calculate_dose_scaling(
    initial_burden_ml,
    tumour_uptake_percent,
)

st.caption(
    f"Dose scaling: uptake ×{uptake_scale:.2f} | "
    f"burden ×{burden_scale:.2f} | "
    f"combined ×{dose_scale:.2f} "
    f"(reference = 234 mL and 1% uptake)"
)


# =============================================================================
# MODEL TRAJECTORIES
# =============================================================================

st.subheader("Model trajectories")

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 9),
)

fig.patch.set_facecolor(
    PLOT_BACKGROUND
)


# =============================================================================
# TOP LEFT — DOSE RATE
# =============================================================================

ax = axes[0, 0]

ax.plot(
    time_days,
    physical_dose_rate_gy_day / 24.0,
    color=COLOR_BLUE,
    linewidth=2.0,
    label="Physical dose rate",
)

ax.plot(
    time_days,
    effective_dose_rate_gy_day / 24.0,
    color=COLOR_ORANGE,
    linewidth=2.0,
    linestyle="--",
    label="Effective dose rate",
)

ax.axhline(
    critical_dose_rate_gy_h,
    color=COLOR_RED,
    linestyle=":",
    linewidth=1.8,
    label="Critical dose rate",
)

add_treatment_markers(
    ax,
    treatment_times,
)

style_axis(
    ax,
    xlabel="Time (days)",
    ylabel="Dose rate (Gy/h)",
)

style_legend(ax)

ax.set_title(
    "Physical vs effective dose rate",
    color=PLOT_TEXT,
)


# =============================================================================
# TOP RIGHT — TUMOUR BURDEN
# =============================================================================

ax = axes[0, 1]

ax.plot(
    time_days,
    tumour["total"],
    color=COLOR_BLUE,
    linewidth=2.0,
    label="Total tumour",
)

ax.plot(
    time_days,
    tumour["sensitive"],
    color=COLOR_GREEN,
    linewidth=1.8,
    linestyle="--",
    label="Sensitive tumour",
)

ax.plot(
    time_days,
    tumour["resistant"],
    color=COLOR_ORANGE,
    linewidth=1.8,
    linestyle=":",
    label="Resistant tumour",
)

add_treatment_markers(
    ax,
    treatment_times,
)

style_axis(
    ax,
    xlabel="Time (days)",
    ylabel="Tumour burden (mL)",
)

style_legend(ax)

ax.set_title(
    "Tumour burden",
    color=PLOT_TEXT,
)


# =============================================================================
# BOTTOM LEFT — SENSITIVE / RESISTANT + RESIDUAL RESISTANT
# =============================================================================

ax = axes[1, 0]

ax.plot(
    time_days,
    tumour["sensitive"],
    color=COLOR_GREEN,
    linewidth=2.0,
    label="Sensitive tumour",
)

ax.plot(
    time_days,
    tumour["resistant"],
    color=COLOR_ORANGE,
    linewidth=2.0,
    linestyle="--",
    label="Resistant tumour",
)

add_treatment_markers(
    ax,
    treatment_times,
)

style_axis(
    ax,
    xlabel="Time (days)",
    ylabel="Tumour burden (mL)",
)

ax2 = ax.twinx()

ax2.plot(
    time_days,
    tumour[
        "residual_resistant_burden_percent"
    ],
    color=COLOR_PURPLE,
    linewidth=1.8,
    linestyle=":",
    label="Residual resistant burden",
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

ax2.set_ylim(
    bottom=0
)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

legend = ax.legend(
    lines1 + lines2,
    labels1 + labels2,
    frameon=True,
    fontsize=8,
)

legend.get_frame().set_facecolor(
    PLOT_BACKGROUND
)

legend.get_frame().set_edgecolor(
    PLOT_GRID
)

for text in legend.get_texts():
    text.set_color(PLOT_TEXT)

ax.set_title(
    "Sensitive and resistant tumour populations",
    color=PLOT_TEXT,
)


# =============================================================================
# BOTTOM RIGHT — TCP
# =============================================================================

ax = axes[1, 1]

ax.plot(
    time_days,
    tcp * 100.0,
    color=COLOR_BLUE,
    linewidth=2.2,
    label="TCP",
)

ax.axhline(
    50.0,
    color=COLOR_ORANGE,
    linestyle="--",
    linewidth=1.6,
    label="50% TCP",
)

ax.axhline(
    90.0,
    color=COLOR_GREEN,
    linestyle=":",
    linewidth=1.6,
    label="90% TCP",
)

add_treatment_markers(
    ax,
    treatment_times,
)

style_axis(
    ax,
    xlabel="Time (days)",
    ylabel="TCP (%)",
)

style_legend(ax)

ax.set_ylim(
    0,
    100,
)

ax.set_title(
    "Tumour control probability",
    color=PLOT_TEXT,
)


plt.tight_layout()


# =============================================================================
# DISPLAY FIGURE
# =============================================================================

st.pyplot(
    fig,
    use_container_width=True,
)


# =============================================================================
# DETAILED MODEL SUMMARY
# =============================================================================

with st.expander(
    "Detailed model summary"
):

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Tumour response")

        st.write(
            f"Initial tumour burden: "
            f"{initial_burden_ml:.1f} mL"
        )

        st.write(
            f"Minimum tumour burden: "
            f"{summary['minimum_tumour_burden_ml']:.1f} mL"
        )

        st.write(
            f"Minimum tumour burden day: "
            f"{summary['minimum_tumour_burden_day']:.1f}"
        )

        st.write(
            f"Final tumour burden: "
            f"{summary['final_tumour_burden_ml']:.1f} mL"
        )

        st.write(
            f"Initial resistant burden: "
            f"{summary['initial_resistant_burden_ml']:.1f} mL"
        )

        st.write(
            f"Final sensitive burden: "
            f"{summary['final_sensitive_burden_ml']:.1f} mL"
        )

        st.write(
            f"Final resistant burden: "
            f"{summary['final_resistant_burden_ml']:.1f} mL"
        )

        st.write(
            f"Final residual resistant burden: "
            f"{summary['final_residual_resistant_burden_percent']:.1f}%"
        )

        st.write(
            f"Minimum residual resistant burden: "
            f"{summary['minimum_residual_resistant_burden_percent']:.1f}%"
        )

        st.write(
            f"Final resistant composition: "
            f"{summary['final_resistant_composition_percent']:.1f}%"
        )

    with col2:

        st.markdown("### Radiation / TCP")

        st.write(
            f"Critical dose rate: "
            f"{summary['critical_dose_rate_gy_h']:.5f} Gy/h"
        )

        st.write(
            f"Peak physical dose rate: "
            f"{summary['peak_physical_rate_gy_h']:.5f} Gy/h"
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
            f"Maximum TCP: "
            f"{summary['maximum_tcp'] * 100.0:.2f}%"
        )

        st.write(
            f"Maximum TCP day: "
            f"{summary['maximum_tcp_day']:.2f}"
        )

        st.write(
            f"Final TCP: "
            f"{summary['final_tcp'] * 100.0:.2f}%"
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

        The initial tumour burden represents total metastatic tumour
        burden in mL rather than a single solid tumour.

        **Tumour uptake**

        Total tumour uptake represents the fraction of administered
        activity attributed to the total metastatic tumour burden.

        The uptake/burden scaling is phenomenological and is intended
        for exploratory modelling. Patient-specific Lu-177 PSMA
        dosimetry would require lesion-level uptake, time-activity
        curves, residence times, absorbed fractions and spatial
        heterogeneity.

        **Radiobiology**

        The sensitive and resistant compartments use an exploratory
        LQ-style radiation-response model with separate effective
        radiosensitivity.

        **Repopulation**

        Repopulation is represented using exponential growth after
        the specified kickoff time. The resistant compartment has a
        different effective doubling time controlled by Tk/Trep.

        **Critical dose rate**

        The critical dose rate is calculated from:

        Rcrit = ln(2) / (alpha × Trep)

        where Trep is converted to hours for the dose-rate calculation.

        **TCP**

        TCP is currently normalised to a 10% baseline and uses relative
        tumour burden as a surrogate for relative surviving clonogenic
        burden. This is an exploratory TCP formulation rather than a
        validated clinical TCP model.

        **Resistant burden**

        Residual resistant burden is reported as a percentage of the
        initial resistant tumour burden. This is distinct from resistant
        composition, which describes what fraction of the remaining
        tumour is resistant.

        A resistant composition approaching 100% does not necessarily
        mean that the absolute resistant tumour burden is increasing.
        It can occur when the sensitive population is eliminated more
        rapidly than the resistant population.
        """
    )


# =============================================================================
# CSV EXPORT
# =============================================================================

csv_data = pd.DataFrame(
    {
        "time_days": time_days,

        "physical_dose_rate_Gy_day":
            physical_dose_rate_gy_day,

        "physical_dose_rate_Gy_h":
            physical_dose_rate_gy_day / 24.0,

        "effective_dose_rate_Gy_day":
            effective_dose_rate_gy_day,

        "effective_dose_rate_Gy_h":
            effective_dose_rate_gy_day / 24.0,

        "critical_dose_rate_Gy_h":
            np.full_like(
                time_days,
                critical_dose_rate_gy_h,
            ),

        "dose_rate_ratio":
            dose_rate_ratio,

        "dose_effectiveness":
            dose_effectiveness,

        "total_tumour_burden_ml":
            tumour["total"],

        "sensitive_tumour_burden_ml":
            tumour["sensitive"],

        "resistant_tumour_burden_ml":
            tumour["resistant"],

        "initial_resistant_burden_ml":
            np.full_like(
                time_days,
                tumour["initial_resistant"],
            ),

        "residual_resistant_burden_percent":
            tumour[
                "residual_resistant_burden_percent"
            ],

        "resistant_composition_percent":
            tumour[
                "resistant_composition_fraction"
            ] * 100.0,

        "TCP_percent":
            tcp * 100.0,

        "initial_burden_ml":
            np.full_like(
                time_days,
                initial_burden_ml,
            ),

        "tumour_uptake_percent":
            np.full_like(
                time_days,
                tumour_uptake_percent,
            ),

        "uptake_scale":
            np.full_like(
                time_days,
                uptake_scale,
            ),

        "burden_scale":
            np.full_like(
                time_days,
                burden_scale,
            ),

        "dose_scale":
            np.full_like(
                time_days,
                dose_scale,
            ),
    }
)

csv_buffer = io.StringIO()

csv_data.to_csv(
    csv_buffer,
    index=False,
)

st.download_button(
    label="Download model results (CSV)",
    data=csv_buffer.getvalue(),
    file_name="Lu177_PSMA_model_results.csv",
    mime="text/csv",
)


# =============================================================================
# HIGH-RESOLUTION PNG EXPORT
# =============================================================================

png_buffer = io.BytesIO()

fig.savefig(
    png_buffer,
    format="png",
    dpi=600,
    bbox_inches="tight",
    facecolor=PLOT_BACKGROUND,
)

png_buffer.seek(0)

st.download_button(
    label="Download model plots (600 dpi PNG)",
    data=png_buffer,
    file_name="Lu177_PSMA_model_trajectories_600dpi.png",
    mime="image/png",
)
