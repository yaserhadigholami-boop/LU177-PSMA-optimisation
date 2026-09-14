import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import streamlit as st


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

    /* ============================================================
       KEY RESULT CARDS
       ============================================================ */

    .key-result-card {
        padding: 0.65rem 0.75rem 0.55rem 0.75rem;
        border-radius: 8px;
        background: #111111;
        border: 1px solid #333333;
        min-height: 125px;
    }

    .key-result-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #D0D0D0;
        margin-bottom: 0.25rem;
    }

    .key-result-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.15;
        margin-bottom: 0.35rem;
    }

    .key-result-status {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        line-height: 1.25;
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

    .key-result-caption {
        font-size: 0.67rem;
        color: #A0A0A0;
        margin-top: 0.35rem;
        line-height: 1.25;
    }


    /* ============================================================
       SECONDARY METRICS
       ============================================================ */

    .secondary-metric {
        padding: 0.20rem 0.30rem 0.15rem 0.30rem;
    }

    .secondary-metric [data-testid="stMetricValue"] {
        font-size: 1.05rem;
        font-weight: 600;
        color: #F2C94C;
        line-height: 1.1;
    }

    .secondary-metric [data-testid="stMetricLabel"] {
        font-size: 0.72rem;
        font-weight: 500;
        color: #B8B8B8;
    }

    .secondary-metric [data-testid="stMetricDelta"] {
        font-size: 0.68rem;
    }


    /* ============================================================
       GENERAL METRIC SPACING
       ============================================================ */

    div[data-testid="stMetric"] {
        padding: 0;
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
# At the reference metastatic burden and 1% tumour uptake,
# 1 GBq corresponds to 0.50 Gy/day.
DOSE_PER_GBQ_GY = 0.50

# Reference metastatic tumour burden
REFERENCE_METASTATIC_BURDEN_ML = 234.0

# Tumour uptake
DEFAULT_TUMOUR_UPTAKE_PERCENT = 1.0
TUMOUR_UPTAKE_MIN_PERCENT = 0.1
TUMOUR_UPTAKE_MAX_PERCENT = 10.0

# Simulation
DT_DAYS = 0.05
FOLLOW_UP_DAYS = 60.0

# Default biological parameters
DEFAULT_BURDEN = 234.0

DEFAULT_ALPHA = 0.10
DEFAULT_BETA_ALPHA = 0.10

DEFAULT_TREP = 30.0
DEFAULT_SENSITIVE_FRACTION = 0.75
DEFAULT_RESISTANT_TK_MULTIPLIER = 1.50
DEFAULT_RESISTANT_RADIO_FACTOR = 0.25
DEFAULT_REPOPULATION_KICKOFF = 5.0

DEFAULT_ACTIVITY = 7.4
DEFAULT_CYCLES = 4
DEFAULT_INTERVAL = 7

DEFAULT_GAMMA = 1.0

# Exploratory TCP normalisation
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

# Secondary metric yellow/gold
COLOR_SECONDARY = "#F2C94C"


# =============================================================================
# MATPLOTLIB GLOBAL STYLE
# =============================================================================

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.facecolor": PLOT_BACKGROUND,
        "figure.facecolor": PLOT_BACKGROUND,
        "savefig.facecolor": PLOT_BACKGROUND,
        "axes.edgecolor": "#888888",
        "axes.labelcolor": PLOT_TEXT,
        "xtick.color": PLOT_TEXT,
        "ytick.color": PLOT_TEXT,
        "text.color": PLOT_TEXT,
        "legend.facecolor": "#111111",
        "legend.edgecolor": "#555555",
        "grid.color": PLOT_GRID,
        "grid.alpha": 0.35,
    }
)


# =============================================================================
# NUMERICAL INTEGRATION COMPATIBILITY
# =============================================================================

def integrate_trapezoid(y, x):
    """
    Compatibility wrapper for NumPy versions where np.trapezoid
    may or may not be available.
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
    Phenomenological scaling of average tumour dose rate.

    Default:
        234 mL tumour burden
        1% tumour uptake

    gives a scaling factor of 1.0.

    Higher uptake increases dose rate.

    At fixed total tumour uptake, larger total tumour burden
    reduces the average dose rate per unit tumour burden.
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
    return np.array(
        [
            i * interval_days
            for i in range(n_cycles)
        ],
        dtype=float,
    )


# =============================================================================
# PHYSICAL DOSE RATE
# =============================================================================

def calculate_physical_dose_rate(
    time_days,
    treatment_times,
    activity_gbq,
    initial_burden_ml,
    tumour_uptake_percent,
):
    """
    Calculate physical tumour dose rate.

    Each treatment contributes exponentially decaying Lu-177
    activity from its administration time.
    """

    (
        uptake_scale,
        burden_scale,
        dose_scale,
    ) = calculate_dose_scaling(
        initial_burden_ml,
        tumour_uptake_percent,
    )

    total_activity = np.zeros_like(time_days)

    for t_admin in treatment_times:

        mask = time_days >= t_admin

        elapsed = time_days[mask] - t_admin

        total_activity[mask] += (
            activity_gbq
            * np.exp(
                -LU177_LAMBDA_PER_DAY
                * elapsed
            )
        )

    dose_rate_gy_day = (
        total_activity
        * DOSE_PER_GBQ_GY
        * dose_scale
    )

    dose_rate_gy_h = dose_rate_gy_day / 24.0

    return (
        total_activity,
        dose_rate_gy_day,
        dose_rate_gy_h,
        uptake_scale,
        burden_scale,
        dose_scale,
    )


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

    Units: Gy/h
    """

    trep_hours = trep_days * 24.0

    if alpha <= 0 or trep_hours <= 0:
        return np.nan

    return np.log(2.0) / (
        alpha * trep_hours
    )


# =============================================================================
# EFFECTIVE DOSE RATE
# =============================================================================

def calculate_effective_dose_rate(
    physical_dose_rate_gy_h,
    critical_dose_rate_gy_h,
    gamma,
):
    """
    Exploratory dose-rate effectiveness model.

        Rratio = Rphysical / Rcritical

        E = min(1, Rratio^gamma)

        Reffective = Rphysical * E
    """

    if critical_dose_rate_gy_h <= 0:
        ratio = np.zeros_like(
            physical_dose_rate_gy_h
        )
    else:
        ratio = (
            physical_dose_rate_gy_h
            / critical_dose_rate_gy_h
        )

    effectiveness = np.minimum(
        1.0,
        np.power(
            np.maximum(ratio, 0.0),
            gamma,
        ),
    )

    effective_dose_rate = (
        physical_dose_rate_gy_h
        * effectiveness
    )

    return (
        ratio,
        effectiveness,
        effective_dose_rate,
    )


# =============================================================================
# TUMOUR DYNAMICS
# =============================================================================

def calculate_tumour_dynamics(
    time_days,
    physical_dose_rate_gy_day,
    initial_burden_ml,
    alpha,
    beta_alpha,
    trep_days,
    sensitive_fraction,
    resistant_tk_multiplier,
    resistant_radio_factor,
    repopulation_kickoff_days,
):
    """
    Two-compartment phenomenological tumour model:

        Sensitive tumour
        Resistant tumour

    Both populations can repopulate after the specified kickoff
    time and are exposed to radiation according to their
    respective radiosensitivity.
    """

    n = len(time_days)

    sensitive = np.zeros(n)
    resistant = np.zeros(n)

    sensitive[0] = (
        initial_burden_ml
        * sensitive_fraction
    )

    resistant[0] = (
        initial_burden_ml
        * (1.0 - sensitive_fraction)
    )

    beta_sensitive = alpha * beta_alpha

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

        s = sensitive[i - 1]
        r = resistant[i - 1]

        # ---------------------------------------------------------
        # Repopulation
        # ---------------------------------------------------------

        if (
            time_days[i]
            >= repopulation_kickoff_days
        ):

            s *= np.exp(
                sensitive_growth_rate * dt
            )

            r *= np.exp(
                resistant_growth_rate * dt
            )

        # ---------------------------------------------------------
        # Radiation dose
        # ---------------------------------------------------------

        dose_interval_gy = (
            physical_dose_rate_gy_day[i]
            * dt
        )

        # ---------------------------------------------------------
        # LQ survival
        # ---------------------------------------------------------

        survival_sensitive = np.exp(
            -alpha * dose_interval_gy
            -beta_sensitive
            * dose_interval_gy**2
        )

        survival_resistant = np.exp(
            -alpha_resistant
            * dose_interval_gy
            -beta_resistant
            * dose_interval_gy**2
        )

        s *= survival_sensitive
        r *= survival_resistant

        sensitive[i] = max(s, 0.0)
        resistant[i] = max(r, 0.0)

    total = sensitive + resistant

    initial_resistant_burden = (
        initial_burden_ml
        * (1.0 - sensitive_fraction)
    )

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

    resistant_composition_fraction = np.divide(
        resistant,
        total,
        out=np.zeros_like(resistant),
        where=total > 0,
    )

    return (
        total,
        sensitive,
        resistant,
        initial_resistant_burden,
        residual_resistant_burden_percent,
        resistant_composition_fraction,
    )


# =============================================================================
# TCP
# =============================================================================

def calculate_tcp(
    total_burden_ml,
    initial_burden_ml,
):
    """
    Exploratory normalized TCP model.

    Baseline TCP is defined as 10%.

    Relative tumour burden is used to scale the
    initial clonogenic burden.
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

    return np.clip(
        tcp,
        0.0,
        1.0,
    )


# =============================================================================
# SUMMARY
# =============================================================================

def calculate_summary(
    time_days,
    physical_dose_rate_gy_h,
    effective_dose_rate_gy_h,
    critical_dose_rate_gy_h,
    total_burden,
    sensitive,
    resistant,
    residual_resistant_burden_percent,
    resistant_composition_fraction,
    tcp,
    initial_burden_ml,
    treatment_times,
):
    # -------------------------------------------------------------
    # Tumour burden
    # -------------------------------------------------------------

    min_index = np.argmin(total_burden)

    minimum_tumour_burden = (
        total_burden[min_index]
    )

    minimum_tumour_burden_day = (
        time_days[min_index]
    )

    final_tumour_burden = (
        total_burden[-1]
    )

    # -------------------------------------------------------------
    # TCP
    # -------------------------------------------------------------

    max_tcp_index = np.argmax(tcp)

    maximum_tcp = tcp[max_tcp_index]

    maximum_tcp_day = (
        time_days[max_tcp_index]
    )

    # -------------------------------------------------------------
    # Resistant burden
    # -------------------------------------------------------------

    final_residual_resistant = (
        residual_resistant_burden_percent[-1]
    )

    minimum_residual_resistant = (
        np.min(
            residual_resistant_burden_percent
        )
    )

    final_resistant_composition = (
        resistant_composition_fraction[-1]
        * 100.0
    )

    # -------------------------------------------------------------
    # Dose-rate metrics
    # -------------------------------------------------------------

    peak_physical_rate = (
        np.max(
            physical_dose_rate_gy_h
        )
    )

    # -------------------------------------------------------------
    # Time above critical dose rate
    # -------------------------------------------------------------

    above = (
        physical_dose_rate_gy_h
        >= critical_dose_rate_gy_h
    )

    time_above_critical_days = (
        np.sum(above)
        * DT_DAYS
    )

    # Longest continuous interval
    longest_continuous = 0.0
    current = 0.0

    for value in above:

        if value:

            current += DT_DAYS

            longest_continuous = max(
                longest_continuous,
                current,
            )

        else:

            current = 0.0

    # -------------------------------------------------------------
    # Cumulative dose
    # -------------------------------------------------------------

    cumulative_physical_dose = (
        integrate_trapezoid(
            physical_dose_rate_gy_h,
            time_days * 24.0,
        )
    )

    cumulative_effective_dose = (
        integrate_trapezoid(
            effective_dose_rate_gy_h,
            time_days * 24.0,
        )
    )

    return {
        "minimum_tumour_burden_ml":
            minimum_tumour_burden,

        "minimum_tumour_burden_day":
            minimum_tumour_burden_day,

        "final_tumour_burden_ml":
            final_tumour_burden,

        "maximum_tcp":
            maximum_tcp,

        "maximum_tcp_percent":
            maximum_tcp * 100.0,

        "maximum_tcp_day":
            maximum_tcp_day,

        "final_residual_resistant_percent":
            final_residual_resistant,

        "minimum_residual_resistant_percent":
            minimum_residual_resistant,

        "final_resistant_composition_percent":
            final_resistant_composition,

        "final_sensitive_burden_ml":
            sensitive[-1],

        "final_resistant_burden_ml":
            resistant[-1],

        "critical_dose_rate_gy_h":
            critical_dose_rate_gy_h,

        "peak_physical_rate_gy_h":
            peak_physical_rate,

        "time_above_critical_days":
            time_above_critical_days,

        "longest_continuous_above_days":
            longest_continuous,

        "cumulative_physical_dose_gy":
            cumulative_physical_dose,

        "cumulative_effective_dose_gy":
            cumulative_effective_dose,
    }


# =============================================================================
# PLOT STYLING
# =============================================================================

def style_axis(ax):

    ax.set_facecolor(
        PLOT_BACKGROUND
    )

    ax.tick_params(
        colors=PLOT_TEXT,
        labelsize=9,
    )

    ax.xaxis.label.set_color(
        PLOT_TEXT
    )

    ax.yaxis.label.set_color(
        PLOT_TEXT
    )

    ax.title.set_color(
        PLOT_TEXT
    )

    ax.grid(
        which="major",
        linestyle="-",
        alpha=0.25,
    )

    ax.grid(
        which="minor",
        linestyle=":",
        alpha=0.15,
    )

    ax.minorticks_on()

    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=7)
    )

    for spine in ax.spines.values():

        spine.set_color(
            "#777777"
        )


def style_legend(ax):

    legend = ax.legend(
        frameon=True,
        fontsize=8,
        loc="best",
    )

    if legend is not None:

        legend.get_frame().set_facecolor(
            "#111111"
        )

        legend.get_frame().set_edgecolor(
            "#555555"
        )

        for text in legend.get_texts():

            text.set_color(
                PLOT_TEXT
            )


def add_treatment_markers(
    ax,
    treatment_times,
):

    for t in treatment_times:

        ax.axvline(
            t,
            color=PLOT_TREATMENT,
            linestyle=":",
            linewidth=0.8,
            alpha=0.55,
        )


# =============================================================================
# KEY RESULT HELPERS
# =============================================================================

def key_result_status(
    text,
    status="neutral",
):
    """
    Create green/red/neutral status pill.
    """

    if status == "positive":

        class_name = (
            "key-status-positive"
        )

    elif status == "negative":

        class_name = (
            "key-status-negative"
        )

    else:

        class_name = (
            "key-status-neutral"
        )

    return (
        f'<span class="key-result-status '
        f'{class_name}">{text}</span>'
    )


def key_result_caption(text):

    return (
        f'<div class="key-result-caption">'
        f'{text}'
        f'</div>'
    )


def key_result_card(
    label,
    value,
    status_html,
    caption,
):

    html = f"""
    <div class="key-result-card">

        <div class="key-result-label">
            {label}
        </div>

        <div class="key-result-value">
            {value}
        </div>

        {status_html}

        {key_result_caption(caption)}

    </div>
    """

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header(
    "Model controls"
)

st.sidebar.markdown(
    "### Tumour"
)

initial_burden_ml = st.sidebar.slider(
    "Initial metastatic burden (mL)",
    min_value=10,
    max_value=1500,
    value=int(DEFAULT_BURDEN),
    step=1,
    help=(
        "Total metastatic tumour burden, "
        "treated as a phenomenological model "
        "input rather than a single solid tumour."
    ),
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
    min_value=10,
    max_value=100,
    value=int(DEFAULT_TREP),
    step=1,
    format="%d days",
)

sensitive_fraction = st.sidebar.slider(
    "Sensitive fraction",
    min_value=0.50,
    max_value=0.95,
    value=DEFAULT_SENSITIVE_FRACTION,
    step=0.01,
    format="%.2f",
)

resistant_tk_multiplier = st.sidebar.slider(
    "Tk/Trep",
    min_value=1.20,
    max_value=1.80,
    value=DEFAULT_RESISTANT_TK_MULTIPLIER,
    step=0.01,
    format="%.2f",
)

resistant_radio_factor = st.sidebar.slider(
    "Resistant radio factor",
    min_value=0.05,
    max_value=1.00,
    value=DEFAULT_RESISTANT_RADIO_FACTOR,
    step=0.01,
    format="%.2f",
)

repopulation_kickoff = st.sidebar.slider(
    "Repopulation kickoff",
    min_value=3,
    max_value=10,
    value=int(DEFAULT_REPOPULATION_KICKOFF),
    step=1,
    format="%d days",
)

st.sidebar.markdown(
    "### Treatment"
)

activity_gbq = st.sidebar.slider(
    "Activity per cycle (GBq)",
    min_value=1.0,
    max_value=15.0,
    value=DEFAULT_ACTIVITY,
    step=0.1,
    format="%.1f GBq",
)

n_cycles = st.sidebar.slider(
    "Number of cycles",
    min_value=1,
    max_value=8,
    value=DEFAULT_CYCLES,
    step=1,
)

interval_days = st.sidebar.slider(
    "Cycle interval",
    min_value=1,
    max_value=42,
    value=DEFAULT_INTERVAL,
    step=1,
    format="%d days",
)

tumour_uptake_percent = st.sidebar.slider(
    "Total tumour uptake (%)",
    min_value=TUMOUR_UPTAKE_MIN_PERCENT,
    max_value=TUMOUR_UPTAKE_MAX_PERCENT,
    value=DEFAULT_TUMOUR_UPTAKE_PERCENT,
    step=0.1,
    format="%.1f%%",
    help=(
        "Phenomenological fraction of administered "
        "activity attributed to the total metastatic "
        "tumour burden."
    ),
)

gamma = st.sidebar.slider(
    "Effectiveness gamma",
    min_value=0.25,
    max_value=2.00,
    value=DEFAULT_GAMMA,
    step=0.05,
    format="%.2f",
)


# =============================================================================
# SIMULATION TIME
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
    )
    + FOLLOW_UP_DAYS,
)

time_days = np.arange(
    0.0,
    simulation_days + DT_DAYS,
    DT_DAYS,
)


# =============================================================================
# DOSE CALCULATION
# =============================================================================

(
    total_activity_gbq,
    physical_dose_rate_gy_day,
    physical_dose_rate_gy_h,
    uptake_scale,
    burden_scale,
    dose_scale,
) = calculate_physical_dose_rate(
    time_days=time_days,
    treatment_times=treatment_times,
    activity_gbq=activity_gbq,
    initial_burden_ml=initial_burden_ml,
    tumour_uptake_percent=tumour_uptake_percent,
)


# =============================================================================
# CRITICAL DOSE RATE
# =============================================================================

critical_dose_rate_gy_h = (
    calculate_critical_dose_rate(
        alpha=alpha,
        trep_days=trep_days,
    )
)


# =============================================================================
# EFFECTIVE DOSE RATE
# =============================================================================

(
    dose_rate_ratio,
    effectiveness,
    effective_dose_rate_gy_h,
) = calculate_effective_dose_rate(
    physical_dose_rate_gy_h,
    critical_dose_rate_gy_h,
    gamma,
)


# =============================================================================
# TUMOUR DYNAMICS
# =============================================================================

(
    total_burden,
    sensitive,
    resistant,
    initial_resistant_burden,
    residual_resistant_burden_percent,
    resistant_composition_fraction,
) = calculate_tumour_dynamics(
    time_days=time_days,
    physical_dose_rate_gy_day=physical_dose_rate_gy_day,
    initial_burden_ml=initial_burden_ml,
    alpha=alpha,
    beta_alpha=DEFAULT_BETA_ALPHA,
    trep_days=trep_days,
    sensitive_fraction=sensitive_fraction,
    resistant_tk_multiplier=resistant_tk_multiplier,
    resistant_radio_factor=resistant_radio_factor,
    repopulation_kickoff_days=repopulation_kickoff,
)


# =============================================================================
# TCP
# =============================================================================

tcp = calculate_tcp(
    total_burden_ml=total_burden,
    initial_burden_ml=initial_burden_ml,
)


# =============================================================================
# SUMMARY
# =============================================================================

summary = calculate_summary(
    time_days=time_days,
    physical_dose_rate_gy_h=physical_dose_rate_gy_h,
    effective_dose_rate_gy_h=effective_dose_rate_gy_h,
    critical_dose_rate_gy_h=critical_dose_rate_gy_h,
    total_burden=total_burden,
    sensitive=sensitive,
    resistant=resistant,
    residual_resistant_burden_percent=(
        residual_resistant_burden_percent
    ),
    resistant_composition_fraction=(
        resistant_composition_fraction
    ),
    tcp=tcp,
    initial_burden_ml=initial_burden_ml,
    treatment_times=treatment_times,
)


# =============================================================================
# PAGE HEADER
# =============================================================================

st.title(
    "Lu-177 PSMA Interactive Model Explorer"
)

st.markdown(
    """
    Explore the effects of tumour burden, tumour uptake,
    Lu-177 activity, cycle spacing, radiosensitivity,
    resistant disease and dose-rate effectiveness on
    tumour response and TCP.
    """
)


# =============================================================================
# KEY RESULTS
# =============================================================================

st.subheader(
    "Key results"
)

key_cols = st.columns(4)


# -------------------------------------------------------------------------
# Minimum tumour burden
# -------------------------------------------------------------------------

burden_change_percent = (
    (
        summary["minimum_tumour_burden_ml"]
        / initial_burden_ml
    )
    - 1.0
) * 100.0

if burden_change_percent < 0:

    burden_status = key_result_status(
        f"↓ {abs(burden_change_percent):.1f}% "
        f"from {initial_burden_ml:.0f} mL baseline",
        "positive",
    )

else:

    burden_status = key_result_status(
        f"↑ {abs(burden_change_percent):.1f}% "
        f"from {initial_burden_ml:.0f} mL baseline",
        "negative",
    )


with key_cols[0]:

    key_result_card(
        "Minimum tumour burden",
        f"{summary['minimum_tumour_burden_ml']:.1f} mL",
        burden_status,
        (
            f"Minimum reached at day "
            f"{summary['minimum_tumour_burden_day']:.1f}"
        ),
    )


# -------------------------------------------------------------------------
# Maximum TCP
# -------------------------------------------------------------------------

tcp_change_points = (
    summary["maximum_tcp_percent"]
    - INITIAL_TCP * 100.0
)

if tcp_change_points > 0:

    tcp_status = key_result_status(
        f"↑ {tcp_change_points:.1f} "
        f"percentage points from 10% baseline",
        "positive",
    )

elif tcp_change_points < 0:

    tcp_status = key_result_status(
        f"↓ {abs(tcp_change_points):.1f} "
        f"percentage points from 10% baseline",
        "negative",
    )

else:

    tcp_status = key_result_status(
        "No change from 10% baseline",
        "neutral",
    )


with key_cols[1]:

    key_result_card(
        "Maximum TCP",
        f"{summary['maximum_tcp_percent']:.1f}%",
        tcp_status,
        (
            f"Maximum reached at day "
            f"{summary['maximum_tcp_day']:.1f}"
        ),
    )


# -------------------------------------------------------------------------
# Residual resistant burden
# -------------------------------------------------------------------------

resistant_status_text = (
    f"{summary['final_residual_resistant_percent']:.1f}% "
    f"of initial resistant burden remaining"
)

if (
    summary["final_residual_resistant_percent"]
    < 100.0
):

    resistant_status = key_result_status(
        f"↓ "
        f"{100.0 - summary['final_residual_resistant_percent']:.1f}% "
        f"from baseline",
        "positive",
    )

elif (
    summary["final_residual_resistant_percent"]
    > 100.0
):

    resistant_status = key_result_status(
        f"↑ "
        f"{summary['final_residual_resistant_percent'] - 100.0:.1f}% "
        f"from baseline",
        "negative",
    )

else:

    resistant_status = key_result_status(
        "No change from baseline",
        "neutral",
    )


with key_cols[2]:

    key_result_card(
        "Residual resistant burden",
        resistant_status_text,
        resistant_status,
        (
            f"Initial resistant burden: "
            f"{initial_resistant_burden:.1f} mL "
            f"({(1.0 - sensitive_fraction) * 100.0:.0f}% "
            f"of total)"
        ),
    )


# -------------------------------------------------------------------------
# Cumulative physical dose
# -------------------------------------------------------------------------

with key_cols[3]:

    key_result_card(
        "Cumulative physical dose",
        (
            f"{summary['cumulative_physical_dose_gy']:.2f} Gy"
        ),
        key_result_status(
            "Integrated physical dose",
            "neutral",
        ),
        (
            f"{activity_gbq:.1f} GBq × "
            f"{n_cycles} cycles"
        ),
    )


# =============================================================================
# SECONDARY METRICS
# =============================================================================

st.subheader(
    "Secondary metrics"
)

sec_cols = st.columns(4)


with sec_cols[0]:

    st.markdown(
        '<div class="secondary-metric">',
        unsafe_allow_html=True,
    )

    st.metric(
        "Critical dose rate",
        (
            f"{summary['critical_dose_rate_gy_h']:.4f} Gy/h"
        ),
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


with sec_cols[1]:

    st.markdown(
        '<div class="secondary-metric">',
        unsafe_allow_html=True,
    )

    st.metric(
        "Peak physical rate",
        (
            f"{summary['peak_physical_rate_gy_h']:.4f} Gy/h"
        ),
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


with sec_cols[2]:

    st.markdown(
        '<div class="secondary-metric">',
        unsafe_allow_html=True,
    )

    st.metric(
        "Time above critical rate",
        (
            f"{summary['time_above_critical_days']:.1f} days"
        ),
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


with sec_cols[3]:

    st.markdown(
        '<div class="secondary-metric">',
        unsafe_allow_html=True,
    )

    st.metric(
        "Effective cumulative dose",
        (
            f"{summary['cumulative_effective_dose_gy']:.2f} Gy"
        ),
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# DOSE SCALING INFORMATION
# =============================================================================

st.caption(
    f""
    f"Dose scaling: uptake scale = {uptake_scale:.2f} × "
    f"burden scale = {burden_scale:.2f} → "
    f"overall dose scale = {dose_scale:.2f}. "
    f"Reference = {REFERENCE_METASTATIC_BURDEN_ML:.0f} mL "
    f"and {DEFAULT_TUMOUR_UPTAKE_PERCENT:.1f}% uptake."
)


# =============================================================================
# MODEL TRAJECTORIES
# =============================================================================

st.subheader(
    "Model trajectories"
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 10),
)

fig.patch.set_facecolor(
    PLOT_BACKGROUND
)


# =============================================================================
# PLOT 1 — DOSE RATE
# =============================================================================

ax = axes[0, 0]

ax.plot(
    time_days,
    physical_dose_rate_gy_h,
    color=COLOR_BLUE,
    linewidth=2.0,
    label="Physical dose rate",
)

ax.plot(
    time_days,
    effective_dose_rate_gy_h,
    color=COLOR_ORANGE,
    linewidth=2.0,
    linestyle="--",
    label="Effective dose rate",
)

ax.axhline(
    critical_dose_rate_gy_h,
    color=COLOR_RED,
    linewidth=1.5,
    linestyle=":",
    label="Critical dose rate",
)

add_treatment_markers(
    ax,
    treatment_times,
)

ax.set_title(
    "Dose rate",
    fontsize=12,
    fontweight="bold",
)

ax.set_xlabel(
    "Time (days)"
)

ax.set_ylabel(
    "Dose rate (Gy/h)"
)

style_axis(ax)
style_legend(ax)


# =============================================================================
# PLOT 2 — TUMOUR BURDEN
# =============================================================================

ax = axes[0, 1]

ax.plot(
    time_days,
    total_burden,
    color=COLOR_BLUE,
    linewidth=2.2,
    label="Total tumour",
)

ax.plot(
    time_days,
    sensitive,
    color=COLOR_GREEN,
    linewidth=1.8,
    linestyle="--",
    label="Sensitive tumour",
)

ax.plot(
    time_days,
    resistant,
    color=COLOR_ORANGE,
    linewidth=1.8,
    linestyle=":",
    label="Resistant tumour",
)

add_treatment_markers(
    ax,
    treatment_times,
)

ax.set_title(
    "Tumour burden",
    fontsize=12,
    fontweight="bold",
)

ax.set_xlabel(
    "Time (days)"
)

ax.set_ylabel(
    "Tumour burden (mL)"
)

style_axis(ax)
style_legend(ax)


# =============================================================================
# PLOT 3 — SENSITIVE / RESISTANT POPULATIONS
# =============================================================================

ax = axes[1, 0]

ax.plot(
    time_days,
    sensitive,
    color=COLOR_GREEN,
    linewidth=2.0,
    label="Sensitive tumour",
)

ax.plot(
    time_days,
    resistant,
    color=COLOR_ORANGE,
    linewidth=2.0,
    linestyle="--",
    label="Resistant tumour",
)

add_treatment_markers(
    ax,
    treatment_times,
)

ax.set_title(
    "Sensitive and resistant tumour",
    fontsize=12,
    fontweight="bold",
)

ax.set_xlabel(
    "Time (days)"
)

ax.set_ylabel(
    "Tumour burden (mL)"
)

style_axis(ax)

ax2 = ax.twinx()

ax2.plot(
    time_days,
    residual_resistant_burden_percent,
    color=COLOR_PURPLE,
    linewidth=1.8,
    linestyle=":",
    label=(
        "Residual resistant burden"
    ),
)

ax2.set_ylabel(
    "Residual resistant burden "
    "(% of initial resistant burden)",
    color=COLOR_PURPLE,
)

ax2.tick_params(
    axis="y",
    colors=COLOR_PURPLE,
)

for spine in ax2.spines.values():

    spine.set_color(
        "#777777"
    )

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

legend = ax.legend(
    lines1 + lines2,
    labels1 + labels2,
    fontsize=8,
    loc="best",
)

legend.get_frame().set_facecolor(
    "#111111"
)

legend.get_frame().set_edgecolor(
    "#555555"
)

for text in legend.get_texts():

    text.set_color(
        PLOT_TEXT
    )


# =============================================================================
# PLOT 4 — TCP
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
    treatment_times,
)

ax.set_title(
    "Tumour control probability",
    fontsize=12,
    fontweight="bold",
)

ax.set_xlabel(
    "Time (days)"
)

ax.set_ylabel(
    "TCP (%)"
)

ax.set_ylim(
    0,
    105,
)

style_axis(ax)
style_legend(ax)


# =============================================================================
# FINAL FIGURE FORMATTING
# =============================================================================

fig.tight_layout(
    pad=2.0
)

st.pyplot(
    fig,
    use_container_width=True,
)

plt.close(fig)


# =============================================================================
# DETAILED MODEL SUMMARY
# =============================================================================

with st.expander(
    "Detailed model summary"
):

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### Tumour response"
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
            f"Maximum TCP: "
            f"{summary['maximum_tcp_percent']:.2f}%"
        )

        st.write(
            f"Final sensitive tumour burden: "
            f"{summary['final_sensitive_burden_ml']:.2f} mL"
        )

        st.write(
            f"Final resistant tumour burden: "
            f"{summary['final_resistant_burden_ml']:.2f} mL"
        )

        st.write(
            f"Residual resistant burden: "
            f"{summary['final_residual_resistant_percent']:.2f}% "
            f"of initial resistant burden"
        )

        st.write(
            f"Final resistant composition: "
            f"{summary['final_resistant_composition_percent']:.2f}%"
        )

    with col2:

        st.markdown(
            "### Radiation response"
        )

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
            f"{summary['cumulative_physical_dose_gy']:.2f} Gy"
        )

        st.write(
            f"Cumulative effective dose: "
            f"{summary['cumulative_effective_dose_gy']:.2f} Gy"
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

        The tumour burden represents the total metastatic tumour
        volume rather than a single solid tumour.

        **Tumour uptake**

        Total tumour uptake is represented as a percentage of
        administered activity. This is a phenomenological scaling
        parameter and is not lesion-specific dosimetry.

        **Dose rate**

        The model assumes exponential Lu-177 physical decay and
        uses a reference dose-rate conversion scaled by tumour
        burden and uptake.

        **Tumour dynamics**

        The tumour is represented by sensitive and resistant
        compartments. Both compartments can repopulate, with
        resistant disease having a modified doubling time.

        **Radiobiology**

        Radiation killing is represented using an exploratory
        linear-quadratic formulation.

        **Critical dose rate**

        The critical dose rate is calculated from alpha and the
        tumour repopulation time.

        **TCP**

        TCP is currently a normalized exploratory model with a
        baseline TCP of 10%. It should not be interpreted as a
        clinically validated patient-specific TCP prediction.

        **Resistant burden**

        Residual resistant burden is expressed relative to the
        initial resistant tumour burden. This is distinct from
        resistant composition, which describes the fraction of
        the remaining tumour that is resistant.

        **Clinical translation**

        Real patient-specific Lu-177 PSMA dosimetry would require
        lesion-level uptake, time-activity curves, residence times,
        absorbed fractions, spatial heterogeneity and potentially
        cross-dose between lesions and surrounding tissues.
        """
    )


# =============================================================================
# CSV EXPORT
# =============================================================================

results_df = pd.DataFrame(
    {
        "time_days": time_days,

        "total_activity_gbq":
            total_activity_gbq,

        "physical_dose_rate_gy_day":
            physical_dose_rate_gy_day,

        "physical_dose_rate_gy_h":
            physical_dose_rate_gy_h,

        "critical_dose_rate_gy_h":
            np.full_like(
                time_days,
                critical_dose_rate_gy_h,
            ),

        "dose_rate_ratio":
            dose_rate_ratio,

        "effectiveness":
            effectiveness,

        "effective_dose_rate_gy_h":
            effective_dose_rate_gy_h,

        "total_tumour_burden_ml":
            total_burden,

        "sensitive_tumour_burden_ml":
            sensitive,

        "resistant_tumour_burden_ml":
            resistant,

        "initial_resistant_burden_ml":
            np.full_like(
                time_days,
                initial_resistant_burden,
            ),

        "residual_resistant_burden_percent":
            residual_resistant_burden_percent,

        "resistant_composition_percent":
            resistant_composition_fraction
            * 100.0,

        "tcp_percent":
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

results_df.to_csv(
    csv_buffer,
    index=False,
)

st.download_button(
    label="Download results CSV",
    data=csv_buffer.getvalue(),
    file_name="Lu177_PSMA_model_results.csv",
    mime="text/csv",
)


# =============================================================================
# HIGH-RESOLUTION PNG EXPORT
# =============================================================================

png_buffer = io.BytesIO()

fig_export, axes_export = plt.subplots(
    2,
    2,
    figsize=(15, 10),
)

fig_export.patch.set_facecolor(
    PLOT_BACKGROUND
)


# -------------------------------------------------------------------------
# Export plot 1
# -------------------------------------------------------------------------

ax = axes_export[0, 0]

ax.plot(
    time_days,
    physical_dose_rate_gy_h,
    color=COLOR_BLUE,
    linewidth=2.0,
    label="Physical dose rate",
)

ax.plot(
    time_days,
    effective_dose_rate_gy_h,
    color=COLOR_ORANGE,
    linewidth=2.0,
    linestyle="--",
    label="Effective dose rate",
)

ax.axhline(
    critical_dose_rate_gy_h,
    color=COLOR_RED,
    linewidth=1.5,
    linestyle=":",
    label="Critical dose rate",
)

add_treatment_markers(
    ax,
    treatment_times,
)

ax.set_title(
    "Dose rate",
    fontsize=12,
    fontweight="bold",
)

ax.set_xlabel(
    "Time (days)"
)

ax.set_ylabel(
    "Dose rate (Gy/h)"
)

style_axis(ax)
style_legend(ax)


# -------------------------------------------------------------------------
# Export plot 2
# -------------------------------------------------------------------------

ax = axes_export[0, 1]

ax.plot(
    time_days,
    total_burden,
    color=COLOR_BLUE,
    linewidth=2.2,
    label="Total tumour",
)

ax.plot(
    time_days,
    sensitive,
    color=COLOR_GREEN,
    linewidth=1.8,
    linestyle="--",
    label="Sensitive tumour",
)

ax.plot(
    time_days,
    resistant,
    color=COLOR_ORANGE,
    linewidth=1.8,
    linestyle=":",
    label="Resistant tumour",
)

add_treatment_markers(
    ax,
    treatment_times,
)

ax.set_title(
    "Tumour burden",
    fontsize=12,
    fontweight="bold",
)

ax.set_xlabel(
    "Time (days)"
)

ax.set_ylabel(
    "Tumour burden (mL)"
)

style_axis(ax)
style_legend(ax)


# -------------------------------------------------------------------------
# Export plot 3
# -------------------------------------------------------------------------

ax = axes_export[1, 0]

ax.plot(
    time_days,
    sensitive,
    color=COLOR_GREEN,
    linewidth=2.0,
    label="Sensitive tumour",
)

ax.plot(
    time_days,
    resistant,
    color=COLOR_ORANGE,
    linewidth=2.0,
    linestyle="--",
    label="Resistant tumour",
)

add_treatment_markers(
    ax,
    treatment_times,
)

ax.set_title(
    "Sensitive and resistant tumour",
    fontsize=12,
    fontweight="bold",
)

ax.set_xlabel(
    "Time (days)"
)

ax.set_ylabel(
    "Tumour burden (mL)"
)

style_axis(ax)

ax2 = ax.twinx()

ax2.plot(
    time_days,
    residual_resistant_burden_percent,
    color=COLOR_PURPLE,
    linewidth=1.8,
    linestyle=":",
    label="Residual resistant burden",
)

ax2.set_ylabel(
    "Residual resistant burden "
    "(% of initial resistant burden)",
    color=COLOR_PURPLE,
)

ax2.tick_params(
    axis="y",
    colors=COLOR_PURPLE,
)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

legend = ax.legend(
    lines1 + lines2,
    labels1 + labels2,
    fontsize=8,
    loc="best",
)

legend.get_frame().set_facecolor(
    "#111111"
)

legend.get_frame().set_edgecolor(
    "#555555"
)

for text in legend.get_texts():

    text.set_color(
        PLOT_TEXT
    )


# -------------------------------------------------------------------------
# Export plot 4
# -------------------------------------------------------------------------

ax = axes_export[1, 1]

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
    treatment_times,
)

ax.set_title(
    "Tumour control probability",
    fontsize=12,
    fontweight="bold",
)

ax.set_xlabel(
    "Time (days)"
)

ax.set_ylabel(
    "TCP (%)"
)

ax.set_ylim(
    0,
    105,
)

style_axis(ax)
style_legend(ax)


# -------------------------------------------------------------------------
# Save PNG
# -------------------------------------------------------------------------

fig_export.tight_layout(
    pad=2.0
)

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

st.download_button(
    label="Download 600 dpi PNG",
    data=png_buffer.getvalue(),
    file_name="Lu177_PSMA_model_trajectories_600dpi.png",
    mime="image/png",
)
