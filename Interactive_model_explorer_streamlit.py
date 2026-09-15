import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import streamlit as st


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Lu-177 PSMA Interactive Model Explorer",
    page_icon="🧬",
    layout="wide",
)


# =============================================================================
# MODEL CONSTANTS
# =============================================================================

LU177_HALF_LIFE_DAYS = 6.647
LU177_LAMBDA_PER_DAY = np.log(2.0) / LU177_HALF_LIFE_DAYS

DOSE_PER_GBQ_GY = 0.50

REFERENCE_METASTATIC_BURDEN_ML = 234.0

DEFAULT_TUMOUR_UPTAKE_PERCENT = 1.0
TUMOUR_UPTAKE_MIN_PERCENT = 0.1
TUMOUR_UPTAKE_MAX_PERCENT = 10.0

DT_DAYS = 0.05
FOLLOW_UP_DAYS = 60.0

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

# Secondary metric accent
COLOR_YELLOW = "#F2C94C"


# =============================================================================
# GLOBAL MATPLOTLIB STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.facecolor": PLOT_BACKGROUND,
    "figure.facecolor": PLOT_BACKGROUND,
    "axes.edgecolor": PLOT_TEXT,
    "axes.labelcolor": PLOT_TEXT,
    "xtick.color": PLOT_TEXT,
    "ytick.color": PLOT_TEXT,
    "text.color": PLOT_TEXT,
    "axes.titlecolor": PLOT_TEXT,
    "legend.facecolor": PLOT_BACKGROUND,
    "legend.edgecolor": PLOT_GRID,
    "legend.labelcolor": PLOT_TEXT,
})


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def integrate_trapezoid(y, x):
    """
    NumPy compatibility helper.
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
    Phenomenological scaling of the reference dose-rate model.

    Reference condition:
        234 mL tumour burden
        1% total tumour uptake

    At the reference condition the scaling factor is 1.0.
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
    Calculate administration times for each treatment cycle.
    """

    return np.arange(n_cycles, dtype=float) * interval_days


# =============================================================================
# PHYSICAL DOSE RATE
# =============================================================================

def calculate_physical_dose_rate(
    time_days,
    activity_gbq,
    treatment_times,
    dose_scale,
):
    """
    Calculate total physical Lu-177 dose rate.

    Dose rate is calculated from the administered activity and then
    scaled according to tumour uptake and metastatic tumour burden.
    """

    dose_rate_gy_day = np.zeros_like(time_days, dtype=float)

    for t_admin in treatment_times:

        mask = time_days >= t_admin

        elapsed = time_days[mask] - t_admin

        activity_remaining = (
            activity_gbq
            * np.exp(-LU177_LAMBDA_PER_DAY * elapsed)
        )

        dose_rate_gy_day[mask] += (
            activity_remaining
            * DOSE_PER_GBQ_GY
            * dose_scale
        )

    dose_rate_gy_h = dose_rate_gy_day / 24.0

    return dose_rate_gy_day, dose_rate_gy_h


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
    """

    trep_hours = trep_days * 24.0

    rcrit_gy_h = (
        np.log(2.0)
        / (alpha * trep_hours)
    )

    rcrit_gy_day = rcrit_gy_h * 24.0

    return rcrit_gy_day, rcrit_gy_h


# =============================================================================
# EFFECTIVE DOSE RATE
# =============================================================================

def calculate_effective_dose_rate(
    physical_dose_rate_gy_day,
    critical_dose_rate_gy_day,
    gamma,
):
    """
    Effectiveness model:

        Rratio = Rphysical / Rcritical

        E = min(1, Rratio^gamma)

        Reffective = Rphysical * E
    """

    ratio = np.divide(
        physical_dose_rate_gy_day,
        critical_dose_rate_gy_day,
        out=np.zeros_like(physical_dose_rate_gy_day),
        where=critical_dose_rate_gy_day > 0,
    )

    effectiveness = np.minimum(
        1.0,
        np.power(
            np.maximum(ratio, 0.0),
            gamma,
        ),
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
    alpha,
    beta_alpha,
    trep_days,
    sensitive_fraction,
    resistant_tk_multiplier,
    resistant_radio_factor,
    repopulation_kickoff_days,
):
    """
    Two-population tumour model:

        Sensitive population
        Resistant population

    Radiation killing follows an LQ model.

    Repopulation begins after the specified kickoff time.
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

        S = sensitive[i - 1]
        R = resistant[i - 1]

        # -------------------------------------------------------------
        # Repopulation
        # -------------------------------------------------------------

        if time_days[i] >= repopulation_kickoff_days:

            S *= np.exp(
                sensitive_growth_rate * dt
            )

            R *= np.exp(
                resistant_growth_rate * dt
            )

        # -------------------------------------------------------------
        # Radiation dose during this interval
        # -------------------------------------------------------------

        dose_interval_gy = (
            physical_dose_rate_gy_day[i]
            * dt
        )

        # -------------------------------------------------------------
        # LQ survival
        # -------------------------------------------------------------

        survival_sensitive = np.exp(
            -alpha * dose_interval_gy
            - beta_sensitive * dose_interval_gy**2
        )

        survival_resistant = np.exp(
            -alpha_resistant * dose_interval_gy
            - beta_resistant * dose_interval_gy**2
        )

        # -------------------------------------------------------------
        # Apply radiation killing
        # -------------------------------------------------------------

        S *= survival_sensitive
        R *= survival_resistant

        sensitive[i] = S
        resistant[i] = R

    total = sensitive + resistant

    return (
        sensitive,
        resistant,
        total,
        initial_resistant,
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

    Initial TCP is fixed at 10%.

    Relative tumour burden is used to scale the surviving
    clonogenic burden.
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
    critical_dose_rate_gy_day,
    total_burden_ml,
    sensitive_burden_ml,
    resistant_burden_ml,
    residual_resistant_burden_percent,
    resistant_composition_percent,
    tcp,
):
    """
    Calculate summary metrics.
    """

    above_critical = (
        physical_dose_rate_gy_day
        >= critical_dose_rate_gy_day
    )

    time_above_critical = (
        np.sum(above_critical)
        * DT_DAYS
    )

    # -------------------------------------------------------------
    # Longest continuous period above critical rate
    # -------------------------------------------------------------

    longest_continuous_above = 0.0
    current_duration = 0.0

    for i in range(len(time_days)):

        if above_critical[i]:

            current_duration += DT_DAYS

            longest_continuous_above = max(
                longest_continuous_above,
                current_duration,
            )

        else:
            current_duration = 0.0

    # -------------------------------------------------------------
    # Dose integration
    # -------------------------------------------------------------

    cumulative_physical_dose = integrate_trapezoid(
        physical_dose_rate_gy_day,
        time_days,
    )

    cumulative_effective_dose = integrate_trapezoid(
        effective_dose_rate_gy_day,
        time_days,
    )

    # -------------------------------------------------------------
    # Main tumour metrics
    # -------------------------------------------------------------

    min_burden_index = np.argmin(
        total_burden_ml
    )

    max_tcp_index = np.argmax(tcp)

    final_burden = total_burden_ml[-1]

    minimum_burden = total_burden_ml[
        min_burden_index
    ]

    maximum_tcp = tcp[
        max_tcp_index
    ]

    final_resistant_percent = (
        residual_resistant_burden_percent[-1]
    )

    minimum_resistant_percent = np.min(
        residual_resistant_burden_percent
    )

    final_resistant_composition = (
        resistant_composition_percent[-1]
    )

    # -------------------------------------------------------------
    # Critical dose-rate metrics
    # -------------------------------------------------------------

    peak_physical_rate = np.max(
        physical_dose_rate_gy_day
    )

    critical_rate = (
        critical_dose_rate_gy_day
    )

    # -------------------------------------------------------------
    # First / last critical crossing
    # -------------------------------------------------------------

    crossing_indices = np.where(
        np.diff(
            above_critical.astype(int)
        ) != 0
    )[0]

    if np.any(above_critical):

        first_above = time_days[
            np.where(above_critical)[0][0]
        ]

        last_above = time_days[
            np.where(above_critical)[0][-1]
        ]

    else:

        first_above = np.nan
        last_above = np.nan

    return {
        "minimum_burden_ml": minimum_burden,
        "minimum_burden_day": time_days[min_burden_index],

        "final_burden_ml": final_burden,

        "maximum_tcp": maximum_tcp,
        "maximum_tcp_day": time_days[max_tcp_index],

        "final_residual_resistant_percent":
            final_resistant_percent,

        "minimum_residual_resistant_percent":
            minimum_resistant_percent,

        "final_resistant_composition_percent":
            final_resistant_composition,

        "final_sensitive_burden_ml":
            sensitive_burden_ml[-1],

        "final_resistant_burden_ml":
            resistant_burden_ml[-1],

        "critical_dose_rate_gy_day":
            critical_rate,

        "peak_physical_rate_gy_day":
            peak_physical_rate,

        "time_above_critical_days":
            time_above_critical,

        "longest_continuous_above_days":
            longest_continuous_above,

        "first_above_day":
            first_above,

        "last_above_day":
            last_above,

        "number_of_crossings":
            len(crossing_indices),

        "cumulative_physical_dose_gy":
            cumulative_physical_dose,

        "cumulative_effective_dose_gy":
            cumulative_effective_dose,
    }


# =============================================================================
# PLOT STYLING
# =============================================================================

def style_axis(ax):

    ax.set_facecolor(PLOT_BACKGROUND)

    ax.tick_params(
        colors=PLOT_TEXT,
        which="both",
    )

    ax.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=0.5,
        alpha=0.30,
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
        MaxNLocator(nbins=7)
    )

    for spine in ax.spines.values():
        spine.set_color(PLOT_GRID)


def style_legend(ax):

    legend = ax.legend(
        frameon=True,
        facecolor=PLOT_BACKGROUND,
        edgecolor=PLOT_GRID,
        fontsize=8,
    )

    if legend is not None:
        for text in legend.get_texts():
            text.set_color(PLOT_TEXT)


def add_treatment_markers(
    ax,
    treatment_times,
):

    for t in treatment_times:

        ax.axvline(
            t,
            color=PLOT_TREATMENT,
            linestyle=":",
            linewidth=1.0,
            alpha=0.65,
        )


# =============================================================================
# KEY RESULT STATUS
# =============================================================================

def key_result_status(
    text,
    status="neutral",
):
    """
    Green/red/neutral status text for key-result captions.
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

    return f"""
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


def key_result_caption(
    text,
):
    return f"""
    <div style="
        font-size:0.72rem;
        color:#6B7280;
        margin-top:4px;
        line-height:1.3;
    ">
        {text}
    </div>
    """


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("Model controls")


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
        "Total metastatic tumour burden represented "
        "as an aggregate volume."
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
        "Linear radiation sensitivity parameter "
        "used in the exploratory LQ model."
    ),
)

trep_days = st.sidebar.slider(
    "Trep",
    min_value=10.0,
    max_value=100.0,
    value=float(DEFAULT_TREP),
    step=1.0,
    help="Tumour repopulation doubling time.",
)

sensitive_fraction = st.sidebar.slider(
    "Sensitive fraction",
    min_value=0.50,
    max_value=0.95,
    value=float(DEFAULT_SENSITIVE_FRACTION),
    step=0.01,
    format="%.2f",
    help=(
        "Initial fraction of tumour represented "
        "by the radiosensitive population."
    ),
)

resistant_tk_multiplier = st.sidebar.slider(
    "Tk / Trep",
    min_value=1.20,
    max_value=1.80,
    value=float(DEFAULT_RESISTANT_TK_MULTIPLIER),
    step=0.01,
    format="%.2f",
    help=(
        "Relative resistant-cell repopulation timescale "
        "compared with the sensitive population."
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
        "Relative radiosensitivity of the resistant "
        "population."
    ),
)

repopulation_kickoff = st.sidebar.slider(
    "Repopulation kickoff",
    min_value=3.0,
    max_value=10.0,
    value=float(DEFAULT_REPOPULATION_KICKOFF),
    step=1.0,
    help=(
        "Day after which tumour repopulation is enabled."
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
)

interval_days = st.sidebar.slider(
    "Cycle interval (days)",
    min_value=1,
    max_value=42,
    value=int(DEFAULT_INTERVAL),
    step=1,
)

tumour_uptake_percent = st.sidebar.slider(
    "Total tumour uptake (%)",
    min_value=float(TUMOUR_UPTAKE_MIN_PERCENT),
    max_value=float(TUMOUR_UPTAKE_MAX_PERCENT),
    value=float(DEFAULT_TUMOUR_UPTAKE_PERCENT),
    step=0.1,
    format="%.1f",
    help=(
        "Phenomenological fraction of administered activity "
        "attributed to the total metastatic tumour burden."
    ),
)

gamma = st.sidebar.slider(
    "Effectiveness gamma",
    min_value=0.25,
    max_value=2.00,
    value=float(DEFAULT_GAMMA),
    step=0.05,
    format="%.2f",
    help=(
        "Controls the dose-rate effectiveness relationship."
    ),
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
# DOSE SCALING
# =============================================================================

(
    uptake_scale,
    burden_scale,
    dose_scale,
) = calculate_dose_scaling(
    initial_burden_ml,
    tumour_uptake_percent,
)


# =============================================================================
# DOSE MODEL
# =============================================================================

(
    physical_dose_rate_gy_day,
    physical_dose_rate_gy_h,
) = calculate_physical_dose_rate(
    time_days,
    activity_gbq,
    treatment_times,
    dose_scale,
)


# =============================================================================
# CRITICAL DOSE RATE
# =============================================================================

(
    critical_dose_rate_gy_day,
    critical_dose_rate_gy_h,
) = calculate_critical_dose_rate(
    alpha,
    trep_days,
)


# =============================================================================
# EFFECTIVE DOSE RATE
# =============================================================================

(
    dose_rate_ratio,
    effectiveness,
    effective_dose_rate_gy_day,
) = calculate_effective_dose_rate(
    physical_dose_rate_gy_day,
    critical_dose_rate_gy_day,
    gamma,
)


# =============================================================================
# TUMOUR DYNAMICS
# =============================================================================

(
    sensitive_burden_ml,
    resistant_burden_ml,
    total_burden_ml,
    initial_resistant_burden,
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
# RESISTANT BURDEN METRICS
# =============================================================================

if initial_resistant_burden > 0:

    residual_resistant_burden_percent = (
        resistant_burden_ml
        / initial_resistant_burden
        * 100.0
    )

else:

    residual_resistant_burden_percent = np.zeros_like(
        resistant_burden_ml
    )


resistant_composition_fraction = np.divide(
    resistant_burden_ml,
    total_burden_ml,
    out=np.zeros_like(resistant_burden_ml),
    where=total_burden_ml > 0,
)

resistant_composition_percent = (
    resistant_composition_fraction
    * 100.0
)


# =============================================================================
# TCP
# =============================================================================

tcp = calculate_tcp(
    total_burden_ml,
    initial_burden_ml,
)

tcp_percent = tcp * 100.0


# =============================================================================
# SUMMARY
# =============================================================================

summary = calculate_summary(
    time_days=time_days,
    physical_dose_rate_gy_day=physical_dose_rate_gy_day,
    effective_dose_rate_gy_day=effective_dose_rate_gy_day,
    critical_dose_rate_gy_day=critical_dose_rate_gy_day,
    total_burden_ml=total_burden_ml,
    sensitive_burden_ml=sensitive_burden_ml,
    resistant_burden_ml=resistant_burden_ml,
    residual_resistant_burden_percent=
        residual_resistant_burden_percent,
    resistant_composition_percent=
        resistant_composition_percent,
    tcp=tcp,
)


# =============================================================================
# HEADER
# =============================================================================

st.title("Lu-177 PSMA Interactive Model Explorer")

st.markdown(
    """
This interactive model explores how Lu-177 PSMA treatment
parameters and tumour characteristics influence physical dose rate,
tumour burden, sensitive/resistant tumour populations and TCP.
"""
)


# =============================================================================
# KEY RESULTS
# =============================================================================

st.markdown("## Key results")

key_cols = st.columns(4)


# -----------------------------------------------------------------------------
# Minimum tumour burden
# -----------------------------------------------------------------------------

burden_reduction_percent = (
    1.0
    - summary["minimum_burden_ml"]
    / initial_burden_ml
) * 100.0

if burden_reduction_percent > 0:

    burden_status = "positive"

    burden_status_text = (
        f"↓ {burden_reduction_percent:.1f}% "
        f"from {initial_burden_ml:.0f} mL baseline"
    )

elif burden_reduction_percent < 0:

    burden_status = "negative"

    burden_status_text = (
        f"↑ {abs(burden_reduction_percent):.1f}% "
        f"from {initial_burden_ml:.0f} mL baseline"
    )

else:

    burden_status = "neutral"

    burden_status_text = (
        "No change from baseline"
    )


with key_cols[0]:

    st.markdown(
        """
        <div style="
            font-size:0.78rem;
            font-weight:600;
            color:#6B7280;
            margin-bottom:2px;
        ">
            Minimum tumour burden
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            font-size:1.55rem;
            font-weight:700;
            line-height:1.1;
            color:#111827;
        ">
            {summary["minimum_burden_ml"]:.1f} mL
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        key_result_status(
            burden_status_text,
            burden_status,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        key_result_caption(
            f"Minimum reached at day "
            f"{summary['minimum_burden_day']:.1f}"
        ),
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Maximum TCP
# -----------------------------------------------------------------------------

tcp_change_pp = (
    summary["maximum_tcp"] * 100.0
    - INITIAL_TCP * 100.0
)

if tcp_change_pp > 0:

    tcp_status = "positive"

    tcp_status_text = (
        f"↑ {tcp_change_pp:.1f} percentage points "
        f"from 10% baseline"
    )

elif tcp_change_pp < 0:

    tcp_status = "negative"

    tcp_status_text = (
        f"↓ {abs(tcp_change_pp):.1f} percentage points "
        f"from 10% baseline"
    )

else:

    tcp_status = "neutral"

    tcp_status_text = (
        "No change from 10% baseline"
    )


with key_cols[1]:

    st.markdown(
        """
        <div style="
            font-size:0.78rem;
            font-weight:600;
            color:#6B7280;
            margin-bottom:2px;
        ">
            Maximum TCP
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            font-size:1.55rem;
            font-weight:700;
            line-height:1.1;
            color:#111827;
        ">
            {summary["maximum_tcp"] * 100.0:.1f}%
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        key_result_status(
            tcp_status_text,
            tcp_status,
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


# -----------------------------------------------------------------------------
# Residual resistant burden
# -----------------------------------------------------------------------------

resistant_change = (
    summary["final_residual_resistant_percent"]
    - 100.0
)

if resistant_change < 0:

    resistant_status = "positive"

    resistant_status_text = (
        f"↓ {abs(resistant_change):.1f}% "
        f"from initial resistant burden"
    )

elif resistant_change > 0:

    resistant_status = "negative"

    resistant_status_text = (
        f"↑ {resistant_change:.1f}% "
        f"from initial resistant burden"
    )

else:

    resistant_status = "neutral"

    resistant_status_text = (
        "No change from initial resistant burden"
    )


with key_cols[2]:

    st.markdown(
        """
        <div style="
            font-size:0.78rem;
            font-weight:600;
            color:#6B7280;
            margin-bottom:2px;
        ">
            Residual resistant burden
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            font-size:1.55rem;
            font-weight:700;
            line-height:1.1;
            color:#111827;
        ">
            {summary["final_residual_resistant_percent"]:.1f}%
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        key_result_status(
            resistant_status_text,
            resistant_status,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        key_result_caption(
            "% of the initial resistant tumour burden remaining"
        ),
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Cumulative physical dose
# -----------------------------------------------------------------------------

with key_cols[3]:

    st.markdown(
        """
        <div style="
            font-size:0.78rem;
            font-weight:600;
            color:#6B7280;
            margin-bottom:2px;
        ">
            Cumulative physical dose
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            font-size:1.55rem;
            font-weight:700;
            line-height:1.1;
            color:#111827;
        ">
            {summary["cumulative_physical_dose_gy"]:.2f} Gy
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        key_result_status(
            f"{activity_gbq:.1f} GBq × {n_cycles} cycles",
            "neutral",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        key_result_caption(
            "Integrated physical dose over the simulation"
        ),
        unsafe_allow_html=True,
    )


# =============================================================================
# SECONDARY METRICS
# =============================================================================

st.markdown("## Secondary metrics")

# Yellow/gold section accent
st.markdown(
    f"""
    <div style="
        height:3px;
        background:{COLOR_YELLOW};
        border-radius:3px;
        margin-top:-8px;
        margin-bottom:16px;
        opacity:0.9;
    ">
    </div>
    """,
    unsafe_allow_html=True,
)

secondary_cols = st.columns(4)


secondary_values = [
    (
        "Critical dose rate",
        f"{summary['critical_dose_rate_gy_day']:.4f} Gy/day",
        "Rcrit based on alpha and Trep",
    ),
    (
        "Peak physical rate",
        f"{summary['peak_physical_rate_gy_day']:.3f} Gy/day",
        "Maximum instantaneous physical dose rate",
    ),
    (
        "Time above critical rate",
        f"{summary['time_above_critical_days']:.1f} days",
        "Time with physical dose rate ≥ Rcrit",
    ),
    (
        "Effective cumulative dose",
        f"{summary['cumulative_effective_dose_gy']:.2f} Gy",
        "Integrated effective dose",
    ),
]


for col, (
    title,
    value,
    caption,
) in zip(
    secondary_cols,
    secondary_values,
):

    with col:

        st.markdown(
            f"""
            <div style="
                border-left:3px solid {COLOR_YELLOW};
                padding-left:12px;
                margin-bottom:12px;
            ">

                <div style="
                    font-size:0.76rem;
                    font-weight:600;
                    color:{COLOR_YELLOW};
                    margin-bottom:3px;
                ">
                    {title}
                </div>

                <div style="
                    font-size:1.18rem;
                    font-weight:700;
                    color:#F2C94C;
                    line-height:1.15;
                ">
                    {value}
                </div>

                <div style="
                    font-size:0.68rem;
                    color:#9CA3AF;
                    margin-top:4px;
                    line-height:1.25;
                ">
                    {caption}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# DOSE SCALING INFORMATION
# =============================================================================

st.caption(
    f"""
    Dose scaling: uptake scale = {uptake_scale:.2f}, 
    burden scale = {burden_scale:.2f}, 
    combined dose scale = {dose_scale:.2f}. 
    Reference condition = {REFERENCE_METASTATIC_BURDEN_ML:.0f} mL 
    and {DEFAULT_TUMOUR_UPTAKE_PERCENT:.1f}% tumour uptake.
    """
)


# =============================================================================
# MODEL TRAJECTORIES
# =============================================================================

st.markdown("## Model trajectories")

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 9),
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
    physical_dose_rate_gy_day,
    color=COLOR_BLUE,
    linewidth=2.2,
    label="Physical dose rate",
)

ax.plot(
    time_days,
    effective_dose_rate_gy_day,
    color=COLOR_ORANGE,
    linewidth=2.0,
    linestyle="--",
    label="Effective dose rate",
)

ax.axhline(
    critical_dose_rate_gy_day,
    color=COLOR_RED,
    linestyle=":",
    linewidth=2.0,
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
    "Dose rate (Gy/day)"
)

style_axis(ax)
style_legend(ax)


# =============================================================================
# PLOT 2 — TUMOUR BURDEN
# =============================================================================

ax = axes[0, 1]

ax.plot(
    time_days,
    total_burden_ml,
    color=COLOR_BLUE,
    linewidth=2.2,
    label="Total tumour",
)

ax.plot(
    time_days,
    sensitive_burden_ml,
    color=COLOR_GREEN,
    linewidth=1.9,
    linestyle="--",
    label="Sensitive tumour",
)

ax.plot(
    time_days,
    resistant_burden_ml,
    color=COLOR_ORANGE,
    linewidth=1.9,
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
    sensitive_burden_ml,
    color=COLOR_GREEN,
    linewidth=2.0,
    label="Sensitive tumour",
)

ax.plot(
    time_days,
    resistant_burden_ml,
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
    "Sensitive and resistant tumour populations",
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

# Secondary y-axis for residual resistant burden

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

ax2.grid(
    False
)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

legend = ax.legend(
    lines1 + lines2,
    labels1 + labels2,
    frameon=True,
    facecolor=PLOT_BACKGROUND,
    edgecolor=PLOT_GRID,
    fontsize=8,
)

for text in legend.get_texts():
    text.set_color(PLOT_TEXT)


# =============================================================================
# PLOT 4 — TCP
# =============================================================================

ax = axes[1, 1]

ax.plot(
    time_days,
    tcp_percent,
    color=COLOR_BLUE,
    linewidth=2.2,
    label="TCP",
)

ax.axhline(
    50.0,
    color=COLOR_ORANGE,
    linestyle="--",
    linewidth=1.8,
    label="50% TCP",
)

ax.axhline(
    90.0,
    color=COLOR_GREEN,
    linestyle=":",
    linewidth=1.8,
    label="90% TCP",
)

add_treatment_markers(
    ax,
    treatment_times,
)

ax.set_ylim(
    0,
    100,
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

style_axis(ax)
style_legend(ax)


# =============================================================================
# FIGURE LAYOUT
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

        st.markdown("### Tumour response")

        st.write(
            f"Minimum tumour burden: "
            f"{summary['minimum_burden_ml']:.2f} mL"
        )

        st.write(
            f"Minimum reached at: "
            f"day {summary['minimum_burden_day']:.2f}"
        )

        st.write(
            f"Final tumour burden: "
            f"{summary['final_burden_ml']:.2f} mL"
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
            f"Final residual resistant burden: "
            f"{summary['final_residual_resistant_percent']:.2f}%"
        )

        st.write(
            f"Final resistant composition: "
            f"{summary['final_resistant_composition_percent']:.2f}%"
        )

        st.write(
            f"Maximum TCP: "
            f"{summary['maximum_tcp'] * 100:.2f}%"
        )

        st.write(
            f"Maximum TCP reached at: "
            f"day {summary['maximum_tcp_day']:.2f}"
        )

    with col2:

        st.markdown("### Radiation response")

        st.write(
            f"Critical dose rate: "
            f"{summary['critical_dose_rate_gy_day']:.5f} Gy/day"
        )

        st.write(
            f"Peak physical dose rate: "
            f"{summary['peak_physical_rate_gy_day']:.4f} Gy/day"
        )

        st.write(
            f"Time above critical rate: "
            f"{summary['time_above_critical_days']:.2f} days"
        )

        st.write(
            f"Longest continuous period above critical: "
            f"{summary['longest_continuous_above_days']:.2f} days"
        )

        st.write(
            f"First above critical rate: "
            f"{summary['first_above_day']:.2f} days"
            if not np.isnan(summary["first_above_day"])
            else "First above critical rate: Never"
        )

        st.write(
            f"Last above critical rate: "
            f"{summary['last_above_day']:.2f} days"
            if not np.isnan(summary["last_above_day"])
            else "Last above critical rate: Never"
        )

        st.write(
            f"Number of critical-rate crossings: "
            f"{summary['number_of_crossings']}"
        )

        st.write(
            f"Cumulative physical dose: "
            f"{summary['cumulative_physical_dose_gy']:.3f} Gy"
        )

        st.write(
            f"Cumulative effective dose: "
            f"{summary['cumulative_effective_dose_gy']:.3f} Gy"
        )


# =============================================================================
# MODEL ASSUMPTIONS AND LIMITATIONS
# =============================================================================

with st.expander(
    "Model assumptions and limitations"
):

    st.markdown(
        """
### Tumour burden

The model treats the metastatic disease burden as an aggregate
tumour volume rather than as a single solid tumour.

### Tumour uptake

The total tumour uptake parameter represents a phenomenological
fraction of administered activity attributed to the aggregate
metastatic tumour burden.

The current dose model does not represent individual lesion
dosimetry, lesion-specific TACs, heterogeneous uptake, or
cross-dose.

### Radiation response

Radiation killing is represented using a two-population
linear-quadratic model.

The sensitive and resistant populations have different
radiosensitivity and repopulation characteristics.

### Critical dose rate

The critical dose rate is calculated from:

Rcrit = ln(2) / (alpha × Trep)

and is used to assess whether the physical dose rate is
sufficient to overcome tumour repopulation.

### Effective dose rate

The effective dose-rate model applies a phenomenological
dose-rate effectiveness function:

E = min(1, (R/Rcrit)^gamma)

This is intended for exploratory modelling rather than as
a validated clinical radiobiological model.

### TCP

TCP is normalized to an initial TCP of 10% and is driven by
the relative tumour burden.

It should therefore be interpreted as an exploratory response
metric rather than a clinically validated probability of cure.

### Resistant disease

The primary resistant-disease metric is the residual resistant
burden expressed as a percentage of the initial resistant
tumour burden.

The resistant composition of the remaining tumour is also
reported separately because a tumour can become predominantly
resistant even while the absolute resistant burden is falling.

### Clinical dosimetry

Patient-specific Lu-177 PSMA dosimetry would require lesion-level
activity measurements, time-activity curves, absorbed fractions,
tumour mass, spatial distribution and potentially cross-dose.
"""
    )


# =============================================================================
# CSV EXPORT
# =============================================================================

csv_df = pd.DataFrame({
    "time_days": time_days,

    "physical_dose_rate_gy_day":
        physical_dose_rate_gy_day,

    "physical_dose_rate_gy_h":
        physical_dose_rate_gy_h,

    "effective_dose_rate_gy_day":
        effective_dose_rate_gy_day,

    "critical_dose_rate_gy_day":
        np.full_like(
            time_days,
            critical_dose_rate_gy_day,
        ),

    "dose_rate_ratio":
        dose_rate_ratio,

    "effectiveness":
        effectiveness,

    "total_tumour_burden_ml":
        total_burden_ml,

    "sensitive_tumour_burden_ml":
        sensitive_burden_ml,

    "resistant_tumour_burden_ml":
        resistant_burden_ml,

    "initial_resistant_burden_ml":
        np.full_like(
            time_days,
            initial_resistant_burden,
        ),

    "residual_resistant_burden_percent":
        residual_resistant_burden_percent,

    "resistant_composition_percent":
        resistant_composition_percent,

    "tcp_percent":
        tcp_percent,

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
})


csv_buffer = io.StringIO()

csv_df.to_csv(
    csv_buffer,
    index=False,
)

st.download_button(
    label="Download model results (CSV)",
    data=csv_buffer.getvalue(),
    file_name="Lu177_PSMA_interactive_model_results.csv",
    mime="text/csv",
)


# =============================================================================
# HIGH-RESOLUTION PNG EXPORT
# =============================================================================

png_buffer = io.BytesIO()

fig_export, axes_export = plt.subplots(
    2,
    2,
    figsize=(14, 9),
)

fig_export.patch.set_facecolor(
    PLOT_BACKGROUND
)


# -----------------------------------------------------------------------------
# Export plot 1
# -----------------------------------------------------------------------------

ax = axes_export[0, 0]

ax.plot(
    time_days,
    physical_dose_rate_gy_day,
    color=COLOR_BLUE,
    linewidth=2.2,
    label="Physical dose rate",
)

ax.plot(
    time_days,
    effective_dose_rate_gy_day,
    color=COLOR_ORANGE,
    linewidth=2.0,
    linestyle="--",
    label="Effective dose rate",
)

ax.axhline(
    critical_dose_rate_gy_day,
    color=COLOR_RED,
    linestyle=":",
    linewidth=2.0,
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
    "Dose rate (Gy/day)"
)

style_axis(ax)
style_legend(ax)


# -----------------------------------------------------------------------------
# Export plot 2
# -----------------------------------------------------------------------------

ax = axes_export[0, 1]

ax.plot(
    time_days,
    total_burden_ml,
    color=COLOR_BLUE,
    linewidth=2.2,
    label="Total tumour",
)

ax.plot(
    time_days,
    sensitive_burden_ml,
    color=COLOR_GREEN,
    linewidth=1.9,
    linestyle="--",
    label="Sensitive tumour",
)

ax.plot(
    time_days,
    resistant_burden_ml,
    color=COLOR_ORANGE,
    linewidth=1.9,
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


# -----------------------------------------------------------------------------
# Export plot 3
# -----------------------------------------------------------------------------

ax = axes_export[1, 0]

ax.plot(
    time_days,
    sensitive_burden_ml,
    color=COLOR_GREEN,
    linewidth=2.0,
    label="Sensitive tumour",
)

ax.plot(
    time_days,
    resistant_burden_ml,
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
    "Sensitive and resistant tumour populations",
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

ax2.grid(False)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

legend = ax.legend(
    lines1 + lines2,
    labels1 + labels2,
    frameon=True,
    facecolor=PLOT_BACKGROUND,
    edgecolor=PLOT_GRID,
    fontsize=8,
)

for text in legend.get_texts():
    text.set_color(PLOT_TEXT)


# -----------------------------------------------------------------------------
# Export plot 4
# -----------------------------------------------------------------------------

ax = axes_export[1, 1]

ax.plot(
    time_days,
    tcp_percent,
    color=COLOR_BLUE,
    linewidth=2.2,
    label="TCP",
)

ax.axhline(
    50.0,
    color=COLOR_ORANGE,
    linestyle="--",
    linewidth=1.8,
    label="50% TCP",
)

ax.axhline(
    90.0,
    color=COLOR_GREEN,
    linestyle=":",
    linewidth=1.8,
    label="90% TCP",
)

add_treatment_markers(
    ax,
    treatment_times,
)

ax.set_ylim(
    0,
    100,
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

style_axis(ax)
style_legend(ax)


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

plt.close(fig_export)

st.download_button(
    label="Download high-resolution plots (600 dpi PNG)",
    data=png_buffer.getvalue(),
    file_name="Lu177_PSMA_interactive_model_600dpi.png",
    mime="image/png",
)
