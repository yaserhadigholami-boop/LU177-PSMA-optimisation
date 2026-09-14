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
    page_title="Lu-177 PSMA Optimisation Model",
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


# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

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


# =============================================================================
# TCP PARAMETERS
# =============================================================================

INITIAL_TCP = 0.10
INITIAL_CLONOGENIC_BURDEN = -np.log(INITIAL_TCP)


# =============================================================================
# PROFESSIONAL PLOT COLOURS
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

# Key-result status colours
STATUS_GREEN = "#39D353"
STATUS_RED = "#FF5C5C"
STATUS_NEUTRAL = "#B8B8B8"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10.5,
        "axes.titleweight": "normal",
        "axes.linewidth": 0.9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "text.color": PLOT_TEXT,
        "axes.labelcolor": PLOT_TEXT,
        "axes.edgecolor": PLOT_TEXT,
        "xtick.color": PLOT_TEXT,
        "ytick.color": PLOT_TEXT,
        "axes.facecolor": PLOT_BACKGROUND,
        "figure.facecolor": PLOT_BACKGROUND,
        "savefig.facecolor": PLOT_BACKGROUND,
    }
)


# =============================================================================
# NUMERICAL INTEGRATION
# =============================================================================

def integrate_trapezoid(y, x):

    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)

    return np.trapz(y, x)


# =============================================================================
# DOSE SCALING
# =============================================================================

def calculate_dose_scaling(
    initial_burden_ml,
    tumour_uptake_percent
):

    uptake_scale = (
        tumour_uptake_percent /
        DEFAULT_TUMOUR_UPTAKE_PERCENT
    )

    burden_scale = (
        REFERENCE_METASTATIC_BURDEN_ML /
        initial_burden_ml
    )

    dose_scale = (
        uptake_scale *
        burden_scale
    )

    return (
        uptake_scale,
        burden_scale,
        dose_scale
    )


# =============================================================================
# TREATMENT TIMES
# =============================================================================

def calculate_treatment_times(
    n_cycles,
    interval_days
):

    return np.array(
        [
            i * interval_days
            for i in range(n_cycles)
        ],
        dtype=float
    )


# =============================================================================
# PHYSICAL DOSE RATE
# =============================================================================

def calculate_physical_dose_rate(
    time_days,
    activity_gbq,
    n_cycles,
    interval_days,
    initial_burden_ml,
    tumour_uptake_percent
):

    (
        uptake_scale,
        burden_scale,
        dose_scale
    ) = calculate_dose_scaling(
        initial_burden_ml,
        tumour_uptake_percent
    )

    treatment_times = calculate_treatment_times(
        n_cycles,
        interval_days
    )

    dose_rate_gy_day = np.zeros_like(
        time_days,
        dtype=float
    )

    for t_admin in treatment_times:

        elapsed = time_days - t_admin

        mask = elapsed >= 0

        activity = np.zeros_like(
            time_days,
            dtype=float
        )

        activity[mask] = (
            activity_gbq *
            np.exp(
                -LU177_LAMBDA_PER_DAY *
                elapsed[mask]
            )
        )

        dose_rate_gy_day += (
            activity *
            DOSE_PER_GBQ_GY *
            dose_scale
        )

    return (
        dose_rate_gy_day,
        treatment_times,
        uptake_scale,
        burden_scale,
        dose_scale
    )


# =============================================================================
# CRITICAL DOSE RATE
# =============================================================================

def calculate_critical_dose_rate(
    alpha,
    trep_days
):

    trep_hours = trep_days * 24.0

    return (
        np.log(2.0) /
        (
            alpha *
            trep_hours
        )
    )


# =============================================================================
# EFFECTIVE DOSE RATE
# =============================================================================

def calculate_effective_dose_rate(
    physical_dose_rate_gy_day,
    critical_dose_rate_gy_h,
    gamma
):

    critical_rate_gy_day = (
        critical_dose_rate_gy_h *
        24.0
    )

    ratio = np.divide(
        physical_dose_rate_gy_day,
        critical_rate_gy_day,
        out=np.zeros_like(
            physical_dose_rate_gy_day
        ),
        where=critical_rate_gy_day > 0
    )

    effectiveness = np.minimum(
        1.0,
        np.power(
            np.maximum(
                ratio,
                0.0
            ),
            gamma
        )
    )

    effective_dose_rate_gy_day = (
        physical_dose_rate_gy_day *
        effectiveness
    )

    return (
        ratio,
        effectiveness,
        effective_dose_rate_gy_day
    )


# =============================================================================
# TUMOUR DYNAMICS
# =============================================================================

def calculate_tumour_dynamics(
    time_days,
    physical_dose_rate_gy_day,
    alpha,
    trep_days,
    sensitive_fraction,
    resistant_tk_multiplier,
    resistant_radio_factor,
    repopulation_kickoff,
    initial_burden_ml
):

    # -------------------------------------------------------------------------
    # LQ PARAMETERS
    # -------------------------------------------------------------------------

    beta_sensitive = (
        alpha *
        DEFAULT_BETA_ALPHA
    )

    alpha_resistant = (
        alpha *
        resistant_radio_factor
    )

    beta_resistant = (
        beta_sensitive *
        resistant_radio_factor
    )

    # -------------------------------------------------------------------------
    # GROWTH RATES
    # -------------------------------------------------------------------------

    sensitive_growth_rate = (
        np.log(2.0) /
        trep_days
    )

    resistant_growth_rate = (
        np.log(2.0) /
        (
            trep_days *
            resistant_tk_multiplier
        )
    )

    # -------------------------------------------------------------------------
    # ARRAYS
    # -------------------------------------------------------------------------

    sensitive = np.zeros_like(
        time_days,
        dtype=float
    )

    resistant = np.zeros_like(
        time_days,
        dtype=float
    )

    # -------------------------------------------------------------------------
    # INITIAL CONDITIONS
    # -------------------------------------------------------------------------

    sensitive[0] = (
        initial_burden_ml *
        sensitive_fraction
    )

    resistant[0] = (
        initial_burden_ml *
        (
            1.0 -
            sensitive_fraction
        )
    )

    # -------------------------------------------------------------------------
    # TIME LOOP
    # -------------------------------------------------------------------------

    for i in range(1, len(time_days)):

        dt = (
            time_days[i] -
            time_days[i - 1]
        )

        s = sensitive[i - 1]
        r = resistant[i - 1]

        # ---------------------------------------------------------------------
        # REPOPULATION
        # ---------------------------------------------------------------------

        if time_days[i] >= repopulation_kickoff:

            s *= np.exp(
                sensitive_growth_rate *
                dt
            )

            r *= np.exp(
                resistant_growth_rate *
                dt
            )

        # ---------------------------------------------------------------------
        # RADIATION KILLING
        # ---------------------------------------------------------------------

        dose_interval_gy = (
            physical_dose_rate_gy_day[i] *
            dt
        )

        survival_sensitive = np.exp(
            -alpha *
            dose_interval_gy
            -
            beta_sensitive *
            dose_interval_gy ** 2
        )

        survival_resistant = np.exp(
            -alpha_resistant *
            dose_interval_gy
            -
            beta_resistant *
            dose_interval_gy ** 2
        )

        s *= survival_sensitive
        r *= survival_resistant

        sensitive[i] = max(
            s,
            0.0
        )

        resistant[i] = max(
            r,
            0.0
        )

    # -------------------------------------------------------------------------
    # TOTAL BURDEN
    # -------------------------------------------------------------------------

    total = (
        sensitive +
        resistant
    )

    # -------------------------------------------------------------------------
    # INITIAL RESISTANT BURDEN
    # -------------------------------------------------------------------------

    initial_resistant_burden = (
        initial_burden_ml *
        (
            1.0 -
            sensitive_fraction
        )
    )

    # -------------------------------------------------------------------------
    # RESISTANT COMPOSITION
    #
    # Percentage of CURRENT tumour that is resistant.
    # -------------------------------------------------------------------------

    resistant_composition_fraction = np.divide(
        resistant,
        total,
        out=np.zeros_like(resistant),
        where=total > 0
    )

    # -------------------------------------------------------------------------
    # RESIDUAL RESISTANT BURDEN
    #
    # Percentage of INITIAL resistant tumour burden remaining.
    # -------------------------------------------------------------------------

    residual_resistant_burden_percent = np.divide(
        resistant,
        initial_resistant_burden,
        out=np.zeros_like(resistant),
        where=initial_resistant_burden > 0
    ) * 100.0

    return (
        total,
        sensitive,
        resistant,
        resistant_composition_fraction,
        residual_resistant_burden_percent,
        initial_resistant_burden
    )


# =============================================================================
# TCP
# =============================================================================

def calculate_tcp(
    total_burden_ml,
    initial_burden_ml
):

    relative_burden = np.divide(
        total_burden_ml,
        initial_burden_ml,
        out=np.zeros_like(
            total_burden_ml
        ),
        where=initial_burden_ml > 0
    )

    surviving_clonogens = (
        INITIAL_CLONOGENIC_BURDEN *
        relative_burden
    )

    tcp = np.exp(
        -surviving_clonogens
    )

    return np.clip(
        tcp,
        0.0,
        1.0
    )


# =============================================================================
# SUMMARY
# =============================================================================

def calculate_summary(
    time_days,
    physical_dose_rate_gy_day,
    effective_dose_rate_gy_day,
    critical_dose_rate_gy_h,
    total_burden,
    sensitive_burden,
    resistant_burden,
    resistant_composition_fraction,
    residual_resistant_burden_percent,
    tcp,
    treatment_times
):

    physical_rate_gy_h = (
        physical_dose_rate_gy_day /
        24.0
    )

    effective_rate_gy_h = (
        effective_dose_rate_gy_day /
        24.0
    )

    # -------------------------------------------------------------------------
    # DOSE
    # -------------------------------------------------------------------------

    cumulative_physical_dose = integrate_trapezoid(
        physical_dose_rate_gy_day,
        time_days
    )

    cumulative_effective_dose = integrate_trapezoid(
        effective_dose_rate_gy_day,
        time_days
    )

    peak_physical_rate = np.max(
        physical_rate_gy_h
    )

    peak_effective_rate = np.max(
        effective_rate_gy_h
    )

    # -------------------------------------------------------------------------
    # CRITICAL DOSE RATE
    # -------------------------------------------------------------------------

    above_critical = (
        physical_rate_gy_h >=
        critical_dose_rate_gy_h
    )

    if len(time_days) > 1:

        dt_days = np.diff(
            time_days
        )

        time_above_critical = np.sum(
            dt_days *
            above_critical[:-1]
        )

    else:

        time_above_critical = 0.0

    # -------------------------------------------------------------------------
    # LONGEST CONTINUOUS PERIOD
    # -------------------------------------------------------------------------

    longest_continuous = 0.0
    current_duration = 0.0

    for i in range(1, len(time_days)):

        if (
            above_critical[i - 1]
            and
            above_critical[i]
        ):

            current_duration += (
                time_days[i] -
                time_days[i - 1]
            )

            longest_continuous = max(
                longest_continuous,
                current_duration
            )

        else:

            current_duration = 0.0

    # -------------------------------------------------------------------------
    # TUMOUR RESPONSE
    # -------------------------------------------------------------------------

    minimum_burden = np.min(
        total_burden
    )

    minimum_burden_day = time_days[
        np.argmin(total_burden)
    ]

    final_burden = total_burden[-1]

    maximum_tcp = np.max(tcp)

    maximum_tcp_percent = (
        maximum_tcp *
        100.0
    )

    maximum_tcp_day = time_days[
        np.argmax(tcp)
    ]

    final_residual_resistant_percent = (
        residual_resistant_burden_percent[-1]
    )

    minimum_residual_resistant_percent = (
        np.min(
            residual_resistant_burden_percent
        )
    )

    final_resistant_composition = (
        resistant_composition_fraction[-1] *
        100.0
    )

    final_sensitive = sensitive_burden[-1]
    final_resistant = resistant_burden[-1]

    treatment_times_text = ", ".join(
        f"{x:.1f}"
        for x in treatment_times
    )

    return {

        "Minimum tumour burden (mL)":
            minimum_burden,

        "Day of minimum tumour burden":
            minimum_burden_day,

        "Final tumour burden (mL)":
            final_burden,

        "Maximum TCP (%)":
            maximum_tcp_percent,

        "Day of maximum TCP":
            maximum_tcp_day,

        "Final residual resistant burden (%)":
            final_residual_resistant_percent,

        "Minimum residual resistant burden (%)":
            minimum_residual_resistant_percent,

        "Final resistant composition (%)":
            final_resistant_composition,

        "Final sensitive burden (mL)":
            final_sensitive,

        "Final resistant burden (mL)":
            final_resistant,

        "Critical dose rate (Gy/h)":
            critical_dose_rate_gy_h,

        "Peak physical dose rate (Gy/h)":
            peak_physical_rate,

        "Peak effective dose rate (Gy/h)":
            peak_effective_rate,

        "Time above critical dose rate (days)":
            time_above_critical,

        "Longest continuous time above critical (days)":
            longest_continuous,

        "Cumulative physical dose (Gy)":
            cumulative_physical_dose,

        "Effective cumulative dose (Gy)":
            cumulative_effective_dose,

        "Treatment times (days)":
            treatment_times_text,
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
        direction="out",
        length=4.5,
        width=0.8
    )

    for spine in ax.spines.values():

        spine.set_color(
            PLOT_TEXT
        )

        spine.set_linewidth(
            0.8
        )

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.45,
        color=PLOT_GRID,
        alpha=0.42
    )

    ax.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.30,
        color=PLOT_GRID,
        alpha=0.20
    )

    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=9)
    )

    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=8)
    )

    ax.minorticks_on()

    ax.tick_params(
        which="minor",
        length=2.5,
        width=0.5,
        color=PLOT_TEXT
    )


def style_legend(legend):

    frame = legend.get_frame()

    frame.set_facecolor(
        PLOT_BACKGROUND
    )

    frame.set_edgecolor(
        "#555555"
    )

    frame.set_alpha(
        0.85
    )

    for text in legend.get_texts():

        text.set_color(
            PLOT_TEXT
        )


def add_treatment_markers(
    ax,
    treatment_times,
    alpha=0.30
):

    for t_admin in treatment_times:

        ax.axvline(
            t_admin,
            linestyle=":",
            linewidth=0.75,
            color=PLOT_TREATMENT,
            alpha=alpha,
            zorder=1
        )


# =============================================================================
# KEY RESULT HELPER
# =============================================================================

def key_result_status(
    text,
    status="neutral"
):

    if status == "positive":

        colour = STATUS_GREEN

    elif status == "negative":

        colour = STATUS_RED

    else:

        colour = STATUS_NEUTRAL

    st.markdown(
        f"""
        <div style="
            color: {colour};
            font-size: 0.82rem;
            font-weight: 600;
            margin-top: -0.25rem;
            margin-bottom: 0.10rem;
            line-height: 1.25;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )


def key_result_caption(text):

    st.markdown(
        f"""
        <div style="
            color: #AFAFAF;
            font-size: 0.72rem;
            margin-top: 0.05rem;
            line-height: 1.25;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header(
    "Model parameters"
)


initial_burden_ml = st.sidebar.slider(
    "Initial metastatic burden (mL)",
    min_value=10.0,
    max_value=1500.0,
    value=DEFAULT_BURDEN,
    step=1.0,
    format="%.0f"
)


alpha = st.sidebar.slider(
    "Alpha",
    min_value=0.001,
    max_value=0.50,
    value=DEFAULT_ALPHA,
    step=0.001,
    format="%.3f",
    help=(
        "Linear radiation sensitivity parameter used in the "
        "exploratory LQ model."
    )
)


trep_days = st.sidebar.slider(
    "Trep (days)",
    min_value=10.0,
    max_value=100.0,
    value=DEFAULT_TREP,
    step=1.0,
    format="%.0f"
)


sensitive_fraction = st.sidebar.slider(
    "Sensitive fraction",
    min_value=0.50,
    max_value=0.95,
    value=DEFAULT_SENSITIVE_FRACTION,
    step=0.01,
    format="%.2f"
)


resistant_tk_multiplier = st.sidebar.slider(
    "Tk / Trep",
    min_value=1.20,
    max_value=1.80,
    value=DEFAULT_RESISTANT_TK_MULTIPLIER,
    step=0.01,
    format="%.2f"
)


resistant_radio_factor = st.sidebar.slider(
    "Resistant radio factor",
    min_value=0.05,
    max_value=1.00,
    value=DEFAULT_RESISTANT_RADIO_FACTOR,
    step=0.01,
    format="%.2f"
)


repopulation_kickoff = st.sidebar.slider(
    "Repopulation kickoff (days)",
    min_value=3.0,
    max_value=10.0,
    value=DEFAULT_REPOPULATION_KICKOFF,
    step=0.5,
    format="%.1f"
)


activity_gbq = st.sidebar.slider(
    "Activity / cycle (GBq)",
    min_value=1.0,
    max_value=15.0,
    value=DEFAULT_ACTIVITY,
    step=0.1,
    format="%.1f"
)


n_cycles = st.sidebar.slider(
    "Number of cycles",
    min_value=1,
    max_value=8,
    value=DEFAULT_CYCLES,
    step=1
)


interval_days = st.sidebar.slider(
    "Cycle interval (days)",
    min_value=1,
    max_value=42,
    value=DEFAULT_INTERVAL,
    step=1
)


gamma = st.sidebar.slider(
    "Effectiveness gamma",
    min_value=0.25,
    max_value=2.00,
    value=DEFAULT_GAMMA,
    step=0.05,
    format="%.2f"
)


tumour_uptake_percent = st.sidebar.slider(
    "Total tumour uptake (%)",
    min_value=TUMOUR_UPTAKE_MIN_PERCENT,
    max_value=TUMOUR_UPTAKE_MAX_PERCENT,
    value=DEFAULT_TUMOUR_UPTAKE_PERCENT,
    step=0.1,
    format="%.1f",
    help=(
        "Exploratory total metastatic tumour uptake of administered "
        "Lu-177 activity. This is a phenomenological uptake scaling, "
        "not a lesion-specific measured uptake or TAC."
    )
)


# =============================================================================
# SIMULATION TIME
# =============================================================================

simulation_days = max(
    FOLLOW_UP_DAYS,
    (
        (
            n_cycles -
            1
        ) *
        interval_days
    )
    +
    FOLLOW_UP_DAYS
)


time_days = np.arange(
    0.0,
    simulation_days + DT_DAYS,
    DT_DAYS
)


# =============================================================================
# DOSE
# =============================================================================

(
    physical_dose_rate_gy_day,
    treatment_times,
    uptake_scale,
    burden_scale,
    dose_scale
) = calculate_physical_dose_rate(
    time_days=time_days,
    activity_gbq=activity_gbq,
    n_cycles=n_cycles,
    interval_days=interval_days,
    initial_burden_ml=initial_burden_ml,
    tumour_uptake_percent=tumour_uptake_percent
)


critical_dose_rate_gy_h = calculate_critical_dose_rate(
    alpha=alpha,
    trep_days=trep_days
)


(
    dose_rate_ratio,
    dose_rate_effectiveness,
    effective_dose_rate_gy_day
) = calculate_effective_dose_rate(
    physical_dose_rate_gy_day,
    critical_dose_rate_gy_h,
    gamma
)


# =============================================================================
# TUMOUR DYNAMICS
# =============================================================================

(
    total_burden,
    sensitive_burden,
    resistant_burden,
    resistant_composition_fraction,
    residual_resistant_burden_percent,
    initial_resistant_burden
) = calculate_tumour_dynamics(
    time_days=time_days,
    physical_dose_rate_gy_day=physical_dose_rate_gy_day,
    alpha=alpha,
    trep_days=trep_days,
    sensitive_fraction=sensitive_fraction,
    resistant_tk_multiplier=resistant_tk_multiplier,
    resistant_radio_factor=resistant_radio_factor,
    repopulation_kickoff=repopulation_kickoff,
    initial_burden_ml=initial_burden_ml
)


# =============================================================================
# TCP
# =============================================================================

tcp = calculate_tcp(
    total_burden_ml=total_burden,
    initial_burden_ml=initial_burden_ml
)


# =============================================================================
# SUMMARY
# =============================================================================

summary = calculate_summary(
    time_days=time_days,
    physical_dose_rate_gy_day=physical_dose_rate_gy_day,
    effective_dose_rate_gy_day=effective_dose_rate_gy_day,
    critical_dose_rate_gy_h=critical_dose_rate_gy_h,
    total_burden=total_burden,
    sensitive_burden=sensitive_burden,
    resistant_burden=resistant_burden,
    resistant_composition_fraction=resistant_composition_fraction,
    residual_resistant_burden_percent=residual_resistant_burden_percent,
    tcp=tcp,
    treatment_times=treatment_times
)


# =============================================================================
# HEADER
# =============================================================================

st.title(
    "Lu-177 PSMA Optimisation Model"
)

st.markdown(
    """
Interactive exploration of Lu-177 PSMA treatment schedules using
physical dose-rate, tumour-response, resistant-disease and TCP models.
"""
)


# =============================================================================
# KEY RESULTS
# =============================================================================

st.subheader(
    "Key results"
)


col1, col2, col3, col4 = st.columns(
    4
)


# =============================================================================
# KEY RESULT 1 — MINIMUM TUMOUR BURDEN
# =============================================================================

minimum_burden = summary[
    "Minimum tumour burden (mL)"
]


burden_reduction_percent = (
    100.0 *
    (
        1.0 -
        minimum_burden /
        initial_burden_ml
    )
)


with col1:

    st.markdown(
        '<div style="font-size:0.78rem; color:#C8C8C8; '
        'margin-bottom:-0.25rem;">Minimum tumour burden</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            font-size: 1.55rem;
            font-weight: 600;
            line-height: 1.15;
            color: #F2F2F2;
            margin-bottom: 0.15rem;
        ">
            {minimum_burden:.1f} mL
        </div>
        """,
        unsafe_allow_html=True
    )

    if burden_reduction_percent > 0.05:

        key_result_status(
            f"↓ {burden_reduction_percent:.1f}% from "
            f"{initial_burden_ml:.0f} mL baseline",
            "positive"
        )

    elif burden_reduction_percent < -0.05:

        key_result_status(
            f"↑ {abs(burden_reduction_percent):.1f}% from "
            f"{initial_burden_ml:.0f} mL baseline",
            "negative"
        )

    else:

        key_result_status(
            "≈ No meaningful change from baseline",
            "neutral"
        )

    key_result_caption(
        f"Minimum reached at day "
        f"{summary['Day of minimum tumour burden']:.1f}"
    )


# =============================================================================
# KEY RESULT 2 — MAXIMUM TCP
# =============================================================================

maximum_tcp_percent = summary[
    "Maximum TCP (%)"
]


tcp_change_percentage_points = (
    maximum_tcp_percent -
    INITIAL_TCP * 100.0
)


with col2:

    st.markdown(
        '<div style="font-size:0.78rem; color:#C8C8C8; '
        'margin-bottom:-0.25rem;">Maximum TCP</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            font-size: 1.55rem;
            font-weight: 600;
            line-height: 1.15;
            color: #F2F2F2;
            margin-bottom: 0.15rem;
        ">
            {maximum_tcp_percent:.1f}%
        </div>
        """,
        unsafe_allow_html=True
    )

    if tcp_change_percentage_points > 0.05:

        key_result_status(
            f"↑ {tcp_change_percentage_points:.1f} "
            f"percentage points from 10% baseline",
            "positive"
        )

    elif tcp_change_percentage_points < -0.05:

        key_result_status(
            f"↓ {abs(tcp_change_percentage_points):.1f} "
            f"percentage points from 10% baseline",
            "negative"
        )

    else:

        key_result_status(
            "≈ No meaningful change from baseline",
            "neutral"
        )

    key_result_caption(
        f"Maximum reached at day "
        f"{summary['Day of maximum TCP']:.1f}"
    )


# =============================================================================
# KEY RESULT 3 — RESIDUAL RESISTANT BURDEN
# =============================================================================

final_residual_resistant_percent = summary[
    "Final residual resistant burden (%)"
]


initial_resistant_fraction_percent = (
    (
        1.0 -
        sensitive_fraction
    ) *
    100.0
)


resistant_reduction_percent = (
    100.0 -
    final_residual_resistant_percent
)


with col3:

    st.markdown(
        '<div style="font-size:0.78rem; color:#C8C8C8; '
        'margin-bottom:-0.25rem;">Residual resistant burden</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            font-size: 1.55rem;
            font-weight: 600;
            line-height: 1.15;
            color: #F2F2F2;
            margin-bottom: 0.15rem;
        ">
            {final_residual_resistant_percent:.1f}%
        </div>
        """,
        unsafe_allow_html=True
    )

    if resistant_reduction_percent > 0.05:

        key_result_status(
            f"↓ {resistant_reduction_percent:.1f}% "
            f"of initial resistant burden",
            "positive"
        )

    elif resistant_reduction_percent < -0.05:

        key_result_status(
            f"↑ {abs(resistant_reduction_percent):.1f}% "
            f"of initial resistant burden",
            "negative"
        )

    else:

        key_result_status(
            "≈ No meaningful change in resistant burden",
            "neutral"
        )

    key_result_caption(
        f"Initial resistant burden: "
        f"{initial_resistant_burden:.1f} mL "
        f"({initial_resistant_fraction_percent:.1f}% of total)"
    )


# =============================================================================
# KEY RESULT 4 — CUMULATIVE PHYSICAL DOSE
# =============================================================================

cumulative_physical_dose = summary[
    "Cumulative physical dose (Gy)"
]


with col4:

    st.markdown(
        '<div style="font-size:0.78rem; color:#C8C8C8; '
        'margin-bottom:-0.25rem;">Cumulative physical dose</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            font-size: 1.55rem;
            font-weight: 600;
            line-height: 1.15;
            color: #F2F2F2;
            margin-bottom: 0.15rem;
        ">
            {cumulative_physical_dose:.2f} Gy
        </div>
        """,
        unsafe_allow_html=True
    )

    key_result_status(
        f"{activity_gbq:.1f} GBq × {n_cycles} cycles",
        "neutral"
    )

    key_result_caption(
        "Integrated physical dose over the simulation"
    )


# =============================================================================
# SECONDARY METRICS
# =============================================================================

st.subheader(
    "Secondary metrics"
)


col1, col2, col3, col4 = st.columns(
    4
)


with col1:

    st.metric(
        "Critical dose rate",
        (
            f"{summary['Critical dose rate (Gy/h)']:.4f} Gy/h"
        )
    )

    st.caption(
        "Rcrit = ln(2) / (α × Trep)"
    )


with col2:

    st.metric(
        "Peak physical rate",
        (
            f"{summary['Peak physical dose rate (Gy/h)']:.3f} Gy/h"
        )
    )

    st.caption(
        f"{summary['Peak physical dose rate (Gy/h)'] / critical_dose_rate_gy_h:.2f}× "
        "critical rate"
    )


with col3:

    st.metric(
        "Time above critical rate",
        (
            f"{summary['Time above critical dose rate (days)']:.1f} d"
        )
    )

    st.caption(
        f"Longest continuous: "
        f"{summary['Longest continuous time above critical (days)']:.1f} d"
    )


with col4:

    st.metric(
        "Effective cumulative dose",
        (
            f"{summary['Effective cumulative dose (Gy)']:.2f} Gy"
        )
    )

    st.caption(
        f"Gamma = {gamma:.2f}"
    )


# =============================================================================
# DOSE SCALING
# =============================================================================

st.caption(
    f"Dose scaling: {dose_scale:.2f}× "
    f"(uptake scaling {uptake_scale:.2f}×; "
    f"burden scaling {burden_scale:.2f}× "
    f"relative to {REFERENCE_METASTATIC_BURDEN_ML:.0f} mL "
    f"and {DEFAULT_TUMOUR_UPTAKE_PERCENT:.1f}% uptake)."
)


# =============================================================================
# MODEL TRAJECTORIES
# =============================================================================

st.subheader(
    "Model trajectories"
)


# =============================================================================
# PLOT 1 — DOSE RATE
# =============================================================================

fig1, ax1 = plt.subplots(
    figsize=(9, 5),
    facecolor=PLOT_BACKGROUND
)


physical_rate_gy_h = (
    physical_dose_rate_gy_day /
    24.0
)


effective_rate_gy_h = (
    effective_dose_rate_gy_day /
    24.0
)


ax1.plot(
    time_days,
    physical_rate_gy_h,
    color=COLOR_BLUE,
    linewidth=2.4,
    label="Physical dose rate",
    zorder=3
)


ax1.plot(
    time_days,
    effective_rate_gy_h,
    color=COLOR_ORANGE,
    linewidth=2.0,
    linestyle="--",
    label="Effective dose rate",
    zorder=3
)


ax1.axhline(
    critical_dose_rate_gy_h,
    color=COLOR_RED,
    linestyle=":",
    linewidth=1.7,
    label="Critical dose rate",
    zorder=2
)


add_treatment_markers(
    ax1,
    treatment_times
)


ax1.set_xlabel(
    "Time (days)"
)

ax1.set_ylabel(
    "Dose rate (Gy/h)"
)

ax1.set_title(
    "Physical and effective dose rate"
)


style_axis(ax1)


legend1 = ax1.legend(
    frameon=True,
    loc="best"
)

style_legend(legend1)

fig1.tight_layout()


# =============================================================================
# PLOT 2 — TUMOUR BURDEN
# =============================================================================

fig2, ax2 = plt.subplots(
    figsize=(9, 5),
    facecolor=PLOT_BACKGROUND
)


ax2.plot(
    time_days,
    total_burden,
    color=COLOR_BLUE,
    linewidth=2.5,
    label="Total burden",
    zorder=3
)


ax2.plot(
    time_days,
    sensitive_burden,
    color=COLOR_GREEN,
    linewidth=1.9,
    linestyle="--",
    label="Sensitive",
    zorder=3
)


ax2.plot(
    time_days,
    resistant_burden,
    color=COLOR_ORANGE,
    linewidth=1.9,
    linestyle=":",
    label="Resistant",
    zorder=3
)


add_treatment_markers(
    ax2,
    treatment_times
)


ax2.set_xlabel(
    "Time (days)"
)

ax2.set_ylabel(
    "Tumour burden (mL)"
)

ax2.set_title(
    "Tumour burden"
)


style_axis(ax2)


legend2 = ax2.legend(
    frameon=True,
    loc="best"
)

style_legend(legend2)

fig2.tight_layout()


# =============================================================================
# PLOT 3 — SENSITIVE / RESISTANT POPULATIONS
# =============================================================================

fig3, ax3 = plt.subplots(
    figsize=(9, 5),
    facecolor=PLOT_BACKGROUND
)


ax3.plot(
    time_days,
    sensitive_burden,
    color=COLOR_GREEN,
    linewidth=2.2,
    label="Sensitive tumour",
    zorder=3
)


ax3.plot(
    time_days,
    resistant_burden,
    color=COLOR_ORANGE,
    linewidth=2.2,
    linestyle="--",
    label="Resistant tumour",
    zorder=3
)


ax3b = ax3.twinx()


ax3b.plot(
    time_days,
    residual_resistant_burden_percent,
    color=COLOR_PURPLE,
    linewidth=2.0,
    linestyle=":",
    label="Residual resistant burden",
    zorder=3
)


add_treatment_markers(
    ax3,
    treatment_times,
    alpha=0.25
)


ax3.set_xlabel(
    "Time (days)"
)

ax3.set_ylabel(
    "Tumour burden (mL)"
)

ax3b.set_ylabel(
    "Residual resistant burden (% of initial)"
)

ax3.set_title(
    "Sensitive and resistant tumour populations"
)


style_axis(ax3)


ax3b.set_facecolor(
    "none"
)

ax3b.spines[
    "top"
].set_visible(False)

ax3b.spines[
    "right"
].set_color(
    PLOT_TEXT
)

ax3b.spines[
    "right"
].set_linewidth(
    0.8
)

ax3b.tick_params(
    colors=PLOT_TEXT,
    direction="out",
    length=4.5,
    width=0.8
)

ax3b.yaxis.set_major_locator(
    MaxNLocator(nbins=8)
)

ax3b.minorticks_on()

ax3b.tick_params(
    which="minor",
    length=2.5,
    width=0.5,
    color=PLOT_TEXT
)


lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()


legend3 = ax3.legend(
    lines1 + lines2,
    labels1 + labels2,
    frameon=True,
    loc="best"
)

style_legend(legend3)

fig3.tight_layout()


# =============================================================================
# PLOT 4 — TCP
# =============================================================================

fig4, ax4 = plt.subplots(
    figsize=(9, 5),
    facecolor=PLOT_BACKGROUND
)


ax4.plot(
    time_days,
    tcp * 100.0,
    color=COLOR_BLUE,
    linewidth=2.5,
    label="TCP",
    zorder=3
)


ax4.axhline(
    50.0,
    color=COLOR_ORANGE,
    linestyle="--",
    linewidth=1.4,
    label="50% TCP",
    zorder=2
)


ax4.axhline(
    90.0,
    color=COLOR_GREEN,
    linestyle=":",
    linewidth=1.3,
    label="90% TCP",
    zorder=2
)


add_treatment_markers(
    ax4,
    treatment_times
)


ax4.set_xlabel(
    "Time (days)"
)

ax4.set_ylabel(
    "TCP (%)"
)

ax4.set_ylim(
    0,
    105
)

ax4.set_title(
    "Tumour control probability"
)


style_axis(ax4)


legend4 = ax4.legend(
    frameon=True,
    loc="best"
)

style_legend(legend4)

fig4.tight_layout()


# =============================================================================
# DISPLAY 2 × 2
# =============================================================================

plot_col1, plot_col2 = st.columns(2)


with plot_col1:

    st.pyplot(
        fig1,
        clear_figure=False
    )


with plot_col2:

    st.pyplot(
        fig2,
        clear_figure=False
    )


plot_col3, plot_col4 = st.columns(2)


with plot_col3:

    st.pyplot(
        fig3,
        clear_figure=False
    )


with plot_col4:

    st.pyplot(
        fig4,
        clear_figure=False
    )


# =============================================================================
# DETAILED MODEL SUMMARY
# =============================================================================

with st.expander(
    "Detailed model summary"
):

    summary_col1, summary_col2 = st.columns(2)


    with summary_col1:

        st.markdown(
            "### Treatment and dosimetry"
        )

        st.write(
            f"Activity per cycle: "
            f"{activity_gbq:.2f} GBq"
        )

        st.write(
            f"Number of cycles: "
            f"{n_cycles}"
        )

        st.write(
            f"Cycle interval: "
            f"{interval_days} days"
        )

        st.write(
            f"Total tumour uptake: "
            f"{tumour_uptake_percent:.1f}%"
        )

        st.write(
            f"Initial metastatic burden: "
            f"{initial_burden_ml:.1f} mL"
        )

        st.write(
            f"Initial resistant burden: "
            f"{initial_resistant_burden:.2f} mL"
        )

        st.write(
            f"Uptake scaling: "
            f"{uptake_scale:.3f}×"
        )

        st.write(
            f"Burden scaling: "
            f"{burden_scale:.3f}×"
        )

        st.write(
            f"Overall dose scaling: "
            f"{dose_scale:.3f}×"
        )

        st.write(
            f"Lu-177 physical half-life: "
            f"{LU177_HALF_LIFE_DAYS:.3f} days"
        )

        st.write(
            f"Critical dose rate: "
            f"{critical_dose_rate_gy_h:.5f} Gy/h"
        )

        st.write(
            f"Peak physical dose rate: "
            f"{summary['Peak physical dose rate (Gy/h)']:.4f} Gy/h"
        )

        st.write(
            f"Peak effective dose rate: "
            f"{summary['Peak effective dose rate (Gy/h)']:.4f} Gy/h"
        )

        st.write(
            f"Cumulative physical dose: "
            f"{summary['Cumulative physical dose (Gy)']:.3f} Gy"
        )

        st.write(
            f"Effective cumulative dose: "
            f"{summary['Effective cumulative dose (Gy)']:.3f} Gy"
        )


    with summary_col2:

        st.markdown(
            "### Tumour response"
        )

        st.write(
            f"Alpha: "
            f"{alpha:.3f} Gy⁻¹"
        )

        st.write(
            f"Trep: "
            f"{trep_days:.1f} days"
        )

        st.write(
            f"Initial sensitive fraction: "
            f"{sensitive_fraction * 100:.1f}%"
        )

        st.write(
            f"Initial resistant fraction: "
            f"{(1.0 - sensitive_fraction) * 100:.1f}%"
        )

        st.write(
            f"Tk / Trep: "
            f"{resistant_tk_multiplier:.2f}"
        )

        st.write(
            f"Resistant radio factor: "
            f"{resistant_radio_factor:.2f}"
        )

        st.write(
            f"Repopulation kickoff: "
            f"{repopulation_kickoff:.1f} days"
        )

        st.write(
            f"Minimum tumour burden: "
            f"{summary['Minimum tumour burden (mL)']:.2f} mL"
        )

        st.write(
            f"Day of minimum burden: "
            f"{summary['Day of minimum tumour burden']:.2f}"
        )

        st.write(
            f"Final tumour burden: "
            f"{summary['Final tumour burden (mL)']:.2f} mL"
        )

        st.write(
            f"Maximum TCP: "
            f"{summary['Maximum TCP (%)']:.2f}%"
        )

        st.write(
            f"Day of maximum TCP: "
            f"{summary['Day of maximum TCP']:.2f}"
        )

        st.write(
            f"Final residual resistant burden: "
            f"{summary['Final residual resistant burden (%)']:.2f}% "
            f"of initial resistant burden"
        )

        st.write(
            f"Final resistant composition: "
            f"{summary['Final resistant composition (%)']:.2f}% "
            f"of remaining tumour"
        )

        st.write(
            f"Final sensitive burden: "
            f"{summary['Final sensitive burden (mL)']:.3f} mL"
        )

        st.write(
            f"Final resistant burden: "
            f"{summary['Final resistant burden (mL)']:.3f} mL"
        )

        st.write(
            f"Time above critical rate: "
            f"{summary['Time above critical dose rate (days)']:.2f} days"
        )

        st.write(
            f"Longest continuous period above critical: "
            f"{summary['Longest continuous time above critical (days)']:.2f} days"
        )


# =============================================================================
# ASSUMPTIONS AND LIMITATIONS
# =============================================================================

with st.expander(
    "Model assumptions and limitations"
):

    st.markdown(
        """
### Tumour burden

The model treats the initial tumour burden as a total metastatic
tumour burden in mL rather than as one solid tumour.

The burden represents an aggregate tumour volume across metastatic
sites.

### Tumour uptake

The total tumour uptake parameter represents an exploratory estimate
of the fraction of administered Lu-177 activity associated with the
total metastatic tumour burden.

The current implementation uses:

**Dose scaling = uptake scaling × inverse burden scaling**

with 234 mL and 1.0% uptake used as the reference condition.

This is a phenomenological model and should not be interpreted as
patient-specific dosimetry.

Actual Lu-177 PSMA dosimetry would ideally incorporate lesion-level
activity concentrations, time-activity curves, residence times,
physical decay, biological clearance, absorbed fractions, cross-dose
and spatial heterogeneity.

### Dose rate

Lu-177 activity is assumed to decay according to its physical
half-life of 6.647 days.

### Critical dose rate

The critical dose rate is calculated as:

**Rcrit = ln(2) / (alpha × Trep)**

where Trep is converted to hours.

### Radiation response

Radiation killing uses a linear-quadratic formulation.

The sensitive and resistant populations have different effective
radiosensitivities.

### Resistant disease

Two different resistant-disease quantities are reported.

**Residual resistant burden**

R(t) / R0 × 100

This represents the percentage of the initial resistant tumour burden
that remains.

This is the preferred metric for assessing whether the resistant
component itself has been reduced by treatment.

**Resistant composition**

R(t) / [S(t) + R(t)] × 100

This represents the percentage of the remaining tumour that is
resistant.

A high resistant composition does not necessarily mean that the
resistant tumour has grown. It can also occur because the sensitive
population has been preferentially eliminated.

### Repopulation

Repopulation begins after the specified repopulation kickoff time.

### TCP

TCP is implemented as a normalised exploratory model based on
relative total tumour burden.

The initial TCP is set to 10%.

This should not be interpreted as a validated clinical TCP model or
as a patient-specific probability of cure.

### Treatment schedule

Each treatment cycle is assumed to deliver the same administered
activity.

The model is intended primarily to explore interactions between
activity, treatment interval, tumour burden, uptake, radiosensitivity,
repopulation and resistant-cell dynamics.
"""
    )


# =============================================================================
# EXPORT RESULTS
# =============================================================================

st.subheader(
    "Export results"
)


# =============================================================================
# CSV
# =============================================================================

results_df = pd.DataFrame(
    {

        "Time_days":
            time_days,

        "Physical_dose_rate_Gy_day":
            physical_dose_rate_gy_day,

        "Physical_dose_rate_Gy_h":
            physical_rate_gy_h,

        "Effective_dose_rate_Gy_day":
            effective_dose_rate_gy_day,

        "Effective_dose_rate_Gy_h":
            effective_rate_gy_h,

        "Critical_dose_rate_Gy_h":
            np.full_like(
                time_days,
                critical_dose_rate_gy_h
            ),

        "Dose_rate_ratio":
            dose_rate_ratio,

        "Dose_rate_effectiveness":
            dose_rate_effectiveness,

        "Total_burden_mL":
            total_burden,

        "Sensitive_burden_mL":
            sensitive_burden,

        "Resistant_burden_mL":
            resistant_burden,

        "Initial_resistant_burden_mL":
            np.full_like(
                time_days,
                initial_resistant_burden
            ),

        "Residual_resistant_burden_percent":
            residual_resistant_burden_percent,

        "Resistant_composition_percent":
            resistant_composition_fraction * 100.0,

        "TCP_percent":
            tcp * 100.0,

        "Initial_burden_mL":
            np.full_like(
                time_days,
                initial_burden_ml
            ),

        "Tumour_uptake_percent":
            np.full_like(
                time_days,
                tumour_uptake_percent
            ),

        "Uptake_scaling":
            np.full_like(
                time_days,
                uptake_scale
            ),

        "Burden_scaling":
            np.full_like(
                time_days,
                burden_scale
            ),

        "Dose_scaling_factor":
            np.full_like(
                time_days,
                dose_scale
            ),
    }
)


csv_data = results_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download model results (CSV)",
    data=csv_data,
    file_name="Lu177_PSMA_model_results.csv",
    mime="text/csv"
)


# =============================================================================
# 600 DPI EXPORT FIGURE
# =============================================================================

export_fig = plt.figure(
    figsize=(16, 12),
    facecolor=PLOT_BACKGROUND
)


# =============================================================================
# EXPORT PANEL 1 — DOSE RATE
# =============================================================================

export_ax1 = export_fig.add_subplot(221)


export_ax1.plot(
    time_days,
    physical_rate_gy_h,
    color=COLOR_BLUE,
    linewidth=2.2,
    label="Physical dose rate"
)


export_ax1.plot(
    time_days,
    effective_rate_gy_h,
    color=COLOR_ORANGE,
    linewidth=1.9,
    linestyle="--",
    label="Effective dose rate"
)


export_ax1.axhline(
    critical_dose_rate_gy_h,
    color=COLOR_RED,
    linestyle=":",
    linewidth=1.5,
    label="Critical dose rate"
)


add_treatment_markers(
    export_ax1,
    treatment_times
)


export_ax1.set_title(
    "Physical vs effective dose rate"
)

export_ax1.set_xlabel(
    "Time (days)"
)

export_ax1.set_ylabel(
    "Dose rate (Gy/h)"
)


style_axis(export_ax1)


export_legend1 = export_ax1.legend(
    frameon=True
)

style_legend(export_legend1)


# =============================================================================
# EXPORT PANEL 2 — TUMOUR BURDEN
# =============================================================================

export_ax2 = export_fig.add_subplot(222)


export_ax2.plot(
    time_days,
    total_burden,
    color=COLOR_BLUE,
    linewidth=2.3,
    label="Total burden"
)


export_ax2.plot(
    time_days,
    sensitive_burden,
    color=COLOR_GREEN,
    linewidth=1.8,
    linestyle="--",
    label="Sensitive"
)


export_ax2.plot(
    time_days,
    resistant_burden,
    color=COLOR_ORANGE,
    linewidth=1.8,
    linestyle=":",
    label="Resistant"
)


add_treatment_markers(
    export_ax2,
    treatment_times
)


export_ax2.set_title(
    "Tumour burden"
)

export_ax2.set_xlabel(
    "Time (days)"
)

export_ax2.set_ylabel(
    "Tumour burden (mL)"
)


style_axis(export_ax2)


export_legend2 = export_ax2.legend(
    frameon=True
)

style_legend(export_legend2)


# =============================================================================
# EXPORT PANEL 3 — RESISTANT DISEASE
# =============================================================================

export_ax3 = export_fig.add_subplot(223)


export_ax3.plot(
    time_days,
    sensitive_burden,
    color=COLOR_GREEN,
    linewidth=2.0,
    label="Sensitive tumour"
)


export_ax3.plot(
    time_days,
    resistant_burden,
    color=COLOR_ORANGE,
    linewidth=2.0,
    linestyle="--",
    label="Resistant tumour"
)


export_ax3b = export_ax3.twinx()


export_ax3b.plot(
    time_days,
    residual_resistant_burden_percent,
    color=COLOR_PURPLE,
    linewidth=1.9,
    linestyle=":",
    label="Residual resistant burden"
)


add_treatment_markers(
    export_ax3,
    treatment_times,
    alpha=0.22
)


export_ax3.set_title(
    "Sensitive and resistant tumour populations"
)

export_ax3.set_xlabel(
    "Time (days)"
)

export_ax3.set_ylabel(
    "Tumour burden (mL)"
)

export_ax3b.set_ylabel(
    "Residual resistant burden (% of initial)"
)


style_axis(export_ax3)


export_ax3b.set_facecolor(
    "none"
)

export_ax3b.spines[
    "top"
].set_visible(False)

export_ax3b.spines[
    "right"
].set_color(
    PLOT_TEXT
)

export_ax3b.tick_params(
    colors=PLOT_TEXT,
    direction="out",
    length=4,
    width=0.8
)

export_ax3b.yaxis.set_major_locator(
    MaxNLocator(nbins=8)
)


lines_a, labels_a = (
    export_ax3.get_legend_handles_labels()
)

lines_b, labels_b = (
    export_ax3b.get_legend_handles_labels()
)


export_legend3 = export_ax3.legend(
    lines_a + lines_b,
    labels_a + labels_b,
    frameon=True
)

style_legend(export_legend3)


# =============================================================================
# EXPORT PANEL 4 — TCP
# =============================================================================

export_ax4 = export_fig.add_subplot(224)


export_ax4.plot(
    time_days,
    tcp * 100.0,
    color=COLOR_BLUE,
    linewidth=2.3,
    label="TCP"
)


export_ax4.axhline(
    50.0,
    color=COLOR_ORANGE,
    linestyle="--",
    linewidth=1.2,
    label="50% TCP"
)


export_ax4.axhline(
    90.0,
    color=COLOR_GREEN,
    linestyle=":",
    linewidth=1.2,
    label="90% TCP"
)


add_treatment_markers(
    export_ax4,
    treatment_times
)


export_ax4.set_title(
    "Tumour control probability"
)

export_ax4.set_xlabel(
    "Time (days)"
)

export_ax4.set_ylabel(
    "TCP (%)"
)

export_ax4.set_ylim(
    0,
    105
)


style_axis(export_ax4)


export_legend4 = export_ax4.legend(
    frameon=True
)

style_legend(export_legend4)


# =============================================================================
# FINALISE EXPORT
# =============================================================================

export_fig.tight_layout(
    pad=2.0
)


png_buffer = io.BytesIO()


export_fig.savefig(
    png_buffer,
    format="png",
    dpi=600,
    bbox_inches="tight",
    facecolor=PLOT_BACKGROUND,
    edgecolor=PLOT_BACKGROUND
)


png_buffer.seek(0)


st.download_button(
    label="Download trajectories (600 dpi PNG)",
    data=png_buffer,
    file_name="Lu177_PSMA_model_trajectories_600dpi.png",
    mime="image/png"
)


plt.close(export_fig)


# =============================================================================
# CLEAN UP DISPLAY FIGURES
# =============================================================================

plt.close(fig1)
plt.close(fig2)
plt.close(fig3)
plt.close(fig4)
