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
LU177_LAMBDA_PER_DAY = (
    np.log(2.0) /
    LU177_HALF_LIFE_DAYS
)

DOSE_PER_GBQ_GY = 0.50

REFERENCE_METASTATIC_BURDEN_ML = 234.0

DEFAULT_TUMOUR_UPTAKE_PERCENT = 1.0

TUMOUR_UPTAKE_MIN_PERCENT = 0.1
TUMOUR_UPTAKE_MAX_PERCENT = 10.0

DT_DAYS = 0.05

FOLLOW_UP_DAYS = 60.0


# =============================================================================
# DEFAULT MODEL PARAMETERS
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

INITIAL_CLONOGENIC_BURDEN = (
    -np.log(INITIAL_TCP)
)


# =============================================================================
# NUMERICAL INTEGRATION
# =============================================================================

def integrate_trapezoid(y, x):

    if hasattr(np, "trapezoid"):

        return np.trapezoid(
            y,
            x
        )

    return np.trapz(
        y,
        x
    )


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

    treatment_times = (
        calculate_treatment_times(
            n_cycles,
            interval_days
        )
    )

    dose_rate_gy_day = np.zeros_like(
        time_days,
        dtype=float
    )

    for t_admin in treatment_times:

        elapsed = (
            time_days -
            t_admin
        )

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

    trep_hours = (
        trep_days *
        24.0
    )

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

    for i in range(
        1,
        len(time_days)
    ):

        dt = (
            time_days[i] -
            time_days[i - 1]
        )

        s = sensitive[i - 1]

        r = resistant[i - 1]

        # ---------------------------------------------------------------------
        # REPOPULATION
        # ---------------------------------------------------------------------

        if (
            time_days[i] >=
            repopulation_kickoff
        ):

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
    # Fraction of the remaining tumour that is resistant.
    # -------------------------------------------------------------------------

    resistant_composition_fraction = np.divide(
        resistant,
        total,
        out=np.zeros_like(
            resistant
        ),
        where=total > 0
    )

    # -------------------------------------------------------------------------
    # RESIDUAL RESISTANT BURDEN
    #
    # Percentage of the INITIAL resistant burden that remains.
    #
    # This is the main resistant-disease response metric.
    # -------------------------------------------------------------------------

    residual_resistant_burden_percent = np.divide(
        resistant,
        initial_resistant_burden,
        out=np.zeros_like(
            resistant
        ),
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

    cumulative_physical_dose = (
        integrate_trapezoid(
            physical_dose_rate_gy_day,
            time_days
        )
    )

    cumulative_effective_dose = (
        integrate_trapezoid(
            effective_dose_rate_gy_day,
            time_days
        )
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

    for i in range(
        1,
        len(time_days)
    ):

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
        np.argmin(
            total_burden
        )
    ]

    final_burden = (
        total_burden[-1]
    )

    maximum_tcp = np.max(
        tcp
    )

    maximum_tcp_percent = (
        maximum_tcp *
        100.0
    )

    maximum_tcp_day = time_days[
        np.argmax(
            tcp
        )
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

    maximum_resistant_composition = (
        np.max(
            resistant_composition_fraction
        ) *
        100.0
    )

    final_sensitive = (
        sensitive_burden[-1]
    )

    final_resistant = (
        resistant_burden[-1]
    )

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

        "Maximum resistant composition (%)":
            maximum_resistant_composition,

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
# PROFESSIONAL DARK PLOT STYLE
# =============================================================================

# A robust font available on most Streamlit/Linux environments.
PLOT_FONT = "DejaVu Sans"

# Dark plot colours.
PLOT_BACKGROUND = "#000000"
PLOT_FOREGROUND = "#F2F2F2"
PLOT_SECONDARY = "#CFCFCF"
PLOT_GRID = "#555555"
PLOT_TREATMENT = "#777777"


plt.rcParams.update(
    {
        "font.family": PLOT_FONT,

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

        "text.color": PLOT_FOREGROUND,

        "axes.labelcolor": PLOT_FOREGROUND,

        "axes.edgecolor": PLOT_FOREGROUND,

        "xtick.color": PLOT_FOREGROUND,

        "ytick.color": PLOT_FOREGROUND,

        "axes.facecolor": PLOT_BACKGROUND,

        "figure.facecolor": PLOT_BACKGROUND,

        "savefig.facecolor": PLOT_BACKGROUND,
    }
)


# =============================================================================
# AXIS STYLE
# =============================================================================

def style_axis(ax):

    ax.set_facecolor(
        PLOT_BACKGROUND
    )

    ax.tick_params(
        colors=PLOT_FOREGROUND,
        direction="out",
        length=4.5,
        width=0.8
    )

    for spine in ax.spines.values():

        spine.set_color(
            PLOT_FOREGROUND
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
        alpha=0.45
    )

    ax.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.35,
        color=PLOT_GRID,
        alpha=0.22
    )

    ax.xaxis.set_major_locator(
        MaxNLocator(
            nbins=8
        )
    )

    ax.yaxis.set_major_locator(
        MaxNLocator(
            nbins=7
        )
    )

    ax.minorticks_on()

    ax.tick_params(
        which="minor",
        length=2.5,
        width=0.5,
        color=PLOT_SECONDARY
    )


# =============================================================================
# TREATMENT MARKERS
# =============================================================================

def add_treatment_markers(
    ax,
    treatment_times,
    alpha=0.35
):

    for t_admin in treatment_times:

        ax.axvline(
            t_admin,
            linestyle=":",
            linewidth=0.8,
            color=PLOT_TREATMENT,
            alpha=alpha,
            zorder=1
        )


# =============================================================================
# STREAMLIT HEADER
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


# IMPORTANT:
# Alpha minimum changed from 0.01 to 0.001.

alpha = st.sidebar.slider(
    "Alpha",
    min_value=0.001,
    max_value=0.50,
    value=DEFAULT_ALPHA,
    step=0.001,
    format="%.3f",
    help=(
        "Effective alpha parameter used by the exploratory "
        "linear-quadratic radiation response model."
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
    simulation_days +
    DT_DAYS,
    DT_DAYS
)


# =============================================================================
# PHYSICAL DOSE RATE
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


# =============================================================================
# CRITICAL DOSE RATE
# =============================================================================

critical_dose_rate_gy_h = (
    calculate_critical_dose_rate(
        alpha=alpha,
        trep_days=trep_days
    )
)


# =============================================================================
# EFFECTIVE DOSE RATE
# =============================================================================

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
# KEY RESULTS
# =============================================================================

st.subheader(
    "Key results"
)


col1, col2, col3, col4 = st.columns(
    4
)


with col1:

    st.metric(
        "Minimum tumour burden",
        (
            f"{summary['Minimum tumour burden (mL)']:.1f} mL"
        )
    )


with col2:

    st.metric(
        "Maximum TCP",
        (
            f"{summary['Maximum TCP (%)']:.1f}%"
        )
    )


with col3:

    st.metric(
        "Residual resistant burden",
        (
            f"{summary['Final residual resistant burden (%)']:.2f}%"
        ),
        help=(
            "Percentage of the initial resistant tumour burden "
            "remaining at the end of the simulation."
        )
    )


with col4:

    st.metric(
        "Cumulative physical dose",
        (
            f"{summary['Cumulative physical dose (Gy)']:.2f} Gy"
        )
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


with col2:

    st.metric(
        "Peak physical rate",
        (
            f"{summary['Peak physical dose rate (Gy/h)']:.3f} Gy/h"
        )
    )


with col3:

    st.metric(
        "Time above critical rate",
        (
            f"{summary['Time above critical dose rate (days)']:.1f} d"
        )
    )


with col4:

    st.metric(
        "Effective cumulative dose",
        (
            f"{summary['Effective cumulative dose (Gy)']:.2f} Gy"
        )
    )


# =============================================================================
# DOSE SCALING INFORMATION
# =============================================================================

st.caption(
    (
        f"Dose scaling: {dose_scale:.2f}× "
        f"(uptake scaling {uptake_scale:.2f}×; "
        f"burden scaling {burden_scale:.2f}× relative to "
        f"{REFERENCE_METASTATIC_BURDEN_ML:.0f} mL and "
        f"{DEFAULT_TUMOUR_UPTAKE_PERCENT:.1f}% uptake)."
    )
)


# =============================================================================
# MODEL TRAJECTORIES
# =============================================================================

st.subheader(
    "Model trajectories"
)


# =============================================================================
# FIGURE 1 — DOSE RATE
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
    color=PLOT_FOREGROUND,
    linewidth=2.2,
    label="Physical dose rate",
    zorder=3
)


ax1.plot(
    time_days,
    effective_rate_gy_h,
    color=PLOT_SECONDARY,
    linewidth=1.8,
    linestyle="--",
    label="Effective dose rate",
    zorder=3
)


ax1.axhline(
    critical_dose_rate_gy_h,
    color=PLOT_SECONDARY,
    linestyle=":",
    linewidth=1.5,
    label="Critical dose rate",
    zorder=2
)


add_treatment_markers(
    ax1,
    treatment_times
)


ax1.set_xlabel(
    "Time (days)",
    color=PLOT_FOREGROUND
)


ax1.set_ylabel(
    "Dose rate (Gy/h)",
    color=PLOT_FOREGROUND
)


ax1.set_title(
    "Physical and effective dose rate",
    color=PLOT_FOREGROUND,
    pad=12
)


style_axis(
    ax1
)


ax1.legend(
    frameon=False,
    labelcolor=PLOT_FOREGROUND,
    loc="best"
)


fig1.tight_layout()


# =============================================================================
# FIGURE 2 — TUMOUR BURDEN
# =============================================================================

fig2, ax2 = plt.subplots(
    figsize=(9, 5),
    facecolor=PLOT_BACKGROUND
)


ax2.plot(
    time_days,
    total_burden,
    color=PLOT_FOREGROUND,
    linewidth=2.4,
    label="Total burden",
    zorder=3
)


ax2.plot(
    time_days,
    sensitive_burden,
    color=PLOT_SECONDARY,
    linewidth=1.8,
    linestyle="--",
    label="Sensitive",
    zorder=3
)


ax2.plot(
    time_days,
    resistant_burden,
    color="#9E9E9E",
    linewidth=1.8,
    linestyle=":",
    label="Resistant",
    zorder=3
)


add_treatment_markers(
    ax2,
    treatment_times
)


ax2.set_xlabel(
    "Time (days)",
    color=PLOT_FOREGROUND
)


ax2.set_ylabel(
    "Tumour burden (mL)",
    color=PLOT_FOREGROUND
)


ax2.set_title(
    "Tumour burden",
    color=PLOT_FOREGROUND,
    pad=12
)


style_axis(
    ax2
)


ax2.legend(
    frameon=False,
    labelcolor=PLOT_FOREGROUND,
    loc="best"
)


fig2.tight_layout()


# =============================================================================
# FIGURE 3 — SENSITIVE / RESISTANT POPULATIONS
# =============================================================================

fig3, ax3 = plt.subplots(
    figsize=(9, 5),
    facecolor=PLOT_BACKGROUND
)


ax3.plot(
    time_days,
    sensitive_burden,
    color=PLOT_FOREGROUND,
    linewidth=2.0,
    label="Sensitive tumour",
    zorder=3
)


ax3.plot(
    time_days,
    resistant_burden,
    color=PLOT_SECONDARY,
    linewidth=2.0,
    linestyle="--",
    label="Resistant tumour",
    zorder=3
)


ax3b = ax3.twinx()


ax3b.plot(
    time_days,
    residual_resistant_burden_percent,
    color="#9E9E9E",
    linewidth=1.9,
    linestyle=":",
    label="Residual resistant burden",
    zorder=3
)


add_treatment_markers(
    ax3,
    treatment_times,
    alpha=0.22
)


ax3.set_xlabel(
    "Time (days)",
    color=PLOT_FOREGROUND
)


ax3.set_ylabel(
    "Tumour burden (mL)",
    color=PLOT_FOREGROUND
)


ax3b.set_ylabel(
    "Residual resistant burden (% of initial)",
    color=PLOT_FOREGROUND
)


ax3.set_title(
    "Sensitive and resistant tumour populations",
    color=PLOT_FOREGROUND,
    pad=12
)


style_axis(
    ax3
)


ax3b.set_facecolor(
    "none"
)


ax3b.spines[
    "top"
].set_visible(False)


ax3b.spines[
    "right"
].set_color(
    PLOT_FOREGROUND
)


ax3b.spines[
    "right"
].set_linewidth(
    0.8
)


ax3b.tick_params(
    colors=PLOT_FOREGROUND,
    direction="out",
    length=4.5,
    width=0.8
)


ax3b.yaxis.set_major_locator(
    MaxNLocator(
        nbins=7
    )
)


ax3b.minorticks_on()


ax3b.tick_params(
    which="minor",
    length=2.5,
    width=0.5,
    color=PLOT_SECONDARY
)


lines1, labels1 = (
    ax3.get_legend_handles_labels()
)


lines2, labels2 = (
    ax3b.get_legend_handles_labels()
)


legend3 = ax3.legend(
    lines1 + lines2,
    labels1 + labels2,
    frameon=False,
    loc="best"
)


for text in legend3.get_texts():

    text.set_color(
        PLOT_FOREGROUND
    )


fig3.tight_layout()


# =============================================================================
# FIGURE 4 — TCP
# =============================================================================

fig4, ax4 = plt.subplots(
    figsize=(9, 5),
    facecolor=PLOT_BACKGROUND
)


ax4.plot(
    time_days,
    tcp * 100.0,
    color=PLOT_FOREGROUND,
    linewidth=2.3,
    label="TCP",
    zorder=3
)


ax4.axhline(
    50.0,
    color=PLOT_SECONDARY,
    linestyle="--",
    linewidth=1.1,
    label="50% TCP",
    zorder=2
)


add_treatment_markers(
    ax4,
    treatment_times
)


ax4.set_xlabel(
    "Time (days)",
    color=PLOT_FOREGROUND
)


ax4.set_ylabel(
    "TCP (%)",
    color=PLOT_FOREGROUND
)


ax4.set_ylim(
    0,
    100
)


ax4.set_title(
    "Tumour control probability",
    color=PLOT_FOREGROUND,
    pad=12
)


style_axis(
    ax4
)


ax4.legend(
    frameon=False,
    labelcolor=PLOT_FOREGROUND,
    loc="best"
)


fig4.tight_layout()


# =============================================================================
# DISPLAY 2 × 2
# =============================================================================

plot_col1, plot_col2 = st.columns(
    2
)


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


plot_col3, plot_col4 = st.columns(
    2
)


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

    summary_col1, summary_col2 = st.columns(
        2
    )


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
            f"Reference metastatic burden: "
            f"{REFERENCE_METASTATIC_BURDEN_ML:.0f} mL"
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
            f"Minimum residual resistant burden: "
            f"{summary['Minimum residual resistant burden (%)']:.2f}% "
            f"of initial resistant burden"
        )

        st.write(
            f"Final resistant composition: "
            f"{summary['Final resistant composition (%)']:.2f}% "
            f"of remaining tumour"
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

        st.write(
            f"Cumulative physical dose: "
            f"{summary['Cumulative physical dose (Gy)']:.3f} Gy"
        )

        st.write(
            f"Effective cumulative dose: "
            f"{summary['Effective cumulative dose (Gy)']:.3f} Gy"
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

The model treats the initial tumour burden as a total metastatic
tumour burden in mL rather than as one solid tumour.

The burden therefore represents an aggregate tumour volume across
metastatic sites.

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

The model uses a reference dose-rate conversion and scales it according
to tumour uptake and aggregate tumour burden.

### Critical dose rate

The critical dose rate is calculated as:

**Rcrit = ln(2) / (alpha × Trep)**

where Trep is converted to hours.

### Radiation response

Radiation killing uses a linear-quadratic formulation.

The sensitive and resistant populations have different effective
radiosensitivities.

### Resistant population

The model contains a sensitive and resistant tumour compartment.

The resistant population has:

- reduced alpha
- reduced beta
- altered proliferation through the Tk/Trep multiplier

The resistant population can therefore become increasingly important
during treatment when the sensitive population is preferentially
eliminated.

### Resistant disease metrics

Two different resistant-disease measures are calculated.

**Residual resistant burden**

This is:

**R(t) / R0 × 100**

and represents the percentage of the original resistant tumour burden
that remains.

This is the primary resistant-disease response metric.

**Resistant composition**

This is:

**R(t) / [S(t) + R(t)] × 100**

and represents the proportion of the remaining tumour burden that is
resistant.

These two metrics answer different biological questions and should
not be interpreted interchangeably.

### Repopulation

Repopulation begins after the specified repopulation kickoff time.

The model does not currently include explicit cell-cycle effects,
immune-mediated killing, tumour microenvironment effects,
reoxygenation or spatially heterogeneous dose deposition.

### TCP

TCP is implemented as a normalised exploratory model based on
relative total tumour burden.

The initial TCP is set to 10%.

This should not be interpreted as a validated clinical TCP model or
as a patient-specific probability of cure.

### Treatment schedule

Each treatment cycle is assumed to deliver the same administered
activity.

The model is intended primarily to explore how activity, treatment
interval, tumour burden, uptake, radiosensitivity and resistant-cell
dynamics interact.
"""
    )


# =============================================================================
# EXPORT RESULTS
# =============================================================================

st.subheader(
    "Export results"
)


# =============================================================================
# CSV EXPORT
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

        "Residual_resistant_burden_percent":
            residual_resistant_burden_percent,

        "Resistant_composition_fraction":
            resistant_composition_fraction,

        "Initial_resistant_burden_mL":
            np.full_like(
                time_days,
                initial_resistant_burden
            ),

        "TCP":
            tcp,

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
).encode(
    "utf-8"
)


st.download_button(
    label="Download model results (CSV)",
    data=csv_data,
    file_name="Lu177_PSMA_model_results.csv",
    mime="text/csv"
)


# =============================================================================
# 600 DPI PNG EXPORT
# =============================================================================

export_fig = plt.figure(
    figsize=(16, 12),
    facecolor=PLOT_BACKGROUND
)


# =============================================================================
# EXPORT PANEL 1 — DOSE RATE
# =============================================================================

export_ax1 = export_fig.add_subplot(
    221
)


export_ax1.plot(
    time_days,
    physical_rate_gy_h,
    color=PLOT_FOREGROUND,
    linewidth=2.1,
    label="Physical dose rate"
)


export_ax1.plot(
    time_days,
    effective_rate_gy_h,
    color=PLOT_SECONDARY,
    linewidth=1.8,
    linestyle="--",
    label="Effective dose rate"
)


export_ax1.axhline(
    critical_dose_rate_gy_h,
    color=PLOT_SECONDARY,
    linestyle=":",
    linewidth=1.4,
    label="Critical dose rate"
)


add_treatment_markers(
    export_ax1,
    treatment_times
)


export_ax1.set_title(
    "Physical vs effective dose rate",
    color=PLOT_FOREGROUND
)


export_ax1.set_xlabel(
    "Time (days)"
)


export_ax1.set_ylabel(
    "Dose rate (Gy/h)"
)


style_axis(
    export_ax1
)


export_ax1.legend(
    frameon=False,
    labelcolor=PLOT_FOREGROUND
)


# =============================================================================
# EXPORT PANEL 2 — TUMOUR BURDEN
# =============================================================================

export_ax2 = export_fig.add_subplot(
    222
)


export_ax2.plot(
    time_days,
    total_burden,
    color=PLOT_FOREGROUND,
    linewidth=2.3,
    label="Total burden"
)


export_ax2.plot(
    time_days,
    sensitive_burden,
    color=PLOT_SECONDARY,
    linewidth=1.8,
    linestyle="--",
    label="Sensitive"
)


export_ax2.plot(
    time_days,
    resistant_burden,
    color="#9E9E9E",
    linewidth=1.8,
    linestyle=":",
    label="Resistant"
)


add_treatment_markers(
    export_ax2,
    treatment_times
)


export_ax2.set_title(
    "Tumour burden",
    color=PLOT_FOREGROUND
)


export_ax2.set_xlabel(
    "Time (days)"
)


export_ax2.set_ylabel(
    "Tumour burden (mL)"
)


style_axis(
    export_ax2
)


export_ax2.legend(
    frameon=False,
    labelcolor=PLOT_FOREGROUND
)


# =============================================================================
# EXPORT PANEL 3 — RESISTANT DISEASE
# =============================================================================

export_ax3 = export_fig.add_subplot(
    223
)


export_ax3.plot(
    time_days,
    sensitive_burden,
    color=PLOT_FOREGROUND,
    linewidth=2.0,
    label="Sensitive tumour"
)


export_ax3.plot(
    time_days,
    resistant_burden,
    color=PLOT_SECONDARY,
    linewidth=2.0,
    linestyle="--",
    label="Resistant tumour"
)


export_ax3b = export_ax3.twinx()


export_ax3b.plot(
    time_days,
    residual_resistant_burden_percent,
    color="#9E9E9E",
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
    "Sensitive and resistant tumour populations",
    color=PLOT_FOREGROUND
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


style_axis(
    export_ax3
)


export_ax3b.set_facecolor(
    "none"
)


export_ax3b.spines[
    "top"
].set_visible(False)


export_ax3b.spines[
    "right"
].set_color(
    PLOT_FOREGROUND
)


export_ax3b.tick_params(
    colors=PLOT_FOREGROUND,
    direction="out",
    length=4,
    width=0.8
)


export_ax3b.yaxis.set_major_locator(
    MaxNLocator(
        nbins=7
    )
)


lines_a, labels_a = (
    export_ax3.get_legend_handles_labels()
)


lines_b, labels_b = (
    export_ax3b.get_legend_handles_labels()
)


export_ax3.legend(
    lines_a + lines_b,
    labels_a + labels_b,
    frameon=False,
    labelcolor=PLOT_FOREGROUND
)


# =============================================================================
# EXPORT PANEL 4 — TCP
# =============================================================================

export_ax4 = export_fig.add_subplot(
    224
)


export_ax4.plot(
    time_days,
    tcp * 100.0,
    color=PLOT_FOREGROUND,
    linewidth=2.2,
    label="TCP"
)


export_ax4.axhline(
    50.0,
    color=PLOT_SECONDARY,
    linestyle="--",
    linewidth=1.0,
    label="50% TCP"
)


add_treatment_markers(
    export_ax4,
    treatment_times
)


export_ax4.set_title(
    "Tumour control probability",
    color=PLOT_FOREGROUND
)


export_ax4.set_xlabel(
    "Time (days)"
)


export_ax4.set_ylabel(
    "TCP (%)"
)


export_ax4.set_ylim(
    0,
    100
)


style_axis(
    export_ax4
)


export_ax4.legend(
    frameon=False,
    labelcolor=PLOT_FOREGROUND
)


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


plt.close(
    export_fig
)
