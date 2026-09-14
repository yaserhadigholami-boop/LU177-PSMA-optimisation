import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Reference dose-rate conversion.
# This is defined at the reference metastatic burden and reference uptake.
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
DEFAULT_TREP = 30.0
DEFAULT_SENSITIVE_FRACTION = 0.75
DEFAULT_RESISTANT_TK_MULTIPLIER = 1.50
DEFAULT_RESISTANT_RADIO_FACTOR = 0.25
DEFAULT_REPOPULATION_KICKOFF = 5.0
DEFAULT_ACTIVITY = 7.4
DEFAULT_CYCLES = 4
DEFAULT_INTERVAL = 7.0
DEFAULT_GAMMA = 1.0

# Normalised initial TCP.
# This is deliberately a modelling parameter rather than a clinical
# pretreatment TCP estimate.
INITIAL_TCP = 0.10
INITIAL_CLONOGENIC_BURDEN = -np.log(INITIAL_TCP)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_dose_scaling(
    initial_burden_ml,
    tumour_uptake_percent
):
    """
    Calculate phenomenological dose scaling.

    Reference condition:
        234 mL metastatic burden
        1.0% total tumour uptake

    Uptake scaling:
        proportional to tumour uptake percentage

    Burden scaling:
        inversely proportional to total tumour burden

    This is an exploratory model and is not a replacement for
    lesion-specific Lu-177 dosimetry/TAC-based absorbed dose.
    """

    uptake_scale = (
        tumour_uptake_percent /
        DEFAULT_TUMOUR_UPTAKE_PERCENT
    )

    burden_scale = (
        REFERENCE_METASTATIC_BURDEN_ML /
        initial_burden_ml
    )

    dose_scale = uptake_scale * burden_scale

    return uptake_scale, burden_scale, dose_scale


def calculate_treatment_times(
    n_cycles,
    interval_days
):
    """Return treatment administration times."""

    return np.array(
        [
            i * interval_days
            for i in range(n_cycles)
        ],
        dtype=float
    )


def calculate_physical_dose_rate(
    time_days,
    activity_gbq,
    n_cycles,
    interval_days,
    initial_burden_ml,
    tumour_uptake_percent
):
    """
    Calculate total physical Lu-177 dose rate.

    Activity from each administration decays exponentially according
    to the Lu-177 physical half-life.

    The administered activity is converted to a phenomenological
    tumour dose rate using the uptake and metastatic burden scaling.
    """

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


def calculate_critical_dose_rate(
    alpha,
    trep_days
):
    """
    Critical dose rate:

        Rcrit = ln(2) / (alpha * Trep_hours)

    Returned in Gy/h.
    """

    trep_hours = trep_days * 24.0

    return (
        np.log(2.0) /
        (alpha * trep_hours)
    )


def calculate_effective_dose_rate(
    physical_dose_rate_gy_day,
    critical_dose_rate_gy_h,
    gamma
):
    """
    Apply dose-rate effectiveness correction.

    R = physical rate / critical rate

    E = min(1, R^gamma)

    Reffective = Rphysical * E
    """

    critical_rate_gy_day = (
        critical_dose_rate_gy_h * 24.0
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
            np.maximum(ratio, 0.0),
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
    """
    Two-compartment sensitive/resistant tumour model.

    Sensitive and resistant compartments grow independently after
    the repopulation kickoff and are reduced by LQ radiation killing.

    The model is phenomenological and intended for treatment-schedule
    exploration.
    """

    beta_sensitive = alpha * DEFAULT_BETA_ALPHA

    alpha_resistant = (
        alpha *
        resistant_radio_factor
    )

    beta_resistant = (
        beta_sensitive *
        resistant_radio_factor
    )

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

    sensitive = np.zeros_like(
        time_days,
        dtype=float
    )

    resistant = np.zeros_like(
        time_days,
        dtype=float
    )

    sensitive[0] = (
        initial_burden_ml *
        sensitive_fraction
    )

    resistant[0] = (
        initial_burden_ml *
        (1.0 - sensitive_fraction)
    )

    for i in range(1, len(time_days)):

        dt = (
            time_days[i] -
            time_days[i - 1]
        )

        s = sensitive[i - 1]
        r = resistant[i - 1]

        # ---------------------------------------------------------------------
        # Repopulation
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
        # Radiation killing
        # ---------------------------------------------------------------------

        dose_interval_gy = (
            physical_dose_rate_gy_day[i] *
            dt
        )

        survival_sensitive = np.exp(
            -alpha * dose_interval_gy
            - beta_sensitive *
            dose_interval_gy ** 2
        )

        survival_resistant = np.exp(
            -alpha_resistant *
            dose_interval_gy
            - beta_resistant *
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

    total = (
        sensitive +
        resistant
    )

    resistant_fraction = np.divide(
        resistant,
        total,
        out=np.zeros_like(total),
        where=total > 0
    )

    return (
        total,
        sensitive,
        resistant,
        resistant_fraction
    )


def calculate_tcp(
    total_burden_ml,
    initial_burden_ml
):
    """
    Normalised TCP model.

    Initial TCP is fixed at INITIAL_TCP.

    Relative tumour burden determines the surviving clonogenic burden.
    """

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


def calculate_summary(
    time_days,
    physical_dose_rate_gy_day,
    effective_dose_rate_gy_day,
    critical_dose_rate_gy_h,
    total_burden,
    sensitive_burden,
    resistant_burden,
    resistant_fraction,
    tcp,
    treatment_times
):
    """Calculate model summary statistics."""

    physical_rate_gy_h = (
        physical_dose_rate_gy_day /
        24.0
    )

    effective_rate_gy_h = (
        effective_dose_rate_gy_day /
        24.0
    )

    # -------------------------------------------------------------------------
    # Dose metrics
    # -------------------------------------------------------------------------

    cumulative_physical_dose = np.trapezoid(
        physical_dose_rate_gy_day,
        time_days
    )

    cumulative_effective_dose = np.trapezoid(
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
    # Critical-rate metrics
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

    # Longest continuous interval above critical rate
    longest_continuous = 0.0
    current_duration = 0.0

    for i in range(1, len(time_days)):

        if (
            above_critical[i - 1]
            and above_critical[i]
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
    # Tumour metrics
    # -------------------------------------------------------------------------

    minimum_burden = np.min(
        total_burden
    )

    minimum_burden_day = time_days[
        np.argmin(total_burden)
    ]

    final_burden = total_burden[-1]

    maximum_tcp = np.max(
        tcp
    )

    maximum_tcp_percent = (
        maximum_tcp *
        100.0
    )

    maximum_tcp_day = time_days[
        np.argmax(tcp)
    ]

    final_resistant_fraction = (
        resistant_fraction[-1] *
        100.0
    )

    maximum_resistant_fraction = (
        np.max(resistant_fraction) *
        100.0
    )

    final_sensitive = sensitive_burden[-1]
    final_resistant = resistant_burden[-1]

    # -------------------------------------------------------------------------
    # Treatment timing
    # -------------------------------------------------------------------------

    treatment_times_text = ", ".join(
        f"{x:.1f}"
        for x in treatment_times
    )

    return {
        "Minimum tumour burden (mL)": minimum_burden,
        "Day of minimum tumour burden": minimum_burden_day,
        "Final tumour burden (mL)": final_burden,
        "Maximum TCP (%)": maximum_tcp_percent,
        "Day of maximum TCP": maximum_tcp_day,
        "Final resistant fraction (%)": final_resistant_fraction,
        "Maximum resistant fraction (%)": maximum_resistant_fraction,
        "Final sensitive burden (mL)": final_sensitive,
        "Final resistant burden (mL)": final_resistant,
        "Critical dose rate (Gy/h)": critical_dose_rate_gy_h,
        "Peak physical dose rate (Gy/h)": peak_physical_rate,
        "Peak effective dose rate (Gy/h)": peak_effective_rate,
        "Time above critical dose rate (days)": time_above_critical,
        "Longest continuous time above critical (days)": longest_continuous,
        "Cumulative physical dose (Gy)": cumulative_physical_dose,
        "Effective cumulative dose (Gy)": cumulative_effective_dose,
        "Treatment times (days)": treatment_times_text,
    }


# =============================================================================
# STREAMLIT SIDEBAR
# =============================================================================

st.title("Lu-177 PSMA Optimisation Model")

st.markdown(
    """
Interactive exploration of Lu-177 PSMA treatment schedules using
physical dose-rate, tumour-response, resistant-cell and TCP models.
"""
)

st.sidebar.header("Model parameters")


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
    min_value=0.01,
    max_value=0.50,
    value=DEFAULT_ALPHA,
    step=0.01,
    format="%.2f",
    help="Effective alpha parameter used by the exploratory LQ model."
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
    value=int(DEFAULT_INTERVAL),
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
# RUN MODEL
# =============================================================================

simulation_days = max(
    FOLLOW_UP_DAYS,
    (
        (n_cycles - 1) *
        interval_days
    ) + FOLLOW_UP_DAYS
)

time_days = np.arange(
    0.0,
    simulation_days + DT_DAYS,
    DT_DAYS
)


# =============================================================================
# DOSE RATE
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

critical_dose_rate_gy_h = calculate_critical_dose_rate(
    alpha=alpha,
    trep_days=trep_days
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
    resistant_fraction
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
    resistant_fraction=resistant_fraction,
    tcp=tcp,
    treatment_times=treatment_times
)


# =============================================================================
# KEY RESULTS
# =============================================================================

st.subheader("Key results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Minimum tumour burden",
        f"{summary['Minimum tumour burden (mL)']:.1f} mL"
    )

with col2:
    st.metric(
        "Maximum TCP",
        f"{summary['Maximum TCP (%)']:.1f}%"
    )

with col3:
    st.metric(
        "Final resistant fraction",
        f"{summary['Final resistant fraction (%)']:.1f}%"
    )

with col4:
    st.metric(
        "Cumulative physical dose",
        f"{summary['Cumulative physical dose (Gy)']:.2f} Gy"
    )


# =============================================================================
# SECONDARY METRICS
# =============================================================================

st.subheader("Secondary metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Critical dose rate",
        f"{summary['Critical dose rate (Gy/h)']:.4f} Gy/h"
    )

with col2:
    st.metric(
        "Peak physical rate",
        f"{summary['Peak physical dose rate (Gy/h)']:.3f} Gy/h"
    )

with col3:
    st.metric(
        "Time above critical rate",
        f"{summary['Time above critical dose rate (days)']:.1f} d"
    )

with col4:
    st.metric(
        "Effective cumulative dose",
        f"{summary['Effective cumulative dose (Gy)']:.2f} Gy"
    )


# =============================================================================
# MODEL SCALING INFORMATION
# =============================================================================

st.caption(
    f"Dose scaling: {dose_scale:.2f}× "
    f"(uptake scaling {uptake_scale:.2f}×; "
    f"burden scaling {burden_scale:.2f}× relative to "
    f"{REFERENCE_METASTATIC_BURDEN_ML:.0f} mL and "
    f"{DEFAULT_TUMOUR_UPTAKE_PERCENT:.1f}% uptake)."
)


# =============================================================================
# MODEL TRAJECTORIES
# =============================================================================

st.subheader("Model trajectories")


# -----------------------------------------------------------------------------
# Plot 1 — Dose rate
# -----------------------------------------------------------------------------

fig1, ax1 = plt.subplots(
    figsize=(9, 5)
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
    linewidth=2.0,
    label="Physical dose rate"
)

ax1.plot(
    time_days,
    effective_rate_gy_h,
    linewidth=1.8,
    linestyle="--",
    label="Effective dose rate"
)

ax1.axhline(
    critical_dose_rate_gy_h,
    linestyle=":",
    linewidth=1.5,
    label="Critical dose rate"
)

# Treatment markers
for t_admin in treatment_times:

    ax1.axvline(
        t_admin,
        linestyle=":",
        linewidth=0.8,
        alpha=0.35,
        color="grey"
    )

ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Dose rate (Gy/h)")
ax1.set_title("Physical and effective dose rate")
ax1.grid(
    alpha=0.25
)
ax1.legend()

fig1.tight_layout()


# -----------------------------------------------------------------------------
# Plot 2 — Tumour burden
# -----------------------------------------------------------------------------

fig2, ax2 = plt.subplots(
    figsize=(9, 5)
)

# Biological curves ONLY:
# Total = solid
# Sensitive = dashed
# Resistant = dotted

ax2.plot(
    time_days,
    total_burden,
    linewidth=2.2,
    label="Total burden"
)

ax2.plot(
    time_days,
    sensitive_burden,
    linewidth=1.6,
    linestyle="--",
    label="Sensitive"
)

ax2.plot(
    time_days,
    resistant_burden,
    linewidth=1.6,
    linestyle=":",
    label="Resistant"
)

# Treatment administration markers are explicitly grey so that they
# cannot be mistaken for another tumour population.
for t_admin in treatment_times:

    ax2.axvline(
        t_admin,
        linestyle=":",
        linewidth=0.8,
        alpha=0.35,
        color="grey"
    )

ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Tumour burden (mL)")
ax2.set_title("Tumour burden")
ax2.grid(
    alpha=0.25
)
ax2.legend()

fig2.tight_layout()


# -----------------------------------------------------------------------------
# Plot 3 — Sensitive/resistant population + resistant fraction
# -----------------------------------------------------------------------------

fig3, ax3 = plt.subplots(
    figsize=(9, 5)
)

ax3.plot(
    time_days,
    sensitive_burden,
    linewidth=1.8,
    label="Sensitive"
)

ax3.plot(
    time_days,
    resistant_burden,
    linewidth=1.8,
    linestyle="--",
    label="Resistant"
)

ax3.set_xlabel("Time (days)")
ax3.set_ylabel("Tumour burden (mL)")
ax3.set_title("Sensitive and resistant tumour populations")
ax3.grid(
    alpha=0.25
)

# Secondary axis for resistant fraction
ax3b = ax3.twinx()

ax3b.plot(
    time_days,
    resistant_fraction * 100.0,
    linewidth=1.5,
    linestyle=":",
    label="Resistant fraction"
)

ax3b.set_ylabel(
    "Resistant fraction (%)"
)

# Combine legends
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()

ax3.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="best"
)

for t_admin in treatment_times:

    ax3.axvline(
        t_admin,
        linestyle=":",
        linewidth=0.8,
        alpha=0.25,
        color="grey"
    )

fig3.tight_layout()


# -----------------------------------------------------------------------------
# Plot 4 — TCP
# -----------------------------------------------------------------------------

fig4, ax4 = plt.subplots(
    figsize=(9, 5)
)

ax4.plot(
    time_days,
    tcp * 100.0,
    linewidth=2.0,
    label="TCP"
)

ax4.axhline(
    50.0,
    linestyle="--",
    linewidth=1.0,
    alpha=0.7,
    label="50% TCP"
)

for t_admin in treatment_times:

    ax4.axvline(
        t_admin,
        linestyle=":",
        linewidth=0.8,
        alpha=0.35,
        color="grey"
    )

ax4.set_xlabel("Time (days)")
ax4.set_ylabel("TCP (%)")
ax4.set_ylim(
    0,
    100
)
ax4.set_title("Tumour control probability")
ax4.grid(
    alpha=0.25
)
ax4.legend()

fig4.tight_layout()


# =============================================================================
# DISPLAY 2 × 2 FIGURE
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

with st.expander("Detailed model summary"):

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.markdown("### Treatment and dosimetry")

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

        st.markdown("### Tumour response")

        st.write(
            f"Initial sensitive fraction: "
            f"{sensitive_fraction * 100:.1f}%"
        )

        st.write(
            f"Initial resistant fraction: "
            f"{(1.0 - sensitive_fraction) * 100:.1f}%"
        )

        st.write(
            f"Trep: "
            f"{trep_days:.1f} days"
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
            f"Final resistant fraction: "
            f"{summary['Final resistant fraction (%)']:.2f}%"
        )

        st.write(
            f"Maximum resistant fraction: "
            f"{summary['Maximum resistant fraction (%)']:.2f}%"
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

with st.expander("Model assumptions and limitations"):

    st.markdown(
        """
### Tumour burden

The model treats the initial tumour burden as a **total metastatic
tumour burden in mL**, rather than as one solid tumour.

The burden variable is therefore intended to represent the aggregate
tumour volume across metastatic sites.

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

The resistant population has:

- reduced alpha
- reduced beta
- slower proliferation through the Tk/Trep multiplier

The resistant population can therefore become increasingly important
when treatment intervals permit repopulation.

### Repopulation

Repopulation begins after the specified repopulation kickoff time.

The model does not currently include explicit cell-cycle effects,
immune-mediated killing, tumour microenvironment effects,
reoxygenation or spatially heterogeneous dose deposition.

### TCP

TCP is implemented as a **normalised exploratory model** based on
relative tumour burden.

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
# DATA EXPORT
# =============================================================================

st.subheader("Export results")


# -----------------------------------------------------------------------------
# CSV export
# -----------------------------------------------------------------------------

results_df = pd.DataFrame(
    {
        "Time_days": time_days,
        "Physical_dose_rate_Gy_day": physical_dose_rate_gy_day,
        "Physical_dose_rate_Gy_h": physical_rate_gy_h,
        "Effective_dose_rate_Gy_day": effective_dose_rate_gy_day,
        "Effective_dose_rate_Gy_h": effective_rate_gy_h,
        "Critical_dose_rate_Gy_h": np.full_like(
            time_days,
            critical_dose_rate_gy_h
        ),
        "Dose_rate_ratio": dose_rate_ratio,
        "Dose_rate_effectiveness": dose_rate_effectiveness,
        "Total_burden_mL": total_burden,
        "Sensitive_burden_mL": sensitive_burden,
        "Resistant_burden_mL": resistant_burden,
        "Resistant_fraction": resistant_fraction,
        "TCP": tcp,
        "Initial_burden_mL": np.full_like(
            time_days,
            initial_burden_ml
        ),
        "Tumour_uptake_percent": np.full_like(
            time_days,
            tumour_uptake_percent
        ),
        "Uptake_scaling": np.full_like(
            time_days,
            uptake_scale
        ),
        "Burden_scaling": np.full_like(
            time_days,
            burden_scale
        ),
        "Dose_scaling_factor": np.full_like(
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
# 600 DPI PNG EXPORT
# =============================================================================

export_fig = plt.figure(
    figsize=(16, 12)
)

export_ax1 = export_fig.add_subplot(221)

export_ax1.plot(
    time_days,
    physical_rate_gy_h,
    linewidth=2.0,
    label="Physical dose rate"
)

export_ax1.plot(
    time_days,
    effective_rate_gy_h,
    linewidth=1.8,
    linestyle="--",
    label="Effective dose rate"
)

export_ax1.axhline(
    critical_dose_rate_gy_h,
    linestyle=":",
    linewidth=1.5,
    label="Critical dose rate"
)

for t_admin in treatment_times:

    export_ax1.axvline(
        t_admin,
        linestyle=":",
        linewidth=0.8,
        alpha=0.35,
        color="grey"
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

export_ax1.grid(
    alpha=0.25
)

export_ax1.legend()


# -----------------------------------------------------------------------------
# Export panel 2 — tumour burden
# -----------------------------------------------------------------------------

export_ax2 = export_fig.add_subplot(222)

export_ax2.plot(
    time_days,
    total_burden,
    linewidth=2.2,
    label="Total burden"
)

export_ax2.plot(
    time_days,
    sensitive_burden,
    linewidth=1.6,
    linestyle="--",
    label="Sensitive"
)

export_ax2.plot(
    time_days,
    resistant_burden,
    linewidth=1.6,
    linestyle=":",
    label="Resistant"
)

for t_admin in treatment_times:

    export_ax2.axvline(
        t_admin,
        linestyle=":",
        linewidth=0.8,
        alpha=0.35,
        color="grey"
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

export_ax2.grid(
    alpha=0.25
)

export_ax2.legend()


# -----------------------------------------------------------------------------
# Export panel 3 — population composition
# -----------------------------------------------------------------------------

export_ax3 = export_fig.add_subplot(223)

export_ax3.plot(
    time_days,
    sensitive_burden,
    linewidth=1.8,
    label="Sensitive"
)

export_ax3.plot(
    time_days,
    resistant_burden,
    linewidth=1.8,
    linestyle="--",
    label="Resistant"
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

export_ax3.grid(
    alpha=0.25
)

export_ax3b = export_ax3.twinx()

export_ax3b.plot(
    time_days,
    resistant_fraction * 100.0,
    linewidth=1.5,
    linestyle=":",
    label="Resistant fraction"
)

export_ax3b.set_ylabel(
    "Resistant fraction (%)"
)

lines_a, labels_a = export_ax3.get_legend_handles_labels()
lines_b, labels_b = export_ax3b.get_legend_handles_labels()

export_ax3.legend(
    lines_a + lines_b,
    labels_a + labels_b,
    loc="best"
)

for t_admin in treatment_times:

    export_ax3.axvline(
        t_admin,
        linestyle=":",
        linewidth=0.8,
        alpha=0.25,
        color="grey"
    )


# -----------------------------------------------------------------------------
# Export panel 4 — TCP
# -----------------------------------------------------------------------------

export_ax4 = export_fig.add_subplot(224)

export_ax4.plot(
    time_days,
    tcp * 100.0,
    linewidth=2.0,
    label="TCP"
)

export_ax4.axhline(
    50.0,
    linestyle="--",
    linewidth=1.0,
    alpha=0.7,
    label="50% TCP"
)

for t_admin in treatment_times:

    export_ax4.axvline(
        t_admin,
        linestyle=":",
        linewidth=0.8,
        alpha=0.35,
        color="grey"
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
    100
)

export_ax4.grid(
    alpha=0.25
)

export_ax4.legend()


export_fig.tight_layout()


png_buffer = io.BytesIO()

export_fig.savefig(
    png_buffer,
    format="png",
    dpi=600,
    bbox_inches="tight"
)

png_buffer.seek(0)

st.download_button(
    label="Download trajectories (600 dpi PNG)",
    data=png_buffer,
    file_name="Lu177_PSMA_model_trajectories_600dpi.png",
    mime="image/png"
)

plt.close(export_fig)
