# =============================================================================
# LU-177 PSMA — STREAMLIT INTERACTIVE METASTATIC TUMOUR DYNAMICS EXPLORER
# =============================================================================
#
# Run locally:
#
#     streamlit run Interactive_model_explorer_streamlit.py
#
# =============================================================================
#
# This is an exploratory mechanistic model.
#
# It is NOT a clinically validated patient-specific dosimetry or TCP model.
#
# =============================================================================


import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Lu-177 PSMA Tumour Dynamics Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# 2. MODEL CONSTANTS
# =============================================================================

LU177_HALF_LIFE_DAYS = 6.647

LU177_LAMBDA_PER_DAY = (
    np.log(2.0)
    / LU177_HALF_LIFE_DAYS
)

# -------------------------------------------------------------------------
# Reference dose conversion
# -------------------------------------------------------------------------
#
# DOSE_PER_GBQ_GY is the reference dose-rate conversion used by the
# exploratory model.
#
# It is referenced to:
#
#     234 mL total metastatic burden
#     1.0% total tumour uptake
#
# The model then scales the dose rate according to:
#
# Dose rate =
#     administered activity
#     × DOSE_PER_GBQ_GY
#     × uptake scaling
#     × burden scaling
#
# -------------------------------------------------------------------------

DOSE_PER_GBQ_GY = 0.50

REFERENCE_METASTATIC_BURDEN_ML = 234.0

DEFAULT_TUMOUR_UPTAKE_PERCENT = 1.0

TUMOUR_UPTAKE_MIN_PERCENT = 0.1
TUMOUR_UPTAKE_MAX_PERCENT = 10.0

DT_DAYS = 0.05

FOLLOW_UP_DAYS = 60.0


# =============================================================================
# 3. DEFAULT BIOLOGICAL PARAMETERS
# =============================================================================

DEFAULT_INITIAL_METASTATIC_BURDEN_ML = 234.0

DEFAULT_ALPHA = 0.10

ALPHA_MIN = 0.01
ALPHA_MAX = 0.50

DEFAULT_BETA_ALPHA = 0.10

DEFAULT_SENSITIVE_FRACTION = 0.75

DEFAULT_RESISTANT_TK_MULTIPLIER = 1.50

DEFAULT_RESISTANT_RADIO_FACTOR = 0.25

DEFAULT_TREP_DAYS = 30.0

DEFAULT_REPOPULATION_KICKOFF_DAYS = 5.0

DEFAULT_ACTIVITY_GBQ = 7.40

DEFAULT_N_CYCLES = 4

DEFAULT_CYCLE_INTERVAL_DAYS = 7.0

DEFAULT_EFFECTIVENESS_GAMMA = 1.0


# =============================================================================
# 4. TCP PARAMETERS
# =============================================================================

INITIAL_TCP = 0.10

INITIAL_CLONOGENIC_BURDEN = (
    -np.log(INITIAL_TCP)
)


# =============================================================================
# 5. PAGE TITLE
# =============================================================================

st.title(
    "Lu-177 PSMA — Metastatic Tumour Dynamics Explorer"
)

st.caption(
    "Exploratory mechanistic model based on the current Sections 01–05 "
    "framework. Not a clinically validated patient-specific model."
)


# =============================================================================
# 6. SIDEBAR
# =============================================================================

st.sidebar.header(
    "Model Controls"
)

st.sidebar.markdown(
    "Adjust the treatment and biological parameters below."
)


# =============================================================================
# 7. INITIAL METASTATIC BURDEN
# =============================================================================

initial_burden_ml = st.sidebar.slider(
    "Initial metastatic burden (mL)",
    min_value=10.0,
    max_value=1500.0,
    value=DEFAULT_INITIAL_METASTATIC_BURDEN_ML,
    step=1.0,
    help=(
        "Total metastatic tumour burden used as the initial condition. "
        "This is a phenomenological burden variable rather than a "
        "single solid tumour mass."
    )
)


# =============================================================================
# 8. TOTAL TUMOUR UPTAKE
# =============================================================================

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
# 9. ALPHA
# =============================================================================

alpha = st.sidebar.slider(
    "Alpha (Gy⁻¹)",
    min_value=ALPHA_MIN,
    max_value=ALPHA_MAX,
    value=DEFAULT_ALPHA,
    step=0.005,
    format="%.3f",
    help=(
        "Linear LQ radiosensitivity parameter."
    )
)


# =============================================================================
# 10. TREP
# =============================================================================

trep_days = st.sidebar.slider(
    "Trep (days)",
    min_value=10.0,
    max_value=100.0,
    value=DEFAULT_TREP_DAYS,
    step=1.0,
    help=(
        "Sensitive-cell population doubling time."
    )
)


# =============================================================================
# 11. SENSITIVE FRACTION
# =============================================================================

sensitive_fraction = st.sidebar.slider(
    "Sensitive fraction",
    min_value=0.50,
    max_value=0.95,
    value=DEFAULT_SENSITIVE_FRACTION,
    step=0.01,
    help=(
        "Initial fraction of the metastatic burden assigned to "
        "the radiosensitive population."
    )
)


# =============================================================================
# 12. RESISTANT TK / TREP
# =============================================================================

resistant_tk_multiplier = st.sidebar.slider(
    "Tk / Trep",
    min_value=1.20,
    max_value=1.80,
    value=DEFAULT_RESISTANT_TK_MULTIPLIER,
    step=0.01,
    help=(
        "Resistant population doubling time relative to the sensitive "
        "population."
    )
)


# =============================================================================
# 13. RESISTANT RADIO FACTOR
# =============================================================================

resistant_radio_factor = st.sidebar.slider(
    "Resistant radio factor",
    min_value=0.05,
    max_value=1.00,
    value=DEFAULT_RESISTANT_RADIO_FACTOR,
    step=0.01,
    help=(
        "Multiplier applied to alpha and beta for the resistant population."
    )
)


# =============================================================================
# 14. REPOPULATION KICKOFF
# =============================================================================

repopulation_kickoff_days = st.sidebar.slider(
    "Repopulation kickoff (days)",
    min_value=3.0,
    max_value=10.0,
    value=DEFAULT_REPOPULATION_KICKOFF_DAYS,
    step=0.5,
    help=(
        "Time after which population regrowth is activated."
    )
)


# =============================================================================
# 15. ACTIVITY
# =============================================================================

activity_gbq = st.sidebar.slider(
    "Activity / cycle (GBq)",
    min_value=1.0,
    max_value=15.0,
    value=DEFAULT_ACTIVITY_GBQ,
    step=0.1,
    help=(
        "Administered Lu-177 activity per treatment cycle. "
        "This is the injected activity before tumour uptake."
    )
)


# =============================================================================
# 16. NUMBER OF CYCLES
# =============================================================================

n_cycles = st.sidebar.slider(
    "Number of cycles",
    min_value=1,
    max_value=8,
    value=DEFAULT_N_CYCLES,
    step=1
)


# =============================================================================
# 17. CYCLE INTERVAL
# =============================================================================

cycle_interval_days = st.sidebar.slider(
    "Cycle interval (days)",
    min_value=1.0,
    max_value=42.0,
    value=DEFAULT_CYCLE_INTERVAL_DAYS,
    step=1.0,
    help=(
        "Time between treatment administrations."
    )
)


# =============================================================================
# 18. EFFECTIVENESS GAMMA
# =============================================================================

gamma = st.sidebar.slider(
    "Effectiveness gamma",
    min_value=0.25,
    max_value=2.00,
    value=DEFAULT_EFFECTIVENESS_GAMMA,
    step=0.05,
    help=(
        "Dose-rate effectiveness parameter from the Section 05 model."
    )
)


# =============================================================================
# 19. RESET
# =============================================================================

if st.sidebar.button(
    "Reset all parameters",
    use_container_width=True
):

    st.session_state.clear()

    st.rerun()


# =============================================================================
# 20. DOSE SCALING
# =============================================================================

def calculate_dose_scaling(
    initial_burden_ml,
    tumour_uptake_percent
):

    uptake_scale = (
        tumour_uptake_percent
        / DEFAULT_TUMOUR_UPTAKE_PERCENT
    )

    burden_scale = (
        REFERENCE_METASTATIC_BURDEN_ML
        / initial_burden_ml
    )

    dose_scale = (
        uptake_scale
        * burden_scale
    )

    return (
        uptake_scale,
        burden_scale,
        dose_scale
    )


(
    uptake_scale,
    burden_scale,
    dose_scaling_factor
) = calculate_dose_scaling(
    initial_burden_ml,
    tumour_uptake_percent
)


# =============================================================================
# 21. CRITICAL DOSE RATE
# =============================================================================

def calculate_critical_dose_rate(
    alpha,
    trep_days
):

    trep_hours = (
        trep_days
        * 24.0
    )

    return (
        np.log(2.0)
        / (
            alpha
            * trep_hours
        )
    )


# =============================================================================
# 22. TREATMENT SCHEDULE
# =============================================================================

def make_treatment_schedule(
    n_cycles,
    interval_days
):

    return (
        np.arange(
            n_cycles,
            dtype=float
        )
        * interval_days
    )


# =============================================================================
# 23. PHYSICAL DOSE RATE
# =============================================================================

def calculate_physical_dose_rate(
    time_days,
    activity_gbq,
    n_cycles,
    interval_days,
    initial_burden_ml,
    tumour_uptake_percent
):

    treatment_times = (
        make_treatment_schedule(
            n_cycles,
            interval_days
        )
    )

    # -------------------------------------------------------------------------
    # Uptake and burden scaling
    # -------------------------------------------------------------------------

    (
        uptake_scale,
        burden_scale,
        dose_scale
    ) = calculate_dose_scaling(
        initial_burden_ml,
        tumour_uptake_percent
    )

    dose_rate_gy_day = np.zeros_like(
        time_days
    )

    for t_admin in treatment_times:

        elapsed = (
            time_days
            - t_admin
        )

        mask = (
            elapsed >= 0.0
        )

        activity = np.zeros_like(
            time_days
        )

        activity[mask] = (
            activity_gbq
            * np.exp(
                -LU177_LAMBDA_PER_DAY
                * elapsed[mask]
            )
        )

        dose_rate_gy_day += (
            activity
            * DOSE_PER_GBQ_GY
            * dose_scale
        )

    return dose_rate_gy_day


# =============================================================================
# 24. EFFECTIVE DOSE RATE
# =============================================================================

def calculate_effective_dose_rate(
    physical_dose_rate_gy_day,
    critical_dose_rate_gy_h,
    gamma
):

    critical_rate_gy_day = (
        critical_dose_rate_gy_h
        * 24.0
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
        ratio ** gamma
    )

    effective_dose_rate_gy_day = (
        physical_dose_rate_gy_day
        * effectiveness
    )

    return effective_dose_rate_gy_day


# =============================================================================
# 25. BIOLOGICAL MODEL
# =============================================================================

def simulate_biology(
    time_days,
    physical_dose_rate_gy_day,
    initial_burden_ml,
    sensitive_fraction,
    alpha,
    beta_alpha,
    trep_days,
    resistant_tk_multiplier,
    resistant_radio_factor,
    repopulation_kickoff_days
):

    n_points = len(
        time_days
    )

    sensitive = np.zeros(
        n_points
    )

    resistant = np.zeros(
        n_points
    )

    # -------------------------------------------------------------------------
    # Initial populations
    # -------------------------------------------------------------------------

    sensitive[0] = (
        initial_burden_ml
        * sensitive_fraction
    )

    resistant[0] = (
        initial_burden_ml
        * (
            1.0
            - sensitive_fraction
        )
    )

    # -------------------------------------------------------------------------
    # LQ parameters
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Growth rates
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Time integration
    # -------------------------------------------------------------------------

    for i in range(
        1,
        n_points
    ):

        dt = (
            time_days[i]
            - time_days[i - 1]
        )

        S = sensitive[i - 1]

        R = resistant[i - 1]

        # ---------------------------------------------------------------------
        # Repopulation
        # ---------------------------------------------------------------------

        if (
            time_days[i]
            >= repopulation_kickoff_days
        ):

            S *= np.exp(
                sensitive_growth_rate
                * dt
            )

            R *= np.exp(
                resistant_growth_rate
                * dt
            )

        # ---------------------------------------------------------------------
        # Dose in interval
        # ---------------------------------------------------------------------

        dose_interval_gy = (
            physical_dose_rate_gy_day[i]
            * dt
        )

        # ---------------------------------------------------------------------
        # LQ survival
        # ---------------------------------------------------------------------

        survival_sensitive = np.exp(
            -alpha
            * dose_interval_gy
            -beta_sensitive
            * dose_interval_gy ** 2
        )

        survival_resistant = np.exp(
            -alpha_resistant
            * dose_interval_gy
            -beta_resistant
            * dose_interval_gy ** 2
        )

        # ---------------------------------------------------------------------
        # Radiation killing
        # ---------------------------------------------------------------------

        S *= survival_sensitive

        R *= survival_resistant

        sensitive[i] = max(
            S,
            1e-12
        )

        resistant[i] = max(
            R,
            1e-12
        )

    total = (
        sensitive
        + resistant
    )

    return (
        sensitive,
        resistant,
        total
    )


# =============================================================================
# 26. TCP
# =============================================================================

def calculate_tcp(
    total_burden_ml,
    initial_burden_ml
):

    relative_burden = np.divide(
        total_burden_ml,
        initial_burden_ml,
        out=np.ones_like(
            total_burden_ml
        ),
        where=initial_burden_ml > 0
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
# 27. SUMMARY METRICS
# =============================================================================

def calculate_summary(
    time_days,
    physical_dose_rate_gy_day,
    effective_dose_rate_gy_day,
    sensitive,
    resistant,
    total,
    tcp,
    critical_dose_rate_gy_h
):

    physical_rate_gy_h = (
        physical_dose_rate_gy_day
        / 24.0
    )

    effective_rate_gy_h = (
        effective_dose_rate_gy_day
        / 24.0
    )

    # -------------------------------------------------------------------------
    # Cumulative dose
    # -------------------------------------------------------------------------

    cumulative_physical_dose = np.trapezoid(
        physical_dose_rate_gy_day,
        time_days
    )

    cumulative_effective_dose = np.trapezoid(
        effective_dose_rate_gy_day,
        time_days
    )

    # -------------------------------------------------------------------------
    # Peak rates
    # -------------------------------------------------------------------------

    peak_physical_rate = np.max(
        physical_rate_gy_h
    )

    peak_effective_rate = np.max(
        effective_rate_gy_h
    )

    # -------------------------------------------------------------------------
    # Tumour burden
    # -------------------------------------------------------------------------

    initial_burden = total[0]

    final_burden = total[-1]

    minimum_burden = np.min(
        total
    )

    minimum_index = np.argmin(
        total
    )

    time_of_minimum = (
        time_days[
            minimum_index
        ]
    )

    burden_reduction = (
        1.0
        - final_burden
        / initial_burden
    )

    # -------------------------------------------------------------------------
    # Resistant fraction
    # -------------------------------------------------------------------------

    resistant_fraction = np.divide(
        resistant,
        total,
        out=np.zeros_like(
            resistant
        ),
        where=total > 0
    )

    initial_resistant_fraction = (
        resistant_fraction[0]
    )

    final_resistant_fraction = (
        resistant_fraction[-1]
    )

    maximum_resistant_fraction = (
        np.max(
            resistant_fraction
        )
    )

    # -------------------------------------------------------------------------
    # TCP
    # -------------------------------------------------------------------------

    initial_tcp = tcp[0]

    final_tcp = tcp[-1]

    maximum_tcp = np.max(
        tcp
    )

    maximum_tcp_index = np.argmax(
        tcp
    )

    time_of_max_tcp = (
        time_days[
            maximum_tcp_index
        ]
    )

    # -------------------------------------------------------------------------
    # Critical dose rate
    # -------------------------------------------------------------------------

    above_critical = (
        physical_rate_gy_h
        >= critical_dose_rate_gy_h
    )

    time_above_critical_hours = (
        np.sum(
            above_critical
        )
        * DT_DAYS
        * 24.0
    )

    return {

        "critical_rate":
            critical_dose_rate_gy_h,

        "peak_physical_rate":
            peak_physical_rate,

        "peak_effective_rate":
            peak_effective_rate,

        "cumulative_physical_dose":
            cumulative_physical_dose,

        "cumulative_effective_dose":
            cumulative_effective_dose,

        "initial_burden":
            initial_burden,

        "final_burden":
            final_burden,

        "minimum_burden":
            minimum_burden,

        "time_of_minimum_burden":
            time_of_minimum,

        "burden_reduction":
            burden_reduction,

        "initial_resistant_fraction":
            initial_resistant_fraction,

        "final_resistant_fraction":
            final_resistant_fraction,

        "maximum_resistant_fraction":
            maximum_resistant_fraction,

        "initial_tcp":
            initial_tcp,

        "final_tcp":
            final_tcp,

        "maximum_tcp":
            maximum_tcp,

        "time_of_max_tcp":
            time_of_max_tcp,

        "time_above_critical":
            time_above_critical_hours
    }


# =============================================================================
# 28. RUN MODEL
# =============================================================================

treatment_times = make_treatment_schedule(
    n_cycles,
    cycle_interval_days
)

last_treatment_day = (
    treatment_times[-1]
)

simulation_days = (
    last_treatment_day
    + FOLLOW_UP_DAYS
)

time_days = np.arange(
    0.0,
    simulation_days + DT_DAYS,
    DT_DAYS
)


# =============================================================================
# 29. CRITICAL DOSE RATE
# =============================================================================

critical_rate = calculate_critical_dose_rate(
    alpha,
    trep_days
)


# =============================================================================
# 30. PHYSICAL DOSE RATE
# =============================================================================

physical_dose_rate = (
    calculate_physical_dose_rate(
        time_days,
        activity_gbq,
        n_cycles,
        cycle_interval_days,
        initial_burden_ml,
        tumour_uptake_percent
    )
)


# =============================================================================
# 31. EFFECTIVE DOSE RATE
# =============================================================================

effective_dose_rate = (
    calculate_effective_dose_rate(
        physical_dose_rate,
        critical_rate,
        gamma
    )
)


# =============================================================================
# 32. BIOLOGICAL MODEL
# =============================================================================

(
    sensitive,
    resistant,
    total
) = simulate_biology(

    time_days=time_days,

    physical_dose_rate_gy_day=
        physical_dose_rate,

    initial_burden_ml=
        initial_burden_ml,

    sensitive_fraction=
        sensitive_fraction,

    alpha=
        alpha,

    beta_alpha=
        DEFAULT_BETA_ALPHA,

    trep_days=
        trep_days,

    resistant_tk_multiplier=
        resistant_tk_multiplier,

    resistant_radio_factor=
        resistant_radio_factor,

    repopulation_kickoff_days=
        repopulation_kickoff_days
)


# =============================================================================
# 33. TCP
# =============================================================================

tcp = calculate_tcp(
    total,
    initial_burden_ml
)


# =============================================================================
# 34. SUMMARY
# =============================================================================

summary = calculate_summary(

    time_days=
        time_days,

    physical_dose_rate_gy_day=
        physical_dose_rate,

    effective_dose_rate_gy_day=
        effective_dose_rate,

    sensitive=
        sensitive,

    resistant=
        resistant,

    total=
        total,

    tcp=
        tcp,

    critical_dose_rate_gy_h=
        critical_rate
)


# =============================================================================
# 35. DATA PREPARATION
# =============================================================================

resistant_fraction = np.divide(
    resistant,
    total,
    out=np.zeros_like(
        resistant
    ),
    where=total > 0
)

physical_rate_gy_h = (
    physical_dose_rate
    / 24.0
)

effective_rate_gy_h = (
    effective_dose_rate
    / 24.0
)

relative_burden = np.divide(
    total,
    initial_burden_ml,
    out=np.ones_like(total),
    where=initial_burden_ml > 0
)


# =============================================================================
# 36. DATAFRAME
# =============================================================================

data = pd.DataFrame({

    "Time_days":
        time_days,

    "Physical_dose_rate_Gy_h":
        physical_rate_gy_h,

    "Effective_dose_rate_Gy_h":
        effective_rate_gy_h,

    "Sensitive_burden_mL":
        sensitive,

    "Resistant_burden_mL":
        resistant,

    "Total_burden_mL":
        total,

    "Relative_burden":
        relative_burden,

    "Resistant_fraction":
        resistant_fraction,

    "TCP":
        tcp,

    # -------------------------------------------------------------------------
    # Model parameters/scaling
    # -------------------------------------------------------------------------

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
            dose_scaling_factor
        )
})


# =============================================================================
# 37. KEY METRICS
# =============================================================================

st.subheader(
    "Key results"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Minimum tumour burden",
        f"{summary['minimum_burden']:.1f} mL",
        f"{100 * summary['burden_reduction']:.1f}% final reduction"
    )

with col2:

    st.metric(
        "Maximum TCP",
        f"{100 * summary['maximum_tcp']:.1f}%",
        f"at {summary['time_of_max_tcp']:.1f} d"
    )

with col3:

    st.metric(
        "Final resistant fraction",
        f"{100 * summary['final_resistant_fraction']:.1f}%"
    )

with col4:

    st.metric(
        "Cumulative physical dose",
        f"{summary['cumulative_physical_dose']:.2f} Gy"
    )


# =============================================================================
# 38. SECONDARY METRICS
# =============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Critical dose rate",
        f"{summary['critical_rate']:.4f} Gy/h"
    )

with col2:

    st.metric(
        "Peak physical rate",
        f"{summary['peak_physical_rate']:.4f} Gy/h"
    )

with col3:

    st.metric(
        "Time above critical rate",
        f"{summary['time_above_critical']:.1f} h"
    )

with col4:

    st.metric(
        "Effective cumulative dose",
        f"{summary['cumulative_effective_dose']:.2f} Gy"
    )


# =============================================================================
# 39. DOSE SCALING INFORMATION
# =============================================================================

st.caption(
    f"Dose scaling: {dose_scaling_factor:.3f}× "
    f"(uptake scaling {uptake_scale:.2f}×; "
    f"burden scaling {burden_scale:.2f}× relative to "
    f"{REFERENCE_METASTATIC_BURDEN_ML:.0f} mL and "
    f"{DEFAULT_TUMOUR_UPTAKE_PERCENT:.1f}% uptake)."
)


# =============================================================================
# 40. MODEL TRAJECTORIES
# =============================================================================

st.subheader(
    "Model trajectories"
)


# =============================================================================
# 41. DOSE RATE FIGURE
# =============================================================================

fig_rate, ax_rate = plt.subplots(
    figsize=(8, 4.5)
)

ax_rate.plot(
    time_days,
    physical_rate_gy_h,
    linewidth=1.8,
    label="Physical"
)

ax_rate.plot(
    time_days,
    effective_rate_gy_h,
    linewidth=1.8,
    label="Effective"
)

ax_rate.axhline(
    critical_rate,
    linestyle="--",
    linewidth=1.0,
    label="Critical rate"
)

for t_admin in treatment_times:

    ax_rate.axvline(
        t_admin,
        linestyle=":",
        linewidth=0.7,
        alpha=0.5
    )

ax_rate.set_title(
    "Lu-177 dose rate"
)

ax_rate.set_xlabel(
    "Time (days)"
)

ax_rate.set_ylabel(
    "Dose rate (Gy/h)"
)

ax_rate.grid(
    alpha=0.2
)

ax_rate.legend()

fig_rate.tight_layout()


# =============================================================================
# 42. TUMOUR BURDEN FIGURE
# =============================================================================

fig_burden, ax_burden = plt.subplots(
    figsize=(8, 4.5)
)

ax_burden.plot(
    time_days,
    total,
    linewidth=2.0,
    label="Total burden"
)

ax_burden.plot(
    time_days,
    sensitive,
    linewidth=1.4,
    linestyle="--",
    label="Sensitive"
)

ax_burden.plot(
    time_days,
    resistant,
    linewidth=1.4,
    linestyle=":",
    label="Resistant"
)

for t_admin in treatment_times:

    ax_burden.axvline(
        t_admin,
        linestyle=":",
        linewidth=0.7,
        alpha=0.5
    )

ax_burden.set_title(
    "Metastatic tumour burden"
)

ax_burden.set_xlabel(
    "Time (days)"
)

ax_burden.set_ylabel(
    "Burden (mL)"
)

ax_burden.grid(
    alpha=0.2
)

ax_burden.legend()

fig_burden.tight_layout()


# =============================================================================
# 43. SENSITIVE / RESISTANT FIGURE
# =============================================================================

fig_population, ax_population = plt.subplots(
    figsize=(8, 4.5)
)

ax_population.plot(
    time_days,
    sensitive,
    linewidth=1.8,
    label="Sensitive"
)

ax_population.plot(
    time_days,
    resistant,
    linewidth=1.8,
    label="Resistant"
)

ax_population.plot(
    time_days,
    total,
    linewidth=1.2,
    linestyle="--",
    label="Total"
)

for t_admin in treatment_times:

    ax_population.axvline(
        t_admin,
        linestyle=":",
        linewidth=0.7,
        alpha=0.5
    )

ax_population.set_title(
    "Sensitive vs resistant metastatic burden"
)

ax_population.set_xlabel(
    "Time (days)"
)

ax_population.set_ylabel(
    "Burden (mL)"
)

ax_population.grid(
    alpha=0.2
)

ax_population.legend(
    loc="upper left"
)


# -----------------------------------------------------------------------------
# Resistant fraction secondary axis
# -----------------------------------------------------------------------------

ax_fraction = ax_population.twinx()

ax_fraction.plot(
    time_days,
    resistant_fraction * 100.0,
    linestyle="-.",
    linewidth=1.1,
    label="Resistant fraction"
)

ax_fraction.set_ylabel(
    "Resistant fraction (%)"
)

ax_fraction.set_ylim(
    0,
    100
)

fig_population.tight_layout()


# =============================================================================
# 44. TCP FIGURE
# =============================================================================

fig_tcp, ax_tcp = plt.subplots(
    figsize=(8, 4.5)
)

ax_tcp.plot(
    time_days,
    tcp * 100.0,
    linewidth=2.0,
    label="TCP"
)

ax_tcp.axhline(
    50.0,
    linestyle="--",
    linewidth=1.0,
    label="50%"
)

for t_admin in treatment_times:

    ax_tcp.axvline(
        t_admin,
        linestyle=":",
        linewidth=0.7,
        alpha=0.5
    )

ax_tcp.set_title(
    "Exploratory tumour control probability"
)

ax_tcp.set_xlabel(
    "Time (days)"
)

ax_tcp.set_ylabel(
    "TCP (%)"
)

ax_tcp.set_ylim(
    0,
    100
)

ax_tcp.grid(
    alpha=0.2
)

ax_tcp.legend(
    loc="lower right"
)

fig_tcp.tight_layout()


# =============================================================================
# 45. DISPLAY PLOTS
# =============================================================================

plot_col1, plot_col2 = st.columns(2)

with plot_col1:

    st.pyplot(
        fig_rate,
        use_container_width=True
    )

with plot_col2:

    st.pyplot(
        fig_burden,
        use_container_width=True
    )

plot_col3, plot_col4 = st.columns(2)

with plot_col3:

    st.pyplot(
        fig_population,
        use_container_width=True
    )

with plot_col4:

    st.pyplot(
        fig_tcp,
        use_container_width=True
    )


# =============================================================================
# 46. DETAILED SUMMARY
# =============================================================================

with st.expander(
    "Detailed model summary",
    expanded=False
):

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.markdown(
            f"""
            **Treatment**

            - Activity/cycle: {activity_gbq:.2f} GBq
            - Number of cycles: {n_cycles}
            - Cycle interval: {cycle_interval_days:.0f} days
            - Treatment days: {", ".join(f"{x:.0f}" for x in treatment_times)}

            **Initial tumour state**

            - Initial metastatic burden: {initial_burden_ml:.1f} mL
            - Total tumour uptake: {tumour_uptake_percent:.1f}%
            - Sensitive fraction: {100*sensitive_fraction:.1f}%
            - Resistant fraction: {100*(1-sensitive_fraction):.1f}%

            **Dose scaling**

            - Reference burden: {REFERENCE_METASTATIC_BURDEN_ML:.0f} mL
            - Reference uptake: {DEFAULT_TUMOUR_UPTAKE_PERCENT:.1f}%
            - Uptake scaling: {uptake_scale:.3f}×
            - Burden scaling: {burden_scale:.3f}×
            - Overall dose scaling: {dose_scaling_factor:.3f}×

            **Radiobiology**

            - α: {alpha:.3f} Gy⁻¹
            - β/α: {DEFAULT_BETA_ALPHA:.3f} Gy
            - β: {alpha * DEFAULT_BETA_ALPHA:.5f} Gy⁻²
            - Trep: {trep_days:.1f} days
            - Resistant Tk/Trep: {resistant_tk_multiplier:.2f}
            - Resistant radio factor: {resistant_radio_factor:.2f}
            - Repopulation kickoff: {repopulation_kickoff_days:.1f} days
            """
        )

    with summary_col2:

        st.markdown(
            f"""
            **Dose-rate behaviour**

            - Critical dose rate: {summary['critical_rate']:.5f} Gy/h
            - Peak physical dose rate: {summary['peak_physical_rate']:.5f} Gy/h
            - Peak effective dose rate: {summary['peak_effective_rate']:.5f} Gy/h
            - Time above critical rate: {summary['time_above_critical']:.1f} h
            - Cumulative physical dose: {summary['cumulative_physical_dose']:.2f} Gy
            - Cumulative effective dose: {summary['cumulative_effective_dose']:.2f} Gy

            **Tumour response**

            - Minimum burden: {summary['minimum_burden']:.2f} mL
            - Time of minimum burden: {summary['time_of_minimum_burden']:.1f} days
            - Final burden: {summary['final_burden']:.2f} mL
            - Final burden reduction: {100*summary['burden_reduction']:.1f}%

            **Resistant population**

            - Initial resistant fraction: {100*summary['initial_resistant_fraction']:.1f}%
            - Final resistant fraction: {100*summary['final_resistant_fraction']:.1f}%
            - Maximum resistant fraction: {100*summary['maximum_resistant_fraction']:.1f}%

            **TCP**

            - Initial TCP: {100*summary['initial_tcp']:.1f}%
            - Maximum TCP: {100*summary['maximum_tcp']:.1f}%
            - Time of maximum TCP: {summary['time_of_max_tcp']:.1f} days
            - Final TCP: {100*summary['final_tcp']:.1f}%
            """
        )


# =============================================================================
# 47. MODEL ASSUMPTIONS
# =============================================================================

with st.expander(
    "Model assumptions and limitations",
    expanded=False
):

    st.markdown(
        """
        ### Important assumptions

        1. **Metastatic tumour burden**

           The initial disease burden is represented as a total metastatic
           burden in mL. It is not treated as one anatomical solid tumour.

        2. **Administered activity**

           The activity slider represents administered Lu-177 activity per
           cycle before tumour uptake.

        3. **Total tumour uptake**

           The tumour uptake slider represents an exploratory total fraction
           of administered activity attributed to the overall metastatic
           tumour burden.

        4. **Burden-dependent dose scaling**

           At fixed administered activity and total tumour uptake, a larger
           total metastatic burden produces a lower average dose rate per
           unit tumour burden, while a smaller burden produces a higher
           average dose rate.

        5. **Reference dose conversion**

           The simplified dose conversion is referenced to 234 mL metastatic
           burden and 1% total tumour uptake. The default model therefore
           retains the previous dose-rate behaviour at these values.

        6. **Tumour uptake and dosimetry**

           Lesion-specific uptake, residence time, time-activity curves,
           spatial heterogeneity, absorbed fractions, cross-dose and
           inter-lesion dose redistribution are not explicitly modelled.

        7. **Sensitive/resistant populations**

           The two populations are phenomenological representations of
           radiosensitive and radioresistant disease.

        8. **TCP**

           TCP is an exploratory normalized clonogenic model. It should not
           be interpreted as a clinically validated probability of cure.

        9. **Dose-rate effectiveness**

           The gamma parameter represents the current exploratory
           dose-rate-effectiveness formulation.

        10. **Clinical interpretation**

            The model is intended for hypothesis generation and sensitivity
            analysis rather than treatment prescription or patient-specific
            outcome prediction.
        """
    )


# =============================================================================
# 48. EXPORT
# =============================================================================

st.subheader(
    "Export"
)


# =============================================================================
# 49. DOWNLOAD CSV
# =============================================================================

csv_data = data.to_csv(
    index=False
)

st.download_button(
    label="Download simulation data (CSV)",
    data=csv_data,
    file_name="Lu177_PSMA_Interactive_Explorer_timeseries.csv",
    mime="text/csv"
)


# =============================================================================
# 50. COMBINED 600-DPI PNG
# =============================================================================

fig_all, axes = plt.subplots(
    2,
    2,
    figsize=(14, 9)
)


# =============================================================================
# 51. DOSE RATE — EXPORT FIGURE
# =============================================================================

axes[0, 0].plot(
    time_days,
    physical_rate_gy_h,
    linewidth=1.7,
    label="Physical"
)

axes[0, 0].plot(
    time_days,
    effective_rate_gy_h,
    linewidth=1.7,
    label="Effective"
)

axes[0, 0].axhline(
    critical_rate,
    linestyle="--",
    linewidth=1.0,
    label="Critical rate"
)

for t_admin in treatment_times:

    axes[0, 0].axvline(
        t_admin,
        linestyle=":",
        linewidth=0.6,
        alpha=0.5
    )

axes[0, 0].set_title(
    "Lu-177 dose rate"
)

axes[0, 0].set_xlabel(
    "Time (days)"
)

axes[0, 0].set_ylabel(
    "Dose rate (Gy/h)"
)

axes[0, 0].grid(
    alpha=0.2
)

axes[0, 0].legend()


# =============================================================================
# 52. TUMOUR BURDEN — EXPORT FIGURE
# =============================================================================

axes[0, 1].plot(
    time_days,
    total,
    linewidth=2.0,
    label="Total"
)

axes[0, 1].plot(
    time_days,
    sensitive,
    linewidth=1.3,
    linestyle="--",
    label="Sensitive"
)

axes[0, 1].plot(
    time_days,
    resistant,
    linewidth=1.3,
    linestyle=":",
    label="Resistant"
)

for t_admin in treatment_times:

    axes[0, 1].axvline(
        t_admin,
        linestyle=":",
        linewidth=0.6,
        alpha=0.5
    )

axes[0, 1].set_title(
    "Metastatic tumour burden"
)

axes[0, 1].set_xlabel(
    "Time (days)"
)

axes[0, 1].set_ylabel(
    "Burden (mL)"
)

axes[0, 1].grid(
    alpha=0.2
)

axes[0, 1].legend()


# =============================================================================
# 53. SENSITIVE / RESISTANT — EXPORT FIGURE
# =============================================================================

axes[1, 0].plot(
    time_days,
    sensitive,
    linewidth=1.6,
    label="Sensitive"
)

axes[1, 0].plot(
    time_days,
    resistant,
    linewidth=1.6,
    label="Resistant"
)

axes[1, 0].set_title(
    "Sensitive vs resistant burden"
)

axes[1, 0].set_xlabel(
    "Time (days)"
)

axes[1, 0].set_ylabel(
    "Burden (mL)"
)

axes[1, 0].grid(
    alpha=0.2
)

axes[1, 0].legend()


# =============================================================================
# 54. TCP — EXPORT FIGURE
# =============================================================================

axes[1, 1].plot(
    time_days,
    tcp * 100.0,
    linewidth=2.0,
    label="TCP"
)

axes[1, 1].axhline(
    50.0,
    linestyle="--",
    linewidth=1.0,
    label="50%"
)

for t_admin in treatment_times:

    axes[1, 1].axvline(
        t_admin,
        linestyle=":",
        linewidth=0.6,
        alpha=0.5
    )

axes[1, 1].set_title(
    "Exploratory tumour control probability"
)

axes[1, 1].set_xlabel(
    "Time (days)"
)

axes[1, 1].set_ylabel(
    "TCP (%)"
)

axes[1, 1].set_ylim(
    0,
    100
)

axes[1, 1].grid(
    alpha=0.2
)

axes[1, 1].legend()


# =============================================================================
# 55. EXPORT FIGURE TITLE
# =============================================================================

fig_all.suptitle(
    "Lu-177 PSMA — Metastatic Tumour Dynamics Explorer",
    fontsize=14,
    fontweight="bold"
)

fig_all.tight_layout(
    rect=[
        0,
        0,
        1,
        0.96
    ]
)


# =============================================================================
# 56. PNG DOWNLOAD BUTTON
# =============================================================================

png_buffer = io.BytesIO()

fig_all.savefig(
    png_buffer,
    format="png",
    dpi=600,
    bbox_inches="tight"
)

png_buffer.seek(0)

st.download_button(
    label="Download plots (600 dpi PNG)",
    data=png_buffer,
    file_name="Lu177_PSMA_Interactive_Explorer_600dpi.png",
    mime="image/png"
)


# =============================================================================
# 57. CLOSE FIGURES
# =============================================================================

plt.close(
    fig_rate
)

plt.close(
    fig_burden
)

plt.close(
    fig_population
)

plt.close(
    fig_tcp
)

plt.close(
    fig_all
)
