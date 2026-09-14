# =============================================================================
# LU-177 PSMA — INTERACTIVE METASTATIC TUMOUR DYNAMICS EXPLORER
# STREAMLIT VERSION
# =============================================================================
#
# Exploratory mechanistic model
#
# Outputs:
#   1. Physical / effective dose rate
#   2. Total metastatic tumour burden
#   3. Sensitive / resistant populations
#   4. Resistant fraction
#   5. Exploratory TCP
#
# New:
#   - Initial metastatic tumour burden affects average tumour dose rate
#   - Tumour uptake (%) affects average tumour dose rate
#
# Reference condition:
#   Initial burden = 234 mL
#   Tumour uptake = 1.0%
#
# IMPORTANT:
# This is an exploratory mechanistic model and is NOT a clinically validated
# patient-specific TCP or dosimetry model.
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
    page_title="Lu-177 PSMA — Metastatic Tumour Dynamics",
    page_icon="☢️",
    layout="wide"
)


# =============================================================================
# 2. MODEL CONSTANTS
# =============================================================================

LU177_HALF_LIFE_DAYS = 6.647

LU177_LAMBDA_PER_DAY = (
    np.log(2.0)
    / LU177_HALF_LIFE_DAYS
)


DOSE_PER_GBQ_GY = 0.50


REFERENCE_BURDEN_ML = 234.0

REFERENCE_TUMOUR_UPTAKE_PERCENT = 1.0


DT_DAYS = 0.05

FOLLOW_UP_DAYS = 60.0


# =============================================================================
# 3. DEFAULT BIOLOGICAL PARAMETERS
# =============================================================================

DEFAULT_INITIAL_METASTATIC_BURDEN_ML = 234.0


DEFAULT_TUMOUR_UPTAKE_PERCENT = 1.0

TUMOUR_UPTAKE_MIN_PERCENT = 0.1

TUMOUR_UPTAKE_MAX_PERCENT = 10.0


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
# 4. TCP MODEL
# =============================================================================

INITIAL_TCP = 0.10

INITIAL_CLONOGENIC_BURDEN = (
    -np.log(INITIAL_TCP)
)


# =============================================================================
# 5. FUNCTIONS
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


def calculate_physical_dose_rate(
    time_days,
    activity_gbq,
    n_cycles,
    interval_days,
    initial_burden_ml,
    tumour_uptake_percent
):
    """
    Average metastatic tumour physical dose rate.

    Reference condition:

        burden = 234 mL
        uptake = 1%

    gives the original:

        dose rate = activity × 0.50 Gy/day
    """

    treatment_times = make_treatment_schedule(
        n_cycles,
        interval_days
    )

    dose_rate_gy_day = np.zeros_like(
        time_days,
        dtype=float
    )

    uptake_scaling = (
        tumour_uptake_percent
        / REFERENCE_TUMOUR_UPTAKE_PERCENT
    )

    burden_scaling = (
        REFERENCE_BURDEN_ML
        / max(
            initial_burden_ml,
            1e-12
        )
    )

    tumour_dose_scaling = (
        uptake_scaling
        * burden_scaling
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
            time_days,
            dtype=float
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
            * tumour_dose_scaling
        )

    return dose_rate_gy_day


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

        dose_interval_gy = (
            physical_dose_rate_gy_day[i]
            * dt
        )

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

    above_critical = (
        physical_rate_gy_h
        >= critical_dose_rate_gy_h
    )

    time_above_critical = (
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
            time_above_critical
    }


# =============================================================================
# 6. SIDEBAR
# =============================================================================

st.title(
    "Lu-177 PSMA — Metastatic Tumour Dynamics Explorer"
)

st.caption(
    "Exploratory mechanistic model — not a clinically validated "
    "patient-specific dosimetry or TCP model."
)


st.sidebar.header(
    "Model controls"
)


# =============================================================================
# 7. CONTROLS
# =============================================================================

initial_burden_ml = st.sidebar.slider(
    "Initial metastatic burden (mL)",
    min_value=10.0,
    max_value=1500.0,
    value=DEFAULT_INITIAL_METASTATIC_BURDEN_ML,
    step=1.0
)


tumour_uptake_percent = st.sidebar.slider(
    "Tumour uptake (%)",
    min_value=TUMOUR_UPTAKE_MIN_PERCENT,
    max_value=TUMOUR_UPTAKE_MAX_PERCENT,
    value=DEFAULT_TUMOUR_UPTAKE_PERCENT,
    step=0.1
)


alpha = st.sidebar.slider(
    "Alpha (Gy⁻¹)",
    min_value=ALPHA_MIN,
    max_value=ALPHA_MAX,
    value=DEFAULT_ALPHA,
    step=0.005,
    format="%.3f"
)


trep_days = st.sidebar.slider(
    "Trep (days)",
    min_value=10.0,
    max_value=100.0,
    value=DEFAULT_TREP_DAYS,
    step=1.0
)


sensitive_fraction = st.sidebar.slider(
    "Sensitive fraction",
    min_value=0.50,
    max_value=0.95,
    value=DEFAULT_SENSITIVE_FRACTION,
    step=0.01
)


resistant_tk_multiplier = st.sidebar.slider(
    "Tk / Trep",
    min_value=1.20,
    max_value=1.80,
    value=DEFAULT_RESISTANT_TK_MULTIPLIER,
    step=0.01
)


resistant_radio_factor = st.sidebar.slider(
    "Resistant radio factor",
    min_value=0.05,
    max_value=1.00,
    value=DEFAULT_RESISTANT_RADIO_FACTOR,
    step=0.01
)


repopulation_kickoff_days = st.sidebar.slider(
    "Repopulation kickoff (days)",
    min_value=3.0,
    max_value=10.0,
    value=DEFAULT_REPOPULATION_KICKOFF_DAYS,
    step=0.5
)


activity_gbq = st.sidebar.slider(
    "Activity / cycle (GBq)",
    min_value=1.0,
    max_value=15.0,
    value=DEFAULT_ACTIVITY_GBQ,
    step=0.1
)


n_cycles = st.sidebar.slider(
    "Number of cycles",
    min_value=1,
    max_value=8,
    value=DEFAULT_N_CYCLES,
    step=1
)


cycle_interval_days = st.sidebar.slider(
    "Cycle interval (days)",
    min_value=1.0,
    max_value=42.0,
    value=DEFAULT_CYCLE_INTERVAL_DAYS,
    step=1.0
)


gamma = st.sidebar.slider(
    "Effectiveness gamma",
    min_value=0.25,
    max_value=2.00,
    value=DEFAULT_EFFECTIVENESS_GAMMA,
    step=0.05
)


# =============================================================================
# 8. RESET
# =============================================================================

if st.sidebar.button(
    "Reset all controls"
):

    st.rerun()


# =============================================================================
# 9. MODEL CALCULATION
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
# 10. CRITICAL DOSE RATE
# =============================================================================

critical_rate = (
    calculate_critical_dose_rate(
        alpha,
        trep_days
    )
)


# =============================================================================
# 11. PHYSICAL DOSE RATE
# =============================================================================

physical_dose_rate = (
    calculate_physical_dose_rate(
        time_days=time_days,
        activity_gbq=activity_gbq,
        n_cycles=n_cycles,
        interval_days=cycle_interval_days,
        initial_burden_ml=initial_burden_ml,
        tumour_uptake_percent=tumour_uptake_percent
    )
)


# =============================================================================
# 12. EFFECTIVE DOSE RATE
# =============================================================================

effective_dose_rate = (
    calculate_effective_dose_rate(
        physical_dose_rate,
        critical_rate,
        gamma
    )
)


# =============================================================================
# 13. BIOLOGICAL MODEL
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
# 14. TCP
# =============================================================================

tcp = calculate_tcp(
    total,
    initial_burden_ml
)


# =============================================================================
# 15. SUMMARY
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
# 16. DERIVED VALUES
# =============================================================================

physical_rate_gy_h = (
    physical_dose_rate
    / 24.0
)

effective_rate_gy_h = (
    effective_dose_rate
    / 24.0
)

resistant_fraction = np.divide(
    resistant,
    total,
    out=np.zeros_like(
        resistant
    ),
    where=total > 0
)

tumour_localised_activity_gbq = (
    activity_gbq
    * tumour_uptake_percent
    / 100.0
)

dose_rate_scaling = (
    tumour_uptake_percent
    / REFERENCE_TUMOUR_UPTAKE_PERCENT
    * REFERENCE_BURDEN_ML
    / initial_burden_ml
)


# =============================================================================
# 17. TOP METRICS
# =============================================================================

st.subheader(
    "Key model results"
)

m1, m2, m3, m4, m5 = st.columns(5)

with m1:

    st.metric(
        "Tumour-localised activity",
        f"{tumour_localised_activity_gbq:.3f} GBq/cycle"
    )


with m2:

    st.metric(
        "Peak physical dose rate",
        f"{summary['peak_physical_rate']:.4f} Gy/h"
    )


with m3:

    st.metric(
        "Maximum TCP",
        f"{100*summary['maximum_tcp']:.1f}%"
    )


with m4:

    st.metric(
        "Minimum tumour burden",
        f"{summary['minimum_burden']:.1f} mL"
    )


with m5:

    st.metric(
        "Final tumour reduction",
        f"{100*summary['burden_reduction']:.1f}%"
    )


# =============================================================================
# 18. DOSE RATE PLOT
# =============================================================================

st.subheader(
    "1. Physical and effective dose rate"
)

fig_rate, ax_rate = plt.subplots(
    figsize=(11, 4.8)
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

for treatment_time in treatment_times:

    ax_rate.axvline(
        treatment_time,
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

st.pyplot(
    fig_rate,
    use_container_width=True
)

plt.close(
    fig_rate
)


# =============================================================================
# 19. TUMOUR BURDEN
# =============================================================================

st.subheader(
    "2. Metastatic tumour burden"
)

fig_burden, ax_burden = plt.subplots(
    figsize=(11, 4.8)
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
    linewidth=1.3,
    linestyle="--",
    label="Sensitive"
)

ax_burden.plot(
    time_days,
    resistant,
    linewidth=1.3,
    linestyle=":",
    label="Resistant"
)

for treatment_time in treatment_times:

    ax_burden.axvline(
        treatment_time,
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

st.pyplot(
    fig_burden,
    use_container_width=True
)

plt.close(
    fig_burden
)


# =============================================================================
# 20. SENSITIVE / RESISTANT
# =============================================================================

st.subheader(
    "3. Sensitive vs resistant tumour populations"
)

fig_population, ax_population = plt.subplots(
    figsize=(11, 4.8)
)

ax_population.plot(
    time_days,
    sensitive,
    linewidth=1.7,
    label="Sensitive"
)

ax_population.plot(
    time_days,
    resistant,
    linewidth=1.7,
    label="Resistant"
)

ax_population.plot(
    time_days,
    total,
    linewidth=1.3,
    linestyle="--",
    label="Total"
)

for treatment_time in treatment_times:

    ax_population.axvline(
        treatment_time,
        linestyle=":",
        linewidth=0.7,
        alpha=0.5
    )

ax_population.set_title(
    "Sensitive vs resistant burden"
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

st.pyplot(
    fig_population,
    use_container_width=True
)

plt.close(
    fig_population
)


# =============================================================================
# 21. TCP
# =============================================================================

st.subheader(
    "4. Exploratory tumour control probability"
)

fig_tcp, ax_tcp = plt.subplots(
    figsize=(11, 4.8)
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

for treatment_time in treatment_times:

    ax_tcp.axvline(
        treatment_time,
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

st.pyplot(
    fig_tcp,
    use_container_width=True
)

plt.close(
    fig_tcp
)


# =============================================================================
# 22. MODEL SUMMARY
# =============================================================================

st.subheader(
    "Model summary"
)

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        f"""
**Tumour**

- Initial burden: **{summary['initial_burden']:.1f} mL**
- Tumour uptake: **{tumour_uptake_percent:.1f}%**
- Tumour-localised activity: **{tumour_localised_activity_gbq:.3f} GBq/cycle**
- Minimum burden: **{summary['minimum_burden']:.1f} mL**
- Time of minimum: **{summary['time_of_minimum_burden']:.1f} days**
- Final burden: **{summary['final_burden']:.1f} mL**
- Final reduction: **{100*summary['burden_reduction']:.1f}%**
"""
    )


with col2:

    st.markdown(
        f"""
**Dose rate**

- Critical dose rate: **{summary['critical_rate']:.4f} Gy/h**
- Peak physical rate: **{summary['peak_physical_rate']:.4f} Gy/h**
- Peak effective rate: **{summary['peak_effective_rate']:.4f} Gy/h**
- Time above critical: **{summary['time_above_critical']:.1f} h**
- Cumulative physical dose: **{summary['cumulative_physical_dose']:.2f} Gy**
- Cumulative effective dose: **{summary['cumulative_effective_dose']:.2f} Gy**
"""
    )


with col3:

    st.markdown(
        f"""
**Tumour response**

- Initial TCP: **{100*summary['initial_tcp']:.1f}%**
- Maximum TCP: **{100*summary['maximum_tcp']:.1f}%**
- Time of maximum TCP: **{summary['time_of_max_tcp']:.1f} days**
- Final TCP: **{100*summary['final_tcp']:.1f}%**
- Initial resistant fraction: **{100*summary['initial_resistant_fraction']:.1f}%**
- Final resistant fraction: **{100*summary['final_resistant_fraction']:.1f}%**
- Maximum resistant fraction: **{100*summary['maximum_resistant_fraction']:.1f}%**
"""
    )


# =============================================================================
# 23. DOSE SCALING INFORMATION
# =============================================================================

with st.expander(
    "Tumour uptake and burden dose-rate scaling"
):

    st.markdown(
        """
The average metastatic tumour dose rate is scaled relative to a reference
condition of **234 mL tumour burden and 1.0% tumour uptake**.

The scaling is:

\[
Dose\\ Rate =
Dose\\ Rate_{reference}
\\times
\\frac{Uptake}{1\\%}
\\times
\\frac{234\\ mL}{Tumour\\ burden}
\]

Therefore:

- Increasing tumour uptake increases the tumour dose rate.
- Increasing metastatic tumour burden decreases the average tumour dose rate.
- Decreasing metastatic tumour burden increases the average tumour dose rate.
"""
    )

    st.write(
        f"Current dose-rate scaling factor: "
        f"**{dose_rate_scaling:.3f} × reference**"
    )


# =============================================================================
# 24. DETAILED MODEL ASSUMPTIONS
# =============================================================================

with st.expander(
    "Model assumptions and limitations"
):

    st.markdown(
        """
### Tumour uptake

The tumour uptake parameter represents the **fraction of administered
activity assumed to localise within the total metastatic tumour burden**.

It is not `%ID/g` and does not represent lesion-specific PSMA uptake.

### Tumour burden

The model treats the metastatic burden as an aggregate tumour volume.

The burden is used to scale the average activity concentration and therefore
the exploratory average tumour dose rate.

### Dose

The model uses a nominal dose conversion of:

**0.50 Gy/day per GBq under the reference condition of 234 mL and 1% uptake.**

### Biology

The tumour contains sensitive and resistant populations.

The sensitive and resistant populations have different radiosensitivity and
repopulation characteristics.

### TCP

TCP is an exploratory normalized quantity and is not a clinically validated
probability of tumour control.

### Clinical limitation

The model does not yet explicitly include lesion-specific:

- PSMA uptake
- time-activity curves
- residence time
- absorbed dose
- tumour heterogeneity
- cross-dose
- spatial dose distribution
"""
    )


# =============================================================================
# 25. DOWNLOAD CSV
# =============================================================================

data = pd.DataFrame({

    "Time_days":
        time_days,

    "Initial_metastatic_burden_mL":
        np.full_like(
            time_days,
            initial_burden_ml,
            dtype=float
        ),

    "Tumour_uptake_percent":
        np.full_like(
            time_days,
            tumour_uptake_percent,
            dtype=float
        ),

    "Tumour_localised_activity_GBq":
        np.full_like(
            time_days,
            tumour_localised_activity_gbq,
            dtype=float
        ),

    "Dose_rate_scaling":
        np.full_like(
            time_days,
            dose_rate_scaling,
            dtype=float
        ),

    "Physical_dose_rate_Gy_day":
        physical_dose_rate,

    "Physical_dose_rate_Gy_h":
        physical_rate_gy_h,

    "Effective_dose_rate_Gy_day":
        effective_dose_rate,

    "Effective_dose_rate_Gy_h":
        effective_rate_gy_h,

    "Sensitive_burden_mL":
        sensitive,

    "Resistant_burden_mL":
        resistant,

    "Total_burden_mL":
        total,

    "Resistant_fraction":
        resistant_fraction,

    "TCP":
        tcp
})


csv_data = data.to_csv(
    index=False
).encode(
    "utf-8"
)


st.download_button(
    label="Download model results (CSV)",
    data=csv_data,
    file_name="Lu177_PSMA_Interactive_Explorer_results.csv",
    mime="text/csv"
)


# =============================================================================
# 26. DOWNLOAD FIGURES
# =============================================================================

def figure_to_png_bytes(
    figure
):

    buffer = io.BytesIO()

    figure.savefig(
        buffer,
        format="png",
        dpi=600,
        bbox_inches="tight"
    )

    buffer.seek(
        0
    )

    return buffer.getvalue()


# -----------------------------------------------------------------------------
# Dose-rate figure
# -----------------------------------------------------------------------------

fig_download_rate, ax = plt.subplots(
    figsize=(11, 5)
)

ax.plot(
    time_days,
    physical_rate_gy_h,
    linewidth=1.8,
    label="Physical"
)

ax.plot(
    time_days,
    effective_rate_gy_h,
    linewidth=1.8,
    label="Effective"
)

ax.axhline(
    critical_rate,
    linestyle="--",
    linewidth=1.0,
    label="Critical rate"
)

for treatment_time in treatment_times:

    ax.axvline(
        treatment_time,
        linestyle=":",
        linewidth=0.7,
        alpha=0.5
    )

ax.set_title(
    "Lu-177 PSMA — Physical and Effective Dose Rate"
)

ax.set_xlabel(
    "Time (days)"
)

ax.set_ylabel(
    "Dose rate (Gy/h)"
)

ax.grid(
    alpha=0.2
)

ax.legend()

fig_download_rate.tight_layout()


st.download_button(
    label="Download dose-rate plot (600 dpi PNG)",
    data=figure_to_png_bytes(
        fig_download_rate
    ),
    file_name="Lu177_PSMA_dose_rate.png",
    mime="image/png"
)

plt.close(
    fig_download_rate
)


# -----------------------------------------------------------------------------
# Burden figure
# -----------------------------------------------------------------------------

fig_download_burden, ax = plt.subplots(
    figsize=(11, 5)
)

ax.plot(
    time_days,
    total,
    linewidth=2.0,
    label="Total"
)

ax.plot(
    time_days,
    sensitive,
    linewidth=1.4,
    label="Sensitive"
)

ax.plot(
    time_days,
    resistant,
    linewidth=1.4,
    label="Resistant"
)

for treatment_time in treatment_times:

    ax.axvline(
        treatment_time,
        linestyle=":",
        linewidth=0.7,
        alpha=0.5
    )

ax.set_title(
    "Lu-177 PSMA — Metastatic Tumour Burden"
)

ax.set_xlabel(
    "Time (days)"
)

ax.set_ylabel(
    "Burden (mL)"
)

ax.grid(
    alpha=0.2
)

ax.legend()

fig_download_burden.tight_layout()


st.download_button(
    label="Download tumour-burden plot (600 dpi PNG)",
    data=figure_to_png_bytes(
        fig_download_burden
    ),
    file_name="Lu177_PSMA_tumour_burden.png",
    mime="image/png"
)

plt.close(
    fig_download_burden
)


# -----------------------------------------------------------------------------
# Population figure
# -----------------------------------------------------------------------------

fig_download_population, ax = plt.subplots(
    figsize=(11, 5)
)

ax.plot(
    time_days,
    sensitive,
    linewidth=1.7,
    label="Sensitive"
)

ax.plot(
    time_days,
    resistant,
    linewidth=1.7,
    label="Resistant"
)

ax.plot(
    time_days,
    total,
    linewidth=1.3,
    linestyle="--",
    label="Total"
)

for treatment_time in treatment_times:

    ax.axvline(
        treatment_time,
        linestyle=":",
        linewidth=0.7,
        alpha=0.5
    )

ax.set_title(
    "Lu-177 PSMA — Sensitive and Resistant Tumour Populations"
)

ax.set_xlabel(
    "Time (days)"
)

ax.set_ylabel(
    "Burden (mL)"
)

ax.grid(
    alpha=0.2
)

ax.legend()

fig_download_population.tight_layout()


st.download_button(
    label="Download population plot (600 dpi PNG)",
    data=figure_to_png_bytes(
        fig_download_population
    ),
    file_name="Lu177_PSMA_sensitive_resistant.png",
    mime="image/png"
)

plt.close(
    fig_download_population
)


# -----------------------------------------------------------------------------
# TCP figure
# -----------------------------------------------------------------------------

fig_download_tcp, ax = plt.subplots(
    figsize=(11, 5)
)

ax.plot(
    time_days,
    tcp * 100.0,
    linewidth=2.0,
    label="TCP"
)

ax.axhline(
    50.0,
    linestyle="--",
    linewidth=1.0,
    label="50%"
)

for treatment_time in treatment_times:

    ax.axvline(
        treatment_time,
        linestyle=":",
        linewidth=0.7,
        alpha=0.5
    )

ax.set_title(
    "Lu-177 PSMA — Exploratory Tumour Control Probability"
)

ax.set_xlabel(
    "Time (days)"
)

ax.set_ylabel(
    "TCP (%)"
)

ax.set_ylim(
    0,
    100
)

ax.grid(
    alpha=0.2
)

ax.legend()

fig_download_tcp.tight_layout()


st.download_button(
    label="Download TCP plot (600 dpi PNG)",
    data=figure_to_png_bytes(
        fig_download_tcp
    ),
    file_name="Lu177_PSMA_TCP.png",
    mime="image/png"
)

plt.close(
    fig_download_tcp
)


# =============================================================================
# 27. FOOTER
# =============================================================================

st.divider()

st.caption(
    "Lu-177 PSMA metastatic tumour dynamics explorer. "
    "Exploratory research model; not for clinical treatment decisions."
)
