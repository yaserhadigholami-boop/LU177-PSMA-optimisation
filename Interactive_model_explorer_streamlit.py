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
    page_title="Lu-177 PSMA Treatment Explorer",
    page_icon="☢️",
    layout="wide",
)


# =============================================================================
# MODEL CONSTANTS
# =============================================================================

REFERENCE_METASTATIC_BURDEN_ML = 234.0

DEFAULT_TUMOUR_UPTAKE_PERCENT = 1.0
TUMOUR_UPTAKE_MIN_PERCENT = 0.1
TUMOUR_UPTAKE_MAX_PERCENT = 10.0

LU177_HALF_LIFE_DAYS = 6.647

# Reference dose conversion:
# 0.50 Gy per GBq at the reference tumour burden / uptake condition.
DOSE_PER_GBQ_GY = 0.50

FOLLOW_UP_DAYS = 60.0

DT_DAYS = 0.05

# Exploratory TCP normalisation
INITIAL_TCP = 0.10
INITIAL_CLONOGENIC_BURDEN = -np.log(INITIAL_TCP)


# =============================================================================
# DEFAULT EXPLORER VALUES
# =============================================================================

DEFAULT_INITIAL_BURDEN = 234.0
DEFAULT_ALPHA = 0.10
DEFAULT_TREP = 30.0
DEFAULT_SENSITIVE_FRACTION = 0.75
DEFAULT_RESISTANT_TK_MULTIPLIER = 1.50
DEFAULT_RESISTANT_RADIO_FACTOR = 0.25
DEFAULT_REPOPULATION_KICKOFF = 5.0
DEFAULT_ACTIVITY = 7.4
DEFAULT_N_CYCLES = 4
DEFAULT_INTERVAL = 7.0
DEFAULT_GAMMA = 1.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def integrate_trapezoid(y, x):
    """
    NumPy compatibility helper.

    Newer NumPy versions use np.trapezoid().
    Older versions use np.trapz().
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)

    return np.trapz(y, x)


def calculate_dose_scaling(
    initial_burden_ml,
    tumour_uptake_percent,
):
    """
    Phenomenological tumour-dose scaling.

    The reference condition is:

        234 mL metastatic tumour burden
        1% total tumour uptake

    Under this reference condition the original dose model is preserved.

    Scaling assumptions:

        uptake_scale = uptake / 1%
        burden_scale = 234 / actual burden

    Therefore:

        dose_scale =
            uptake_scale * burden_scale
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

    return {
        "uptake_scale": uptake_scale,
        "burden_scale": burden_scale,
        "dose_scale": dose_scale,
    }


def calculate_treatment_times(
    n_cycles,
    interval_days,
):
    """
    Return treatment administration times.
    """

    return np.array(
        [
            i * interval_days
            for i in range(n_cycles)
        ],
        dtype=float,
    )


def calculate_physical_dose_rate(
    time_days,
    activity_gbq,
    n_cycles,
    interval_days,
    dose_scale,
):
    """
    Calculate total physical dose rate from all Lu-177 administrations.

    Output:
        Gy/day
    """

    treatment_times = calculate_treatment_times(
        n_cycles,
        interval_days,
    )

    lambda_lu = np.log(2.0) / LU177_HALF_LIFE_DAYS

    dose_rate = np.zeros_like(
        time_days,
        dtype=float,
    )

    for t_admin in treatment_times:

        elapsed = time_days - t_admin

        active = elapsed >= 0.0

        activity = np.zeros_like(
            time_days,
            dtype=float,
        )

        activity[active] = (
            activity_gbq
            * np.exp(
                -lambda_lu * elapsed[active]
            )
        )

        dose_rate += (
            activity
            * DOSE_PER_GBQ_GY
            * dose_scale
        )

    return dose_rate


def calculate_critical_dose_rate(
    alpha,
    trep_days,
):
    """
    Critical dose rate:

        Rcrit = ln(2) / (alpha * Trep)

    Returned in Gy/hour.
    """

    trep_hours = trep_days * 24.0

    rcrit_gy_h = (
        np.log(2.0)
        / (alpha * trep_hours)
    )

    return rcrit_gy_h


def calculate_effective_dose_rate(
    physical_dose_rate_gy_day,
    critical_dose_rate_gy_h,
    gamma,
):
    """
    Apply the dose-rate effectiveness model.

    R = physical rate / critical rate

    E = min(1, R^gamma)

    Effective dose rate:

        Reffective = Rphysical * E

    Returns:
        effective dose rate in Gy/day
        effectiveness factor
        physical-to-critical ratio
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
        where=critical_rate_gy_day > 0,
    )

    effectiveness = np.minimum(
        1.0,
        np.power(
            np.maximum(ratio, 0.0),
            gamma,
        ),
    )

    effective_dose_rate = (
        physical_dose_rate_gy_day
        * effectiveness
    )

    return (
        effective_dose_rate,
        effectiveness,
        ratio,
    )


# =============================================================================
# TUMOUR DYNAMICS
# =============================================================================

def calculate_tumour_dynamics(
    time_days,
    physical_dose_rate_gy_day,
    initial_burden_ml,
    alpha,
    trep_days,
    sensitive_fraction,
    resistant_tk_multiplier,
    resistant_radio_factor,
    repopulation_kickoff_days,
):
    """
    Two-compartment tumour model:

        Sensitive tumour
        Resistant tumour

    Growth occurs after the repopulation kickoff.

    Radiation killing is modelled using an LQ formulation.

    Resistant cells have:

        lower alpha
        lower beta
        faster/slower growth according to Tk multiplier
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
        * DEFAULT_BETA_ALPHA
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

        s = sensitive[i - 1]
        r = resistant[i - 1]

        # ---------------------------------------------------------
        # REPUPULATION
        # ---------------------------------------------------------

        if (
            time_days[i]
            >= repopulation_kickoff_days
        ):

            s *= np.exp(
                sensitive_growth_rate
                * dt
            )

            r *= np.exp(
                resistant_growth_rate
                * dt
            )

        # ---------------------------------------------------------
        # RADIATION DOSE
        # ---------------------------------------------------------

        dose_interval_gy = (
            physical_dose_rate_gy_day[i]
            * dt
        )

        # ---------------------------------------------------------
        # LQ SURVIVAL
        # ---------------------------------------------------------

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

        s *= survival_sensitive
        r *= survival_resistant

        sensitive[i] = max(
            s,
            0.0,
        )

        resistant[i] = max(
            r,
            0.0,
        )

    total = (
        sensitive
        + resistant
    )

    resistant_composition = np.divide(
        resistant,
        total,
        out=np.zeros_like(resistant),
        where=total > 0,
    )

    residual_resistant_burden_percent = np.divide(
        resistant,
        initial_resistant,
        out=np.zeros_like(resistant),
        where=initial_resistant > 0,
    ) * 100.0

    return {
        "sensitive": sensitive,
        "resistant": resistant,
        "total": total,
        "resistant_composition": (
            resistant_composition
        ),
        "residual_resistant_burden_percent": (
            residual_resistant_burden_percent
        ),
        "initial_sensitive": initial_sensitive,
        "initial_resistant": initial_resistant,
    }


# =============================================================================
# TCP
# =============================================================================

def calculate_tcp(
    total_burden_ml,
    initial_burden_ml,
):
    """
    Exploratory normalised TCP model.

    Initial TCP is fixed at 10%.

    The relative tumour burden determines the
    remaining clonogenic burden.
    """

    relative_burden = np.divide(
        total_burden_ml,
        initial_burden_ml,
        out=np.zeros_like(
            total_burden_ml
        ),
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
# SUMMARY CALCULATIONS
# =============================================================================

def calculate_summary(
    time_days,
    physical_dose_rate_gy_day,
    effective_dose_rate_gy_day,
    total_burden_ml,
    resistant_burden_ml,
    tcp,
    critical_dose_rate_gy_h,
    initial_burden_ml,
    initial_resistant_burden_ml,
):
    """
    Calculate headline and secondary model outputs.
    """

    physical_dose_rate_gy_h = (
        physical_dose_rate_gy_day
        / 24.0
    )

    effective_dose_rate_gy_h = (
        effective_dose_rate_gy_day
        / 24.0
    )

    # ---------------------------------------------------------
    # CUMULATIVE DOSE
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # MINIMUM TUMOUR BURDEN
    # ---------------------------------------------------------

    min_idx = int(
        np.argmin(total_burden_ml)
    )

    minimum_tumour_burden_ml = (
        total_burden_ml[min_idx]
    )

    minimum_tumour_burden_day = (
        time_days[min_idx]
    )

    # ---------------------------------------------------------
    # MAXIMUM TCP
    # ---------------------------------------------------------

    max_tcp_idx = int(
        np.argmax(tcp)
    )

    maximum_tcp = tcp[max_tcp_idx]

    maximum_tcp_day = (
        time_days[max_tcp_idx]
    )

    # ---------------------------------------------------------
    # RESIDUAL RESISTANT BURDEN
    # ---------------------------------------------------------

    final_resistant_burden_ml = (
        resistant_burden_ml[-1]
    )

    residual_resistant_burden_percent = (
        final_resistant_burden_ml
        / initial_resistant_burden_ml
        * 100.0
        if initial_resistant_burden_ml > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # CRITICAL DOSE RATE
    # ---------------------------------------------------------

    above_critical = (
        physical_dose_rate_gy_h
        >= critical_dose_rate_gy_h
    )

    time_above_critical_days = (
        np.sum(above_critical)
        * DT_DAYS
    )

    # ---------------------------------------------------------
    # PEAK DOSE RATE
    # ---------------------------------------------------------

    peak_idx = int(
        np.argmax(
            physical_dose_rate_gy_h
        )
    )

    peak_physical_rate = (
        physical_dose_rate_gy_h[peak_idx]
    )

    peak_physical_rate_day = (
        time_days[peak_idx]
    )

    # ---------------------------------------------------------
    # FINAL VALUES
    # ---------------------------------------------------------

    final_total_burden = (
        total_burden_ml[-1]
    )

    final_tcp = tcp[-1]

    # ---------------------------------------------------------
    # CHANGE FROM BASELINE
    # ---------------------------------------------------------

    tumour_reduction_percent = (
        1.0
        - (
            minimum_tumour_burden_ml
            / initial_burden_ml
        )
    ) * 100.0

    return {
        "minimum_tumour_burden_ml":
            minimum_tumour_burden_ml,

        "minimum_tumour_burden_day":
            minimum_tumour_burden_day,

        "maximum_tcp":
            maximum_tcp,

        "maximum_tcp_day":
            maximum_tcp_day,

        "final_tcp":
            final_tcp,

        "final_total_burden_ml":
            final_total_burden,

        "final_resistant_burden_ml":
            final_resistant_burden_ml,

        "residual_resistant_burden_percent":
            residual_resistant_burden_percent,

        "critical_dose_rate_gy_h":
            critical_dose_rate_gy_h,

        "peak_physical_rate_gy_h":
            peak_physical_rate,

        "peak_physical_rate_day":
            peak_physical_rate_day,

        "time_above_critical_days":
            time_above_critical_days,

        "cumulative_physical_dose_gy":
            cumulative_physical_dose,

        "cumulative_effective_dose_gy":
            cumulative_effective_dose,

        "tumour_reduction_percent":
            tumour_reduction_percent,
    }


# =============================================================================
# PLOT STYLING
# =============================================================================

def style_axis(ax):
    ax.grid(
        True,
        alpha=0.20,
        linewidth=0.7,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        labelsize=9,
    )

    ax.xaxis.set_major_locator(
        MaxNLocator(
            nbins=7,
            integer=True,
        )
    )


def style_legend(ax):
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="best",
    )


def add_treatment_markers(
    ax,
    treatment_times,
):
    for t in treatment_times:

        ax.axvline(
            t,
            linestyle="--",
            linewidth=0.8,
            alpha=0.30,
        )


# =============================================================================
# DEFAULT BETA PARAMETER
# =============================================================================

DEFAULT_BETA_ALPHA = 0.10


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
    <style>

    /* ---------------------------------------------------------
       GENERAL
       --------------------------------------------------------- */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }


    /* ---------------------------------------------------------
       SECTION HEADINGS
       --------------------------------------------------------- */

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 1.0rem;
        margin-bottom: 0.65rem;
    }


    /* ---------------------------------------------------------
       KEY RESULTS
       --------------------------------------------------------- */

    .key-card {
        padding: 0.15rem 0.2rem 0.45rem 0.2rem;
        min-height: 118px;
    }

    .key-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #BDBDBD;
        line-height: 1.15;
        margin-bottom: 0.25rem;
    }

    .key-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.15;
        margin-bottom: 0.40rem;
    }

    .key-status {
        display: inline-block;
        padding: 0.16rem 0.45rem;
        border-radius: 0.35rem;
        font-size: 0.68rem;
        font-weight: 600;
        line-height: 1.1;
        margin-bottom: 0.35rem;
    }

    .key-status-positive {
        background-color: rgba(46, 204, 113, 0.14);
        color: #6FCF97;
    }

    .key-status-negative {
        background-color: rgba(235, 87, 87, 0.14);
        color: #EB5757;
    }

    .key-status-neutral {
        background-color: rgba(189, 189, 189, 0.12);
        color: #BDBDBD;
    }

    .key-caption {
        font-size: 0.67rem;
        color: #858585;
        line-height: 1.2;
    }


    /* ---------------------------------------------------------
       SECONDARY METRICS
       --------------------------------------------------------- */

    .secondary-card {
        padding: 0.15rem 0.15rem 0.35rem 0.15rem;
        min-height: 55px;
    }

    .secondary-label {
        font-size: 0.68rem;
        font-weight: 500;
        color: #9E9E9E;
        line-height: 1.15;
        margin-bottom: 0.16rem;
    }

    .secondary-value {
        font-size: 1.00rem;
        font-weight: 600;
        color: #F2C94C;
        line-height: 1.15;
    }

    .secondary-unit {
        font-size: 0.68rem;
        font-weight: 500;
        color: #888888;
    }


    /* ---------------------------------------------------------
       DOWNLOAD BUTTONS
       --------------------------------------------------------- */

    .download-note {
        font-size: 0.78rem;
        color: #8A8A8A;
        margin-bottom: 0.3rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header("Model controls")

st.sidebar.markdown(
    "Adjust the treatment and tumour parameters below."
)


initial_burden_ml = st.sidebar.slider(
    "Initial metastatic burden (mL)",
    min_value=10.0,
    max_value=1500.0,
    value=DEFAULT_INITIAL_BURDEN,
    step=1.0,
)

alpha = st.sidebar.slider(
    "Alpha (Gy⁻¹)",
    min_value=0.01,
    max_value=0.50,
    value=DEFAULT_ALPHA,
    step=0.01,
)

trep_days = st.sidebar.slider(
    "Trep (days)",
    min_value=10.0,
    max_value=100.0,
    value=DEFAULT_TREP,
    step=1.0,
)

sensitive_fraction = st.sidebar.slider(
    "Sensitive fraction",
    min_value=0.50,
    max_value=0.95,
    value=DEFAULT_SENSITIVE_FRACTION,
    step=0.01,
)

resistant_tk_multiplier = st.sidebar.slider(
    "Tk / Trep",
    min_value=1.20,
    max_value=1.80,
    value=DEFAULT_RESISTANT_TK_MULTIPLIER,
    step=0.05,
)

resistant_radio_factor = st.sidebar.slider(
    "Resistant radio factor",
    min_value=0.05,
    max_value=1.00,
    value=DEFAULT_RESISTANT_RADIO_FACTOR,
    step=0.05,
)

repopulation_kickoff_days = st.sidebar.slider(
    "Repopulation kickoff (days)",
    min_value=3.0,
    max_value=10.0,
    value=DEFAULT_REPOPULATION_KICKOFF,
    step=0.5,
)

activity_gbq = st.sidebar.slider(
    "Activity / cycle (GBq)",
    min_value=1.0,
    max_value=15.0,
    value=DEFAULT_ACTIVITY,
    step=0.1,
)

n_cycles = st.sidebar.slider(
    "Number of cycles",
    min_value=1,
    max_value=8,
    value=DEFAULT_N_CYCLES,
    step=1,
)

interval_days = st.sidebar.slider(
    "Cycle interval (days)",
    min_value=1,
    max_value=42,
    value=int(DEFAULT_INTERVAL),
    step=1,
)

gamma = st.sidebar.slider(
    "Effectiveness gamma",
    min_value=0.25,
    max_value=2.00,
    value=DEFAULT_GAMMA,
    step=0.05,
)

tumour_uptake_percent = st.sidebar.slider(
    "Total tumour uptake (%)",
    min_value=TUMOUR_UPTAKE_MIN_PERCENT,
    max_value=TUMOUR_UPTAKE_MAX_PERCENT,
    value=DEFAULT_TUMOUR_UPTAKE_PERCENT,
    step=0.1,
)


# =============================================================================
# SIMULATION TIME
# =============================================================================

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
# TREATMENT TIMES
# =============================================================================

treatment_times = calculate_treatment_times(
    n_cycles,
    interval_days,
)


# =============================================================================
# DOSE SCALING
# =============================================================================

dose_scaling = calculate_dose_scaling(
    initial_burden_ml,
    tumour_uptake_percent,
)

dose_scale = dose_scaling["dose_scale"]


# =============================================================================
# PHYSICAL DOSE RATE
# =============================================================================

physical_dose_rate_gy_day = (
    calculate_physical_dose_rate(
        time_days=time_days,
        activity_gbq=activity_gbq,
        n_cycles=n_cycles,
        interval_days=interval_days,
        dose_scale=dose_scale,
    )
)

physical_dose_rate_gy_h = (
    physical_dose_rate_gy_day
    / 24.0
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
    effective_dose_rate_gy_day,
    effectiveness_factor,
    dose_rate_ratio,
) = calculate_effective_dose_rate(
    physical_dose_rate_gy_day=(
        physical_dose_rate_gy_day
    ),
    critical_dose_rate_gy_h=(
        critical_dose_rate_gy_h
    ),
    gamma=gamma,
)

effective_dose_rate_gy_h = (
    effective_dose_rate_gy_day
    / 24.0
)


# =============================================================================
# TUMOUR DYNAMICS
# =============================================================================

tumour = calculate_tumour_dynamics(
    time_days=time_days,
    physical_dose_rate_gy_day=(
        physical_dose_rate_gy_day
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

sensitive = tumour["sensitive"]
resistant = tumour["resistant"]
total_burden = tumour["total"]

resistant_composition = (
    tumour["resistant_composition"]
)

residual_resistant_burden_percent = (
    tumour[
        "residual_resistant_burden_percent"
    ]
)

initial_resistant_burden = (
    tumour["initial_resistant"]
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
    physical_dose_rate_gy_day=(
        physical_dose_rate_gy_day
    ),
    effective_dose_rate_gy_day=(
        effective_dose_rate_gy_day
    ),
    total_burden_ml=total_burden,
    resistant_burden_ml=resistant,
    tcp=tcp,
    critical_dose_rate_gy_h=(
        critical_dose_rate_gy_h
    ),
    initial_burden_ml=initial_burden_ml,
    initial_resistant_burden_ml=(
        initial_resistant_burden
    ),
)


# =============================================================================
# HEADER
# =============================================================================

st.title(
    "Lu-177 PSMA Treatment Model Explorer"
)

st.markdown(
    """
    Exploratory model linking Lu-177 PSMA dose-rate dynamics,
    tumour response, resistant tumour evolution and TCP.
    """
)


# =============================================================================
# KEY RESULTS
# =============================================================================

st.markdown(
    '<div class="section-title">Key results</div>',
    unsafe_allow_html=True,
)

key_cols = st.columns(4)


# -----------------------------------------------------------------------------
# KEY RESULT 1 — MINIMUM TUMOUR BURDEN
# -----------------------------------------------------------------------------

with key_cols[0]:

    st.markdown(
        '<div class="key-card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="key-label">'
        'Minimum tumour burden'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="key-value">'
        f'{summary["minimum_tumour_burden_ml"]:.1f} mL'
        f'</div>',
        unsafe_allow_html=True,
    )

    if (
        summary["tumour_reduction_percent"]
        > 0
    ):
        status_class = (
            "key-status-positive"
        )

        status_text = (
            f'↓ '
            f'{summary["tumour_reduction_percent"]:.1f}% '
            f'from {initial_burden_ml:.0f} mL baseline'
        )

    elif (
        summary["tumour_reduction_percent"]
        < 0
    ):
        status_class = (
            "key-status-negative"
        )

        status_text = (
            f'↑ '
            f'{abs(summary["tumour_reduction_percent"]):.1f}% '
            f'from baseline'
        )

    else:
        status_class = (
            "key-status-neutral"
        )

        status_text = "No change from baseline"

    st.markdown(
        f'<span class="key-status '
        f'{status_class}">'
        f'{status_text}'
        f'</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="key-caption">'
        f'Minimum reached at day '
        f'{summary["minimum_tumour_burden_day"]:.1f}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# KEY RESULT 2 — MAXIMUM TCP
# -----------------------------------------------------------------------------

with key_cols[1]:

    st.markdown(
        '<div class="key-card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="key-label">'
        'Maximum TCP'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="key-value">'
        f'{summary["maximum_tcp"] * 100.0:.1f}%'
        f'</div>',
        unsafe_allow_html=True,
    )

    if summary["maximum_tcp"] >= 0.50:

        status_class = (
            "key-status-positive"
        )

        status_text = (
            "≥50% TCP threshold"
        )

    else:

        status_class = (
            "key-status-negative"
        )

        status_text = (
            "<50% TCP threshold"
        )

    st.markdown(
        f'<span class="key-status '
        f'{status_class}">'
        f'{status_text}'
        f'</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="key-caption">'
        f'Maximum reached at day '
        f'{summary["maximum_tcp_day"]:.1f}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# KEY RESULT 3 — RESIDUAL RESISTANT BURDEN
# -----------------------------------------------------------------------------

with key_cols[2]:

    st.markdown(
        '<div class="key-card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="key-label">'
        'Residual resistant burden'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="key-value">'
        f'{summary["residual_resistant_burden_percent"]:.1f}%'
        f'</div>',
        unsafe_allow_html=True,
    )

    resistant_reduction = (
        100.0
        - summary[
            "residual_resistant_burden_percent"
        ]
    )

    if resistant_reduction > 0:

        status_class = (
            "key-status-positive"
        )

        status_text = (
            f'↓ '
            f'{resistant_reduction:.1f}% '
            f'from initial resistant burden'
        )

    elif resistant_reduction < 0:

        status_class = (
            "key-status-negative"
        )

        status_text = (
            f'↑ '
            f'{abs(resistant_reduction):.1f}% '
            f'from initial resistant burden'
        )

    else:

        status_class = (
            "key-status-neutral"
        )

        status_text = (
            "No change from initial resistant burden"
        )

    st.markdown(
        f'<span class="key-status '
        f'{status_class}">'
        f'{status_text}'
        f'</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="key-caption">'
        '% of initial resistant tumour burden remaining'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# KEY RESULT 4 — CUMULATIVE PHYSICAL DOSE
# -----------------------------------------------------------------------------

with key_cols[3]:

    st.markdown(
        '<div class="key-card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="key-label">'
        'Cumulative physical dose'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="key-value">'
        f'{summary["cumulative_physical_dose_gy"]:.1f} Gy'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<span class="key-status '
        'key-status-neutral">'
        'Physical dose'
        '</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="key-caption">'
        f'Across {n_cycles} treatment cycles'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# SECONDARY METRICS
# =============================================================================

st.markdown(
    '<div class="section-title">'
    'Secondary metrics'
    '</div>',
    unsafe_allow_html=True,
)

secondary_cols = st.columns(4)


# -----------------------------------------------------------------------------
# SECONDARY 1
# -----------------------------------------------------------------------------

with secondary_cols[0]:

    st.markdown(
        '<div class="secondary-card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="secondary-label">'
        'Critical dose rate'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="secondary-value">'
        f'{summary["critical_dose_rate_gy_h"]:.4f} '
        f'<span class="secondary-unit">Gy/h</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# SECONDARY 2
# -----------------------------------------------------------------------------

with secondary_cols[1]:

    st.markdown(
        '<div class="secondary-card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="secondary-label">'
        'Peak physical rate'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="secondary-value">'
        f'{summary["peak_physical_rate_gy_h"]:.4f} '
        f'<span class="secondary-unit">Gy/h</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# SECONDARY 3
# -----------------------------------------------------------------------------

with secondary_cols[2]:

    st.markdown(
        '<div class="secondary-card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="secondary-label">'
        'Time above critical rate'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="secondary-value">'
        f'{summary["time_above_critical_days"]:.1f} '
        f'<span class="secondary-unit">days</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# SECONDARY 4
# -----------------------------------------------------------------------------

with secondary_cols[3]:

    st.markdown(
        '<div class="secondary-card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="secondary-label">'
        'Effective cumulative dose'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="secondary-value">'
        f'{summary["cumulative_effective_dose_gy"]:.1f} '
        f'<span class="secondary-unit">Gy</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# DOSE SCALING INFORMATION
# =============================================================================

st.caption(
    f"Dose scaling: {dose_scale:.3f}× "
    f"(tumour uptake {tumour_uptake_percent:.1f}% | "
    f"metastatic burden {initial_burden_ml:.1f} mL | "
    f"reference {REFERENCE_METASTATIC_BURDEN_ML:.0f} mL at "
    f"{DEFAULT_TUMOUR_UPTAKE_PERCENT:.1f}% uptake)"
)


# =============================================================================
# MODEL TRAJECTORIES
# =============================================================================

st.markdown(
    '<div class="section-title">'
    'Model trajectories'
    '</div>',
    unsafe_allow_html=True,
)


# =============================================================================
# CREATE FIGURES
# =============================================================================

fig1, ax1 = plt.subplots(
    figsize=(7.2, 4.4)
)

ax1.plot(
    time_days,
    physical_dose_rate_gy_h,
    linewidth=2.0,
    label="Physical dose rate",
)

ax1.plot(
    time_days,
    effective_dose_rate_gy_h,
    linewidth=2.0,
    linestyle="--",
    label="Effective dose rate",
)

ax1.axhline(
    critical_dose_rate_gy_h,
    linestyle=":",
    linewidth=1.5,
    label="Critical dose rate",
)

add_treatment_markers(
    ax1,
    treatment_times,
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
style_legend(ax1)

fig1.tight_layout()


# =============================================================================
# TUMOUR BURDEN FIGURE
# =============================================================================

fig2, ax2 = plt.subplots(
    figsize=(7.2, 4.4)
)

ax2.plot(
    time_days,
    total_burden,
    linewidth=2.2,
    label="Total tumour burden",
)

ax2.axhline(
    initial_burden_ml,
    linestyle=":",
    linewidth=1.0,
    alpha=0.7,
    label="Initial burden",
)

add_treatment_markers(
    ax2,
    treatment_times,
)

ax2.set_xlabel(
    "Time (days)"
)

ax2.set_ylabel(
    "Tumour burden (mL)"
)

ax2.set_title(
    "Total tumour burden"
)

style_axis(ax2)
style_legend(ax2)

fig2.tight_layout()


# =============================================================================
# SENSITIVE / RESISTANT FIGURE
# =============================================================================

fig3, ax3 = plt.subplots(
    figsize=(7.2, 4.4)
)

ax3.plot(
    time_days,
    sensitive,
    linewidth=2.0,
    label="Sensitive tumour",
)

ax3.plot(
    time_days,
    resistant,
    linewidth=2.0,
    label="Resistant tumour",
)

add_treatment_markers(
    ax3,
    treatment_times,
)

ax3.set_xlabel(
    "Time (days)"
)

ax3.set_ylabel(
    "Tumour burden (mL)"
)

ax3.set_title(
    "Sensitive and resistant tumour populations"
)

style_axis(ax3)

ax3b = ax3.twinx()

ax3b.plot(
    time_days,
    residual_resistant_burden_percent,
    linewidth=1.8,
    linestyle="--",
    label="Residual resistant burden",
)

ax3b.set_ylabel(
    "Residual resistant burden (% of initial)",
)

ax3b.spines["top"].set_visible(False)

ax3b.tick_params(
    labelsize=9,
)

lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()

ax3.legend(
    lines1 + lines2,
    labels1 + labels2,
    frameon=False,
    fontsize=8,
    loc="best",
)

fig3.tight_layout()


# =============================================================================
# TCP FIGURE
# =============================================================================

fig4, ax4 = plt.subplots(
    figsize=(7.2, 4.4)
)

ax4.plot(
    time_days,
    tcp * 100.0,
    linewidth=2.2,
    label="TCP",
)

ax4.axhline(
    50.0,
    linestyle=":",
    linewidth=1.2,
    label="50% TCP",
)

add_treatment_markers(
    ax4,
    treatment_times,
)

ax4.set_xlabel(
    "Time (days)"
)

ax4.set_ylabel(
    "TCP (%)"
)

ax4.set_ylim(
    0,
    100.5,
)

ax4.set_title(
    "Tumour control probability"
)

style_axis(ax4)
style_legend(ax4)

fig4.tight_layout()


# =============================================================================
# DISPLAY 2 × 2
# =============================================================================

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

st.markdown(
    '<div class="section-title">'
    'Detailed model summary'
    '</div>',
    unsafe_allow_html=True,
)


with st.expander(
    "Show detailed results",
    expanded=False,
):

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:

        st.markdown("### Treatment")

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
            f"{interval_days:.1f} days"
        )

        st.write(
            f"Total tumour uptake: "
            f"{tumour_uptake_percent:.1f}%"
        )

        st.write(
            f"Dose scaling factor: "
            f"{dose_scale:.3f}×"
        )

        st.write(
            f"Cumulative physical dose: "
            f"{summary['cumulative_physical_dose_gy']:.2f} Gy"
        )

        st.write(
            f"Cumulative effective dose: "
            f"{summary['cumulative_effective_dose_gy']:.2f} Gy"
        )

        st.markdown("### Dose-rate metrics")

        st.write(
            f"Critical dose rate: "
            f"{summary['critical_dose_rate_gy_h']:.5f} Gy/h"
        )

        st.write(
            f"Peak physical dose rate: "
            f"{summary['peak_physical_rate_gy_h']:.5f} Gy/h"
        )

        st.write(
            f"Peak dose-rate day: "
            f"{summary['peak_physical_rate_day']:.2f}"
        )

        st.write(
            f"Time above critical rate: "
            f"{summary['time_above_critical_days']:.2f} days"
        )

    with detail_col2:

        st.markdown("### Tumour response")

        st.write(
            f"Initial tumour burden: "
            f"{initial_burden_ml:.2f} mL"
        )

        st.write(
            f"Minimum tumour burden: "
            f"{summary['minimum_tumour_burden_ml']:.2f} mL"
        )

        st.write(
            f"Minimum burden day: "
            f"{summary['minimum_tumour_burden_day']:.2f}"
        )

        st.write(
            f"Final tumour burden: "
            f"{summary['final_total_burden_ml']:.2f} mL"
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

        st.markdown(
            "### Resistant disease"
        )

        st.write(
            f"Initial resistant burden: "
            f"{initial_resistant_burden:.2f} mL"
        )

        st.write(
            f"Final resistant burden: "
            f"{summary['final_resistant_burden_ml']:.2f} mL"
        )

        st.write(
            f"Residual resistant burden: "
            f"{summary['residual_resistant_burden_percent']:.2f}% "
            f"of initial resistant burden"
        )

        st.markdown(
            "### Biological parameters"
        )

        st.write(
            f"Alpha: {alpha:.3f} Gy⁻¹"
        )

        st.write(
            f"Trep: {trep_days:.1f} days"
        )

        st.write(
            f"Sensitive fraction: "
            f"{sensitive_fraction * 100.0:.1f}%"
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
            f"{repopulation_kickoff_days:.1f} days"
        )

        st.write(
            f"Effectiveness gamma: "
            f"{gamma:.2f}"
        )


# =============================================================================
# MODEL ASSUMPTIONS AND LIMITATIONS
# =============================================================================

with st.expander(
    "Model assumptions and limitations",
    expanded=False,
):

    st.markdown(
        """
        **Tumour burden**

        The initial tumour burden represents total metastatic tumour
        burden rather than a single solid tumour. The model treats this
        as a phenomenological aggregate volume.

        **Tumour uptake**

        Total tumour uptake is represented as a percentage of the
        administered activity. The model then distributes this uptake
        over the specified total metastatic burden.

        **Dose scaling**

        Dose rate is scaled relative to a reference condition of
        234 mL tumour burden and 1% total tumour uptake. This is an
        exploratory scaling relationship and is not a substitute for
        patient-specific lesion dosimetry.

        **Radiobiology**

        Radiation killing is represented using a linear-quadratic model.
        Sensitive and resistant populations have different radiosensitivity
        and growth characteristics.

        **Resistant disease**

        The residual resistant burden is reported relative to the
        initial resistant tumour burden. This is different from the
        resistant composition of the remaining tumour.

        A tumour can therefore become 100% resistant in composition
        while the absolute resistant tumour burden is simultaneously
        becoming very small.

        **TCP**

        TCP is normalised to an initial TCP of 10% and is driven by the
        relative tumour burden. This is an exploratory model rather than
        a clinically validated TCP model.

        **Dose-rate effectiveness**

        The model applies a phenomenological dose-rate effectiveness
        relationship using the critical dose rate and the gamma parameter.

        **Clinical interpretation**

        Actual Lu-177 PSMA dosimetry requires lesion-specific activity
        measurements, time-activity curves, residence times, absorbed
        fractions, spatial heterogeneity, cross-dose and patient-specific
        biological parameters.

        Therefore, the explorer should be interpreted as a mechanistic
        modelling and hypothesis-generation tool rather than a clinical
        treatment-planning tool.
        """
    )


# =============================================================================
# EXPORT DATA
# =============================================================================

st.markdown(
    '<div class="section-title">'
    'Export results'
    '</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# DATAFRAME
# -----------------------------------------------------------------------------

results_df = pd.DataFrame(
    {
        "Time_days": time_days,

        "Physical_dose_rate_Gy_h":
            physical_dose_rate_gy_h,

        "Effective_dose_rate_Gy_h":
            effective_dose_rate_gy_h,

        "Critical_dose_rate_Gy_h":
            np.full_like(
                time_days,
                critical_dose_rate_gy_h,
            ),

        "Dose_rate_ratio":
            dose_rate_ratio,

        "Effectiveness_factor":
            effectiveness_factor,

        "Sensitive_tumour_mL":
            sensitive,

        "Resistant_tumour_mL":
            resistant,

        "Total_tumour_burden_mL":
            total_burden,

        "Resistant_composition_percent":
            resistant_composition * 100.0,

        "Residual_resistant_burden_percent":
            residual_resistant_burden_percent,

        "TCP_percent":
            tcp * 100.0,
    }
)


# =============================================================================
# CSV DOWNLOAD
# =============================================================================

csv_buffer = io.StringIO()

results_df.to_csv(
    csv_buffer,
    index=False,
)

st.download_button(
    label="Download model results CSV",
    data=csv_buffer.getvalue(),
    file_name=(
        "Lu177_PSMA_model_results.csv"
    ),
    mime="text/csv",
)


# =============================================================================
# 600 DPI PNG EXPORT
# =============================================================================

png_buffer = io.BytesIO()

fig_export, ax_export = plt.subplots(
    figsize=(14, 8),
)

ax_export.plot(
    time_days,
    physical_dose_rate_gy_h,
    linewidth=2.0,
    label="Physical dose rate",
)

ax_export.plot(
    time_days,
    effective_dose_rate_gy_h,
    linewidth=2.0,
    linestyle="--",
    label="Effective dose rate",
)

ax_export.axhline(
    critical_dose_rate_gy_h,
    linestyle=":",
    linewidth=1.5,
    label="Critical dose rate",
)

add_treatment_markers(
    ax_export,
    treatment_times,
)

ax_export.set_xlabel(
    "Time (days)"
)

ax_export.set_ylabel(
    "Dose rate (Gy/h)"
)

ax_export.set_title(
    "Lu-177 PSMA dose-rate trajectory"
)

style_axis(ax_export)
style_legend(ax_export)

fig_export.tight_layout()

fig_export.savefig(
    png_buffer,
    format="png",
    dpi=600,
    bbox_inches="tight",
)

plt.close(fig_export)

st.download_button(
    label="Download dose-rate plot (600 dpi PNG)",
    data=png_buffer.getvalue(),
    file_name=(
        "Lu177_PSMA_dose_rate_600dpi.png"
    ),
    mime="image/png",
)


# =============================================================================
# END
# =============================================================================
