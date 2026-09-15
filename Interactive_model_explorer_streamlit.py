# ============================================================
# LU-177 PSMA OPTIMISATION — INTERACTIVE MODEL EXPLORER
# Streamlit version
# ============================================================

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Lu-177 PSMA Optimisation",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MODEL CONSTANTS
# ============================================================

LU177_HALF_LIFE_DAYS = 6.647
DOSE_PER_GBQ_GY = 0.50

REFERENCE_METASTATIC_BURDEN_ML = 234.0
DEFAULT_TUMOUR_UPTAKE_PERCENT = 1.0

TUMOUR_UPTAKE_MIN_PERCENT = 0.1
TUMOUR_UPTAKE_MAX_PERCENT = 10.0

DEFAULT_BURDEN = 234.0
DEFAULT_ALPHA = 0.10
DEFAULT_BETA_ALPHA = 0.10
DEFAULT_TREP = 30.0
DEFAULT_SENSITIVE_FRACTION = 0.75
DEFAULT_RESISTANT_TK_MULTIPLIER = 1.50
DEFAULT_RESISTANT_RADIO_FACTOR = 0.25
DEFAULT_REPOPULATION_KICKOFF = 5.0
DEFAULT_ACTIVITY = 7.4
DEFAULT_N_CYCLES = 4
DEFAULT_INTERVAL_DAYS = 7

INITIAL_TCP = 0.10
INITIAL_CLONOGENIC_BURDEN = -np.log(INITIAL_TCP)

DT_DAYS = 0.05
SIMULATION_DAYS = 180.0

PLOT_TREATMENT = "#A0A0A0"
COLOR_TREATMENT = "#A0A0A0"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def integrate_trapezoid(y, x):
    """
    Compatibility wrapper for NumPy versions where np.trapz
    has been removed/deprecated.
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)

    return np.trapz(y, x)


# ============================================================
# KEY RESULT RENDERING
# ============================================================

def render_key_metric(
    title,
    value,
    status_text="",
    status="neutral",
    caption="",
):
    """
    Render one complete key-result card as a single HTML block.

    IMPORTANT:
    Do not construct nested HTML fragments here.
    Streamlit's Markdown parser can interpret nested/generated
    fragments as code blocks. Keeping the entire card as one
    continuous HTML block avoids that problem.
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

    html = f"""
<div class="metric-card">
    <div class="metric-title">{title}</div>

    <div class="metric-value">
        {value}
    </div>

    <span style="display:inline-block; padding:3px 8px; border-radius:999px; font-size:0.75rem; font-weight:600; color:{colour}; background:{background}; margin-top:3px;">
        {status_text}
    </span>

    <div style="font-size:0.72rem; color:#6B7280; margin-top:5px; line-height:1.25;">
        {caption}
    </div>
</div>
"""

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


# ============================================================
# SECONDARY METRIC RENDERING
# ============================================================

def render_secondary_metric(
    title,
    value,
    caption="",
):
    """
    Render one complete secondary metric as a single HTML block.
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


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.main-title {
    font-size: 2.1rem;
    font-weight: 700;
    margin-bottom: 0.15rem;
}

.main-subtitle {
    color: #6B7280;
    font-size: 0.95rem;
    margin-bottom: 1.2rem;
}

.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 1.0rem;
    margin-bottom: 0.65rem;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 14px 16px 13px 16px;
    min-height: 125px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

.metric-title {
    font-size: 0.80rem;
    font-weight: 600;
    color: #6B7280;
    margin-bottom: 3px;
}

.metric-value {
    font-size: 1.55rem;
    line-height: 1.15;
    font-weight: 700;
    color: #111827;
    margin-bottom: 4px;
}

.metric-card-secondary {
    background: #FFFDF4;
    border: 1px solid #F2C94C;
    border-left: 4px solid #F2C94C;
    border-radius: 10px;
    padding: 11px 14px 10px 14px;
    min-height: 100px;
}

.metric-title-secondary {
    font-size: 0.77rem;
    font-weight: 600;
    color: #A16207;
    margin-bottom: 3px;
}

.metric-value-secondary {
    font-size: 1.25rem;
    line-height: 1.15;
    font-weight: 700;
    color: #92400E;
    margin-bottom: 3px;
}

.metric-caption {
    font-size: 0.70rem;
    color: #78716C;
    line-height: 1.2;
}

.plot-container {
    margin-top: 0.2rem;
}

.sidebar-section {
    font-size: 1.0rem;
    font-weight: 700;
    margin-top: 0.8rem;
    margin-bottom: 0.3rem;
}

div[data-testid="stDownloadButton"] button {
    width: 100%;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL CALCULATION
# ============================================================

def calculate_model(
    initial_burden_ml,
    alpha,
    trep_days,
    sensitive_fraction,
    resistant_tk_multiplier,
    resistant_radio_factor,
    repopulation_kickoff,
    activity_gbq,
    n_cycles,
    interval_days,
    tumour_uptake_percent,
):
    """
    Run the complete tumour-dynamics model.

    Initial tumour burden represents the total metastatic
    tumour burden as an aggregate volume.
    """

    # --------------------------------------------------------
    # Time grid
    # --------------------------------------------------------

    time_days = np.arange(
        0.0,
        SIMULATION_DAYS + DT_DAYS,
        DT_DAYS,
    )

    n_points = len(time_days)

    # --------------------------------------------------------
    # Treatment administration times
    # --------------------------------------------------------

    administration_times = np.array(
        [
            i * interval_days
            for i in range(n_cycles)
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # Critical dose rate
    # --------------------------------------------------------

    trep_hours = trep_days * 24.0

    critical_dose_rate_gy_h = (
        np.log(2.0)
        / (alpha * trep_hours)
    )

    critical_dose_rate_gy_day = (
        critical_dose_rate_gy_h * 24.0
    )

    # --------------------------------------------------------
    # Tumour uptake / burden scaling
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Physical dose rate
    # --------------------------------------------------------

    physical_dose_rate_gy_day = np.zeros(
        n_points,
        dtype=float,
    )

    lambda_lu = (
        np.log(2.0)
        / LU177_HALF_LIFE_DAYS
    )

    for administration_time in administration_times:

        elapsed = (
            time_days
            - administration_time
        )

        active = elapsed >= 0.0

        activity = np.zeros(
            n_points,
            dtype=float,
        )

        activity[active] = (
            activity_gbq
            * np.exp(
                -lambda_lu
                * elapsed[active]
            )
        )

        physical_dose_rate_gy_day += (
            activity
            * DOSE_PER_GBQ_GY
            * dose_scale
        )

    physical_dose_rate_gy_h = (
        physical_dose_rate_gy_day
        / 24.0
    )

    # --------------------------------------------------------
    # Effective dose rate
    #
    # No gamma slider.
    #
    # effectiveness = min(1, physical rate / critical rate)
    # --------------------------------------------------------

    dose_rate_ratio = np.divide(
        physical_dose_rate_gy_day,
        critical_dose_rate_gy_day,
        out=np.zeros_like(
            physical_dose_rate_gy_day
        ),
        where=critical_dose_rate_gy_day > 0,
    )

    effectiveness = np.minimum(
        1.0,
        np.maximum(
            0.0,
            dose_rate_ratio,
        ),
    )

    effective_dose_rate_gy_day = (
        physical_dose_rate_gy_day
        * effectiveness
    )

    effective_dose_rate_gy_h = (
        effective_dose_rate_gy_day
        / 24.0
    )

    # --------------------------------------------------------
    # Biological parameters
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Initial populations
    # --------------------------------------------------------

    initial_sensitive_burden = (
        initial_burden_ml
        * sensitive_fraction
    )

    initial_resistant_burden = (
        initial_burden_ml
        * (1.0 - sensitive_fraction)
    )

    sensitive = np.zeros(
        n_points,
        dtype=float,
    )

    resistant = np.zeros(
        n_points,
        dtype=float,
    )

    total_burden = np.zeros(
        n_points,
        dtype=float,
    )

    tcp = np.zeros(
        n_points,
        dtype=float,
    )

    sensitive[0] = (
        initial_sensitive_burden
    )

    resistant[0] = (
        initial_resistant_burden
    )

    total_burden[0] = (
        sensitive[0]
        + resistant[0]
    )

    # --------------------------------------------------------
    # Biological simulation
    # --------------------------------------------------------

    for i in range(1, n_points):

        dt = (
            time_days[i]
            - time_days[i - 1]
        )

        S = sensitive[i - 1]
        R = resistant[i - 1]

        # ----------------------------------------------------
        # Repopulation
        # ----------------------------------------------------

        if time_days[i] >= repopulation_kickoff:

            S *= np.exp(
                sensitive_growth_rate
                * dt
            )

            R *= np.exp(
                resistant_growth_rate
                * dt
            )

        # ----------------------------------------------------
        # Radiation dose
        # ----------------------------------------------------

        dose_interval_gy = (
            physical_dose_rate_gy_day[i]
            * dt
        )

        # ----------------------------------------------------
        # LQ survival
        # ----------------------------------------------------

        survival_sensitive = np.exp(
            -alpha
            * dose_interval_gy
            -beta_sensitive
            * dose_interval_gy**2
        )

        survival_resistant = np.exp(
            -alpha_resistant
            * dose_interval_gy
            -beta_resistant
            * dose_interval_gy**2
        )

        S *= survival_sensitive
        R *= survival_resistant

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        sensitive[i] = max(
            S,
            0.0,
        )

        resistant[i] = max(
            R,
            0.0,
        )

        total_burden[i] = (
            sensitive[i]
            + resistant[i]
        )

    # --------------------------------------------------------
    # Resistant composition
    # --------------------------------------------------------

    resistant_composition_fraction = np.divide(
        resistant,
        total_burden,
        out=np.zeros_like(resistant),
        where=total_burden > 0,
    )

    # --------------------------------------------------------
    # Residual resistant burden
    # --------------------------------------------------------

    residual_resistant_burden_percent = (
        np.divide(
            resistant,
            initial_resistant_burden,
            out=np.zeros_like(resistant),
            where=initial_resistant_burden > 0,
        )
        * 100.0
    )

    # --------------------------------------------------------
    # TCP
    # --------------------------------------------------------

    relative_burden = np.divide(
        total_burden,
        initial_burden_ml,
        out=np.zeros_like(total_burden),
        where=initial_burden_ml > 0,
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

    # --------------------------------------------------------
    # Summary calculations
    # --------------------------------------------------------

    cumulative_physical_dose_gy = integrate_trapezoid(
        physical_dose_rate_gy_day,
        time_days,
    )

    cumulative_effective_dose_gy = integrate_trapezoid(
        effective_dose_rate_gy_day,
        time_days,
    )

    above_critical = (
        physical_dose_rate_gy_h
        >= critical_dose_rate_gy_h
    )

    time_above_critical_days = (
        np.sum(above_critical)
        * DT_DAYS
    )

    longest_continuous_days = 0.0
    current_duration = 0.0

    for flag in above_critical:

        if flag:

            current_duration += DT_DAYS

            longest_continuous_days = max(
                longest_continuous_days,
                current_duration,
            )

        else:

            current_duration = 0.0

    peak_index = int(
        np.argmax(
            physical_dose_rate_gy_h
        )
    )

    peak_dose_rate_gy_h = (
        physical_dose_rate_gy_h[
            peak_index
        ]
    )

    peak_day = (
        time_days[
            peak_index
        ]
    )

    minimum_burden_index = int(
        np.argmin(
            total_burden
        )
    )

    minimum_burden_ml = (
        total_burden[
            minimum_burden_index
        ]
    )

    minimum_burden_day = (
        time_days[
            minimum_burden_index
        ]
    )

    maximum_tcp_index = int(
        np.argmax(
            tcp
        )
    )

    maximum_tcp = (
        tcp[
            maximum_tcp_index
        ]
    )

    maximum_tcp_day = (
        time_days[
            maximum_tcp_index
        ]
    )

    final_resistant_burden_percent = (
        residual_resistant_burden_percent[-1]
    )

    return {
        "time_days":
            time_days,

        "administration_times":
            administration_times,

        "physical_dose_rate_gy_day":
            physical_dose_rate_gy_day,

        "physical_dose_rate_gy_h":
            physical_dose_rate_gy_h,

        "effective_dose_rate_gy_day":
            effective_dose_rate_gy_day,

        "effective_dose_rate_gy_h":
            effective_dose_rate_gy_h,

        "critical_dose_rate_gy_h":
            critical_dose_rate_gy_h,

        "critical_dose_rate_gy_day":
            critical_dose_rate_gy_day,

        "dose_rate_ratio":
            dose_rate_ratio,

        "effectiveness":
            effectiveness,

        "sensitive":
            sensitive,

        "resistant":
            resistant,

        "total_burden":
            total_burden,

        "tcp":
            tcp,

        "resistant_composition_fraction":
            resistant_composition_fraction,

        "residual_resistant_burden_percent":
            residual_resistant_burden_percent,

        "initial_resistant_burden":
            initial_resistant_burden,

        "cumulative_physical_dose_gy":
            cumulative_physical_dose_gy,

        "cumulative_effective_dose_gy":
            cumulative_effective_dose_gy,

        "time_above_critical_days":
            time_above_critical_days,

        "longest_continuous_above_days":
            longest_continuous_days,

        "peak_dose_rate_gy_h":
            peak_dose_rate_gy_h,

        "peak_day":
            peak_day,

        "minimum_burden_ml":
            minimum_burden_ml,

        "minimum_burden_day":
            minimum_burden_day,

        "maximum_tcp":
            maximum_tcp,

        "maximum_tcp_day":
            maximum_tcp_day,

        "final_resistant_burden_percent":
            final_resistant_burden_percent,

        "initial_burden_ml":
            initial_burden_ml,

        "sensitive_fraction":
            sensitive_fraction,

        "activity_gbq":
            activity_gbq,

        "n_cycles":
            n_cycles,

        "interval_days":
            interval_days,

        "tumour_uptake_percent":
            tumour_uptake_percent,

        "dose_scale":
            dose_scale,
    }


# ============================================================
# PLOT HELPERS
# ============================================================

def add_treatment_markers(
    ax,
    administration_times,
):

    for t in administration_times:

        ax.axvline(
            t,
            color=PLOT_TREATMENT,
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
        )


def make_dose_rate_plot(results):

    fig, ax = plt.subplots(
        figsize=(8, 5),
    )

    time = results["time_days"]

    ax.plot(
        time,
        results["physical_dose_rate_gy_h"],
        linewidth=2.0,
        label="Physical dose rate",
    )

    ax.plot(
        time,
        results["effective_dose_rate_gy_h"],
        linewidth=2.0,
        linestyle="--",
        label="Effective dose rate",
    )

    ax.axhline(
        results["critical_dose_rate_gy_h"],
        linestyle=":",
        linewidth=1.8,
        label="Critical dose rate",
    )

    add_treatment_markers(
        ax,
        results["administration_times"],
    )

    ax.set_xlabel(
        "Time (days)"
    )

    ax.set_ylabel(
        "Dose rate (Gy/h)"
    )

    ax.set_title(
        "Physical vs effective dose rate"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    return fig


def make_burden_plot(results):

    fig, ax = plt.subplots(
        figsize=(8, 5),
    )

    time = results["time_days"]

    ax.plot(
        time,
        results["total_burden"],
        linewidth=2.2,
        label="Total tumour burden",
    )

    add_treatment_markers(
        ax,
        results["administration_times"],
    )

    ax.set_xlabel(
        "Time (days)"
    )

    ax.set_ylabel(
        "Tumour burden (mL)"
    )

    ax.set_title(
        "Total tumour burden"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    return fig


def make_population_plot(results):

    fig, ax1 = plt.subplots(
        figsize=(8, 5),
    )

    time = results["time_days"]

    ax1.plot(
        time,
        results["sensitive"],
        linewidth=2.0,
        label="Sensitive tumour",
    )

    ax1.plot(
        time,
        results["resistant"],
        linewidth=2.0,
        label="Resistant tumour",
    )

    ax1.set_xlabel(
        "Time (days)"
    )

    ax1.set_ylabel(
        "Tumour burden (mL)"
    )

    ax1.set_title(
        "Sensitive and resistant tumour populations"
    )

    ax1.grid(
        alpha=0.25
    )

    ax2 = ax1.twinx()

    ax2.plot(
        time,
        results[
            "residual_resistant_burden_percent"
        ],
        linewidth=1.8,
        linestyle="--",
        label="Residual resistant burden",
    )

    ax2.set_ylabel(
        "Residual resistant burden (% of initial)"
    )

    add_treatment_markers(
        ax1,
        results["administration_times"],
    )

    lines1, labels1 = (
        ax1.get_legend_handles_labels()
    )

    lines2, labels2 = (
        ax2.get_legend_handles_labels()
    )

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=False,
        loc="upper right",
    )

    fig.tight_layout()

    return fig


def make_tcp_plot(results):

    fig, ax = plt.subplots(
        figsize=(8, 5),
    )

    time = results["time_days"]

    ax.plot(
        time,
        results["tcp"] * 100.0,
        linewidth=2.2,
        label="TCP",
    )

    ax.axhline(
        50.0,
        linestyle=":",
        linewidth=1.5,
        label="50% TCP",
    )

    add_treatment_markers(
        ax,
        results["administration_times"],
    )

    ax.set_xlabel(
        "Time (days)"
    )

    ax.set_ylabel(
        "Tumour control probability (%)"
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_title(
        "Tumour control probability"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    return fig


# ============================================================
# CSV EXPORT
# ============================================================

def create_csv(results):

    df = pd.DataFrame(
        {
            "Time_days":
                results["time_days"],

            "Physical_dose_rate_Gy_h":
                results[
                    "physical_dose_rate_gy_h"
                ],

            "Physical_dose_rate_Gy_day":
                results[
                    "physical_dose_rate_gy_day"
                ],

            "Effective_dose_rate_Gy_h":
                results[
                    "effective_dose_rate_gy_h"
                ],

            "Effective_dose_rate_Gy_day":
                results[
                    "effective_dose_rate_gy_day"
                ],

            "Critical_dose_rate_Gy_h":
                results[
                    "critical_dose_rate_gy_h"
                ],

            "Dose_rate_ratio":
                results[
                    "dose_rate_ratio"
                ],

            "Dose_rate_effectiveness":
                results[
                    "effectiveness"
                ],

            "Sensitive_tumour_mL":
                results[
                    "sensitive"
                ],

            "Resistant_tumour_mL":
                results[
                    "resistant"
                ],

            "Total_tumour_burden_mL":
                results[
                    "total_burden"
                ],

            "Resistant_composition_percent":
                results[
                    "resistant_composition_fraction"
                ]
                * 100.0,

            "Residual_resistant_burden_percent":
                results[
                    "residual_resistant_burden_percent"
                ],

            "TCP_percent":
                results[
                    "tcp"
                ]
                * 100.0,
        }
    )

    return df.to_csv(
        index=False
    ).encode("utf-8")


# ============================================================
# HIGH-RESOLUTION PNG EXPORT
# ============================================================

def create_png_export(results):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 11),
    )

    time = results["time_days"]

    # --------------------------------------------------------
    # Dose rate
    # --------------------------------------------------------

    ax = axes[0, 0]

    ax.plot(
        time,
        results["physical_dose_rate_gy_h"],
        linewidth=2.0,
        label="Physical dose rate",
    )

    ax.plot(
        time,
        results["effective_dose_rate_gy_h"],
        linewidth=2.0,
        linestyle="--",
        label="Effective dose rate",
    )

    ax.axhline(
        results["critical_dose_rate_gy_h"],
        linestyle=":",
        linewidth=1.8,
        label="Critical dose rate",
    )

    add_treatment_markers(
        ax,
        results["administration_times"],
    )

    ax.set_xlabel(
        "Time (days)"
    )

    ax.set_ylabel(
        "Dose rate (Gy/h)"
    )

    ax.set_title(
        "Physical vs effective dose rate"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )

    # --------------------------------------------------------
    # Tumour burden
    # --------------------------------------------------------

    ax = axes[0, 1]

    ax.plot(
        time,
        results["total_burden"],
        linewidth=2.2,
    )

    add_treatment_markers(
        ax,
        results["administration_times"],
    )

    ax.set_xlabel(
        "Time (days)"
    )

    ax.set_ylabel(
        "Tumour burden (mL)"
    )

    ax.set_title(
        "Total tumour burden"
    )

    ax.grid(
        alpha=0.25
    )

    # --------------------------------------------------------
    # Sensitive / resistant
    # --------------------------------------------------------

    ax = axes[1, 0]

    ax.plot(
        time,
        results["sensitive"],
        linewidth=2.0,
        label="Sensitive tumour",
    )

    ax.plot(
        time,
        results["resistant"],
        linewidth=2.0,
        label="Resistant tumour",
    )

    add_treatment_markers(
        ax,
        results["administration_times"],
    )

    ax.set_xlabel(
        "Time (days)"
    )

    ax.set_ylabel(
        "Tumour burden (mL)"
    )

    ax.set_title(
        "Sensitive and resistant tumour"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )

    # --------------------------------------------------------
    # TCP
    # --------------------------------------------------------

    ax = axes[1, 1]

    ax.plot(
        time,
        results["tcp"] * 100.0,
        linewidth=2.2,
        label="TCP",
    )

    ax.axhline(
        50.0,
        linestyle=":",
        linewidth=1.5,
        label="50% TCP",
    )

    add_treatment_markers(
        ax,
        results["administration_times"],
    )

    ax.set_xlabel(
        "Time (days)"
    )

    ax.set_ylabel(
        "TCP (%)"
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_title(
        "Tumour control probability"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    """
<div class="main-title">
    Lu-177 PSMA Optimisation
</div>

<div class="main-subtitle">
    Interactive exploration of dose-rate effects,
    tumour dynamics, resistant disease and tumour
    control probability
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header(
    "Model controls"
)


# ------------------------------------------------------------
# Tumour parameters
# ------------------------------------------------------------

st.sidebar.markdown(
    '<div class="sidebar-section">Tumour parameters</div>',
    unsafe_allow_html=True,
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
    "Alpha (Gy⁻¹)",
    min_value=0.001,
    max_value=0.50,
    value=float(DEFAULT_ALPHA),
    step=0.001,
    help=(
        "Linear radiation sensitivity parameter "
        "used in the LQ model."
    ),
)

trep_days = st.sidebar.slider(
    "Trep (days)",
    min_value=10.0,
    max_value=100.0,
    value=float(DEFAULT_TREP),
    step=1.0,
    help=(
        "Sensitive tumour doubling time."
    ),
)

sensitive_fraction = st.sidebar.slider(
    "Sensitive fraction",
    min_value=0.50,
    max_value=0.95,
    value=float(DEFAULT_SENSITIVE_FRACTION),
    step=0.01,
    help=(
        "Initial fraction of tumour burden "
        "assigned to the radiation-sensitive population."
    ),
)

resistant_tk_multiplier = st.sidebar.slider(
    "Tk/Trep",
    min_value=1.20,
    max_value=1.80,
    value=float(DEFAULT_RESISTANT_TK_MULTIPLIER),
    step=0.01,
    help=(
        "Resistant population doubling time relative "
        "to the sensitive population."
    ),
)

resistant_radio_factor = st.sidebar.slider(
    "Resistant radio factor",
    min_value=0.05,
    max_value=1.00,
    value=float(DEFAULT_RESISTANT_RADIO_FACTOR),
    step=0.01,
    help=(
        "Relative radiosensitivity of the resistant "
        "population compared with the sensitive population."
    ),
)

repopulation_kickoff = st.sidebar.slider(
    "Repopulation kickoff (days)",
    min_value=3.0,
    max_value=10.0,
    value=float(DEFAULT_REPOPULATION_KICKOFF),
    step=1.0,
    help=(
        "Time after treatment initiation when tumour "
        "repopulation begins."
    ),
)


# ------------------------------------------------------------
# Treatment parameters
# ------------------------------------------------------------

st.sidebar.markdown(
    '<div class="sidebar-section">Treatment parameters</div>',
    unsafe_allow_html=True,
)

activity_gbq = st.sidebar.slider(
    "Activity per cycle (GBq)",
    min_value=1.0,
    max_value=15.0,
    value=float(DEFAULT_ACTIVITY),
    step=0.1,
    help=(
        "Administered Lu-177 activity per treatment cycle."
    ),
)

n_cycles = st.sidebar.slider(
    "Number of cycles",
    min_value=1,
    max_value=8,
    value=int(DEFAULT_N_CYCLES),
    step=1,
    help=(
        "Number of Lu-177 treatment administrations."
    ),
)

interval_days = st.sidebar.slider(
    "Cycle interval (days)",
    min_value=1,
    max_value=42,
    value=int(DEFAULT_INTERVAL_DAYS),
    step=1,
    help=(
        "Interval between treatment administrations."
    ),
)

tumour_uptake_percent = st.sidebar.slider(
    "Total tumour uptake (%)",
    min_value=float(TUMOUR_UPTAKE_MIN_PERCENT),
    max_value=float(TUMOUR_UPTAKE_MAX_PERCENT),
    value=float(DEFAULT_TUMOUR_UPTAKE_PERCENT),
    step=0.1,
    help=(
        "Phenomenological fraction of administered activity "
        "associated with the total metastatic tumour burden."
    ),
)


# ============================================================
# RUN MODEL
# ============================================================

results = calculate_model(
    initial_burden_ml=initial_burden_ml,
    alpha=alpha,
    trep_days=trep_days,
    sensitive_fraction=sensitive_fraction,
    resistant_tk_multiplier=resistant_tk_multiplier,
    resistant_radio_factor=resistant_radio_factor,
    repopulation_kickoff=repopulation_kickoff,
    activity_gbq=activity_gbq,
    n_cycles=n_cycles,
    interval_days=interval_days,
    tumour_uptake_percent=tumour_uptake_percent,
)


# ============================================================
# KEY RESULTS
# ============================================================

st.markdown(
    '<div class="section-header">Key results</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# Minimum tumour burden
# ------------------------------------------------------------

with col1:

    burden_reduction = (
        1.0
        - (
            results["minimum_burden_ml"]
            / initial_burden_ml
        )
    ) * 100.0

    if burden_reduction > 0:

        burden_status = "positive"

        burden_text = (
            f"↓ {burden_reduction:.1f}% "
            f"from baseline"
        )

    else:

        burden_status = "negative"

        burden_text = (
            f"↑ {abs(burden_reduction):.1f}% "
            f"from baseline"
        )

    render_key_metric(
        "Minimum tumour burden",
        f'{results["minimum_burden_ml"]:.1f} mL',
        burden_text,
        burden_status,
        (
            f'Minimum reached at day '
            f'{results["minimum_burden_day"]:.1f}'
        ),
    )


# ------------------------------------------------------------
# Maximum TCP
# ------------------------------------------------------------

with col2:

    maximum_tcp_percent = (
        results["maximum_tcp"]
        * 100.0
    )

    tcp_change_pp = (
        maximum_tcp_percent
        - INITIAL_TCP * 100.0
    )

    if tcp_change_pp > 0:

        tcp_status = "positive"

        tcp_text = (
            f"↑ {tcp_change_pp:.1f} "
            f"percentage points"
        )

    else:

        tcp_status = "negative"

        tcp_text = (
            f"↓ {abs(tcp_change_pp):.1f} "
            f"percentage points"
        )

    render_key_metric(
        "Maximum TCP",
        f"{maximum_tcp_percent:.1f}%",
        tcp_text,
        tcp_status,
        (
            f'Maximum reached at day '
            f'{results["maximum_tcp_day"]:.1f}'
        ),
    )


# ------------------------------------------------------------
# Residual resistant burden
# ------------------------------------------------------------

with col3:

    residual_resistant = (
        results[
            "final_resistant_burden_percent"
        ]
    )

    resistant_reduction = (
        100.0
        - residual_resistant
    )

    if resistant_reduction > 0:

        resistant_status = "positive"

        resistant_text = (
            f"↓ {resistant_reduction:.1f}% "
            f"from initial resistant burden"
        )

    else:

        resistant_status = "negative"

        resistant_text = (
            f"↑ {abs(resistant_reduction):.1f}% "
            f"from initial resistant burden"
        )

    render_key_metric(
        "Residual resistant burden",
        f"{residual_resistant:.1f}%",
        resistant_text,
        resistant_status,
        (
            f'Initial resistant burden: '
            f'{results["initial_resistant_burden"]:.1f} mL '
            f'({(1.0 - sensitive_fraction) * 100.0:.0f}% of total)'
        ),
    )


# ------------------------------------------------------------
# Cumulative physical dose
# ------------------------------------------------------------

with col4:

    render_key_metric(
        "Cumulative physical dose",
        (
            f'{results["cumulative_physical_dose_gy"]:.2f} Gy'
        ),
        (
            f'{activity_gbq:.1f} GBq × '
            f'{n_cycles} cycles'
        ),
        "neutral",
        "Integrated over the 180-day simulation",
    )


# ============================================================
# SECONDARY METRICS
# ============================================================

st.markdown(
    '<div class="section-header">Secondary metrics</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    render_secondary_metric(
        "Critical dose rate",
        (
            f'{results["critical_dose_rate_gy_h"]:.4f} Gy/h'
        ),
        (
            f'{results["critical_dose_rate_gy_day"]:.4f} Gy/day'
        ),
    )


with col2:

    render_secondary_metric(
        "Peak physical rate",
        (
            f'{results["peak_dose_rate_gy_h"]:.4f} Gy/h'
        ),
        (
            f'Peak at day '
            f'{results["peak_day"]:.1f}'
        ),
    )


with col3:

    render_secondary_metric(
        "Time above critical rate",
        (
            f'{results["time_above_critical_days"]:.1f} days'
        ),
        (
            f'Longest continuous: '
            f'{results["longest_continuous_above_days"]:.1f} days'
        ),
    )


with col4:

    render_secondary_metric(
        "Effective cumulative dose",
        (
            f'{results["cumulative_effective_dose_gy"]:.2f} Gy'
        ),
        "Dose-rate effectiveness applied",
    )


# ============================================================
# MODEL TRAJECTORIES
# ============================================================

st.markdown(
    '<div class="section-header">Model trajectories</div>',
    unsafe_allow_html=True,
)

plot_col1, plot_col2 = st.columns(2)


with plot_col1:

    fig = make_dose_rate_plot(
        results
    )

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)


with plot_col2:

    fig = make_burden_plot(
        results
    )

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)


plot_col3, plot_col4 = st.columns(2)


with plot_col3:

    fig = make_population_plot(
        results
    )

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)


with plot_col4:

    fig = make_tcp_plot(
        results
    )

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)


# ============================================================
# DETAILED MODEL SUMMARY
# ============================================================

with st.expander(
    "Detailed model summary"
):

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.markdown(
            "### Treatment"
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
            f"Dose scaling factor: "
            f"{results['dose_scale']:.3f}"
        )

        st.write(
            f"Cumulative physical dose: "
            f"{results['cumulative_physical_dose_gy']:.2f} Gy"
        )

        st.write(
            f"Cumulative effective dose: "
            f"{results['cumulative_effective_dose_gy']:.2f} Gy"
        )

    with summary_col2:

        st.markdown(
            "### Tumour dynamics"
        )

        st.write(
            f"Initial burden: "
            f"{initial_burden_ml:.1f} mL"
        )

        st.write(
            f"Sensitive fraction: "
            f"{sensitive_fraction * 100.0:.1f}%"
        )

        st.write(
            f"Initial resistant burden: "
            f"{results['initial_resistant_burden']:.1f} mL"
        )

        st.write(
            f"Minimum tumour burden: "
            f"{results['minimum_burden_ml']:.2f} mL"
        )

        st.write(
            f"Maximum TCP: "
            f"{results['maximum_tcp'] * 100.0:.2f}%"
        )

        st.write(
            f"Residual resistant burden: "
            f"{results['final_resistant_burden_percent']:.2f}%"
        )

        st.write(
            f"Resistant composition at end: "
            f"{results['resistant_composition_fraction'][-1] * 100.0:.2f}%"
        )


# ============================================================
# MODEL ASSUMPTIONS AND LIMITATIONS
# ============================================================

with st.expander(
    "Model assumptions and limitations"
):

    st.markdown(
        """
        **Tumour burden**

        Initial metastatic burden is treated as an aggregate tumour
        volume rather than a single solid tumour. The model therefore
        represents an average metastatic disease burden.

        **Tumour uptake**

        Total tumour uptake is represented phenomenologically as a
        percentage of administered activity associated with the total
        tumour burden. The default condition is 234 mL total burden
        and 1% tumour uptake.

        Increasing uptake increases the average tumour dose rate.
        Increasing total tumour burden at fixed total uptake decreases
        the average dose rate per unit tumour burden.

        This is not a substitute for lesion-level Lu-177 PSMA
        dosimetry using measured activity-time curves.

        **Radiation response**

        Tumour killing follows a linear-quadratic formulation with
        separate parameters for sensitive and resistant populations.

        **Resistant population**

        The resistant radio factor controls relative radiation
        sensitivity. Tk/Trep controls relative repopulation kinetics.

        **Critical dose rate**

        The critical dose rate is calculated from the linear
        radiosensitivity parameter and tumour repopulation time:

        rcrit = ln(2) / (alpha × Trep)

        **Dose-rate effectiveness**

        The effective dose rate uses a direct relationship with the
        physical dose rate relative to the calculated critical dose
        rate. There is no independent gamma parameter in the user
        interface.

        **TCP**

        TCP is represented using a normalised exploratory clonogenic
        burden model. It is intended for comparative exploration of
        treatment schedules rather than clinical prediction.

        **Clinical interpretation**

        The model does not include patient-specific lesion geometry,
        heterogeneous PSMA expression, measured lesion TACs,
        cross-dose, organ dosimetry, spatial dose heterogeneity,
        fractionation repair, repopulation biology beyond the
        phenomenological terms above, or individual patient TCP
        calibration.
        """
    )


# ============================================================
# DOWNLOADS
# ============================================================

st.markdown(
    '<div class="section-header">Export results</div>',
    unsafe_allow_html=True,
)

download_col1, download_col2 = st.columns(2)


with download_col1:

    csv_data = create_csv(
        results
    )

    st.download_button(
        label="Download trajectory CSV",
        data=csv_data,
        file_name=(
            "Lu177_PSMA_model_trajectory.csv"
        ),
        mime="text/csv",
    )


with download_col2:

    png_data = create_png_export(
        results
    )

    st.download_button(
        label="Download 600 dpi PNG",
        data=png_data,
        file_name=(
            "Lu177_PSMA_model_trajectories_600dpi.png"
        ),
        mime="image/png",
    )
