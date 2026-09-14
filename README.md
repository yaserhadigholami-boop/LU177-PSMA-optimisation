# LU177-PSMA-optimisation

# Lu-177 PSMA Tumour Response Modelling

## Overview

This project contains a mechanistic and radiobiological modelling framework for investigating tumour response to repeated **Lu-177 PSMA radioligand therapy (RLT)**.

The model combines:

* patient-level biological parameter generation
* Lu-177 physical decay and dose-rate modelling
* critical dose-rate analysis
* dose-rate effectiveness modelling
* sensitive and resistant tumour-cell dynamics
* tumour repopulation between treatment cycles
* fractionated treatment schedules
* tumour burden evolution
* tumour control probability (TCP)
* interactive parameter exploration

The framework is intended primarily as a **research and hypothesis-generation tool** for investigating how treatment activity, treatment-cycle spacing, tumour biology and radioresistance may influence tumour response.

It is not intended to provide clinical treatment recommendations or patient-specific treatment prescriptions.

---

## Project Structure

The modelling workflow is divided into sequential sections.

### Section 01 — Patient Parameters

Generates a virtual cohort of patients with heterogeneous biological characteristics.

Parameters include:

* initial metastatic tumour burden
* sensitive tumour-cell fraction
* radiosensitivity (`alpha`)
* `beta/alpha`
* tumour repopulation time (`Trep`)
* resistant-cell doubling-time multiplier
* repopulation kickoff time
* DNA-repair half-life
* ctDNA clearance half-life
* ctDNA burden fraction
* ctDNA death-release scaling

The default cohort contains 100 virtual patients.

---

### Section 02 — Treatment Schedules

Defines the treatment schedules used in the simulations.

The framework supports repeated administrations of Lu-177 with configurable:

* activity per cycle
* number of treatment cycles
* interval between cycles

Treatment schedules can be evaluated over a range of cycle intervals.

---

### Section 03 — Physical Lu-177 Dose Rate

Calculates the physical Lu-177 activity and corresponding tumour dose rate following each administration.

The model accounts for the physical half-life of Lu-177 and the cumulative contribution from multiple treatment administrations.

The default Lu-177 physical half-life is:

```text
6.647 days
```

The model currently uses a nominal physical dose conversion of:

```text
0.50 Gy per GBq
```

unless otherwise specified in the relevant section.

---

### Section 04 — Critical Dose Rate

Calculates the biological critical dose rate:

```text
Rcrit = ln(2) / (alpha × Trep)
```

where:

* `alpha` = tumour radiosensitivity
* `Trep` = tumour repopulation time

The physical dose rate is compared with the critical dose rate to determine whether the radiation effect is expected to overcome tumour repopulation.

The section evaluates:

* critical dose rate
* peak dose rate
* dose-rate ratio
* time above critical dose rate
* time below critical dose rate
* longest continuous period above the critical dose rate
* first crossing above the critical dose rate
* first crossing below the critical dose rate
* cumulative physical dose

---

### Section 05 — Dose-Rate Effectiveness

Introduces a dose-rate effectiveness relationship based on the ratio between the physical dose rate and the critical dose rate.

The effectiveness term is currently:

```text
E(t) = min(1, R(t)^gamma)
```

where:

```text
R(t) = Rphysical(t) / Rcrit
```

and `gamma` controls the strength of the dose-rate effectiveness relationship.

The model then calculates an effective dose rate and cumulative effective dose.

---

### Section 07 — Radiation Cell-Killing Trajectories

Simulates radiation-induced cell killing using a linear-quadratic (LQ) formulation.

Separate tumour populations are considered:

* radiation-sensitive cells
* radiation-resistant cells

The model applies different radiosensitivity and repopulation characteristics to the two populations.

---

### Section 09 — Sensitive / Resistant Tumour Dynamics

Models the interaction between:

* radiation killing
* tumour-cell repopulation
* treatment-cycle timing
* sensitive-cell depletion
* resistant-cell enrichment

The resistant population can become increasingly important when treatment intervals allow substantial tumour repopulation between cycles.

This section is particularly important for investigating the effect of treatment-cycle spacing.

---

## Interactive Model Explorer

The project includes an interactive Streamlit application for exploring the model behaviour.

The application allows the user to vary:

1. Initial metastatic tumour burden
2. `alpha`
3. Tumour repopulation time (`Trep`)
4. Sensitive tumour-cell fraction
5. Resistant `Tk/Trep` multiplier
6. Resistant radiosensitivity factor
7. Repopulation kickoff time
8. Lu-177 activity per cycle
9. Number of treatment cycles
10. Treatment-cycle interval
11. Dose-rate effectiveness `gamma`

The explorer dynamically recalculates the model and displays:

* physical and effective dose rate
* metastatic tumour burden
* sensitive and resistant tumour populations
* resistant fraction
* tumour control probability (TCP)

### Running the Streamlit Application

From the project directory:

```bash
cd /Users/yaser/Documents/Python_scripts/Playground/Lu-177_PSMA/Final_code
```

Run:

```bash
streamlit run Interactive_model_explorer_streamlit.py
```

The application should then open in a web browser.

---

## Output Directory

Results generated by the modelling sections are stored under:

```text
/Users/yaser/Documents/Python_scripts/Playground/Lu-177_PSMA/Final_code/Output
```

Each modelling section has its own output directory.

The interactive explorer also provides downloadable CSV results and high-resolution PNG figures.

Figures are generated at **600 dpi** where applicable.

---

## Tumour Burden

The interactive explorer uses metastatic tumour burden in **mL** rather than assuming that the initial tumour mass represents a single solid tumour.

The default value is:

```text
234 mL
```

This is intended as a representative metastatic tumour burden for exploratory modelling.

The burden parameter represents the aggregate modelled metastatic tumour volume rather than a single anatomical lesion.

The current default exploration range is:

```text
10–1500 mL
```

---

## Radiobiological Model

The sensitive and resistant populations are modelled separately.

For a radiation dose `D`, the LQ survival formulation is:

```text
S = exp(-alpha × D - beta × D²)
```

with:

```text
beta = alpha × (beta/alpha)
```

The resistant population uses a modified radiosensitivity parameter.

Tumour repopulation is represented using exponential growth based on the corresponding tumour doubling time.

The resistant population can therefore have both:

* reduced radiation sensitivity
* different repopulation kinetics

This allows the model to investigate treatment schedules that may produce different levels of resistant-cell enrichment.

---

## Treatment-Cycle Spacing

Treatment-cycle interval is an important parameter in this model.

Shorter intervals may provide greater continuity of radiation exposure and reduce the opportunity for tumour repopulation.

Longer intervals allow more time for surviving tumour cells, particularly resistant cells, to repopulate before the next treatment.

Consequently, two schedules with the same total administered activity can produce substantially different predicted tumour trajectories.

The model therefore evaluates treatment schedules based not only on cumulative dose but also on the **temporal structure of dose delivery**.

---

## TCP Model

The interactive explorer includes a normalized tumour control probability (TCP) calculation.

The current exploratory implementation assumes an initial TCP of:

```text
10%
```

and scales the effective clonogenic burden according to the relative tumour burden.

The TCP is therefore intended to provide a useful comparative metric for exploring model behaviour rather than representing a clinically validated probability of cure for an individual patient.

TCP should not be interpreted independently of the underlying assumptions about:

* clonogenic burden
* radiosensitivity
* tumour heterogeneity
* absorbed dose
* repopulation
* treatment schedule

---

## Important Modelling Assumptions

The current model is intentionally simplified.

### 1. Homogeneous tumour dose

The current exploratory implementation uses administered activity and a nominal dose conversion to estimate dose rate.

It does not yet explicitly model:

* PSMA uptake heterogeneity
* lesion-specific uptake
* tumour-to-background ratios
* residence time differences
* heterogeneous intratumoural activity
* cross-dose between lesions
* marrow or organ dosimetry

Therefore, the calculated tumour dose should not be interpreted as a patient-specific absorbed dose.

### 2. Simplified tumour kinetics

Tumour growth and repopulation are represented using simplified exponential kinetics.

Real metastatic prostate cancer is expected to exhibit:

* heterogeneous growth rates
* treatment-dependent growth kinetics
* changing PSMA expression
* phenotypic evolution
* spatial heterogeneity

These effects are not yet explicitly represented.

### 3. Simplified resistant population

The resistant population represents a phenomenological resistant compartment.

It does not currently model a specific molecular mechanism of resistance.

### 4. TCP is exploratory

The TCP calculation is calibrated for model exploration and comparison between treatment scenarios.

It is not a validated clinical TCP model.

---

## Interpretation

The main purpose of the model is to investigate relationships between:

```text
Lu-177 activity
       ↓
physical dose rate
       ↓
critical dose-rate relationship
       ↓
radiation cell killing
       ↓
sensitive/resistant population dynamics
       ↓
tumour repopulation
       ↓
overall tumour burden
       ↓
TCP
```

In particular, the framework can be used to investigate whether **treatment-cycle spacing**, rather than cumulative administered activity alone, substantially changes the predicted tumour response.

---

## Current Development Priorities

Future versions of the model may incorporate:

* patient-specific PSMA PET tumour volumes
* lesion-specific PSMA uptake
* time-activity curves (TACs)
* biologically derived residence times
* lesion-specific absorbed dose
* heterogeneous tumour dose distributions
* fractionation effects
* DNA repair
* sublethal damage repair
* treatment-induced changes in radiosensitivity
* dynamic resistant-cell evolution
* ctDNA response modelling
* patient-specific TCP parameterisation
* comparison of Lu-177 and Ga-68 treatment scenarios
* Monte Carlo-derived absorbed dose and microdosimetric parameters

---

## Reproducibility

The virtual patient generation uses a fixed random seed where specified so that simulations can be reproduced.

For the patient-parameter generation:

```text
Random seed: 20260825
```

Changing the seed will generate a different virtual patient cohort.

---

## Disclaimer

This software is a research modelling framework.

The outputs are simulations based on predefined assumptions and parameter ranges and should not be used for clinical decision-making, patient treatment planning, activity prescription, or estimation of an individual patient's probability of tumour control without appropriate clinical and dosimetric validation.

---

## Author

Dr Yaser Hadi Gholami
School of Biomedical Engineering
Faculty of Engineering
The University of Sydney

Lu-177 PSMA radiobiological and tumour-response modelling project.
