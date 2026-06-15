"""Text template generation for clinical events.

Converts discretized clinical events into natural-language text strings
for the encoder. Called by the trajectory builder to generate per-stay text.

Format: "{Display Name}: {value} {unit}", sorted by category, separated by ". ".
"""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# =============================================================================
# Display Name Mapping
# =============================================================================
# Maps (source_table, variable_name) -> human-readable display name.

DISPLAY_NAMES: dict[tuple[str, str], str] = {
    # --- vitalsign ---
    ("vitalsign", "heart_rate"): "Heart rate",
    ("vitalsign", "sbp"): "Systolic BP",
    ("vitalsign", "dbp"): "Diastolic BP",
    ("vitalsign", "mbp"): "MAP",
    ("vitalsign", "sbp_ni"): "Systolic BP (non-invasive)",
    ("vitalsign", "dbp_ni"): "Diastolic BP (non-invasive)",
    ("vitalsign", "mbp_ni"): "MAP (non-invasive)",
    ("vitalsign", "resp_rate"): "Respiratory rate",
    ("vitalsign", "temperature"): "Temperature",
    ("vitalsign", "spo2"): "SpO2",
    ("vitalsign", "glucose"): "Glucose",
    # --- bg (blood gas) ---
    ("bg", "so2"): "SaO2",
    ("bg", "po2"): "PaO2",
    ("bg", "pco2"): "PaCO2",
    ("bg", "fio2"): "FiO2 (ABG)",
    ("bg", "ph"): "pH",
    ("bg", "baseexcess"): "Base excess",
    ("bg", "bicarbonate"): "Bicarbonate (ABG)",
    ("bg", "totalco2"): "Total CO2",
    ("bg", "hematocrit"): "Hematocrit (ABG)",
    ("bg", "hemoglobin"): "Hemoglobin (ABG)",
    ("bg", "chloride"): "Chloride (ABG)",
    ("bg", "calcium"): "Calcium (ABG)",
    ("bg", "potassium"): "Potassium (ABG)",
    ("bg", "sodium"): "Sodium (ABG)",
    ("bg", "lactate"): "Lactate",
    ("bg", "glucose"): "Glucose (ABG)",
    ("bg", "aado2"): "A-a gradient",
    ("bg", "aado2_calc"): "A-a gradient (calc)",
    ("bg", "fio2_chartevents"): "FiO2 (charted)",
    ("bg", "pao2fio2ratio"): "PaO2/FiO2 ratio",
    # --- chemistry ---
    ("chemistry", "albumin"): "Albumin",
    ("chemistry", "globulin"): "Globulin",
    ("chemistry", "total_protein"): "Total protein",
    ("chemistry", "aniongap"): "Anion gap",
    ("chemistry", "bicarbonate"): "Bicarbonate",
    ("chemistry", "bun"): "BUN",
    ("chemistry", "calcium"): "Calcium",
    ("chemistry", "chloride"): "Chloride",
    ("chemistry", "creatinine"): "Creatinine",
    ("chemistry", "glucose"): "Glucose (chem)",
    ("chemistry", "sodium"): "Sodium",
    ("chemistry", "potassium"): "Potassium",
    # --- complete_blood_count ---
    ("complete_blood_count", "hematocrit"): "Hematocrit",
    ("complete_blood_count", "hemoglobin"): "Hemoglobin",
    ("complete_blood_count", "mch"): "MCH",
    ("complete_blood_count", "mchc"): "MCHC",
    ("complete_blood_count", "mcv"): "MCV",
    ("complete_blood_count", "platelet"): "Platelet count",
    ("complete_blood_count", "rbc"): "RBC count",
    ("complete_blood_count", "rdw"): "RDW",
    ("complete_blood_count", "rdwsd"): "RDW-SD",
    ("complete_blood_count", "wbc"): "WBC count",
    # --- coagulation ---
    ("coagulation", "d_dimer"): "D-dimer",
    ("coagulation", "fibrinogen"): "Fibrinogen",
    ("coagulation", "thrombin"): "Thrombin time",
    ("coagulation", "inr"): "INR",
    ("coagulation", "pt"): "PT",
    ("coagulation", "ptt"): "PTT",
    # --- enzyme ---
    ("enzyme", "alt"): "ALT",
    ("enzyme", "alp"): "ALP",
    ("enzyme", "ast"): "AST",
    ("enzyme", "amylase"): "Amylase",
    ("enzyme", "bilirubin_total"): "Total bilirubin",
    ("enzyme", "bilirubin_direct"): "Direct bilirubin",
    ("enzyme", "bilirubin_indirect"): "Indirect bilirubin",
    ("enzyme", "ck_cpk"): "CK",
    ("enzyme", "ck_mb"): "CK-MB",
    ("enzyme", "ggt"): "GGT",
    ("enzyme", "ld_ldh"): "LDH",
    # --- inflammation ---
    ("inflammation", "crp"): "CRP",
    # --- cardiac_marker ---
    ("cardiac_marker", "troponin_t"): "Troponin T",
    ("cardiac_marker", "ck_mb"): "CK-MB (cardiac)",
    ("cardiac_marker", "ntprobnp"): "NT-proBNP",
    # --- gcs ---
    ("gcs", "gcs"): "GCS total",
    ("gcs", "gcs_motor"): "GCS motor",
    ("gcs", "gcs_verbal"): "GCS verbal",
    ("gcs", "gcs_eyes"): "GCS eyes",
    ("gcs", "gcs_unable"): "GCS unable",
    # --- urine_output ---
    ("urine_output", "urineoutput"): "Urine output",
    # --- urine_output_rate ---
    ("urine_output_rate", "uo"): "Urine output",
    ("urine_output_rate", "urineoutput_6hr"): "Urine output 6hr",
    ("urine_output_rate", "urineoutput_12hr"): "Urine output 12hr",
    ("urine_output_rate", "urineoutput_24hr"): "Urine output 24hr",
    ("urine_output_rate", "uo_mlkghr_6hr"): "Urine rate 6hr",
    ("urine_output_rate", "uo_mlkghr_12hr"): "Urine rate 12hr",
    ("urine_output_rate", "uo_mlkghr_24hr"): "Urine rate 24hr",
    ("urine_output_rate", "weight"): "Weight",
    # --- kdigo_stages ---
    ("kdigo_stages", "creat"): "Creatinine (KDIGO)",
    ("kdigo_stages", "creat_low_past_7day"): "Creatinine low 7d",
    ("kdigo_stages", "creat_low_past_48hr"): "Creatinine low 48hr",
    ("kdigo_stages", "aki_stage_creat"): "AKI stage (creatinine)",
    ("kdigo_stages", "uo_rt_6hr"): "Urine rate 6hr (KDIGO)",
    ("kdigo_stages", "uo_rt_12hr"): "Urine rate 12hr (KDIGO)",
    ("kdigo_stages", "uo_rt_24hr"): "Urine rate 24hr (KDIGO)",
    ("kdigo_stages", "aki_stage_uo"): "AKI stage (urine)",
    ("kdigo_stages", "aki_stage_crrt"): "AKI stage (CRRT)",
    ("kdigo_stages", "aki_stage"): "AKI stage",
    ("kdigo_stages", "aki_stage_smoothed"): "AKI stage (smoothed)",
    # --- oxygen_delivery ---
    ("oxygen_delivery", "o2_flow"): "O2 flow",
    ("oxygen_delivery", "o2_flow_additional"): "O2 flow (additional)",
    ("oxygen_delivery", "o2_delivery_device_1"): "O2 delivery device",
    # --- blood_differential ---
    ("blood_differential", "wbc"): "WBC (diff)",
    ("blood_differential", "neutrophils_abs"): "Neutrophils",
    ("blood_differential", "lymphocytes_abs"): "Lymphocytes",
    ("blood_differential", "monocytes_abs"): "Monocytes",
    ("blood_differential", "eosinophils_abs"): "Eosinophils",
    ("blood_differential", "basophils_abs"): "Basophils",
    ("blood_differential", "bands"): "Bands",
    ("blood_differential", "immature_granulocytes"): "Immature granulocytes",
    ("blood_differential", "nrbc"): "NRBC",
    # --- height ---
    ("height", "height"): "Height",
    # --- weight ---
    ("weight", "weight"): "Weight",
    ("weight", "weight_type"): "Weight type",
    # --- sofa ---
    ("sofa", "sofa_24hours"): "SOFA total",
    ("sofa", "respiration_24hours"): "SOFA respiration",
    ("sofa", "coagulation_24hours"): "SOFA coagulation",
    ("sofa", "liver_24hours"): "SOFA liver",
    ("sofa", "cardiovascular_24hours"): "SOFA cardiovascular",
    ("sofa", "cns_24hours"): "SOFA CNS",
    ("sofa", "renal_24hours"): "SOFA renal",
    # =====================================================================
    # ACTION VARIABLES
    # =====================================================================
    # --- vasoactive_agent ---
    ("vasoactive_agent", "dopamine"): "Dopamine",
    ("vasoactive_agent", "epinephrine"): "Epinephrine",
    ("vasoactive_agent", "norepinephrine"): "Norepinephrine",
    ("vasoactive_agent", "phenylephrine"): "Phenylephrine",
    ("vasoactive_agent", "vasopressin"): "Vasopressin",
    ("vasoactive_agent", "dobutamine"): "Dobutamine",
    ("vasoactive_agent", "milrinone"): "Milrinone",
    # --- norepinephrine_equivalent_dose ---
    ("norepinephrine_equivalent_dose", "norepinephrine_equivalent_dose"):
        "Norepinephrine equivalent dose",
    # --- antibiotic ---
    ("antibiotic", "antibiotic"): "Antibiotic",
    ("antibiotic", "route"): "Antibiotic route",
    # --- ventilation ---
    ("ventilation", "ventilation_status"): "Ventilation status",
    # --- ventilator_setting ---
    ("ventilator_setting", "respiratory_rate_set"): "RR set",
    ("ventilator_setting", "respiratory_rate_total"): "RR total",
    ("ventilator_setting", "respiratory_rate_spontaneous"): "RR spontaneous",
    ("ventilator_setting", "minute_volume"): "Minute volume",
    ("ventilator_setting", "tidal_volume_set"): "Tidal volume set",
    ("ventilator_setting", "tidal_volume_observed"): "Tidal volume observed",
    ("ventilator_setting", "tidal_volume_spontaneous"): "Tidal volume spontaneous",
    ("ventilator_setting", "plateau_pressure"): "Plateau pressure",
    ("ventilator_setting", "peep"): "PEEP",
    ("ventilator_setting", "fio2"): "FiO2",
    ("ventilator_setting", "flow_rate"): "Flow rate",
    ("ventilator_setting", "ventilator_mode"): "Ventilator mode",
    ("ventilator_setting", "ventilator_mode_hamilton"): "Ventilator mode (Hamilton)",
    ("ventilator_setting", "ventilator_type"): "Ventilator type",
    # --- rrt ---
    ("rrt", "dialysis_present"): "Dialysis present",
    ("rrt", "dialysis_active"): "Dialysis active",
    ("rrt", "dialysis_type"): "Dialysis type",
    # --- crrt ---
    ("crrt", "crrt_mode"): "CRRT mode",
    ("crrt", "access_pressure"): "CRRT access pressure",
    ("crrt", "blood_flow"): "CRRT blood flow",
    ("crrt", "citrate"): "CRRT citrate",
    ("crrt", "current_goal"): "CRRT current goal",
    ("crrt", "dialysate_fluid"): "CRRT dialysate fluid",
    ("crrt", "dialysate_rate"): "CRRT dialysate rate",
    ("crrt", "effluent_pressure"): "CRRT effluent pressure",
    ("crrt", "filter_pressure"): "CRRT filter pressure",
    ("crrt", "heparin_concentration"): "CRRT heparin concentration",
    ("crrt", "heparin_dose"): "CRRT heparin dose",
    ("crrt", "hourly_patient_fluid_removal"): "CRRT hourly fluid removal",
    ("crrt", "prefilter_replacement_rate"): "CRRT prefilter replacement rate",
    ("crrt", "postfilter_replacement_rate"): "CRRT postfilter replacement rate",
    ("crrt", "replacement_fluid"): "CRRT replacement fluid",
    ("crrt", "replacement_rate"): "CRRT replacement rate",
    ("crrt", "return_pressure"): "CRRT return pressure",
    ("crrt", "ultrafiltrate_output"): "CRRT ultrafiltrate output",
    ("crrt", "system_active"): "CRRT system active",
    ("crrt", "clots"): "CRRT clots",
    ("crrt", "clots_increasing"): "CRRT clots increasing",
    ("crrt", "clotted"): "CRRT clotted",
    # --- invasive_line ---
    ("invasive_line", "line_type"): "Invasive line type",
    ("invasive_line", "line_site"): "Invasive line site",
    # --- neuroblock ---
    ("neuroblock", "drug_rate"): "Neuromuscular blocker rate",
    ("neuroblock", "drug_amount"): "Neuromuscular blocker amount",
    # --- rhythm ---
    ("rhythm", "heart_rhythm"): "Heart rhythm",
    ("rhythm", "ectopy_type"): "Ectopy type",
    ("rhythm", "ectopy_frequency"): "Ectopy frequency",
}


CATEGORY_SORT_ORDER: dict[str, int] = {
    "vitals": 0,
    "blood_gas": 1,
    "labs": 2,
    "assessment": 3,
    "score": 4,
    "output": 5,
    "organ_failure": 6,
    "diagnosis": 7,
    "vasopressor": 8,
    "neuromuscular_blocker": 9,
    "ventilation": 10,
    "ventilation_settings": 11,
    "dialysis": 12,
    "antibiotic": 13,
    "fluid_med_input": 14,
    "medication_order": 15,
    "medication_admin": 16,
    "procedure": 17,
}

# Regex for parsing structured text_value from inputevents
_RATE_PATTERN = re.compile(r"rate:([\d.]+)\s*([^|]*)")


# =============================================================================
# Formatting Functions
# =============================================================================


def get_display_name(source_table: str, variable_name: str) -> str:
    """Look up human-readable display name for a variable.

    Args:
        source_table: Source table name (e.g., "vitalsign", "inputevents").
        variable_name: Variable name (e.g., "heart_rate", "NaCl 0.9%").

    Returns:
        Display name string. Falls back to title-cased underscore replacement
        for unmapped variables (common for raw table drug/procedure names).
    """
    name = DISPLAY_NAMES.get((source_table, variable_name))
    if name is not None:
        return name
    return variable_name.replace("_", " ").strip()


def format_numeric(value: int | float) -> str:
    """Format a numeric value with smart rounding.

    - Integers displayed without decimal (88.0 -> "88")
    - Values with decimals keep up to 2 decimal places (37.2 -> "37.2")

    Args:
        value: Numeric value to format.

    Returns:
        Formatted string.
    """
    if value == int(value):
        return str(int(value))
    # Round to 2 decimal places, strip trailing zeros
    formatted = f"{value:.2f}".rstrip("0").rstrip(".")
    return formatted


def parse_raw_text_value(
    variable_name: str,
    numeric_value: Optional[float],
    text_value: str,
    unit: str,
    source_table: str,
) -> str:
    """Parse structured text_value from raw tables into natural language.

    Input patterns from Step 03:
      inputevents:    "rate:75.0 mL/hour | Continuous IV"
      prescriptions:  "route:IV | 20mg Premix Bag"
      procedureevents: "ContinuousProcess" / "Task" / category description
      emar:           plain text (event_txt like "Administered")

    Args:
        variable_name: The drug/procedure name.
        numeric_value: Numeric amount (may be NaN).
        text_value: Structured text string from Step 03.
        unit: Amount unit (e.g., "mL", "mg").
        source_table: One of "inputevents", "prescriptions",
                      "procedureevents", "emar".

    Returns:
        Natural language text fragment.
    """
    name = get_display_name(source_table, variable_name)
    parts = [name]

    has_numeric = numeric_value is not None and not (
        isinstance(numeric_value, float) and np.isnan(numeric_value)
    )

    if source_table == "inputevents":
        # Format: "rate:75.0 mL/hour | Continuous IV"
        if has_numeric:
            parts.append(f": {format_numeric(numeric_value)}")
            if unit:
                parts.append(f" {unit}")

        if text_value:
            rate_match = _RATE_PATTERN.search(text_value)
            if rate_match:
                try:
                    rate_val = format_numeric(float(rate_match.group(1).strip()))
                except ValueError:
                    rate_val = rate_match.group(1).strip()
                rate_unit = rate_match.group(2).strip()
                parts.append(f", {rate_val}")
                if rate_unit:
                    parts.append(f" {rate_unit}")

            # Extract category after pipe
            if "|" in text_value:
                category = text_value.split("|", 1)[1].strip()
                if category:
                    parts.append(f" {category}")

    elif source_table == "prescriptions":
        # Format: "route:IV | 20mg Premix Bag"
        route = ""
        strength = ""
        if text_value:
            if "route:" in text_value:
                after_route = text_value.split("route:", 1)[1]
                if "|" in after_route:
                    route = after_route.split("|", 1)[0].strip()
                    strength = after_route.split("|", 1)[1].strip()
                else:
                    route = after_route.strip()
            elif "|" in text_value:
                strength = text_value.split("|", 1)[1].strip()
            else:
                strength = text_value.strip()

        if has_numeric:
            parts.append(f": {format_numeric(numeric_value)}")
            if unit:
                parts.append(f" {unit}")

        if strength:
            parts.append(f" {strength}")
        if route:
            parts.append(f" {route}")

    elif source_table == "procedureevents":
        # text_value is ordercategorydescription (e.g., "ContinuousProcess")
        if has_numeric:
            parts.append(f": {format_numeric(numeric_value)}")
            if unit:
                parts.append(f" {unit}")
        if text_value and text_value not in ("", "nan"):
            parts.append(f" {text_value}")

    elif source_table == "emar":
        # text_value is event_txt (e.g., "Administered")
        if text_value and text_value not in ("", "nan"):
            parts.append(f": {text_value}")

    else:
        # Fallback for unknown raw tables
        if has_numeric:
            parts.append(f": {format_numeric(numeric_value)}")
            if unit:
                parts.append(f" {unit}")
        if text_value and text_value not in ("", "nan"):
            parts.append(f" ({text_value})")

    return "".join(parts)


def format_event(
    source_table: str,
    variable_name: str,
    numeric_value: Optional[float],
    text_value: Optional[str],
    unit: str,
) -> Optional[str]:
    """Convert a single event into a text fragment.

    Handles three cases:
    1. Numeric only: "Heart rate: 88 bpm"
    2. Text only: "Ventilation status: InvasiveMV"
    3. Both (raw tables): delegates to parse_raw_text_value()

    Args:
        source_table: Source table name.
        variable_name: Variable name.
        numeric_value: Numeric value (may be None/NaN).
        text_value: Text value (may be None/empty).
        unit: Unit string.

    Returns:
        Text fragment string, or None if both values are missing.
    """
    has_numeric = numeric_value is not None and not (
        isinstance(numeric_value, float) and np.isnan(numeric_value)
    )
    has_text = (
        text_value is not None
        and str(text_value) not in ("", "nan", "None")
    )

    if not has_numeric and not has_text:
        return None

    # ICD diagnosis events: "Diagnosis: Sepsis, unspecified organism (A419)"
    if source_table == "diagnoses":
        if has_text:
            return f"Diagnosis: {text_value} ({variable_name})"
        return f"Diagnosis: {variable_name}"

    # Raw table events have structured text_value — use dedicated parser
    if source_table in ("inputevents", "prescriptions", "procedureevents", "emar"):
        return parse_raw_text_value(
            variable_name, numeric_value,
            str(text_value) if has_text else "",
            unit, source_table,
        )

    # Concepts table events: simple format
    name = get_display_name(source_table, variable_name)

    if has_numeric and has_text:
        text_str = str(text_value)
        if unit:
            return f"{name}: {format_numeric(numeric_value)} {unit} ({text_str})"
        return f"{name}: {format_numeric(numeric_value)} ({text_str})"

    if has_numeric:
        if unit:
            return f"{name}: {format_numeric(numeric_value)} {unit}"
        return f"{name}: {format_numeric(numeric_value)}"

    # Text only
    return f"{name}: {text_value}"


def events_to_text_list(events_df: pd.DataFrame) -> list[str]:
    """Convert a DataFrame of step events into a list of individual text strings.

    Events are sorted by CATEGORY_SORT_ORDER -> source_table -> variable_name
    for deterministic ordering. Each event becomes one text fragment.

    Args:
        events_df: DataFrame with columns: variable_name, numeric_value,
                   text_value, unit, category, source_table.

    Returns:
        List of text fragment strings. Empty list if no valid events.
    """
    if len(events_df) == 0:
        return []

    # Sort for deterministic ordering
    sort_keys = events_df["category"].map(
        lambda c: CATEGORY_SORT_ORDER.get(c, 99)
    )
    sorted_df = events_df.assign(_sort_key=sort_keys).sort_values(
        ["_sort_key", "source_table", "variable_name"],
        kind="stable",
    )

    fragments = []
    for _, row in sorted_df.iterrows():
        fragment = format_event(
            row["source_table"],
            row["variable_name"],
            row.get("numeric_value"),
            row.get("text_value"),
            row.get("unit", ""),
        )
        if fragment is not None:
            fragments.append(fragment)

    return fragments


def events_to_text(events_df: pd.DataFrame) -> str:
    """Convert a DataFrame of step events into a concatenated text string.

    Events are sorted by CATEGORY_SORT_ORDER -> source_table -> variable_name
    for deterministic implicit grouping. Joined with ". " (period-space).

    Args:
        events_df: DataFrame with columns: variable_name, numeric_value,
                   text_value, unit, category, source_table.

    Returns:
        Concatenated text string. Empty string if no valid events.
    """
    return ". ".join(events_to_text_list(events_df))


def step_to_texts(stay_data: dict, step_idx: int) -> tuple[str, str]:
    """Generate (obs_text, act_text) for one time step.

    Args:
        stay_data: Discretized stay dict from Step 04 pickle file.
        step_idx: Time step index.

    Returns:
        Tuple of (obs_text, act_text). Either may be empty string.
    """
    obs_events = stay_data["obs_events"]
    act_events = stay_data["act_events"]

    # Filter to this step
    step_obs = obs_events[obs_events["step_idx"] == step_idx]
    step_act = act_events[act_events["step_idx"] == step_idx]

    obs_text = events_to_text(step_obs)
    act_text = events_to_text(step_act)

    return obs_text, act_text


def stay_to_all_texts(stay_data: dict) -> tuple[list[str], list[str]]:
    """Generate per-step text lists for an entire stay.

    Args:
        stay_data: Discretized stay dict from Step 04 pickle file.

    Returns:
        Tuple of (obs_texts, act_texts), each a list of length n_steps.
    """
    n_steps = stay_data["n_steps"]
    obs_events = stay_data["obs_events"]
    act_events = stay_data["act_events"]

    # Pre-group by step_idx for efficiency
    obs_groups = dict(list(obs_events.groupby("step_idx"))) if len(obs_events) > 0 else {}
    act_groups = dict(list(act_events.groupby("step_idx"))) if len(act_events) > 0 else {}

    obs_texts = []
    act_texts = []

    empty_df_obs = pd.DataFrame(columns=obs_events.columns)
    empty_df_act = pd.DataFrame(columns=act_events.columns)

    for t in range(n_steps):
        step_obs = obs_groups.get(t, empty_df_obs)
        step_act = act_groups.get(t, empty_df_act)

        obs_texts.append(events_to_text(step_obs))
        act_texts.append(events_to_text(step_act))

    return obs_texts, act_texts
