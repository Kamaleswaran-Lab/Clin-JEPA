"""Canonical per-hour event ordering and value-presence helpers.

Used by the trajectory builder (:mod:`clin_jepa.data.step06_trajectories`) to
order within-hour clinical events by a fixed clinical priority and to drop
empty event rows.

``CANONICAL_VARIABLE_ORDER`` maps ``(source_table, variable_name)`` to an
integer sort priority (lower = earlier). Unmapped variables get 9999.
Ordering: SOFA -> Vitals -> Blood gas -> Labs -> Assessment -> Output ->
Organ failure -> Vasopressors -> Ventilation -> Dialysis -> Antibiotic ->
Neuromuscular blockade -> Invasive lines.
"""

import numpy as np
import pandas as pd


CANONICAL_VARIABLE_ORDER: dict[tuple[str, str], int] = {
    # --- SOFA scores (0-99) ---
    ("sofa", "sofa_24hours"): 0,
    ("sofa", "respiration_24hours"): 1,
    ("sofa", "coagulation_24hours"): 2,
    ("sofa", "liver_24hours"): 3,
    ("sofa", "cardiovascular_24hours"): 4,
    ("sofa", "cns_24hours"): 5,
    ("sofa", "renal_24hours"): 6,
    # --- Vitals (100-199) ---
    ("vitalsign", "heart_rate"): 100,
    ("vitalsign", "sbp"): 101,
    ("vitalsign", "dbp"): 102,
    ("vitalsign", "mbp"): 103,
    ("vitalsign", "sbp_ni"): 104,
    ("vitalsign", "dbp_ni"): 105,
    ("vitalsign", "mbp_ni"): 106,
    ("vitalsign", "resp_rate"): 107,
    ("vitalsign", "temperature"): 108,
    ("vitalsign", "spo2"): 109,
    ("vitalsign", "glucose"): 110,
    ("rhythm", "heart_rhythm"): 111,
    ("rhythm", "ectopy_type"): 112,
    ("rhythm", "ectopy_frequency"): 113,
    ("oxygen_delivery", "o2_flow"): 114,
    ("oxygen_delivery", "o2_flow_additional"): 115,
    ("oxygen_delivery", "o2_delivery_device_1"): 116,
    ("height", "height"): 117,
    ("weight", "weight"): 118,
    ("weight", "weight_type"): 119,
    # --- Blood gas (200-299) ---
    ("bg", "po2"): 200,
    ("bg", "pco2"): 201,
    ("bg", "ph"): 202,
    ("bg", "lactate"): 203,
    ("bg", "so2"): 204,
    ("bg", "fio2"): 205,
    ("bg", "baseexcess"): 206,
    ("bg", "bicarbonate"): 207,
    ("bg", "totalco2"): 208,
    ("bg", "hematocrit"): 209,
    ("bg", "hemoglobin"): 210,
    ("bg", "chloride"): 211,
    ("bg", "calcium"): 212,
    ("bg", "potassium"): 213,
    ("bg", "sodium"): 214,
    ("bg", "glucose"): 215,
    ("bg", "aado2"): 216,
    ("bg", "aado2_calc"): 217,
    ("bg", "fio2_chartevents"): 218,
    ("bg", "pao2fio2ratio"): 219,
    # --- Labs: Chemistry (300-319) ---
    ("chemistry", "creatinine"): 300,
    ("chemistry", "bun"): 301,
    ("chemistry", "albumin"): 302,
    ("chemistry", "globulin"): 303,
    ("chemistry", "total_protein"): 304,
    ("chemistry", "aniongap"): 305,
    ("chemistry", "bicarbonate"): 306,
    ("chemistry", "calcium"): 307,
    ("chemistry", "chloride"): 308,
    ("chemistry", "glucose"): 309,
    ("chemistry", "sodium"): 310,
    ("chemistry", "potassium"): 311,
    # --- Labs: CBC (320-339) ---
    ("complete_blood_count", "wbc"): 320,
    ("complete_blood_count", "hemoglobin"): 321,
    ("complete_blood_count", "hematocrit"): 322,
    ("complete_blood_count", "platelet"): 323,
    ("complete_blood_count", "rbc"): 324,
    ("complete_blood_count", "mch"): 325,
    ("complete_blood_count", "mchc"): 326,
    ("complete_blood_count", "mcv"): 327,
    ("complete_blood_count", "rdw"): 328,
    ("complete_blood_count", "rdwsd"): 329,
    # --- Labs: Coagulation (340-349) ---
    ("coagulation", "inr"): 340,
    ("coagulation", "pt"): 341,
    ("coagulation", "ptt"): 342,
    ("coagulation", "d_dimer"): 343,
    ("coagulation", "fibrinogen"): 344,
    ("coagulation", "thrombin"): 345,
    # --- Labs: Enzymes (350-369) ---
    ("enzyme", "alt"): 350,
    ("enzyme", "ast"): 351,
    ("enzyme", "alp"): 352,
    ("enzyme", "bilirubin_total"): 353,
    ("enzyme", "bilirubin_direct"): 354,
    ("enzyme", "bilirubin_indirect"): 355,
    ("enzyme", "amylase"): 356,
    ("enzyme", "ck_cpk"): 357,
    ("enzyme", "ck_mb"): 358,
    ("enzyme", "ggt"): 359,
    ("enzyme", "ld_ldh"): 360,
    # --- Labs: Inflammation (370-374) ---
    ("inflammation", "crp"): 370,
    # --- Labs: Cardiac markers (375-379) ---
    ("cardiac_marker", "troponin_t"): 375,
    ("cardiac_marker", "ck_mb"): 376,
    ("cardiac_marker", "ntprobnp"): 377,
    # --- Labs: Blood differential (380-399) ---
    ("blood_differential", "wbc"): 380,
    ("blood_differential", "neutrophils_abs"): 381,
    ("blood_differential", "lymphocytes_abs"): 382,
    ("blood_differential", "monocytes_abs"): 383,
    ("blood_differential", "eosinophils_abs"): 384,
    ("blood_differential", "basophils_abs"): 385,
    ("blood_differential", "bands"): 386,
    ("blood_differential", "immature_granulocytes"): 387,
    ("blood_differential", "nrbc"): 388,
    # --- Assessment: GCS (400-499) ---
    ("gcs", "gcs"): 400,
    ("gcs", "gcs_motor"): 401,
    ("gcs", "gcs_verbal"): 402,
    ("gcs", "gcs_eyes"): 403,
    ("gcs", "gcs_unable"): 404,
    # --- Output (500-599) ---
    ("urine_output", "urineoutput"): 500,
    ("urine_output_rate", "uo"): 501,
    ("urine_output_rate", "urineoutput_6hr"): 502,
    ("urine_output_rate", "urineoutput_12hr"): 503,
    ("urine_output_rate", "urineoutput_24hr"): 504,
    ("urine_output_rate", "uo_mlkghr_6hr"): 505,
    ("urine_output_rate", "uo_mlkghr_12hr"): 506,
    ("urine_output_rate", "uo_mlkghr_24hr"): 507,
    ("urine_output_rate", "weight"): 508,
    # --- Organ failure: KDIGO (600-699) ---
    ("kdigo_stages", "creat"): 600,
    ("kdigo_stages", "creat_low_past_7day"): 601,
    ("kdigo_stages", "creat_low_past_48hr"): 602,
    ("kdigo_stages", "aki_stage_creat"): 603,
    ("kdigo_stages", "uo_rt_6hr"): 604,
    ("kdigo_stages", "uo_rt_12hr"): 605,
    ("kdigo_stages", "uo_rt_24hr"): 606,
    ("kdigo_stages", "aki_stage_uo"): 607,
    ("kdigo_stages", "aki_stage_crrt"): 608,
    ("kdigo_stages", "aki_stage"): 609,
    ("kdigo_stages", "aki_stage_smoothed"): 610,
    # --- Vasopressors (700-749) ---
    ("vasoactive_agent", "norepinephrine"): 700,
    ("vasoactive_agent", "epinephrine"): 701,
    ("vasoactive_agent", "dopamine"): 702,
    ("vasoactive_agent", "dobutamine"): 703,
    ("vasoactive_agent", "phenylephrine"): 704,
    ("vasoactive_agent", "vasopressin"): 705,
    ("vasoactive_agent", "milrinone"): 706,
    ("norepinephrine_equivalent_dose", "norepinephrine_equivalent_dose"): 710,
    # --- Ventilation (750-799) ---
    ("ventilation", "ventilation_status"): 750,
    ("ventilator_setting", "ventilator_mode"): 751,
    ("ventilator_setting", "ventilator_mode_hamilton"): 752,
    ("ventilator_setting", "ventilator_type"): 753,
    ("ventilator_setting", "fio2"): 754,
    ("ventilator_setting", "peep"): 755,
    ("ventilator_setting", "plateau_pressure"): 756,
    ("ventilator_setting", "tidal_volume_set"): 757,
    ("ventilator_setting", "tidal_volume_observed"): 758,
    ("ventilator_setting", "tidal_volume_spontaneous"): 759,
    ("ventilator_setting", "respiratory_rate_set"): 760,
    ("ventilator_setting", "respiratory_rate_total"): 761,
    ("ventilator_setting", "respiratory_rate_spontaneous"): 762,
    ("ventilator_setting", "minute_volume"): 763,
    ("ventilator_setting", "flow_rate"): 764,
    # --- Dialysis (800-849) ---
    ("rrt", "dialysis_present"): 800,
    ("rrt", "dialysis_active"): 801,
    ("rrt", "dialysis_type"): 802,
    ("crrt", "crrt_mode"): 810,
    ("crrt", "blood_flow"): 811,
    ("crrt", "dialysate_rate"): 812,
    ("crrt", "dialysate_fluid"): 813,
    ("crrt", "access_pressure"): 814,
    ("crrt", "effluent_pressure"): 815,
    ("crrt", "filter_pressure"): 816,
    ("crrt", "return_pressure"): 817,
    ("crrt", "citrate"): 818,
    ("crrt", "current_goal"): 819,
    ("crrt", "heparin_concentration"): 820,
    ("crrt", "heparin_dose"): 821,
    ("crrt", "hourly_patient_fluid_removal"): 822,
    ("crrt", "prefilter_replacement_rate"): 823,
    ("crrt", "postfilter_replacement_rate"): 824,
    ("crrt", "replacement_fluid"): 825,
    ("crrt", "replacement_rate"): 826,
    ("crrt", "ultrafiltrate_output"): 827,
    ("crrt", "system_active"): 828,
    ("crrt", "clots"): 829,
    ("crrt", "clots_increasing"): 830,
    ("crrt", "clotted"): 831,
    # --- Antibiotic (850-899) ---
    ("antibiotic", "antibiotic"): 850,
    ("antibiotic", "route"): 851,
    # --- Neuromuscular blockade (900-949) ---
    ("neuroblock", "drug_rate"): 900,
    ("neuroblock", "drug_amount"): 901,
    # --- Invasive lines (950-999) ---
    ("invasive_line", "line_type"): 950,
    ("invasive_line", "line_site"): 951,
}


def _sort_key(source_table: str, variable_name: str) -> int:
    """Get canonical sort priority for a variable. Unmapped variables get 9999."""
    return CANONICAL_VARIABLE_ORDER.get((source_table, variable_name), 9999)


def _sort_events(deduped: pd.DataFrame) -> pd.DataFrame:
    """Sort deduplicated events by canonical variable order.

    Args:
        deduped: Deduplicated events DataFrame with source_table, variable_name.

    Returns:
        DataFrame sorted by canonical priority.
    """
    if len(deduped) == 0:
        return deduped
    priorities = deduped.apply(
        lambda row: _sort_key(row["source_table"], row["variable_name"]),
        axis=1,
    )
    return deduped.assign(_priority=priorities).sort_values(
        "_priority", kind="stable",
    ).drop(columns=["_priority"])


def _has_value(numeric_value, text_value) -> bool:
    """Check if an event row has a meaningful value (numeric or text)."""
    has_numeric = numeric_value is not None and not (
        isinstance(numeric_value, float) and np.isnan(numeric_value)
    )
    has_text = (
        text_value is not None
        and str(text_value) not in ("", "nan", "None")
    )
    return has_numeric or has_text
