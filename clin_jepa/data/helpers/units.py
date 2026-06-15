"""Variable-to-unit mapping for MIMIC-IV observation and action events.

Maps (source_table, variable_name) pairs to their unit strings.
Units sourced from MIMIC-IV documentation and concepts table definitions.
Empty string indicates dimensionless or unknown units.
"""

VARIABLE_UNITS: dict[tuple[str, str], str] = {
    # --- vitalsign ---
    ("vitalsign", "heart_rate"): "bpm",
    ("vitalsign", "sbp"): "mmHg",
    ("vitalsign", "dbp"): "mmHg",
    ("vitalsign", "mbp"): "mmHg",
    ("vitalsign", "sbp_ni"): "mmHg",
    ("vitalsign", "dbp_ni"): "mmHg",
    ("vitalsign", "mbp_ni"): "mmHg",
    ("vitalsign", "resp_rate"): "breaths/min",
    ("vitalsign", "temperature"): "°C",
    ("vitalsign", "spo2"): "%",
    ("vitalsign", "glucose"): "mg/dL",
    # --- bg (blood gas) ---
    ("bg", "so2"): "%",
    ("bg", "po2"): "mmHg",
    ("bg", "pco2"): "mmHg",
    ("bg", "fio2"): "",
    ("bg", "ph"): "",
    ("bg", "baseexcess"): "mEq/L",
    ("bg", "bicarbonate"): "mEq/L",
    ("bg", "totalco2"): "mEq/L",
    ("bg", "hematocrit"): "%",
    ("bg", "hemoglobin"): "g/dL",
    ("bg", "chloride"): "mEq/L",
    ("bg", "calcium"): "mg/dL",
    ("bg", "potassium"): "mEq/L",
    ("bg", "sodium"): "mEq/L",
    ("bg", "lactate"): "mmol/L",
    ("bg", "glucose"): "mg/dL",
    ("bg", "aado2"): "mmHg",
    ("bg", "aado2_calc"): "mmHg",
    ("bg", "fio2_chartevents"): "%",
    ("bg", "pao2fio2ratio"): "",
    # --- chemistry ---
    ("chemistry", "albumin"): "g/dL",
    ("chemistry", "globulin"): "g/dL",
    ("chemistry", "total_protein"): "g/dL",
    ("chemistry", "aniongap"): "mEq/L",
    ("chemistry", "bicarbonate"): "mEq/L",
    ("chemistry", "bun"): "mg/dL",
    ("chemistry", "calcium"): "mg/dL",
    ("chemistry", "chloride"): "mEq/L",
    ("chemistry", "creatinine"): "mg/dL",
    ("chemistry", "glucose"): "mg/dL",
    ("chemistry", "sodium"): "mEq/L",
    ("chemistry", "potassium"): "mEq/L",
    # --- complete_blood_count ---
    ("complete_blood_count", "hematocrit"): "%",
    ("complete_blood_count", "hemoglobin"): "g/dL",
    ("complete_blood_count", "mch"): "pg",
    ("complete_blood_count", "mchc"): "g/dL",
    ("complete_blood_count", "mcv"): "fL",
    ("complete_blood_count", "platelet"): "K/uL",
    ("complete_blood_count", "rbc"): "M/uL",
    ("complete_blood_count", "rdw"): "%",
    ("complete_blood_count", "rdwsd"): "fL",
    ("complete_blood_count", "wbc"): "K/uL",
    # --- coagulation ---
    ("coagulation", "d_dimer"): "ng/mL",
    ("coagulation", "fibrinogen"): "mg/dL",
    ("coagulation", "thrombin"): "sec",
    ("coagulation", "inr"): "",
    ("coagulation", "pt"): "sec",
    ("coagulation", "ptt"): "sec",
    # --- enzyme ---
    ("enzyme", "alt"): "IU/L",
    ("enzyme", "alp"): "IU/L",
    ("enzyme", "ast"): "IU/L",
    ("enzyme", "amylase"): "IU/L",
    ("enzyme", "bilirubin_total"): "mg/dL",
    ("enzyme", "bilirubin_direct"): "mg/dL",
    ("enzyme", "bilirubin_indirect"): "mg/dL",
    ("enzyme", "ck_cpk"): "IU/L",
    ("enzyme", "ck_mb"): "ng/mL",
    ("enzyme", "ggt"): "IU/L",
    ("enzyme", "ld_ldh"): "IU/L",
    # --- inflammation ---
    ("inflammation", "crp"): "mg/L",
    # --- cardiac_marker ---
    ("cardiac_marker", "troponin_t"): "ng/mL",
    ("cardiac_marker", "ck_mb"): "ng/mL",
    ("cardiac_marker", "ntprobnp"): "pg/mL",
    # --- gcs ---
    ("gcs", "gcs"): "",
    ("gcs", "gcs_motor"): "",
    ("gcs", "gcs_verbal"): "",
    ("gcs", "gcs_eyes"): "",
    ("gcs", "gcs_unable"): "",
    # --- urine_output ---
    ("urine_output", "urineoutput"): "mL",
    # --- urine_output_rate ---
    ("urine_output_rate", "uo"): "mL",
    ("urine_output_rate", "urineoutput_6hr"): "mL",
    ("urine_output_rate", "urineoutput_12hr"): "mL",
    ("urine_output_rate", "urineoutput_24hr"): "mL",
    ("urine_output_rate", "uo_mlkghr_6hr"): "mL/kg/hr",
    ("urine_output_rate", "uo_mlkghr_12hr"): "mL/kg/hr",
    ("urine_output_rate", "uo_mlkghr_24hr"): "mL/kg/hr",
    ("urine_output_rate", "weight"): "kg",
    # --- kdigo_stages ---
    ("kdigo_stages", "creat"): "mg/dL",
    ("kdigo_stages", "creat_low_past_7day"): "mg/dL",
    ("kdigo_stages", "creat_low_past_48hr"): "mg/dL",
    ("kdigo_stages", "aki_stage_creat"): "",
    ("kdigo_stages", "uo_rt_6hr"): "mL/kg/hr",
    ("kdigo_stages", "uo_rt_12hr"): "mL/kg/hr",
    ("kdigo_stages", "uo_rt_24hr"): "mL/kg/hr",
    ("kdigo_stages", "aki_stage_uo"): "",
    ("kdigo_stages", "aki_stage_crrt"): "",
    ("kdigo_stages", "aki_stage"): "",
    ("kdigo_stages", "aki_stage_smoothed"): "",
    # --- oxygen_delivery ---
    ("oxygen_delivery", "o2_flow"): "L/min",
    ("oxygen_delivery", "o2_flow_additional"): "L/min",
    ("oxygen_delivery", "o2_delivery_device_1"): "",
    # --- blood_differential ---
    ("blood_differential", "wbc"): "K/uL",
    ("blood_differential", "neutrophils_abs"): "K/uL",
    ("blood_differential", "lymphocytes_abs"): "K/uL",
    ("blood_differential", "monocytes_abs"): "K/uL",
    ("blood_differential", "eosinophils_abs"): "K/uL",
    ("blood_differential", "basophils_abs"): "K/uL",
    ("blood_differential", "bands"): "%",
    ("blood_differential", "immature_granulocytes"): "%",
    ("blood_differential", "nrbc"): "/100 WBC",
    # --- height ---
    ("height", "height"): "cm",
    # --- weight ---
    ("weight", "weight"): "kg",
    ("weight", "weight_type"): "",
    # --- sofa ---
    ("sofa", "sofa_24hours"): "",
    ("sofa", "respiration_24hours"): "",
    ("sofa", "coagulation_24hours"): "",
    ("sofa", "liver_24hours"): "",
    ("sofa", "cardiovascular_24hours"): "",
    ("sofa", "cns_24hours"): "",
    ("sofa", "renal_24hours"): "",
    # =====================================================================
    # ACTION VARIABLES (Step 03)
    # =====================================================================
    # --- vasoactive_agent ---
    ("vasoactive_agent", "dopamine"): "mcg/kg/min",
    ("vasoactive_agent", "epinephrine"): "mcg/kg/min",
    ("vasoactive_agent", "norepinephrine"): "mcg/kg/min",
    ("vasoactive_agent", "phenylephrine"): "mcg/kg/min",
    ("vasoactive_agent", "vasopressin"): "units/hour",
    ("vasoactive_agent", "dobutamine"): "mcg/kg/min",
    ("vasoactive_agent", "milrinone"): "mcg/kg/min",
    # --- norepinephrine_equivalent_dose ---
    ("norepinephrine_equivalent_dose", "norepinephrine_equivalent_dose"): "mcg/kg/min",
    # --- antibiotic ---
    ("antibiotic", "antibiotic"): "",
    ("antibiotic", "route"): "",
    # --- ventilation ---
    ("ventilation", "ventilation_status"): "",
    # --- ventilator_setting ---
    ("ventilator_setting", "respiratory_rate_set"): "breaths/min",
    ("ventilator_setting", "respiratory_rate_total"): "breaths/min",
    ("ventilator_setting", "respiratory_rate_spontaneous"): "breaths/min",
    ("ventilator_setting", "minute_volume"): "L/min",
    ("ventilator_setting", "tidal_volume_set"): "mL",
    ("ventilator_setting", "tidal_volume_observed"): "mL",
    ("ventilator_setting", "tidal_volume_spontaneous"): "mL",
    ("ventilator_setting", "plateau_pressure"): "cmH2O",
    ("ventilator_setting", "peep"): "cmH2O",
    ("ventilator_setting", "fio2"): "%",
    ("ventilator_setting", "flow_rate"): "L/min",
    ("ventilator_setting", "ventilator_mode"): "",
    ("ventilator_setting", "ventilator_mode_hamilton"): "",
    ("ventilator_setting", "ventilator_type"): "",
    # --- rrt ---
    ("rrt", "dialysis_present"): "",
    ("rrt", "dialysis_active"): "",
    ("rrt", "dialysis_type"): "",
    # --- crrt ---
    ("crrt", "crrt_mode"): "",
    ("crrt", "access_pressure"): "mmHg",
    ("crrt", "blood_flow"): "mL/min",
    ("crrt", "citrate"): "mL/hr",
    ("crrt", "current_goal"): "mL",
    ("crrt", "dialysate_fluid"): "",
    ("crrt", "dialysate_rate"): "mL/hr",
    ("crrt", "effluent_pressure"): "mmHg",
    ("crrt", "filter_pressure"): "mmHg",
    ("crrt", "heparin_concentration"): "units/mL",
    ("crrt", "heparin_dose"): "units/hr",
    ("crrt", "hourly_patient_fluid_removal"): "mL/hr",
    ("crrt", "prefilter_replacement_rate"): "mL/hr",
    ("crrt", "postfilter_replacement_rate"): "mL/hr",
    ("crrt", "replacement_fluid"): "",
    ("crrt", "replacement_rate"): "mL/hr",
    ("crrt", "return_pressure"): "mmHg",
    ("crrt", "ultrafiltrate_output"): "mL",
    ("crrt", "system_active"): "",
    ("crrt", "clots"): "",
    ("crrt", "clots_increasing"): "",
    ("crrt", "clotted"): "",
    # --- invasive_line ---
    ("invasive_line", "line_type"): "",
    ("invasive_line", "line_site"): "",
    # --- neuroblock ---
    ("neuroblock", "drug_rate"): "mcg/kg/min",
    ("neuroblock", "drug_amount"): "mg",
    # --- rhythm ---
    ("rhythm", "heart_rhythm"): "",
    ("rhythm", "ectopy_type"): "",
    ("rhythm", "ectopy_frequency"): "",
}


def get_unit(source_table: str, variable_name: str) -> str:
    """Look up the unit for a given (source_table, variable_name) pair.

    Returns empty string if not found.
    """
    return VARIABLE_UNITS.get((source_table, variable_name), "")
