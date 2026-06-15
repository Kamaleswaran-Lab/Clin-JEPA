"""Noise filter for ``act_events`` before action-text generation.

Applied to a stay's ``act_events`` DataFrame before the per-step
action-text formatter runs:

1. Drop noise ``source_table`` rows (equipment duration tracking, static
   device descriptors — not clinical decisions).
2. Drop a small allowlist of ``(source_table, variable_name)`` pairs that
   are pure carriers, plumbing, flushes, or vendor metadata inside
   otherwise kept tables.
3. Collapse the temporal-discretization step's ``is_continuous=True``
   fan-out on the ``prescriptions`` source table: a 5-day Aspirin order
   is expanded into 120 hourly rows by the temporal-discretization step
   (:mod:`clin_jepa.data.step04_discretization`); we emit it at the
   earliest ``step_idx`` only (one row per unique
   ``(variable_name, text_value, numeric_value, unit)`` combination
   per stay).
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Whole-table drops
# ---------------------------------------------------------------------------

ACTION_SOURCE_TABLES_TO_DROP: set[str] = {
    "procedureevents",
    "invasive_line",
}


# ---------------------------------------------------------------------------
# Variable-level drops within KEPT tables
# ---------------------------------------------------------------------------
# Each entry is (source_table, variable_name).

ACTION_VARIABLES_TO_DROP: set[tuple[str, str]] = {
    # -----------------------------------------------------------------------
    # inputevents: carrier / plumbing / flush rows
    # -----------------------------------------------------------------------
    ("inputevents", "Solution"),        # 4267 events — generic "Solution" label
    ("inputevents", "Sterile Water"),   # 1484 events — diluent carrier
    ("inputevents", "Piggyback"),       #  528 events — generic piggyback infusion
    ("inputevents", "GT Flush"),        #  415 events — gastrostomy tube maintenance flush

    # -----------------------------------------------------------------------
    # prescriptions: carriers / flushes / diluent abbreviations.
    # -----------------------------------------------------------------------
    ("prescriptions", "Bag"),                                  # 20803 — generic IV bag
    ("prescriptions", "Sterile Water"),                        # 13734 — diluent
    ("prescriptions", "SW"),                                   #  8919 — Sterile Water abbreviation
    ("prescriptions", "Iso-Osmotic Dextrose"),                 #  6825 — drug diluent
    ("prescriptions", "Iso-Osmotic Sodium Chloride"),          #  5642 — drug diluent
    ("prescriptions", "Sodium Chloride 0.9%  Flush"),          # 19770 — IV line flush (note double-space after "0.9%")
    ("prescriptions", "Sodium Chloride 0.9%  Flush for CRRT"), #   200 — CRRT line flush
    ("prescriptions", "Heparin Flush (10 units/ml)"),          #  2937 — line anticoagulation flush
    ("prescriptions", "Heparin Flush (100 units/ml)"),         #   569 — line anticoagulation flush
    ("prescriptions", "Heparin Flush (1000 units/mL)"),        #   899 — line anticoagulation flush
    ("prescriptions", "Heparin Flush (5000 Units/mL)"),        #    24 — line anticoagulation flush

    # -----------------------------------------------------------------------
    # ventilator_setting: equipment vendor metadata, not a clinical decision
    # -----------------------------------------------------------------------
    ("ventilator_setting", "ventilator_type"),       # 1689

    # -----------------------------------------------------------------------
    # emar: content-free placeholders + shadow entries duplicating prescriptions
    # -----------------------------------------------------------------------
    ("emar", "unknown_medication"),                  #   73
    ("emar", "Sodium Chloride 0.9%  Flush"),         #    8 — emar shadow of prescriptions flush
                                                      #    (note double-space, matches prescriptions entry)
    ("emar", "Heparin Flush (10 units/ml)"),         #  404 — emar shadow

    # -----------------------------------------------------------------------
    # prescriptions: additional carrier variants.
    # -----------------------------------------------------------------------
    ("prescriptions", "Syringe (Iso-Osmotic Dextrose)"),  # 183 — syringe-form diluent
    ("prescriptions", "Solution"),                        # 125 — bare carrier

    # -----------------------------------------------------------------------
    # Additional noise entries
    # -----------------------------------------------------------------------

    # --- Bare delivery devices (~2.4M total noise rows) ---
    ("prescriptions", "Vial"),                            # 2,034,524 — "Send Vial" placeholder
    ("prescriptions", "Syringe"),                         #   325,637 — bare syringe
    ("prescriptions", "Bottle"),                          #    10,608 — bare bottle

    # --- Heparin Flush variants by line type (line maintenance, not therapy) ---
    ("prescriptions", "Heparin Flush CVL  (100 units/ml)"),    # 6,266  (note double-space "CVL  ")
    ("prescriptions", "Heparin Flush CRRT (5000 Units/mL)"),   # 3,946
    ("prescriptions", "Heparin Flush PICC (100 units/ml)"),    # 2,128
    ("prescriptions", "Heparin Flush Hickman (100 units/ml)"), #   297
    ("prescriptions", "Heparin Flush Port (10 units/mL)"),     #   279
    ("prescriptions", "Heparin Flush"),                        #    46  — bare heparin flush
    ("prescriptions", "Heparin Flush "),                       #    45  — bare with TRAILING SPACE
    ("prescriptions", "Heparin Flush (100 units/mL)"),         #    43  — case variant (capital M)
    ("prescriptions", "Heparin Flush (10 Units/mL)"),          #     5  — case variant (caps U+M)

    # --- Syringe (X) carrier variants — drug name absent, just delivery vehicle ---
    ("prescriptions", "Syringe (0.9% Sodium Chloride)"),   # 63,434
    ("prescriptions", "Syringe (SW)"),                     # 28,631  — sterile water syringe
    ("prescriptions", "Syringe (Sterile Water)"),          # 22,325
    ("prescriptions", "Syringe (Chemo)"),                  #  4,313  — chemo carrier
    ("prescriptions", "Syringe (NS)"),                     #  2,930  — normal saline carrier
    ("prescriptions", "Syringe (Intraventricular)"),       #  1,205  — IVT route carrier
    ("prescriptions", "Syringe (Intraperitoneal)"),        #    988  — IP route carrier
    ("prescriptions", "Syringe (Inhalation)"),             #    848
    ("prescriptions", "Syringe (Intrathecal)"),            #    480
    ("prescriptions", "Syringe (LR)"),                     #    357  — Lactated Ringers carrier
    ("prescriptions", "Syringe (PD)"),                     #    232  — peritoneal dialysis carrier
    ("prescriptions", "Syringe (PCA)"),                    #    212  — patient-controlled analgesia carrier
    ("prescriptions", "Syringe (Intrapleural)"),           #     72
    ("prescriptions", "Syringe (0.9% Sodium Chloride"),    #      1  — TYPO: missing close paren

    # --- Sterile Water / Diluent variants ---
    ("prescriptions", "VIal (Sterile Water)"),             #    446  — TYPO: capital "VI"
    ("prescriptions", "Sterile Water For Irrigation"),     #     72  — irrigation, not therapy
    ("prescriptions", "Sterile Diluent for Flolan"),       #  2,074  — explicit diluent
    ("prescriptions", "Sterile Diluent for Remodulin"),    #  1,730  — explicit diluent

    # --- Saline 0.45% Flush ---
    ("prescriptions", "Sodium Chloride 0.45% Flush"),      #     55  — explicit flush
}


_PRESCRIPTIONS_COLLAPSE_SUBSET = [
    "variable_name",
    "text_value",
    "numeric_value",
    "unit",
]


def filter_act_events(df: pd.DataFrame) -> pd.DataFrame:
    """Apply noise filter + prescriptions continuous-fan-out collapse.

    Args:
        df: ``act_events`` DataFrame for a single stay. Columns must include
            ``step_idx``, ``source_table``, ``variable_name``, ``numeric_value``,
            ``text_value``, ``unit``. Other columns (``category``,
            ``is_continuous``) are preserved but not inspected by the filter.

    Returns:
        Filtered DataFrame with the same columns. Row order is
        ``(step_idx, source_table, variable_name)`` after collapse to make
        downstream per-step grouping deterministic.
    """
    if len(df) == 0:
        return df

    # Stage 1-2: boolean mask combining whole-table and variable-level drops.
    mask = ~df["source_table"].isin(ACTION_SOURCE_TABLES_TO_DROP)
    for source_table, variable_name in ACTION_VARIABLES_TO_DROP:
        mask &= ~(
            (df["source_table"] == source_table)
            & (df["variable_name"] == variable_name)
        )
    df = df.loc[mask].reset_index(drop=True)

    if len(df) == 0 or not (df["source_table"] == "prescriptions").any():
        return df

    presc_mask = df["source_table"] == "prescriptions"
    non_presc = df.loc[~presc_mask]
    presc = df.loc[presc_mask]

    presc_first = (
        presc.sort_values("step_idx", kind="stable")
             .drop_duplicates(
                 subset=_PRESCRIPTIONS_COLLAPSE_SUBSET,
                 keep="first",
             )
    )

    df = pd.concat([non_presc, presc_first], ignore_index=True)
    df = df.sort_values(
        ["step_idx", "source_table", "variable_name"],
        kind="stable",
    ).reset_index(drop=True)
    return df
