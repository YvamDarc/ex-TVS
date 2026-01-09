# app.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import date
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import streamlit as st


# ==========================================================
# CONFIG STREAMLIT
# ==========================================================
st.set_page_config(
    page_title="Taxe véhicules de tourisme (ex-TVS) — 2026",
    layout="wide",
)

# ==========================================================
# BARÈMES 2026 (selon ton cadrage)
# ==========================================================

# CO2 WLTP 2026 : tranches (start_g, end_g_inclusive, rate_eur_per_g)
WLTP_2026: List[Tuple[int, int, int]] = [
    (0, 4, 0),
    (5, 45, 1),
    (46, 53, 2),
    (54, 85, 3),
    (86, 105, 4),
    (106, 125, 10),
    (126, 145, 50),
    (146, 165, 60),
    (166, 10**9, 65),
]

# CO2 NEDC 2026
NEDC_2026: List[Tuple[int, int, int]] = [
    (0, 3, 0),
    (4, 37, 1),
    (38, 44, 2),
    (45, 70, 3),
    (71, 87, 4),
    (88, 103, 10),
    (104, 120, 50),
    (121, 136, 60),
    (137, 10**9, 65),
]

# Barème Puissance Administrative 2026 : (start_cv, end_cv_inclusive, rate_eur_per_cv)
PA_2026: List[Tuple[int, int, int]] = [
    (1, 3, 2000),
    (4, 6, 3000),
    (7, 10, 4500),
    (11, 15, 5250),
    (16, 10**9, 6500),
]

# Polluants 2026 par groupe
POLLUTANTS_2026: Dict[str, int] = {"E": 0, "1": 100, "P": 500}

# Coeff IK (frais kilométriques)
IK_COEFF_TABLE = [
    (0, 15000, 0.00, "0–15 000 km => coeff 0 %"),
    (15001, 25000, 0.25, "15 001–25 000 km => coeff 25 %"),
    (25001, 35000, 0.50, "25 001–35 000 km => coeff 50 %"),
    (35001, 45000, 0.75, "35 001–45 000 km => coeff 75 %"),
    (45001, 10**9, 1.00, "> 45 000 km => coeff 100 %"),
]


# ==========================================================
# OUTILS / HELPERS ROBUSTES
# ==========================================================
def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def euro_round(x: float) -> int:
    """Arrondi fiscal à l'euro : >= 0,50 vers le haut."""
    try:
        return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(abs(x) + 0.5))
    except Exception:
        return 0


def is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def days_in_year(year: int) -> int:
    return 366 if is_leap_year(year) else 365


def clamp_date_to_year(d: date, year: int) -> date:
    if d < date(year, 1, 1):
        return date(year, 1, 1)
    if d > date(year, 12, 31):
        return date(year, 12, 31)
    return d


def overlap_days_in_year(start: date, end: date, year: int) -> int:
    """Nombre de jours inclusifs entre start et end bornés à l'année."""
    s = clamp_date_to_year(start, year)
    e = clamp_date_to_year(end, year)
    if e < s:
        return 0
    return (e - s).days + 1


def json_dumps_safe(obj: Any) -> str:
    """Sérialisation JSON robuste (dates -> str)."""
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def bracket_progressive_integer(value: int, brackets: List[Tuple[int, int, int]]) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Calcul progressif par tranches sur une valeur entière.
    Ex : CV ou autres
    """
    total = 0
    details: List[Dict[str, Any]] = []
    v = max(0, safe_int(value, 0))

    for a, b, rate in brackets:
        if v < a:
            continue
        upper = min(v, b)
        qty = upper - a + 1
        if qty <= 0:
            continue
        part = qty * rate
        total += part
        details.append({"tranche": f"{a}–{upper}", "unites": qty, "taux": rate, "montant": part})
        if v <= b:
            break

    return total, details


def bracket_progressive_co2(value: float, brackets: List[Tuple[int, int, int]]) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Calcul progressif par gramme CO2.
    On arrondit la valeur à l'entier le plus proche car la CG donne un entier en général.
    """
    v = max(0, int(round(safe_float(value, 0.0))))
    total = 0
    details: List[Dict[str, Any]] = []

    for a, b, rate in brackets:
        if v < a:
            continue
        upper = min(v, b)
        qty = upper - a + 1
        if qty <= 0:
            continue
        part = qty * rate
        total += part
        details.append({"tranche_g": f"{a}–{upper}", "grammes": qty, "taux_€/g": rate, "montant": part})
        if v <= b:
            break

    return total, details


def ik_coefficient(km: int) -> Tuple[float, str]:
    k = max(0, safe_int(km, 0))
    for a, b, coeff, label in IK_COEFF_TABLE:
        if a <= k <= b:
            return coeff, label
    return 1.00, "> 45 000 km => coeff 100 %"


def critair_group(label: str) -> str:
    """
    Retourne E / 1 / P
    """
    txt = (label or "").strip().upper()
    if txt in {"E", "EV", "ELECTRIQUE", "ÉLECTRIQUE", "HYDROGENE", "HYDROGÈNE", "VERT", "VERTE"}:
        return "E"
    if txt in {"1", "CRIT1", "CRIT'1", "CRIT’1", "VIOLET", "VIOLETTE"}:
        return "1"
    # tout le reste => P
    return "P"


# ==========================================================
# MODELES DONNEES
# ==========================================================
@dataclass
class VehicleInput:
    label: str
    year: int

    # questionnaire entreprise
    is_french_company: bool
    is_entrepreneur_individuel: bool
    is_osbl_exempt_vat: bool

    # exonérations
    exempt_usage: bool
    exempt_disability_adapted: bool
    exempt_rental_company_vehicle: bool
    exempt_temporary_replacement: bool
    exempt_short_rental_le_30d: bool

    # type véhicule
    vehicle_kind: str  # "M1" / "N1"
    n1_config_taxable: bool

    # carte grise
    energy: str  # "Essence", "Diesel", "Hybride", "GPL/GNV", "EV/H2"
    co2_norm: str  # "WLTP", "NEDC", "PA"
    co2_value: Optional[float]
    fiscal_power_cv: Optional[int]
    critair_label: str

    # E85
    has_e85: bool

    # affectation
    affect_start: date
    affect_end: date

    # IK
    is_ik_vehicle: bool
    ik_km_reimbursed: int

    # minoration flotte
    is_non_owned_with_expenses: bool


@dataclass
class VehicleResult:
    taxable: bool
    reason: str

    days: int
    proportion: float
    ik_coeff: float

    co2_mode: str
    co2_input: Optional[float]
    co2_base_used: float
    co2_tariff: int
    co2_tranches: List[Dict[str, Any]]
    e85_note: Optional[str]
    co2_warning: Optional[str]

    poll_group: str
    poll_tariff: int

    annual_total_before_prorata: int
    total_before_rounding: float
    total_rounded: int

    is_non_owned_with_expenses: bool

    details: Dict[str, Any]


# ==========================================================
# MOTEUR D'ASSUJETTISSEMENT + CALCUL
# ==========================================================
def determine_taxability(v: VehicleInput) -> Tuple[bool, str]:
    if not v.is_french_company:
        return False, "Non calculé : app France uniquement (entreprise non française)."

    # Exonérations entreprise
    if v.is_entrepreneur_individuel:
        return False, "Exonération : entrepreneur individuel (EI)."
    if v.is_osbl_exempt_vat:
        return False, "Exonération : OSBL d’intérêt général exonéré de TVA."

    # Type véhicule
    if v.vehicle_kind == "N1" and not v.n1_config_taxable:
        return False, "Non assujetti : N1 non assimilé véhicule de tourisme (configuration non taxable)."

    # Exonérations usage / situation
    if v.exempt_usage:
        return False, "Exonération : usage exonéré (taxi/VTC, transport public, auto-école, agricole/forestier, compétition…)."
    if v.exempt_disability_adapted:
        return False, "Exonération : véhicule aménagé handicap."
    if v.exempt_rental_company_vehicle:
        return False, "Exonération : véhicule affecté à l’activité de location (au bénéfice du loueur)."
    if v.exempt_temporary_replacement:
        return False, "Exonération : véhicule prêté temporairement en remplacement (garage)."
    if v.exempt_short_rental_le_30d:
        return False, "Exonération : location ≤ 30 jours consécutifs / 1 mois."

    # Exonération énergie
    if v.energy == "EV/H2":
        return False, "Exonération : motorisation 100 % électrique et/ou hydrogène (0€ CO₂ et 0€ polluants)."

    return True, "Assujetti : véhicule de tourisme affecté à des fins économiques (aucune exonération détectée)."


def compute_co2_tariff(v: VehicleInput) -> Tuple[int, float, List[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    Retourne (tarif_co2_annuel, base_utilisée, detail_tranches, note_e85, warning)
    """
    mode = (v.co2_norm or "WLTP").upper().strip()
    if mode not in {"WLTP", "NEDC", "PA"}:
        mode = "WLTP"

    note_e85: Optional[str] = None
    warning: Optional[str] = None

    # base init
    co2_val = v.co2_value if v.co2_value is not None else None
    cv_val = v.fiscal_power_cv if v.fiscal_power_cv is not None else None

    # Abattement E85 : -40% CO2 si <= 250 ; -2 CV si PA et CV <= 12
    if v.has_e85:
        if mode in {"WLTP", "NEDC"}:
            if co2_val is None:
                warning = "E85 coché, mais CO₂ absent : impossible d'appliquer l'abattement CO₂."
            else:
                if co2_val <= 250:
                    co2_val = co2_val * 0.60
                    note_e85 = f"E85 : abattement -40% sur CO₂ (CO₂ <= 250) => CO₂ retenu = {co2_val:.1f} g/km"
                else:
                    note_e85 = "E85 : pas d’abattement (CO₂ > 250 g/km)"
        else:  # PA
            if cv_val is None:
                warning = "E85 coché, mais puissance fiscale absente : impossible d'appliquer -2 CV."
            else:
                if cv_val <= 12:
                    cv_val = max(0, cv_val - 2)
                    note_e85 = f"E85 : abattement -2 CV (CV <= 12) => CV retenus = {cv_val}"
                else:
                    note_e85 = "E85 : pas d’abattement (CV > 12)"

    # Calcul barème
    if mode == "WLTP":
        if co2_val is None:
            warning = warning or "Mode WLTP choisi mais CO₂ absent : tarif CO₂ forcé à 0 (à vérifier)."
            return 0, 0.0, [], note_e85, warning
        tariff, tr = bracket_progressive_co2(co2_val, WLTP_2026)
        return tariff, float(co2_val), tr, note_e85, warning

    if mode == "NEDC":
        if co2_val is None:
            warning = warning or "Mode NEDC choisi mais CO₂ absent : tarif CO₂ forcé à 0 (à vérifier)."
            return 0, 0.0, [], note_e85, warning
        tariff, tr = bracket_progressive_co2(co2_val, NEDC_2026)
        return tariff, float(co2_val), tr, note_e85, warning

    # PA
    if cv_val is None or cv_val <= 0:
        warning = warning or "Mode PA (puissance fiscale) : CV absents ou nuls => tarif CO₂ forcé à 0 (à vérifier)."
        return 0, 0.0, [], note_e85, warning

    tariff, tr = bracket_progressive_integer(int(cv_val), PA_2026)
    return tariff, float(cv_val), tr, note_e85, warning


def compute_pollutants_tariff(v: VehicleInput) -> Tuple[int, str]:
    # énergie EV/H2 : groupe E
    if v.energy == "EV/H2":
        return 0, "E"
    g = critair_group(v.critair_label)
    return POLLUTANTS_2026.get(g, 500), g


def compute_vehicle(v: VehicleInput) -> VehicleResult:
    taxable, reason = determine_taxability(v)

    # affectation
    d = overlap_days_in_year(v.affect_start, v.affect_end, v.year)
    denom = days_in_year(v.year)
    prop = (d / denom) if denom else 0.0

    # IK coeff
    ik_coeff = 1.0
    ik_note = None
    if v.is_ik_vehicle:
        ik_coeff, ik_note = ik_coefficient(v.ik_km_reimbursed)

    if not taxable:
        details = {
            "assujettissement": {"taxable": False, "raison": reason},
            "affectation": {
                "annee": v.year,
                "debut": str(v.affect_start),
                "fin": str(v.affect_end),
                "jours": d,
                "jours_dans_annee": denom,
                "proportion": prop,
            },
            "ik": {
                "actif": v.is_ik_vehicle,
                "km": v.ik_km_reimbursed,
                "coeff": ik_coeff,
                "regle": ik_note,
            },
            "resultat": {"total_arrondi": 0},
        }
        return VehicleResult(
            taxable=False,
            reason=reason,
            days=d,
            proportion=prop,
            ik_coeff=ik_coeff,
            co2_mode=v.co2_norm,
            co2_input=v.co2_value,
            co2_base_used=0.0,
            co2_tariff=0,
            co2_tranches=[],
            e85_note=None,
            co2_warning=None,
            poll_group="",
            poll_tariff=0,
            annual_total_before_prorata=0,
            total_before_rounding=0.0,
            total_rounded=0,
            is_non_owned_with_expenses=v.is_non_owned_with_expenses,
            details=details,
        )

    # CO2 + Polluants
    co2_tariff, co2_base, co2_tr, e85_note, co2_warning = compute_co2_tariff(v)
    poll_tariff, poll_group = compute_pollutants_tariff(v)

    annual_total = int(co2_tariff + poll_tariff)

    total = annual_total * prop * ik_coeff
    total_rounded = euro_round(total)

    details = {
        "assujettissement": {"taxable": True, "raison": reason},
        "carte_grise": {
            "energie": v.energy,
            "co2_norme": v.co2_norm,
            "co2_saisi": v.co2_value,
            "puissance_cv": v.fiscal_power_cv,
            "critair": v.critair_label,
            "e85": v.has_e85,
        },
        "affectation": {
            "annee": v.year,
            "debut": str(v.affect_start),
            "fin": str(v.affect_end),
            "jours": d,
            "jours_dans_annee": denom,
            "proportion": prop,
        },
        "co2": {
            "tarif_annuel": co2_tariff,
            "base_retendue": co2_base,
            "detail_tranches": co2_tr,
            "note_e85": e85_note,
            "warning": co2_warning,
        },
        "polluants": {
            "groupe": poll_group,
            "tarif_annuel": poll_tariff,
            "regle": "E=0€, Crit’Air 1=100€, autres=500€ (2026)",
        },
        "ik": {
            "actif": v.is_ik_vehicle,
            "km": v.ik_km_reimbursed,
            "coeff": ik_coeff,
            "regle": ik_note,
        },
        "calcul": {
            "annuel_avant_prorata": annual_total,
            "total_avant_arrondi": total,
            "total_arrondi": total_rounded,
            "arrondi": "Arrondi à l’euro le plus proche (>=0,50 vers le haut).",
        },
        "minoration_15000": {
            "eligible": v.is_non_owned_with_expenses,
            "note": "Applicable au niveau flotte sur les véhicules 'non détenus + frais pris en charge'.",
        },
    }

    return VehicleResult(
        taxable=True,
        reason=reason,
        days=d,
        proportion=prop,
        ik_coeff=ik_coeff,
        co2_mode=v.co2_norm,
        co2_input=v.co2_value,
        co2_base_used=co2_base,
        co2_tariff=co2_tariff,
        co2_tranches=co2_tr,
        e85_note=e85_note,
        co2_warning=co2_warning,
        poll_group=poll_group,
        poll_tariff=poll_tariff,
        annual_total_before_prorata=annual_total,
        total_before_rounding=total,
        total_rounded=total_rounded,
        is_non_owned_with_expenses=v.is_non_owned_with_expenses,
        details=details,
    )


# ==========================================================
# UI : ETAT SESSION
# ==========================================================
if "fleet" not in st.session_state:
    st.session_state["fleet"] = []  # list of {"input":..., "result":...}

if "last_vehicle" not in st.session_state:
    st.session_state["last_vehicle"] = None
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None


# ==========================================================
# UI : EN-TÊTE
# ==========================================================
st.title("Calcul taxe annuelle sur l’affectation des véhicules de tourisme (ex-TVS) — Barèmes 2026")
st.caption("App France uniquement • Questionnaire d’assujettissement • Détail complet des calculs (CO₂ + Polluants).")

tabs = st.tabs(["Questionnaire & carte grise", "Résultat (détail)", "Parc véhicules + minoration 15 000 €"])


# ==========================================================
# TAB 1 : SAISIE
# ==========================================================
with tabs[0]:
    st.subheader("1) Questionnaire d’assujettissement (entreprise / usage / type de véhicule)")

    c1, c2, c3 = st.columns(3)
    with c1:
        is_french_company = st.checkbox("Entreprise française (France uniquement)", value=True)
        is_entrepreneur_individuel = st.checkbox("Entrepreneur individuel (EI) — exonéré", value=False)
        is_osbl_exempt_vat = st.checkbox("OSBL d’intérêt général exonéré de TVA — exonéré", value=False)

    with c2:
        exempt_usage = st.checkbox("Usage exonéré (taxi/VTC, transport public, auto-école, agricole/forestier, compétition)", value=False)
        exempt_disability_adapted = st.checkbox("Véhicule aménagé handicap — exonéré", value=False)

    with c3:
        exempt_rental_company_vehicle = st.checkbox("Véhicule affecté à la location (au bénéfice du loueur) — exonéré", value=False)
        exempt_temporary_replacement = st.checkbox("Véhicule de remplacement (garage) — exonéré", value=False)
        exempt_short_rental_le_30d = st.checkbox("Location ≤ 30 jours consécutifs (ou 1 mois) — exonéré", value=False)

    st.divider()
    st.subheader("2) Type de véhicule (pour savoir si c’est un véhicule de tourisme taxable)")

    colA, colB = st.columns([1, 2])
    with colA:
        vehicle_kind_ui = st.selectbox("Catégorie du véhicule", ["M1 (VP - voiture particulière)", "N1 (utilitaire léger)"])

    vehicle_kind = "M1" if vehicle_kind_ui.startswith("M1") else "N1"
    n1_config_taxable = True

    with colB:
        if vehicle_kind == "N1":
            n1_config_taxable = st.checkbox(
                "N1 assimilé à véhicule de tourisme (ex : pick-up double cabine ≥5 places, fourgonnette 'passagers')",
                value=False,
            )
        else:
            st.info("M1 : considéré véhicule de tourisme.")

    st.caption("N1 n’est taxable que s’il est assimilé à un véhicule de tourisme.")

    st.divider()
    st.subheader("3) Données carte grise (CO₂ / norme / Crit’Air / énergie / puissance fiscale)")

    today = date.today()
    default_year = today.year - 1  # en pratique déclaration en N+1
    row1, row2, row3, row4 = st.columns(4)

    with row1:
        label = st.text_input("Libellé véhicule (ex : 'Peugeot 308 - AB-123-CD')", value="Véhicule 1")
        energy = st.selectbox("Énergie", ["Essence", "Diesel", "Hybride", "GPL/GNV", "EV/H2"])
        has_e85 = st.checkbox("Carburant E85 (exclusif ou partiel)", value=False)

    with row2:
        co2_norm_ui = st.selectbox("Norme CO₂ (selon carte grise)", ["WLTP", "NEDC", "PA (pas de CO₂ => puissance fiscale)"])
        co2_norm = "PA" if co2_norm_ui.startswith("PA") else co2_norm_ui

        co2_value: Optional[float] = None
        if co2_norm in {"WLTP", "NEDC"}:
            co2_value = st.number_input("CO₂ (g/km) — champ V.7", min_value=0.0, value=100.0, step=1.0)

        fiscal_power_cv = st.number_input("Puissance fiscale (CV) — champ P.6", min_value=0, value=6, step=1)

    with row3:
        critair_label = st.selectbox("Crit’Air", ["1", "2", "3", "4", "5", "Non classé", "E"])
        st.caption("Polluants : E=0€, Crit’Air 1=100€, autres=500€ (2026).")

    with row4:
        year = st.number_input("Année d’affectation (année N)", min_value=2022, value=int(default_year), step=1)
        st.caption("La taxe est calculée sur l’année d’affectation (N), déclarée ensuite en N+1.")

    st.divider()
    st.subheader("4) Affectation dans l’année (proratisation)")

    # valeurs par défaut cohérentes avec l'année choisie
    default_start = date(int(year), 1, 1)
    default_end = date(int(year), 12, 31)

    d1, d2, d3 = st.columns(3)
    with d1:
        affect_start = st.date_input("Début d’affectation", value=default_start)
    with d2:
        affect_end = st.date_input("Fin d’affectation", value=default_end)
    with d3:
        st.info("Prorata = nb jours affectés / nb jours dans l’année.")

    if affect_end < affect_start:
        st.error("La date de fin d’affectation ne peut pas être antérieure à la date de début.")
        st.stop()

    st.divider()
    st.subheader("5) Cas indemnités kilométriques (véhicule non détenu, remboursement km)")

    is_ik_vehicle = st.checkbox("Véhicule concerné par remboursement de frais kilométriques (IK)", value=False)
    ik_km_reimbursed = 0
    if is_ik_vehicle:
        ik_km_reimbursed = int(st.number_input("Km remboursés sur l’année", min_value=0, value=12000, step=1000))
        coeff, msg = ik_coefficient(ik_km_reimbursed)
        st.info(f"Coefficient IK appliqué : {coeff:.2f} — {msg}")

    st.divider()
    st.subheader("6) Cas minoration 15 000 € (niveau flotte)")

    is_non_owned_with_expenses = st.checkbox(
        "Véhicule non détenu + frais d’utilisation/acquisition pris en charge (éligible à la minoration 15 000 € sur le TOTAL flotte)",
        value=False,
    )

    st.divider()

    if st.button("Calculer et afficher le détail", type="primary"):
        v = VehicleInput(
            label=label.strip() or "Véhicule",
            year=int(year),

            is_french_company=is_french_company,
            is_entrepreneur_individuel=is_entrepreneur_individuel,
            is_osbl_exempt_vat=is_osbl_exempt_vat,

            exempt_usage=exempt_usage,
            exempt_disability_adapted=exempt_disability_adapted,
            exempt_rental_company_vehicle=exempt_rental_company_vehicle,
            exempt_temporary_replacement=exempt_temporary_replacement,
            exempt_short_rental_le_30d=exempt_short_rental_le_30d,

            vehicle_kind=vehicle_kind,
            n1_config_taxable=n1_config_taxable,

            energy=energy,
            co2_norm=co2_norm,
            co2_value=float(co2_value) if co2_value is not None else None,
            fiscal_power_cv=int(fiscal_power_cv),
            critair_label=critair_label,

            has_e85=has_e85,

            affect_start=affect_start,
            affect_end=affect_end,

            is_ik_vehicle=is_ik_vehicle,
            ik_km_reimbursed=int(ik_km_reimbursed),

            is_non_owned_with_expenses=is_non_owned_with_expenses,
        )

        res = compute_vehicle(v)
        st.session_state["last_vehicle"] = v
        st.session_state["last_result"] = res
        st.success("Calcul effectué. Va dans l’onglet « Résultat (détail) ».")

    st.caption("Astuce : ajoute ensuite le véhicule au parc dans l’onglet 3 (multi véhicules + minoration 15k).")


# ==========================================================
# TAB 2 : RESULTAT DETAIL
# ==========================================================
with tabs[1]:
    st.subheader("Résultat — détail pas-à-pas")

    v: Optional[VehicleInput] = st.session_state.get("last_vehicle")
    res: Optional[VehicleResult] = st.session_state.get("last_result")

    if v is None or res is None:
        st.info("Fais un calcul dans l’onglet « Questionnaire & carte grise ».")
    else:
        a, b, c = st.columns([2, 2, 3])
        with a:
            st.metric("Véhicule", v.label)
            st.write(f"**Assujetti :** {'OUI' if res.taxable else 'NON'}")
            st.write(res.reason)

        with b:
            st.metric("Montant (arrondi)", f"{res.total_rounded} €")
            st.write(f"CO₂ annuel : **{res.co2_tariff} €**")
            st.write(f"Polluants annuel : **{res.poll_tariff} €**")

        with c:
            st.write("**Facteurs appliqués**")
            st.write(f"- Jours affectés : {res.days} j")
            st.write(f"- Proportion : {res.proportion:.6f}")
            st.write(f"- Coefficient IK : {res.ik_coeff:.2f}")

        st.divider()
        st.write("## Détail du calcul")

        # A - affectation
        aff = res.details.get("affectation", {})
        st.write("### A) Proratisation (affectation)")
        st.write(f"Période : **{aff.get('debut')}** → **{aff.get('fin')}**")
        st.write(f"Jours retenus : **{aff.get('jours')}** / {aff.get('jours_dans_annee')} => proportion **{aff.get('proportion'):.6f}**")

        # B - CO2
        st.write("### B) Taxe CO₂ (barème 2026)")
        co2 = res.details.get("co2", {})
        st.write(f"Mode : **{v.co2_norm}**")

        if v.co2_norm in {"WLTP", "NEDC"}:
            st.write(f"CO₂ saisi : **{v.co2_value if v.co2_value is not None else '—'} g/km**")
            st.write(f"CO₂ retenu : **{co2.get('base_retendue')}**")
        else:
            st.write(f"Puissance fiscale saisie : **{v.fiscal_power_cv} CV**")
            st.write(f"CV retenus : **{co2.get('base_retendue')}**")

        if co2.get("note_e85"):
            st.info(co2.get("note_e85"))
        if co2.get("warning"):
            st.warning(co2.get("warning"))

        tr = co2.get("detail_tranches", [])
        if isinstance(tr, list) and len(tr) > 0:
            st.write("Détail par tranches :")
            st.dataframe(pd.DataFrame(tr), use_container_width=True, hide_index=True)

        st.write(f"➡️ **Tarif CO₂ annuel = {co2.get('tarif_annuel', 0)} €**")

        # C - polluants
        st.write("### C) Taxe polluants (barème 2026)")
        pol = res.details.get("polluants", {})
        st.write(f"Crit’Air saisi : **{v.critair_label}**")
        st.write(f"Groupe retenu : **{pol.get('groupe')}** (E=0€, 1=100€, autres=500€)")
        st.write(f"➡️ **Tarif polluants annuel = {pol.get('tarif_annuel', 0)} €**")

        # D - somme
        st.write("### D) Somme annuelle avant prorata")
        calc = res.details.get("calcul", {})
        st.write(f"Annuel avant prorata = **{calc.get('annuel_avant_prorata', 0)} €**")

        # E - IK
        st.write("### E) Coefficient IK")
        ik = res.details.get("ik", {})
        if ik.get("actif"):
            st.write(f"Km remboursés : **{ik.get('km')}**")
            st.write(f"Règle : {ik.get('regle')}")
        else:
            st.write("Non applicable (pas de remboursement IK déclaré).")
        st.write(f"➡️ **Coeff IK = {ik.get('coeff', 1.0)}**")

        # F - final
        st.write("### F) Calcul final + arrondi")
        st.code(
            f"Total = Annuel({calc.get('annuel_avant_prorata', 0)}) x Proportion({res.proportion:.6f}) x CoeffIK({res.ik_coeff:.2f})\n"
            f"     = {res.total_before_rounding:.2f} €  -> arrondi => {res.total_rounded} €\n"
            f"Règle : {calc.get('arrondi')}",
            language="text",
        )

        # note minoration
        if res.is_non_owned_with_expenses:
            st.warning("⚠️ Ce véhicule est marqué éligible à la minoration 15 000 € (appliquée au niveau flotte dans l’onglet 3).")

        st.divider()

        # actions
        left, right = st.columns(2)

        with left:
            if st.button("➕ Ajouter ce véhicule au parc (onglet 3)"):
                st.session_state["fleet"].append({"input": asdict(v), "result": asdict(res)})
                st.success("Ajouté au parc.")

        with right:
            json_data = json_dumps_safe(res.details)
            st.download_button(
                "Télécharger le détail (JSON)",
                data=json_data,
                file_name=f"detail_calcul_{v.label.replace(' ', '_')}.json",
                mime="application/json",
            )


# ==========================================================
# TAB 3 : PARC + MINORATION 15 000 €
# ==========================================================
with tabs[2]:
    st.subheader("Parc véhicules + minoration 15 000 €")

    fleet: List[Dict[str, Any]] = st.session_state.get("fleet", [])

    if not fleet:
        st.info("Parc vide. Fais un calcul (onglet 1) puis ajoute le véhicule depuis l’onglet 2.")
    else:
        rows = []
        for i, item in enumerate(fleet, start=1):
            vin = item.get("input", {})
            r = item.get("result", {})

            rows.append({
                "#": i,
                "Véhicule": vin.get("label", f"Véhicule {i}"),
                "Année": vin.get("year"),
                "Assujetti": "OUI" if r.get("taxable") else "NON",
                "Montant arrondi (€)": safe_int(r.get("total_rounded"), 0),
                "Éligible minoration 15k": "OUI" if vin.get("is_non_owned_with_expenses") else "NON",
                "Énergie": vin.get("energy"),
                "Norme CO₂": vin.get("co2_norm"),
                "CO₂": vin.get("co2_value"),
                "CV": vin.get("fiscal_power_cv"),
                "Crit’Air": vin.get("critair_label"),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        total_all = float(df["Montant arrondi (€)"].sum())

        elig_df = df[df["Éligible minoration 15k"] == "OUI"]
        elig_total = float(elig_df["Montant arrondi (€)"].sum())

        minoration = min(15000.0, elig_total)
        elig_net = max(0.0, elig_total - minoration)

        total_net = (total_all - elig_total) + elig_net

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total parc (arrondi)", f"{int(total_all)} €")
        c2.metric("Sous-total éligible 15k", f"{int(elig_total)} €")
        c3.metric("Minoration appliquée", f"{int(minoration)} €")
        c4.metric("Total parc net", f"{int(total_net)} €")

        st.write("### Détail minoration (niveau flotte)")
        st.code(
            f"Sous-total éligible = {int(elig_total)} €\n"
            f"Minoration = min(15 000, {int(elig_total)}) = {int(minoration)} €\n"
            f"Sous-total éligible net = {int(elig_net)} €\n"
            f"Total net = (Total parc - Sous-total éligible) + Sous-total éligible net\n"
            f"         = ({int(total_all)} - {int(elig_total)}) + {int(elig_net)}\n"
            f"         = {int(total_net)} €",
            language="text",
        )

        st.divider()
        left, right = st.columns(2)
        with left:
            if st.button("🧹 Vider le parc"):
                st.session_state["fleet"] = []
                st.success("Parc vidé (rafraîchis la page si besoin).")

        with right:
            st.download_button(
                "Télécharger le parc (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="parc_vehicules_taxe.csv",
                mime="text/csv",
            )

st.caption(
    "Implémentation robuste : JSON via json.dumps, validation dates, champs optionnels sécurisés, "
    "barèmes 2026 (CO₂ WLTP/NEDC/PA + polluants E/1/autres) + prorata jours + coeff IK + abattement E85 + minoration 15k flotte."
)
