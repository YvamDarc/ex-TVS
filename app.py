import math
from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import List, Tuple, Optional, Dict

import streamlit as st
import pandas as pd


# =========================
# Barèmes 2026 (à jour selon ton cadrage)
# =========================

# Barème WLTP 2026 : tranches (start_g, end_g_inclusive, rate_per_g)
WLTP_2026 = [
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

# Barème NEDC 2026 : tranches (start_g, end_g_inclusive, rate_per_g)
NEDC_2026 = [
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

# Barème Puissance Administrative (PA) 2026 : tranches (start_cv, end_cv_inclusive, rate_per_cv)
PA_2026 = [
    (1, 3, 2000),
    (4, 6, 3000),
    (7, 10, 4500),
    (11, 15, 5250),
    (16, 10**9, 6500),
]

# Taxe polluants 2026 (catégorie Crit'Air simplifiée en 3 groupes)
# - "E" : électrique / hydrogène (vignette verte) -> 0 €
# - "1" : essence/hybride/gaz Crit’Air 1 -> 100 €
# - "P" : autres (Crit’Air 2/3/4/5/non classé) -> 500 €
POLLUTANTS_2026 = {"E": 0, "1": 100, "P": 500}


# =========================
# Helpers
# =========================

def euro_round(x: float) -> int:
    """Arrondi fiscal à l'euro : >= 0,50 à l'euro supérieur."""
    if x >= 0:
        return int(math.floor(x + 0.5))
    return -int(math.floor(abs(x) + 0.5))


def clamp_date(d: date, year: int) -> date:
    if d < date(year, 1, 1):
        return date(year, 1, 1)
    if d > date(year, 12, 31):
        return date(year, 12, 31)
    return d


def days_in_year(year: int) -> int:
    # Année bissextile
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return 366
    return 365


def overlap_days(start: date, end: date, year: int) -> int:
    """Nombre de jours (inclusifs) entre start/end, bornés à l'année."""
    s = clamp_date(start, year)
    e = clamp_date(end, year)
    if e < s:
        return 0
    return (e - s).days + 1


def bracket_progressive(value: int, brackets: List[Tuple[int, int, int]]) -> Tuple[int, List[Dict]]:
    """
    Calcule un montant progressif par tranches (valeur entière).
    Retourne (total, détails par tranche).
    """
    total = 0
    details = []
    for a, b, rate in brackets:
        if value < a:
            continue
        upper = min(value, b)
        qty = upper - a + 1
        if qty <= 0:
            continue
        part = qty * rate
        total += part
        details.append(
            {"tranche": f"{a}–{upper}", "unites": qty, "taux": rate, "montant": part}
        )
        if value <= b:
            break
    return total, details


def bracket_progressive_co2(value: float, brackets: List[Tuple[int, int, int]]) -> Tuple[int, List[Dict]]:
    """
    Progressif par gramme de CO2 : value peut être float, on considère le g/km en entier (comme sur carte grise).
    On calcule par gramme dans la tranche.
    """
    v = int(round(value))
    total = 0
    details = []
    for a, b, rate in brackets:
        if v < a:
            continue
        upper = min(v, b)
        qty = upper - a + 1
        if qty <= 0:
            continue
        part = qty * rate
        total += part
        details.append(
            {"tranche_g": f"{a}–{upper}", "grammes": qty, "taux_€/g": rate, "montant": part}
        )
        if v <= b:
            break
    return total, details


def ik_coefficient(km: int) -> Tuple[float, str]:
    """
    Coefficient pondérateur pour véhicules avec remboursement de frais kilométriques.
    """
    if km <= 15000:
        return 0.00, "0–15 000 km => coeff 0 %"
    if km <= 25000:
        return 0.25, "15 001–25 000 km => coeff 25 %"
    if km <= 35000:
        return 0.50, "25 001–35 000 km => coeff 50 %"
    if km <= 45000:
        return 0.75, "35 001–45 000 km => coeff 75 %"
    return 1.00, "> 45 000 km => coeff 100 %"


def critair_group(label: str) -> str:
    """
    Convertit une saisie Crit'Air en groupe: E / 1 / P.
    - E : électrique/hydrogène (vignette verte)
    - 1 : Crit'Air 1
    - P : autres (2/3/4/5/non classé)
    """
    label = (label or "").strip().upper()
    if label in {"E", "EV", "ELECTRIQUE", "ÉLECTRIQUE", "HYDROGENE", "HYDROGÈNE", "VERT"}:
        return "E"
    if label in {"1", "CRIT1", "CRIT'1", "CRIT'AI R 1", "VIOLET"}:
        return "1"
    return "P"


# =========================
# Data model
# =========================

@dataclass
class VehicleInput:
    label: str

    # Assujettissement / cas
    is_french_company: bool
    is_entrepreneur_individuel: bool
    is_osbl_exempt_vat: bool
    exempt_usage: bool  # taxi, VTC, etc.
    exempt_disability_adapted: bool
    exempt_rental_company_vehicle: bool
    exempt_temporary_replacement: bool
    exempt_short_rental_le_30d: bool

    # Véhicule concerné ?
    vehicle_kind: str  # "M1" / "N1"
    n1_config_taxable: bool  # si N1, correspond à un véhicule de tourisme taxable

    # Données carte grise
    energy: str  # "EV/H2", "Essence", "Diesel", "Hybride", "GPL/GNV", ...
    co2_value: Optional[float]
    co2_norm: str  # "WLTP" / "NEDC" / "PA"
    fiscal_power_cv: Optional[int]
    critair_label: str

    # Abattement E85
    has_e85: bool  # véhicule roulant E85 (exclusif/partiel)
    e85_abattement_applicable: bool  # auto-calcul / override

    # Affectation
    year: int
    affect_start: date
    affect_end: date

    # Cas indemnités kilométriques
    is_ik_vehicle: bool
    ik_km_reimbursed: int

    # Cas minoration 15 000 € (sur flotte) : véhicule non détenu + frais pris en charge
    is_non_owned_with_expenses: bool


@dataclass
class VehicleResult:
    taxable: bool
    taxable_reason: str
    days: int
    proportion: float
    ik_coeff: float
    co2_base: float
    co2_tariff: int
    poll_group: str
    poll_tariff: int
    annual_total_before_prorata: int
    total_after_prorata: float
    total_rounded: int
    details: Dict


# =========================
# Core computation
# =========================

def determine_taxability(v: VehicleInput) -> Tuple[bool, str]:
    if not v.is_french_company:
        return False, "Entreprise non française (app configurée France uniquement)."

    if v.is_entrepreneur_individuel:
        return False, "Exonération : entrepreneur individuel (EI)."

    if v.is_osbl_exempt_vat:
        return False, "Exonération : organisme sans but lucratif bénéficiant d’exonération de TVA."

    # Véhicule concerné ?
    if v.vehicle_kind == "N1" and not v.n1_config_taxable:
        return False, "Véhicule N1 non assimilé véhicule de tourisme (configuration non taxable)."

    # Exonérations d’usage / caractéristiques
    if v.exempt_usage:
        return False, "Exonération : usage exonéré (taxi/VTC/transports publics, auto-école, agricole/forestier, compétition…)."
    if v.exempt_disability_adapted:
        return False, "Exonération : véhicule aménagé pour personnes handicapées."
    if v.exempt_rental_company_vehicle:
        return False, "Exonération : véhicule affecté à l’activité de location (au bénéfice du loueur)."
    if v.exempt_temporary_replacement:
        return False, "Exonération : véhicule prêté temporairement en remplacement (garage)."
    if v.exempt_short_rental_le_30d:
        return False, "Exonération : location ≤ 30 jours consécutifs (ou 1 mois) sur l’année."

    # Électrique / hydrogène : exonération des 2 taxes
    if v.energy.upper() in {"EV/H2", "ELECTRIQUE/HYDROGENE", "ÉLECTRIQUE/HYDROGÈNE"}:
        return False, "Exonération : motorisation 100 % électrique et/ou hydrogène."

    return True, "Assujetti : véhicule de tourisme affecté à des fins économiques (aucune exonération détectée)."


def compute_co2_tariff_2026(v: VehicleInput) -> Tuple[int, float, Dict]:
    """
    Retourne (tarif_CO2_annuel, co2_base_utilisée, détails)
    co2_base_utilisée = valeur après abattement E85 si applicable.
    """
    details = {"mode": v.co2_norm, "tranches": [], "abattement_e85": None}

    # Si CO2 absent -> PA (puissance fiscale)
    mode = v.co2_norm.upper()
    if mode not in {"WLTP", "NEDC", "PA"}:
        mode = "WLTP"

    # Abattement E85 : -40% CO2 (si <=250 g) OU -2CV (si PA) ; sinon pas d’abattement
    co2_base = v.co2_value if v.co2_value is not None else 0.0
    cv_base = v.fiscal_power_cv if v.fiscal_power_cv is not None else 0

    if v.has_e85:
        if mode in {"WLTP", "NEDC"}:
            if co2_base <= 250:
                co2_base = co2_base * 0.60
                details["abattement_e85"] = f"E85 : -40% sur CO2 (car CO2 <= 250) => CO2 retenu = {co2_base:.1f} g/km"
            else:
                details["abattement_e85"] = "E85 : pas d’abattement (CO2 > 250 g/km)"
        elif mode == "PA":
            if cv_base <= 12:
                cv_base = max(0, cv_base - 2)
                details["abattement_e85"] = f"E85 : -2 CV (car CV <= 12) => CV retenus = {cv_base}"
            else:
                details["abattement_e85"] = "E85 : pas d’abattement (CV > 12)"

    if mode == "WLTP":
        tariff, tr = bracket_progressive_co2(co2_base, WLTP_2026)
        details["tranches"] = tr
        return tariff, co2_base, details

    if mode == "NEDC":
        tariff, tr = bracket_progressive_co2(co2_base, NEDC_2026)
        details["tranches"] = tr
        return tariff, co2_base, details

    # PA
    if cv_base <= 0:
        return 0, 0.0, {**details, "warning": "Puissance fiscale absente ou nulle : tarif CO2 = 0 (à vérifier)."}
    tariff, tr = bracket_progressive(cv_base, PA_2026)
    details["tranches"] = tr
    return tariff, float(cv_base), details


def compute_pollutants_tariff_2026(v: VehicleInput) -> Tuple[int, str, Dict]:
    """
    Retourne (tarif_polluants_annuel, groupe, détails)
    """
    g = critair_group(v.critair_label)

    # Si énergie électrique/hydrogène => groupe E (normalement déjà exonéré en amont)
    if v.energy.upper() in {"EV/H2", "ELECTRIQUE/HYDROGENE", "ÉLECTRIQUE/HYDROGÈNE"}:
        g = "E"

    tariff = POLLUTANTS_2026[g]
    details = {"critair_saisie": v.critair_label, "groupe": g, "tarif": tariff}
    return tariff, g, details


def compute_vehicle_tax_2026(v: VehicleInput) -> VehicleResult:
    taxable, reason = determine_taxability(v)

    # affectation
    d = overlap_days(v.affect_start, v.affect_end, v.year)
    prop = d / days_in_year(v.year) if days_in_year(v.year) else 0.0

    # IK
    ik_coeff = 1.0
    ik_detail = None
    if v.is_ik_vehicle:
        ik_coeff, ik_detail = ik_coefficient(v.ik_km_reimbursed)

    if not taxable:
        return VehicleResult(
            taxable=False,
            taxable_reason=reason,
            days=d,
            proportion=prop,
            ik_coeff=ik_coeff,
            co2_base=0.0,
            co2_tariff=0,
            poll_group="",
            poll_tariff=0,
            annual_total_before_prorata=0,
            total_after_prorata=0.0,
            total_rounded=0,
            details={
                "assujettissement": reason,
                "affectation": {"jours": d, "proportion": prop},
                "ik": {"actif": v.is_ik_vehicle, "km": v.ik_km_reimbursed, "coeff": ik_coeff, "detail": ik_detail},
            },
        )

    co2_tariff, co2_base, co2_details = compute_co2_tariff_2026(v)
    poll_tariff, poll_group, poll_details = compute_pollutants_tariff_2026(v)

    annual_total = int(co2_tariff + poll_tariff)

    total = annual_total * prop * ik_coeff
    total_rounded = euro_round(total)

    details = {
        "assujettissement": reason,
        "affectation": {"debut": str(v.affect_start), "fin": str(v.affect_end), "jours": d, "proportion": prop},
        "co2": {
            "mode": v.co2_norm,
            "valeur_saisie": v.co2_value,
            "co2_base_retendue": co2_base,
            "tarif_annuel": co2_tariff,
            "detail_tranches": co2_details.get("tranches", []),
            "e85": co2_details.get("abattement_e85"),
            "note": co2_details.get("warning"),
        },
        "polluants": poll_details,
        "ik": {"actif": v.is_ik_vehicle, "km": v.ik_km_reimbursed, "coeff": ik_coeff, "detail": ik_detail},
        "somme": {"co2": co2_tariff, "polluants": poll_tariff, "annuel_avant_prorata": annual_total},
        "calcul_final": {
            "annuel": annual_total,
            "x_proportion": prop,
            "x_coeff_ik": ik_coeff,
            "total_avant_arrondi": total,
            "total_arrondi": total_rounded,
            "arrondi": "Arrondi à l’euro le plus proche (>=0,50 vers le haut).",
        },
        "minoration_15000": {
            "eligible": v.is_non_owned_with_expenses,
            "note": "La minoration forfaitaire de 15 000 € s’applique sur le TOTAL des véhicules 'non détenus + frais pris en charge' (niveau flotte).",
        },
    }

    return VehicleResult(
        taxable=True,
        taxable_reason=reason,
        days=d,
        proportion=prop,
        ik_coeff=ik_coeff,
        co2_base=co2_base,
        co2_tariff=co2_tariff,
        poll_group=poll_group,
        poll_tariff=poll_tariff,
        annual_total_before_prorata=annual_total,
        total_after_prorata=total,
        total_rounded=total_rounded,
        details=details,
    )


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Taxe véhicule tourisme (ex-TVS) — Calcul 2026", layout="wide")
st.title("Calcul taxe annuelle sur l’affectation des véhicules de tourisme (ex-TVS) — Barèmes 2026")
st.caption("App France uniquement • Questionnaire d’assujettissement • Détail complet des calculs (CO₂ + Polluants).")

today = date.today()
default_year = today.year - 1  # en pratique : déclaration en N+1
if "fleet" not in st.session_state:
    st.session_state["fleet"] = []  # liste de dicts véhicule + résultat

tabs = st.tabs(["1) Questionnaire & carte grise", "2) Résultat (détail)", "3) Parc véhicules + minoration 15 000 €"])

# ---------- TAB 1 ----------
with tabs[0]:
    st.subheader("1) Questionnaire d’assujettissement (entreprise / usage / type de véhicule)")
    colA, colB, colC = st.columns(3)

    with colA:
        is_french_company = st.checkbox("Entreprise française (France uniquement)", value=True)
        is_entrepreneur_individuel = st.checkbox("Entrepreneur individuel (EI) — exonéré", value=False)
        is_osbl_exempt_vat = st.checkbox("OSBL d’intérêt général exonéré de TVA — exonéré", value=False)

    with colB:
        exempt_usage = st.checkbox("Usage exonéré (taxi/VTC, transport public, auto-école, agricole/forestier, compétition)", value=False)
        exempt_disability_adapted = st.checkbox("Véhicule aménagé handicap — exonéré", value=False)

    with colC:
        exempt_rental_company_vehicle = st.checkbox("Véhicule affecté à la location (au bénéfice du loueur) — exonéré", value=False)
        exempt_temporary_replacement = st.checkbox("Véhicule de remplacement (garage) — exonéré", value=False)
        exempt_short_rental_le_30d = st.checkbox("Location ≤ 30 jours consécutifs (ou 1 mois) — exonéré", value=False)

    st.divider()
    st.subheader("2) Type de véhicule (pour savoir si c’est un véhicule de tourisme taxable)")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        vehicle_kind = st.selectbox("Catégorie du véhicule", ["M1 (VP - voiture particulière)", "N1 (utilitaire léger)"])
    with col2:
        if vehicle_kind.startswith("N1"):
            n1_config_taxable = st.checkbox(
                "N1 assimilé à véhicule de tourisme (ex : pick-up double cabine ≥5 places, ou fourgonnette 'passagers')",
                value=False
            )
        else:
            n1_config_taxable = True
            st.info("M1 : considéré véhicule de tourisme.")
    with col3:
        st.markdown(
            "- **M1** : véhicule de tourisme (taxable si affecté à l’activité).\n"
            "- **N1** : taxable seulement si **assimilé tourisme** (configuration passagers)."
        )

    st.divider()
    st.subheader("3) Données carte grise (CO₂ / norme / Crit’Air / énergie / puissance fiscale)")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        label = st.text_input("Libellé véhicule (ex : 'Peugeot 308 - AB-123-CD')", value="Véhicule 1")
        energy = st.selectbox("Énergie", ["Essence", "Diesel", "Hybride", "GPL/GNV", "EV/H2 (100% électrique/hydrogène)"])
        has_e85 = st.checkbox("Carburant E85 (exclusif ou partiel)", value=False)

    with c2:
        co2_norm = st.selectbox("Norme CO₂ (selon carte grise)", ["WLTP", "NEDC", "PA (pas de CO₂ => puissance fiscale)"])
        co2_value = None
        fiscal_power_cv = None

        if co2_norm != "PA (pas de CO₂ => puissance fiscale)":
            co2_value = st.number_input("CO₂ (g/km) — champ V.7", min_value=0.0, value=100.0, step=1.0)
        fiscal_power_cv = st.number_input("Puissance fiscale (CV) — champ P.6", min_value=0, value=6, step=1)

    with c3:
        critair_label = st.selectbox("Crit’Air", ["1", "2", "3", "4", "5", "Non classé", "E (électrique/hydrogène)"])
        st.caption("Polluants : E=0€, Crit’Air 1=100€, autres=500€ (barème 2026).")

    with c4:
        year = st.number_input("Année d’affectation (année N)", min_value=2022, value=int(default_year), step=1)
        st.caption("La taxe est calculée sur l’année d’affectation (N), déclarée ensuite en N+1.")

    st.divider()
    st.subheader("4) Affectation dans l’année (proratisation)")

    ca, cb, cc = st.columns(3)
    with ca:
        affect_start = st.date_input("Début d’affectation", value=date(int(year), 1, 1))
    with cb:
        affect_end = st.date_input("Fin d’affectation", value=date(int(year), 12, 31))
    with cc:
        st.write("")
        st.write("")
        st.info("Prorata = nb jours d’affectation / nb jours dans l’année.")

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
        "Véhicule non détenu par l’entreprise + frais d’utilisation/acquisition pris en charge (éligible à la minoration 15 000 € sur le TOTAL flotte)",
        value=False
    )

    st.divider()
    if st.button("Calculer et afficher le détail", type="primary"):
        v = VehicleInput(
            label=label,
            is_french_company=is_french_company,
            is_entrepreneur_individuel=is_entrepreneur_individuel,
            is_osbl_exempt_vat=is_osbl_exempt_vat,
            exempt_usage=exempt_usage,
            exempt_disability_adapted=exempt_disability_adapted,
            exempt_rental_company_vehicle=exempt_rental_company_vehicle,
            exempt_temporary_replacement=exempt_temporary_replacement,
            exempt_short_rental_le_30d=exempt_short_rental_le_30d,
            vehicle_kind="M1" if vehicle_kind.startswith("M1") else "N1",
            n1_config_taxable=n1_config_taxable,
            energy=energy,
            co2_value=co2_value,
            co2_norm=("PA" if co2_norm.startswith("PA") else co2_norm),
            fiscal_power_cv=int(fiscal_power_cv) if fiscal_power_cv is not None else None,
            critair_label=critair_label,
            has_e85=has_e85,
            e85_abattement_applicable=True,
            year=int(year),
            affect_start=affect_start,
            affect_end=affect_end,
            is_ik_vehicle=is_ik_vehicle,
            ik_km_reimbursed=int(ik_km_reimbursed),
            is_non_owned_with_expenses=is_non_owned_with_expenses
        )

        res = compute_vehicle_tax_2026(v)
        st.session_state["last_vehicle"] = v
        st.session_state["last_result"] = res
        st.success("Calcul effectué. Va dans l’onglet « Résultat (détail) ».")

    st.caption("Astuce : tu peux ensuite ajouter le véhicule au parc dans l’onglet 3 pour gérer la minoration 15 000 €.")


# ---------- TAB 2 ----------
with tabs[1]:
    st.subheader("Résultat — détail pas-à-pas")

    res: VehicleResult = st.session_state.get("last_result")
    v: VehicleInput = st.session_state.get("last_vehicle")

    if not res or not v:
        st.info("Fais un calcul dans l’onglet 1.")
    else:
        top1, top2, top3 = st.columns([2, 2, 3])
        with top1:
            st.metric("Véhicule", v.label)
            st.write(f"**Assujettissement :** {'OUI' if res.taxable else 'NON'}")
            st.write(res.taxable_reason)

        with top2:
            st.metric("Montant (arrondi)", f"{res.total_rounded} €")
            st.write(f"CO₂ annuel : **{res.co2_tariff} €**")
            st.write(f"Polluants annuel : **{res.poll_tariff} €**")

        with top3:
            st.write("**Facteurs appliqués**")
            st.write(f"- Jours affectés : {res.days} j")
            st.write(f"- Proportion : {res.proportion:.6f}")
            if v.is_ik_vehicle:
                st.write(f"- Coefficient IK : {res.ik_coeff:.2f}")
            else:
                st.write("- Coefficient IK : 1.00 (non applicable)")

        st.divider()
        st.write("## Détail du calcul")

        # Affectation
        aff = res.details["affectation"]
        st.write("### A) Proratisation (affectation)")
        st.write(f"Période : **{aff.get('debut')}** → **{aff.get('fin')}**")
        st.write(f"Jours retenus : **{aff.get('jours')}**")
        st.write(f"Proportion : **{aff.get('proportion'):.6f}**")

        # CO2
        st.write("### B) Taxe CO₂ (barème 2026)")
        co2 = res.details["co2"]
        st.write(f"Mode : **{co2.get('mode')}**")
        if co2.get("valeur_saisie") is not None:
            st.write(f"CO₂ saisi : **{co2.get('valeur_saisie')} g/km**")
            st.write(f"CO₂ retenu : **{co2.get('co2_base_retendue'):.1f} g/km**")
        else:
            st.write(f"Puissance retenue (PA) : **{co2.get('co2_base_retendue'):.0f} CV**")

        if co2.get("e85"):
            st.info(co2.get("e85"))

        if co2.get("note"):
            st.warning(co2.get("note"))

        tr = co2.get("detail_tranches", [])
        if tr:
            st.write("Détail par tranches :")
            st.dataframe(pd.DataFrame(tr), use_container_width=True, hide_index=True)
        st.write(f"➡️ **Tarif CO₂ annuel = {co2.get('tarif_annuel')} €**")

        # Polluants
        st.write("### C) Taxe polluants (barème 2026)")
        pol = res.details["polluants"]
        st.write(f"Crit’Air saisi : **{pol.get('critair_saisie')}**")
        st.write(f"Groupe retenu : **{pol.get('groupe')}** (E=0€, 1=100€, autres=500€)")
        st.write(f"➡️ **Tarif polluants annuel = {pol.get('tarif')} €**")

        # Somme
        st.write("### D) Somme annuelle avant prorata")
        s = res.details["somme"]
        st.write(f"CO₂ : {s['co2']} € + Polluants : {s['polluants']} € = **{s['annuel_avant_prorata']} €**")

        # IK
        st.write("### E) Coefficient IK (si remboursement km)")
        ik = res.details["ik"]
        if ik["actif"]:
            st.write(f"Km remboursés : **{ik['km']}**")
            st.write(f"Règle : {ik['detail']}")
            st.write(f"➡️ **Coefficient IK = {ik['coeff']:.2f}**")
        else:
            st.write("Non applicable (pas de remboursement IK déclaré).")

        # Final
        st.write("### F) Calcul final + arrondi")
        f = res.details["calcul_final"]
        st.code(
            f"Total = {f['annuel']}  x  {f['x_proportion']:.6f}  x  {f['x_coeff_ik']:.2f}\n"
            f"     = {f['total_avant_arrondi']:.2f} €  -> arrondi => {f['total_arrondi']} €\n"
            f"Règle d'arrondi : {f['arrondi']}",
            language="text",
        )

        # Minoration note
        m = res.details["minoration_15000"]
        if m["eligible"]:
            st.warning("⚠️ Ce véhicule est marqué éligible à la minoration 15 000 €, mais celle-ci s’applique sur le TOTAL flotte (onglet 3).")


        st.divider()
        cA, cB = st.columns(2)
        with cA:
            if st.button("➕ Ajouter ce véhicule au parc (onglet 3)"):
                st.session_state["fleet"].append({"input": asdict(v), "result": asdict(res)})
                st.success("Ajouté au parc.")
        with cB:
            st.download_button(
                "Télécharger le détail (JSON)",
                data=pd.Series(res.details).to_json(ensure_ascii=False, indent=2),
                file_name=f"detail_calcul_{v.label.replace(' ', '_')}.json",
                mime="application/json"
            )


# ---------- TAB 3 ----------
with tabs[2]:
    st.subheader("Parc véhicules (multi) + application de la minoration 15 000 €")
    st.write(
        "Ici, tu peux cumuler plusieurs véhicules. La **minoration forfaitaire 15 000 €** "
        "s’applique sur le **TOTAL** des véhicules dont tu coches : "
        "**« non détenu + frais pris en charge »**."
    )

    fleet = st.session_state["fleet"]

    if not fleet:
        st.info("Ton parc est vide. Fais un calcul dans l’onglet 1 puis ajoute le véhicule depuis l’onglet 2.")
    else:
        # Construire un tableau
        rows = []
        for i, item in enumerate(fleet, start=1):
            vin = item["input"]
            res = item["result"]
            rows.append({
                "#": i,
                "Véhicule": vin["label"],
                "Assujetti": "OUI" if res["taxable"] else "NON",
                "Montant arrondi (€)": res["total_rounded"],
                "Éligible - minoration 15k (flotte)": "OUI" if vin["is_non_owned_with_expenses"] else "NON",
                "Année": vin["year"],
                "Énergie": vin["energy"],
                "Norme CO2": vin["co2_norm"],
                "CO2 (g/km)": vin["co2_value"],
                "CritAir": vin["critair_label"],
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        total_all = df["Montant arrondi (€)"].sum()

        # Total minoration : uniquement sur sous-ensemble éligible
        elig = df[df["Éligible - minoration 15k (flotte)"] == "OUI"]["Montant arrondi (€)"].sum()
        minoration = min(15000, elig)
        net_elig = max(0, elig - minoration)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total parc (arrondi)", f"{int(total_all)} €")
        c2.metric("Sous-total véhicules éligibles 15k", f"{int(elig)} €")
        c3.metric("Minoration appliquée", f"{int(minoration)} €")

        st.write("### Total net après minoration (sur sous-total éligible)")
        st.code(
            f"Sous-total éligible = {int(elig)} €\n"
            f"Minoration = min(15 000, {int(elig)}) = {int(minoration)} €\n"
            f"Sous-total éligible net = {int(net_elig)} €\n"
            f"Total parc net = (Total parc - Sous-total éligible) + Sous-total éligible net\n"
            f"              = ({int(total_all)} - {int(elig)}) + {int(net_elig)}\n"
            f"              = {int((total_all - elig) + net_elig)} €",
            language="text"
        )

        st.divider()
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🧹 Vider le parc"):
                st.session_state["fleet"] = []
                st.success("Parc vidé.")
        with b2:
            st.download_button(
                "Télécharger le parc (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="parc_vehicules_taxe.csv",
                mime="text/csv"
            )


# Footer
st.caption(
    "Note : cette app implémente le calcul barèmes 2026 (CO₂ WLTP/NEDC/PA, polluants E/1/autres), "
    "proratisation jours, coefficient IK, abattement E85, et minoration 15k au niveau flotte."
)
