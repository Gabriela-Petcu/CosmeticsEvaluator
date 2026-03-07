from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from Src.user_profile import UserProfile


CATEGORY_COLUMNS = {
    "anti_aging": "category_Anti-Aging",
    "acne_treatments": "category_Blemish_&_Acne_Treatments",
    "exfoliators": "category_Exfoliators",
    "eye_treatments": "category_Eye_Creams_&_Treatments",
    "face_masks": "category_Face_Masks",
    "face_oils": "category_Face_Oils",
    "face_serums": "category_Face_Serums",
    "face_sunscreen": "category_Face_Sunscreen",
    "face_wash": "category_Face_Wash_&_Cleansers",
    "facial_peels": "category_Facial_Peels",
    "mists_essences": "category_Mists_&_Essences",
    "moisturizer_treatments": "category_Moisturizer_&_Treatments",
    "moisturizers": "category_Moisturizers",
    "night_creams": "category_Night_Creams",
    "toners": "category_Toners",
    "blotting_papers": "category_Blotting_Papers",
}


@dataclass
class MatchResult:
    FitScore: int
    SePotriveste: int
    ReasonsPositive: list[str]
    ReasonsNegative: list[str]


def _get_category(product: pd.Series, key: str) -> int:
    col = CATEGORY_COLUMNS[key]
    if col not in product.index:
        return 0
    return int(product.get(col, 0))


def _name_contains(product_name: str, keywords: list[str]) -> bool:
    name = (product_name or "").lower()
    return any(keyword in name for keyword in keywords)


def _apply_skin_type_rules(profile: UserProfile, product: pd.Series, score: int,
                           reasons_pos: list[str], reasons_neg: list[str]) -> int:
    name = str(product.get("name", ""))

    if profile.skin_type == "oily":
        if _get_category(product, "face_wash") or _get_category(product, "toners") or _get_category(product, "face_sunscreen") or _get_category(product, "acne_treatments"):
            score += 15
            reasons_pos.append("Categoria produsului este potrivită pentru ten gras.")
        if _name_contains(name, ["gel", "water", "matte", "oil-free"]):
            score += 10
            reasons_pos.append("Denumirea produsului sugerează o textură mai lejeră, potrivită pentru ten gras.")
        if _get_category(product, "face_oils") or _get_category(product, "night_creams"):
            score -= 15
            reasons_neg.append("Categoria produsului poate fi prea grea pentru ten gras.")

    elif profile.skin_type == "dry":
        if _get_category(product, "moisturizers") or _get_category(product, "moisturizer_treatments") or _get_category(product, "night_creams") or _get_category(product, "face_oils") or _get_category(product, "face_masks"):
            score += 15
            reasons_pos.append("Categoria produsului este potrivită pentru ten uscat.")
        if _name_contains(name, ["cream", "hydrat", "moistur", "dewy"]):
            score += 10
            reasons_pos.append("Denumirea produsului sugerează hidratare sau nutriție.")
        if _get_category(product, "blotting_papers"):
            score -= 10
            reasons_neg.append("Produsul nu pare relevant pentru nevoile unui ten uscat.")

    elif profile.skin_type == "combination":
        if _get_category(product, "moisturizers") or _get_category(product, "face_wash") or _get_category(product, "toners") or _get_category(product, "face_sunscreen"):
            score += 12
            reasons_pos.append("Categoria produsului este potrivită pentru ten mixt.")
        if _name_contains(name, ["gel", "water", "balance", "matte", "oil-free"]):
            score += 10
            reasons_pos.append("Produsul pare să aibă o formulă ușoară, bună pentru ten mixt.")
        if _get_category(product, "face_oils"):
            score -= 8
            reasons_neg.append("Produsul ar putea fi prea greu pentru anumite zone ale tenului mixt.")

    elif profile.skin_type == "sensitive":
        if _get_category(product, "moisturizers") or _get_category(product, "face_masks") or _get_category(product, "face_wash"):
            score += 12
            reasons_pos.append("Categoria produsului este relativ potrivită pentru ten sensibil.")
        if _get_category(product, "exfoliators") or _get_category(product, "facial_peels"):
            score -= 18
            reasons_neg.append("Categoria produsului poate fi prea agresivă pentru ten sensibil.")

    elif profile.skin_type == "normal":
        score += 5
        reasons_pos.append("Tenul normal este compatibil cu o gamă mai largă de produse.")

    return score


def _apply_concern_rules(profile: UserProfile, product: pd.Series, score: int,
                         reasons_pos: list[str], reasons_neg: list[str]) -> int:
    name = str(product.get("name", ""))

    if profile.main_concern == "acne":
        if _get_category(product, "acne_treatments") or _get_category(product, "face_wash") or _get_category(product, "toners"):
            score += 15
            reasons_pos.append("Produsul este relevant pentru nevoi asociate cu acneea.")
        if _name_contains(name, ["acne", "blemish", "clarifying", "oil-free", "matte"]):
            score += 10
            reasons_pos.append("Denumirea produsului sugerează caracteristici utile pentru un profil acneic.")
        if _get_category(product, "face_oils"):
            score -= 12
            reasons_neg.append("Produsul poate fi mai puțin potrivit pentru un profil cu acnee.")

    elif profile.main_concern == "dehydration":
        if _get_category(product, "moisturizers") or _get_category(product, "night_creams") or _get_category(product, "face_masks") or _get_category(product, "face_oils") or _get_category(product, "mists_essences"):
            score += 15
            reasons_pos.append("Produsul este compatibil cu nevoia de hidratare.")
        if _name_contains(name, ["hydrat", "moistur", "dewy"]):
            score += 10
            reasons_pos.append("Denumirea produsului sugerează un efect hidratant.")

    elif profile.main_concern == "anti_aging":
        if _get_category(product, "anti_aging") or _get_category(product, "face_serums") or _get_category(product, "night_creams") or _get_category(product, "eye_treatments"):
            score += 15
            reasons_pos.append("Produsul este relevant pentru nevoi anti-aging.")
        if _name_contains(name, ["retinol", "peptide", "firm", "repair"]):
            score += 10
            reasons_pos.append("Denumirea produsului sugerează efect anti-aging.")

    elif profile.main_concern == "dark_spots":
        if _get_category(product, "face_serums") or _get_category(product, "facial_peels"):
            score += 12
            reasons_pos.append("Categoria produsului poate ajuta în rutina pentru pete pigmentare.")
        if _name_contains(name, ["bright", "vitamin c", "glow"]):
            score += 10
            reasons_pos.append("Denumirea produsului sugerează luminozitate sau uniformizare.")

    elif profile.main_concern == "redness":
        if _name_contains(name, ["cica", "calm", "repair", "soothing"]):
            score += 10
            reasons_pos.append("Denumirea produsului sugerează efect calmant.")

    elif profile.main_concern == "dullness":
        if _name_contains(name, ["glow", "bright", "radiance"]):
            score += 10
            reasons_pos.append("Denumirea produsului sugerează efect de luminozitate.")

    return score


def _apply_budget_rules(profile: UserProfile, product: pd.Series, score: int,
                        reasons_pos: list[str], reasons_neg: list[str]) -> int:
    price = product.get("price", None)
    price_per_ounce = product.get("price_per_ounce", None)

    if pd.isna(price):
        return score

    if profile.budget_level == "low":
        if price <= 20:
            score += 12
            reasons_pos.append("Prețul este potrivit pentru un buget redus.")
        elif price <= 40:
            score += 5
            reasons_pos.append("Prețul este acceptabil pentru un buget redus.")
        elif price > 60:
            score -= 18
            reasons_neg.append("Prețul este ridicat pentru un buget redus.")

        if pd.notna(price_per_ounce) and price_per_ounce > 50:
            score -= 10
            reasons_neg.append("Raportul preț/cantitate este nefavorabil pentru un buget redus.")

    elif profile.budget_level == "medium":
        if price <= 50:
            score += 6
            reasons_pos.append("Prețul este rezonabil pentru un buget mediu.")
        elif price > 90:
            score -= 8
            reasons_neg.append("Prețul este destul de mare pentru un buget mediu.")

    elif profile.budget_level == "high":
        score += 3
        reasons_pos.append("Bugetul ridicat permite accesul la acest produs fără restricții majore.")

    return score


def match_product_to_user(profile: UserProfile, product: pd.Series | dict[str, Any]) -> MatchResult:
    if isinstance(product, dict):
        product = pd.Series(product)

    score = 50
    reasons_pos: list[str] = []
    reasons_neg: list[str] = []

    score = _apply_skin_type_rules(profile, product, score, reasons_pos, reasons_neg)
    score = _apply_concern_rules(profile, product, score, reasons_pos, reasons_neg)
    score = _apply_budget_rules(profile, product, score, reasons_pos, reasons_neg)

    score = max(0, min(100, score))
    se_potriveste = 1 if score >= 60 else 0

    return MatchResult(
        FitScore=score,
        SePotriveste=se_potriveste,
        ReasonsPositive=reasons_pos,
        ReasonsNegative=reasons_neg
    )