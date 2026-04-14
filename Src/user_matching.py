from __future__ import annotations

"""
Modul euristic de potrivire produs-utilizator.

Scorul de compatibilitate este calculat pe baza unor reguli definite manual,
inspirate de caracteristici generale ale produselor și de profilul utilizatorului.

Această componentă:
- nu este un model ML antrenat
- nu învață ponderi din date
- nu produce probabilități statistice
- folosește un sistem de reguli explicabil pentru compatibilitate
"""

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
    PositiveSignals: list[str]
    NegativeSignals: list[str]


SKIN_TYPE_RULES = {
    "oily": {
        "category_rules": [
            {
                "keys": ["face_wash", "toners", "face_sunscreen", "acne_treatments"],
                "score": 15,
                "message": "Categoria produsului este potrivită pentru ten gras.",
                "positive": True,
            },
            {
                "keys": ["face_oils", "night_creams"],
                "score": -15,
                "message": "Categoria produsului poate fi prea grea pentru ten gras.",
                "positive": False,
            },
        ],
        "keyword_rules": [
            {
                "keywords": ["gel", "water", "matte", "oil-free"],
                "score": 10,
                "message": (
                    "Denumirea produsului sugerează o textură mai lejeră, "
                    "potrivită pentru ten gras."
                ),
                "positive": True,
            }
        ],
        "base_adjustment": None,
    },
    "dry": {
        "category_rules": [
            {
                "keys": [
                    "moisturizers",
                    "moisturizer_treatments",
                    "night_creams",
                    "face_oils",
                    "face_masks",
                ],
                "score": 15,
                "message": "Categoria produsului este potrivită pentru ten uscat.",
                "positive": True,
            },
            {
                "keys": ["blotting_papers"],
                "score": -10,
                "message": "Produsul nu pare relevant pentru nevoile unui ten uscat.",
                "positive": False,
            },
        ],
        "keyword_rules": [
            {
                "keywords": ["cream", "hydrat", "moistur", "dewy"],
                "score": 10,
                "message": "Denumirea produsului sugerează hidratare sau nutriție.",
                "positive": True,
            }
        ],
        "base_adjustment": None,
    },
    "combination": {
        "category_rules": [
            {
                "keys": ["moisturizers", "face_wash", "toners", "face_sunscreen"],
                "score": 12,
                "message": "Categoria produsului este potrivită pentru ten mixt.",
                "positive": True,
            },
            {
                "keys": ["face_oils"],
                "score": -8,
                "message": (
                    "Produsul ar putea fi prea greu pentru anumite zone ale tenului mixt."
                ),
                "positive": False,
            },
        ],
        "keyword_rules": [
            {
                "keywords": ["gel", "water", "balance", "matte", "oil-free"],
                "score": 10,
                "message": "Produsul pare să aibă o formulă ușoară, bună pentru ten mixt.",
                "positive": True,
            }
        ],
        "base_adjustment": None,
    },
    "sensitive": {
        "category_rules": [
            {
                "keys": ["moisturizers", "face_masks", "face_wash"],
                "score": 12,
                "message": "Categoria produsului este relativ potrivită pentru ten sensibil.",
                "positive": True,
            },
            {
                "keys": ["exfoliators", "facial_peels"],
                "score": -18,
                "message": "Categoria produsului poate fi prea agresivă pentru ten sensibil.",
                "positive": False,
            },
        ],
        "keyword_rules": [],
        "base_adjustment": None,
    },
    "normal": {
        "category_rules": [],
        "keyword_rules": [],
        "base_adjustment": {
            "score": 5,
            "message": "Tenul normal este compatibil cu o gamă mai largă de produse.",
            "positive": True,
        },
    },
}


CONCERN_RULES = {
    "acne": {
        "category_rules": [
            {
                "keys": ["acne_treatments", "face_wash", "toners"],
                "score": 15,
                "message": "Produsul este relevant pentru nevoi asociate cu acneea.",
                "positive": True,
            },
            {
                "keys": ["face_oils"],
                "score": -12,
                "message": "Produsul poate fi mai puțin potrivit pentru un profil cu acnee.",
                "positive": False,
            },
        ],
        "keyword_rules": [
            {
                "keywords": ["acne", "blemish", "clarifying", "oil-free", "matte"],
                "score": 10,
                "message": (
                    "Denumirea produsului sugerează caracteristici utile "
                    "pentru un profil acneic."
                ),
                "positive": True,
            }
        ],
    },
    "dehydration": {
        "category_rules": [
            {
                "keys": [
                    "moisturizers",
                    "night_creams",
                    "face_masks",
                    "face_oils",
                    "mists_essences",
                ],
                "score": 15,
                "message": "Produsul este compatibil cu nevoia de hidratare.",
                "positive": True,
            }
        ],
        "keyword_rules": [
            {
                "keywords": ["hydrat", "moistur", "dewy"],
                "score": 10,
                "message": "Denumirea produsului sugerează un efect hidratant.",
                "positive": True,
            }
        ],
    },
    "anti_aging": {
        "category_rules": [
            {
                "keys": ["anti_aging", "face_serums", "night_creams", "eye_treatments"],
                "score": 15,
                "message": "Produsul este relevant pentru nevoi anti-aging.",
                "positive": True,
            }
        ],
        "keyword_rules": [
            {
                "keywords": ["retinol", "peptide", "firm", "repair"],
                "score": 10,
                "message": "Denumirea produsului sugerează efect anti-aging.",
                "positive": True,
            }
        ],
    },
    "dark_spots": {
        "category_rules": [
            {
                "keys": ["face_serums", "facial_peels"],
                "score": 12,
                "message": (
                    "Categoria produsului poate ajuta în rutina pentru pete pigmentare."
                ),
                "positive": True,
            }
        ],
        "keyword_rules": [
            {
                "keywords": ["bright", "vitamin c", "glow"],
                "score": 10,
                "message": "Denumirea produsului sugerează luminozitate sau uniformizare.",
                "positive": True,
            }
        ],
    },
    "redness": {
        "category_rules": [],
        "keyword_rules": [
            {
                "keywords": ["cica", "calm", "repair", "soothing"],
                "score": 10,
                "message": "Denumirea produsului sugerează efect calmant.",
                "positive": True,
            }
        ],
    },
    "dullness": {
        "category_rules": [],
        "keyword_rules": [
            {
                "keywords": ["glow", "bright", "radiance"],
                "score": 10,
                "message": "Denumirea produsului sugerează efect de luminozitate.",
                "positive": True,
            }
        ],
    },
}


# Verifică dacă produsul aparține unei categorii.
def _get_category(product: pd.Series, key: str) -> int:
    col = CATEGORY_COLUMNS[key]
    if col not in product.index:
        return 0
    return int(product.get(col, 0))


# Verifică dacă denumirea produsului conține unul dintre keyword-urile date.
def _name_contains(product_name: str, keywords: list[str]) -> bool:
    name = (product_name or "").lower()
    return any(keyword in name for keyword in keywords)


def _validate_category_columns(product: pd.Series) -> None:
    available = [col for col in CATEGORY_COLUMNS.values() if col in product.index]
    if not available:
        raise ValueError(
            "Produsul nu conține coloane de categorie. Modulul de user matching "
            "are nevoie de aceste informații pentru a aplica regulile euristice."
        )


def _append_reason(
    positive: bool,
    message: str,
    reasons_pos: list[str],
    reasons_neg: list[str],
) -> None:
    if positive:
        reasons_pos.append(message)
    else:
        reasons_neg.append(message)


def _apply_category_rules(
    product: pd.Series,
    score: int,
    reasons_pos: list[str],
    reasons_neg: list[str],
    rules: list[dict[str, Any]],
) -> int:
    for rule in rules:
        if any(_get_category(product, key) for key in rule["keys"]):
            score += rule["score"]
            _append_reason(rule["positive"], rule["message"], reasons_pos, reasons_neg)
    return score


def _apply_keyword_rules(
    product_name: str,
    score: int,
    reasons_pos: list[str],
    reasons_neg: list[str],
    rules: list[dict[str, Any]],
) -> int:
    for rule in rules:
        if _name_contains(product_name, rule["keywords"]):
            score += rule["score"]
            _append_reason(rule["positive"], rule["message"], reasons_pos, reasons_neg)
    return score


def _apply_base_adjustment(
    score: int,
    reasons_pos: list[str],
    reasons_neg: list[str],
    adjustment: dict[str, Any] | None,
) -> int:
    if adjustment is None:
        return score

    score += adjustment["score"]
    _append_reason(adjustment["positive"], adjustment["message"], reasons_pos, reasons_neg)
    return score


def _apply_skin_type_rules(
    profile: UserProfile,
    product: pd.Series,
    score: int,
    reasons_pos: list[str],
    reasons_neg: list[str]
) -> int:
    name = str(product.get("name", ""))
    rules = SKIN_TYPE_RULES.get(profile.skin_type)

    if rules is None:
        return score

    score = _apply_base_adjustment(
        score,
        reasons_pos,
        reasons_neg,
        rules.get("base_adjustment"),
    )
    score = _apply_category_rules(
        product,
        score,
        reasons_pos,
        reasons_neg,
        rules.get("category_rules", []),
    )
    score = _apply_keyword_rules(
        name,
        score,
        reasons_pos,
        reasons_neg,
        rules.get("keyword_rules", []),
    )

    return score


def _apply_concern_rules(
    profile: UserProfile,
    product: pd.Series,
    score: int,
    reasons_pos: list[str],
    reasons_neg: list[str]
) -> int:
    name = str(product.get("name", ""))
    rules = CONCERN_RULES.get(profile.main_concern)

    if rules is None:
        return score

    score = _apply_category_rules(
        product,
        score,
        reasons_pos,
        reasons_neg,
        rules.get("category_rules", []),
    )
    score = _apply_keyword_rules(
        name,
        score,
        reasons_pos,
        reasons_neg,
        rules.get("keyword_rules", []),
    )

    return score


def _apply_budget_rules(
    profile: UserProfile,
    product: pd.Series,
    score: int,
    reasons_pos: list[str],
    reasons_neg: list[str]
) -> int:
    price = product.get("price", None)
    price_per_ounce = product.get("price_per_ounce", None)

    # Adaugă o verificare la începutul funcției
    if pd.isna(price) or price <= 0:
        # Dacă nu avem preț, penalizăm ușor pentru incertitudine sau ignorăm
        score -= 5 
        reasons_neg.append("Preț indisponibil pentru verificare buget.")
        return score

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
            reasons_neg.append(
                "Raportul preț/cantitate este nefavorabil pentru un buget redus."
            )

    elif profile.budget_level == "medium":
        if price <= 50:
            score += 6
            reasons_pos.append("Prețul este rezonabil pentru un buget mediu.")
        elif price > 90:
            score -= 8
            reasons_neg.append("Prețul este destul de mare pentru un buget mediu.")

    elif profile.budget_level == "high":
        score += 3
        reasons_pos.append(
            "Bugetul ridicat permite accesul la acest produs fără restricții majore."
        )

    return score


def match_product_to_user(profile: UserProfile, product: pd.Series | dict[str, Any]) -> MatchResult:
    """
    Evaluează compatibilitatea dintre un produs și profilul utilizatorului
    folosind un sistem euristic de reguli.

    Scorul pornește de la 50 și este ajustat pe baza:
    - tipului de ten
    - preocupării principale
    - nivelului de buget

    Verdictul binar SePotriveste este obținut prin pragul euristic FitScore >= 60.
    """
    if isinstance(product, dict):
        product = pd.Series(product)
    elif not isinstance(product, pd.Series):
        raise TypeError("product trebuie să fie dict sau pandas.Series")

    _validate_category_columns(product)

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
        PositiveSignals=reasons_pos,
        NegativeSignals=reasons_neg
    )



