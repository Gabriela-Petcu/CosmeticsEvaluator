import pandas as pd

from Src.inference import build_baseline_ml_analysis_df
from Src.config import PROCESSED_DIR
from Src.user_profile import UserProfile
from Src.user_matching import match_product_to_user
from Src.recommendation import build_final_recommendation


def find_product(df: pd.DataFrame, brand_contains: str, name_contains: str) -> pd.Series:
    matches = df[
        (df["brand"].str.contains(brand_contains, case=False, na=False, regex=False)) &
        (df["name"].str.contains(name_contains, case=False, na=False, regex=False))
    ]

    if matches.empty:
        raise ValueError(
            f"Nu am găsit produsul pentru brand='{brand_contains}' și name='{name_contains}'."
        )

    return matches.iloc[0]


def build_row(user: UserProfile, product: pd.Series) -> dict:
    match_result = match_product_to_user(user, product)

    final_result = build_final_recommendation(
        merita=int(product["Merita"]),
        merita_ml=int(product["MeritaML"]),
        se_potriveste=match_result.SePotriveste
    )

    return {
        "skin_type": user.skin_type,
        "main_concern": user.main_concern,
        "budget_level": user.budget_level,
        "brand": product.get("brand", ""),
        "name": product.get("name", ""),
        "price": product.get("price", None),
        "price_per_ounce": product.get("price_per_ounce", None),
        "ScorFinal": float(product.get("ScorFinal", 0)),
        "Merita": int(product.get("Merita", 0)),
        "MeritaML": int(product.get("MeritaML", 0)),
        "ProbabilitateML": float(product.get("ProbabilitateML", 0)),
        "FitScore": match_result.FitScore,
        "SePotriveste": match_result.SePotriveste,
        "VerdictFinal": final_result.verdict,
        "ExplicatieFinala": final_result.explanation,
        "MotivePozitive": " | ".join(match_result.PositiveSignals),
        "MotiveNegative": " | ".join(match_result.NegativeSignals),
    }


def print_scenario_result(index: int, row: dict) -> None:
    print(f"\n=== SCENARIUL {index} ===")
    print(
        f"Profil: skin_type={row['skin_type']}, "
        f"main_concern={row['main_concern']}, "
        f"budget_level={row['budget_level']}"
    )
    print(f"Produs: {row['brand']} - {row['name']}")
    print(f"ScorFinal: {row['ScorFinal']:.4f}")
    print(f"Merita: {row['Merita']}")
    print(f"MeritaML: {row['MeritaML']}")
    print(f"ProbabilitateML: {row['ProbabilitateML']:.4f}")
    print(f"FitScore: {row['FitScore']}")
    print(f"SePotriveste: {row['SePotriveste']}")
    print(f"VerdictFinal: {row['VerdictFinal']}")
    print(f"ExplicatieFinala: {row['ExplicatieFinala']}")
    print(f"MotivePozitive: {row['MotivePozitive']}")
    print(f"MotiveNegative: {row['MotiveNegative']}")


def main():
    df = build_baseline_ml_analysis_df()

    scenarios = [
        {
            "user": UserProfile(
                skin_type="combination",
                main_concern="acne",
                budget_level="low"
            ),
            "brand_contains": "IT Cosmetics",
            "name_contains": "Oil-Free Matte",
        },
        {
            "user": UserProfile(
                skin_type="oily",
                main_concern="acne",
                budget_level="low"
            ),
            "brand_contains": "Drunk Elephant",
            "name_contains": "Virgin Marula Luxury Facial Oil",
        },
        {
            "user": UserProfile(
                skin_type="dry",
                main_concern="dehydration",
                budget_level="low"
            ),
            "brand_contains": "Fresh",
            "name_contains": "Ancienne",
        },
    ]

    rows = []

    for scenario in scenarios:
        user = scenario["user"]
        product = find_product(
            df,
            brand_contains=scenario["brand_contains"],
            name_contains=scenario["name_contains"]
        )
        row = build_row(user, product)
        rows.append(row)

    results_df = pd.DataFrame(rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "user_recommendation_examples.csv"
    results_df.to_csv(output_path, index=False)

    print("=== USER RECOMMENDATION SCENARIOS ===")
    for i, row in enumerate(rows, start=1):
        print_scenario_result(i, row)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()