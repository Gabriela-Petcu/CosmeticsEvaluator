from Src.inference import build_full_analysis_df
from Src.config import PROCESSED_DIR


def main():
    df_scored = build_full_analysis_df()

    name_col = None
    for c in ["name", "product_name", "product", "title"]:
        if c in df_scored.columns:
            name_col = c
            break

    output_cols = []
    if name_col:
        output_cols.append(name_col)

    output_cols += ["ScorFinal", "Merita", "MeritaML", "ProbabilitateML"]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "comparison_results.csv"
    df_scored[output_cols].to_csv(out_path, index=False)

    print(f"✅ Rezultate comparate salvate în: {out_path}")


if __name__ == "__main__":
    main()