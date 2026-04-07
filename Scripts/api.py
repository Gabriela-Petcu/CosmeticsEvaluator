from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Importăm funcțiile tale din inference.py
from Src.inference import load_bundle, prepare_baseline_dataframe, prepare_ml_dataframe, add_ml_predictions

app = FastAPI(title="CosmeticsEvaluator ML Engine")

# --- MODELE DE DATE (Ce primim de la ASP.NET) ---
class ProductData(BaseModel):
    review_score: float
    n_of_reviews: int
    n_of_loves: int
    price_per_ounce: float

class EvaluationRequest(BaseModel):
    product_id: str
    data: ProductData

# --- LOGICA DE ÎNCĂRCARE ---
# Încărcăm bundle-ul o singură dată la pornire, pentru performanță
try:
    BUNDLE = load_bundle()
    print("✅ Model bundle loaded successfully.")
except Exception as e:
    print(f"❌ Error loading bundle: {e}")
    BUNDLE = None

# --- ENDPOINT-UL PRINCIPAL ---
@app.post("/evaluate")
async def evaluate_product(request: EvaluationRequest):
    if BUNDLE is None:
        raise HTTPException(status_code=500, detail="ML Model not loaded.")

    try:
        # 1. Convertim datele primite în DataFrame (formatul așteptat de funcțiile tale)
        input_dict = request.data.dict()
        df = pd.DataFrame([input_dict])

        # 2. Rulăm flow-ul tău existent
        # Calculăm Baseline (ScorFinal, Merita)
        df_baseline = prepare_baseline_dataframe(df, BUNDLE)
        
        # Calculăm ML Features
        df_ml_prep = prepare_ml_dataframe(df_baseline)
        
        # Obținem Predicțiile ML
        df_final = add_ml_predictions(df_ml_prep, BUNDLE)

        # 3. Extragem rezultatele pentru a le trimite înapoi
        result = {
            "product_id": request.product_id,
            "baseline": {
                "score": float(df_final["ScorFinal"].iloc[0]),
                "merita": bool(df_final["Merita"].iloc[0])
            },
            "ml": {
                "merita_ml": bool(df_final["MeritaML"].iloc[0]),
                "probability": float(df_final["ProbabilitateML"].iloc[0])
            }
        }
        
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Pornire: uvicorn api:app --reload