from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from Src.pipeline import evaluate_product_for_user, UserProfile

app = FastAPI(title="CosmeticsEvaluator ML Engine")

# Modelele de date pentru request
class ProductData(BaseModel):
    review_score: float
    n_of_reviews: int
    n_of_loves: int
    price_per_ounce: float
    # Adăugate pentru a suporta User Matching din .NET
    brand: str = "Unknown"
    name: str = "Unknown"
    price: float = 0.0
    # Categorii (opțional, dacă le trimiți din .NET)
    # category_Moisturizers: int = 0 etc.

class UserProfileData(BaseModel):
    skin_type: str
    main_concern: str
    budget_level: str

class EvaluationRequest(BaseModel):
    product_id: str
    data: ProductData
    user_profile: UserProfileData | None = None # Profil opțional

@app.post("/evaluate")
async def evaluate_product(request: EvaluationRequest):
    try:
        # 1. Setăm profilul (din request sau default)
        if request.user_profile:
            profile = UserProfile(
                skin_type=request.user_profile.skin_type,
                main_concern=request.user_profile.main_concern,
                budget_level=request.user_profile.budget_level
            )
        else:
            profile = UserProfile(skin_type="normal", main_concern="anti_aging", budget_level="medium")
        
        # 2. Rulăm pipeline-ul complet
        response = evaluate_product_for_user(request.data.dict(), profile)
        
        if response.status != "ok":
            raise HTTPException(status_code=400, detail=response.message)
            
        return response.result 
        
    except Exception as e:
        print(f"Internal Error: {e}")
        raise HTTPException(status_code=500, detail="Eroare internă de procesare AI.")