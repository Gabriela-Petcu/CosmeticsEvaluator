import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from Src.pipeline import evaluate_product_for_user
from Src.user_profile import UserProfile
from Src.explainability import _get_background_data
from Src.inference import load_bundle

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = load_bundle()
    preprocessor = bundle["full_system"].named_steps["preprocessor"]
    _get_background_data(preprocessor)
    yield


app = FastAPI(title="CosmeticsEvaluator ML Engine", lifespan=lifespan)


class ProductData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    review_score: float
    n_of_reviews: int
    n_of_loves: int
    price_per_ounce: float
    brand: str = "Unknown"
    name: str = "Unknown"
    price: float = 0.0

    category_Anti_Aging: int = Field(0, alias="category_Anti-Aging")
    category_Acne_Treatments: int = Field(0, alias="category_Blemish_&_Acne_Treatments")
    category_Exfoliators: int = Field(0, alias="category_Exfoliators")
    category_Eye_Treatments: int = Field(0, alias="category_Eye_Creams_&_Treatments")
    category_Face_Masks: int = Field(0, alias="category_Face_Masks")
    category_Face_Oils: int = Field(0, alias="category_Face_Oils")
    category_Face_Serums: int = Field(0, alias="category_Face_Serums")
    category_Face_Sunscreen: int = Field(0, alias="category_Face_Sunscreen")
    category_Face_Wash: int = Field(0, alias="category_Face_Wash_&_Cleansers")
    category_Facial_Peels: int = Field(0, alias="category_Facial_Peels")
    category_Mists_Essences: int = Field(0, alias="category_Mists_&_Essences")
    category_Moisturizer_Treatments: int = Field(0, alias="category_Moisturizer_&_Treatments")
    category_Moisturizers: int = Field(0, alias="category_Moisturizers")
    category_Night_Creams: int = Field(0, alias="category_Night_Creams")
    category_Toners: int = Field(0, alias="category_Toners")
    category_Blotting_Papers: int = Field(0, alias="category_Blotting_Papers")


class UserProfileData(BaseModel):
    skin_type: str
    main_concern: str
    budget_level: str


class EvaluationRequest(BaseModel):
    product_id: str
    data: ProductData
    user_profile: UserProfileData | None = None


@app.post("/evaluate")
async def evaluate_product(request: EvaluationRequest):
    try:
        if request.user_profile:
            profile = UserProfile(
                skin_type=request.user_profile.skin_type,
                main_concern=request.user_profile.main_concern,
                budget_level=request.user_profile.budget_level,
            )
        else:
            profile = UserProfile(
                skin_type="normal",
                main_concern="anti_aging",
                budget_level="medium",
            )

        response = evaluate_product_for_user(
            request.data.model_dump(by_alias=True), profile
        )

        if response.status != "ok":
            raise HTTPException(status_code=400, detail=response.message)

        return response.result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Internal error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Eroare internă.")