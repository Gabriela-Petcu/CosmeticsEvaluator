using CsvHelper.Configuration;
using CosmeticsEvaluator.Api.Models;

public class ProductCatalogMap : ClassMap<ProductCatalog>
{
    // Această clasă definește cum să mapăm coloanele din fișierul CSV la proprietățile clasei ProductCatalog, folosind CsvHelper pentru a facilita importul datelor din CSV în baza de date
    public ProductCatalogMap()
    {
        Map(m => m.Brand).Name("brand");
        Map(m => m.Name).Name("name");
        Map(m => m.Price).Name("price");
        Map(m => m.NOfReviews).Name("n_of_reviews");
        Map(m => m.NOfLoves).Name("n_of_loves");
        Map(m => m.ReviewScore).Name("review_score");
        Map(m => m.PricePerOunce).Name("price_per_ounce");

        Map(m => m.CategoryAntiAging).Name("category_Anti-Aging");
        Map(m => m.CategoryAcneTreatments).Name("category_Blemish_&_Acne_Treatments");
        Map(m => m.CategoryExfoliators).Name("category_Exfoliators");
        Map(m => m.CategoryEyeTreatments).Name("category_Eye_Creams_&_Treatments");
        Map(m => m.CategoryFaceMasks).Name("category_Face_Masks");
        Map(m => m.CategoryFaceOils).Name("category_Face_Oils");
        Map(m => m.CategoryFaceSerums).Name("category_Face_Serums");
        Map(m => m.CategoryFaceSunscreen).Name("category_Face_Sunscreen");
        Map(m => m.CategoryFaceWash).Name("category_Face_Wash_&_Cleansers");
        Map(m => m.CategoryFacialPeels).Name("category_Facial_Peels");
        Map(m => m.CategoryMistsEssences).Name("category_Mists_&_Essences");
        Map(m => m.CategoryMoisturizerTreatments).Name("category_Moisturizer_&_Treatments");
        Map(m => m.CategoryMoisturizers).Name("category_Moisturizers");
        Map(m => m.CategoryNightCreams).Name("category_Night_Creams");
        Map(m => m.CategoryToners).Name("category_Toners");
        Map(m => m.CategoryBlottingPapers).Name("category_Blotting_Papers");
    }
}