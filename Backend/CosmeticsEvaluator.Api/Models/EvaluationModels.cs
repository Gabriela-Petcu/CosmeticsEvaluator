using System.Text.Json.Serialization;

namespace CosmeticsEvaluator.Api.Models
{
    // Shared contract between C# and Python for structuring input and output data,
    // ensuring consistent communication between the two application components.

    //REQUEST (sent to Python) 

    public class ProductEvaluationRequest
    {
        [JsonPropertyName("product_id")]
        public string ProductId { get; set; } = string.Empty;

        [JsonPropertyName("data")]
        public ProductData Data { get; set; } = new();

        [JsonPropertyName("user_profile")]
        public UserProfileData? UserProfile { get; set; }
    }

    public class UserProfileData
    {
        // snake_case intentional — matches the field names expected by FastAPI
        [JsonPropertyName("skin_type")]
        public string SkinType { get; set; } = "normal";

        [JsonPropertyName("main_concern")]
        public string MainConcern { get; set; } = "anti_aging";

        [JsonPropertyName("budget_level")]
        public string BudgetLevel { get; set; } = "medium";
    }

    public class ProductData
    {
        [JsonPropertyName("review_score")]
        public double review_score { get; set; }

        [JsonPropertyName("n_of_reviews")]
        public int n_of_reviews { get; set; }

        [JsonPropertyName("n_of_loves")]
        public int n_of_loves { get; set; }

        [JsonPropertyName("price_per_ounce")]
        public double price_per_ounce { get; set; }

        [JsonPropertyName("category_Anti-Aging")]
        public int category_Anti_Aging { get; set; }

        [JsonPropertyName("category_Blemish_&_Acne_Treatments")]
        public int category_Acne_Treatments { get; set; }

        [JsonPropertyName("category_Exfoliators")]
        public int category_Exfoliators { get; set; }

        [JsonPropertyName("category_Eye_Creams_&_Treatments")]
        public int category_Eye_Treatments { get; set; }

        [JsonPropertyName("category_Face_Masks")]
        public int category_Face_Masks { get; set; }

        [JsonPropertyName("category_Face_Oils")]
        public int category_Face_Oils { get; set; }

        [JsonPropertyName("category_Face_Serums")]
        public int category_Face_Serums { get; set; }

        [JsonPropertyName("category_Face_Sunscreen")]
        public int category_Face_Sunscreen { get; set; }

        [JsonPropertyName("category_Face_Wash_&_Cleansers")]
        public int category_Face_Wash { get; set; }

        [JsonPropertyName("category_Facial_Peels")]
        public int category_Facial_Peels { get; set; }

        [JsonPropertyName("category_Mists_&_Essences")]
        public int category_Mists_Essences { get; set; }

        [JsonPropertyName("category_Moisturizer_&_Treatments")]
        public int category_Moisturizer_Treatments { get; set; }

        [JsonPropertyName("category_Moisturizers")]
        public int category_Moisturizers { get; set; }

        [JsonPropertyName("category_Night_Creams")]
        public int category_Night_Creams { get; set; }

        [JsonPropertyName("category_Toners")]
        public int category_Toners { get; set; }

        [JsonPropertyName("category_Blotting_Papers")]
        public int category_Blotting_Papers { get; set; }
    }

    // RESPONSE (received from Python) 
    // Property names must match exactly the fields in FullPipelineResult (pipeline.py)
    public class EvaluationResponse
    {
        public double FinalScore { get; set; }
        public int IsRecommended { get; set; }
        public int IsRecommendedML { get; set; }
        public double MLProbability { get; set; }
        public int FitScore { get; set; }
        public int IsCompatible { get; set; }
        public string FinalVerdict { get; set; } = string.Empty;
        public string FinalExplanation { get; set; } = string.Empty;
        public List<string> PositiveSignals { get; set; } = new();
        public List<string> NegativeSignals { get; set; } = new();
    }
}