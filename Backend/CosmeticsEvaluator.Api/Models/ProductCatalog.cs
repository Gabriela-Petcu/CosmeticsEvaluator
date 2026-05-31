namespace CosmeticsEvaluator.Api.Models
{
    //Model for storing the product catalog. Used to populate the database
    //and provide detailed product data during evaluation.
    public class ProductCatalog
    {
        public int Id { get; set; }
        public string Brand { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public double Price { get; set; }
        public int NOfReviews { get; set; }
        public int NOfLoves { get; set; }
        public double ReviewScore { get; set; }
        public double? PricePerOunce { get; set; }

        public int CategoryAntiAging { get; set; }
        public int CategoryAcneTreatments { get; set; }
        public int CategoryExfoliators { get; set; }
        public int CategoryEyeTreatments { get; set; }
        public int CategoryFaceMasks { get; set; }
        public int CategoryFaceOils { get; set; }
        public int CategoryFaceSerums { get; set; }
        public int CategoryFaceSunscreen { get; set; }
        public int CategoryFaceWash { get; set; }
        public int CategoryFacialPeels { get; set; }
        public int CategoryMistsEssences { get; set; }
        public int CategoryMoisturizerTreatments { get; set; }
        public int CategoryMoisturizers { get; set; }
        public int CategoryNightCreams { get; set; }
        public int CategoryToners { get; set; }
        public int CategoryBlottingPapers { get; set; }
    }
}