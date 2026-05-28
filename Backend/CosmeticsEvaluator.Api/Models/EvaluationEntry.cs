using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;
namespace CosmeticsEvaluator.Api.Models
{
    public class EvaluationEntry
    {
        //rezultatul complet al unei evaluari, stocat în baza de date pentru istoric și analiză ulterioară
        [Key]
        public int Id { get; set; }
        public string Brand { get; set; } = string.Empty;
        public string Name { get; set; }  = string.Empty;
        public double Price { get; set; }
        public int NOfReviews { get; set; }
        public int NOfLoves { get; set; }
        public double PricePerOunce { get; set; }
        public string ProductId { get; set; } = string.Empty;
        public double ReviewScore { get; set; }
        public double MlProbability { get; set; }
        public string FinalVerdict { get; set; } = string.Empty;
        public DateTime CreatedAt { get; set; } = DateTime.Now;
        [Required]
        public int UserId { get; set; } 
        [JsonIgnore]
        public virtual User? User { get; set; }
    }
}