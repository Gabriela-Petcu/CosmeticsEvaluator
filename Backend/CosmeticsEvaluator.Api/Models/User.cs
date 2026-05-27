using System.ComponentModel.DataAnnotations;

namespace CosmeticsEvaluator.Api.Models
{
    public class User
    {
        [Key]
        public int Id { get; set; }

        public string SkinType { get; set; } = "normal";
        public string MainConcern { get; set; } = "anti_aging";
        public string BudgetLevel { get; set; } = "medium";
        public virtual ICollection<EvaluationEntry> Evaluations { get; set; } = new List<EvaluationEntry>();

        [Required]
        public string Email { get; set; } = string.Empty;

        [Required]
        public string PasswordHash { get; set; } = string.Empty;

        public string Role { get; set; } = "User";

        public DateTime CreatedAt { get; set; } = DateTime.Now;
        public string? PasswordResetToken { get; set; }
        public DateTime? PasswordResetTokenExpiry { get; set; }
    }
}