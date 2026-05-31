using System.ComponentModel.DataAnnotations;

namespace CosmeticsEvaluator.Api.Models
{
    // Models for authentication and user profile management
 
    public class RegisterRequest
    {
        [Required(ErrorMessage = "Email is required.")]
        [EmailAddress(ErrorMessage = "Invalid email format.")]
        public string Email { get; set; } = string.Empty;
        [Required(ErrorMessage = "Password is required.")]
        [MinLength(8, ErrorMessage = "Password must be at least 8 characters long.")]
        [RegularExpression(@"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).+$", 
            ErrorMessage = "Password must contain an uppercase letter, a lowercase letter, and a digit.")]
        public string Password { get; set; } = string.Empty;
    }

public class UpdateProfileRequest
{
    [Required]
    public string SkinType { get; set; } = string.Empty;

    [Required]
    public string MainConcern { get; set; } = string.Empty;

    [Required]
    public string BudgetLevel { get; set; } = string.Empty;
}

    public class LoginRequest
    {
        [Required(ErrorMessage = "Email is required.")]
        public string Email { get; set; } = string.Empty;

        [Required(ErrorMessage = "Password is required.")]
        public string Password { get; set; } = string.Empty;
    }

    public class ForgotPasswordRequest
{
    [Required]
    [EmailAddress]
    public string Email { get; set; } = string.Empty;
}

public class ResetPasswordRequest
{
    [Required]
    public string Token { get; set; } = string.Empty;

    [Required]
    [EmailAddress]
    public string Email { get; set; } = string.Empty;

    [Required]
    [MinLength(8)]
    [RegularExpression(@"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).+$",
        ErrorMessage = "Password must contain an uppercase letter, a lowercase letter and a digit.")]
    public string NewPassword { get; set; } = string.Empty;
}
}