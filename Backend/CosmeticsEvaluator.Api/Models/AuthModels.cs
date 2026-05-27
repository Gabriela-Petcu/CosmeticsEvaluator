using System.ComponentModel.DataAnnotations;

namespace CosmeticsEvaluator.Api.Models
{
    // Modele pentru autentificare și gestionarea profilului utilizatorului
    public class RegisterRequest
    {
        [Required(ErrorMessage = "Email-ul este obligatoriu.")]
        [EmailAddress(ErrorMessage = "Formatul email-ului este invalid.")]
        public string Email { get; set; } = string.Empty;

        [Required(ErrorMessage = "Parola este obligatorie.")]
        [MinLength(8, ErrorMessage = "Parola trebuie să aibă cel puțin 8 caractere.")]
        [RegularExpression(@"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).+$", 
            ErrorMessage = "Parola trebuie să conțină o literă mare, una mică și o cifră.")]
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
        [Required(ErrorMessage = "Email necesar.")]
        public string Email { get; set; } = string.Empty;

        [Required(ErrorMessage = "Parolă necesară.")]
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
        ErrorMessage = "Parola trebuie să conțină o literă mare, una mică și o cifră.")]
    public string NewPassword { get; set; } = string.Empty;
}
}