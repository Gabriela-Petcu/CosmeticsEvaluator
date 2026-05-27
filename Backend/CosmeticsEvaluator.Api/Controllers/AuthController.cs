using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CosmeticsEvaluator.Api.Data;
using CosmeticsEvaluator.Api.Models;
using CosmeticsEvaluator.Api.Services;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using Google.Apis.Auth;
using Microsoft.AspNetCore.Authorization;

namespace CosmeticsEvaluator.Api.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class AuthController : ControllerBase
    {
        private readonly AppDbContext _context;
        private readonly IConfiguration _config;
        private readonly IEmailService _emailService;

        public AuthController(AppDbContext context, IConfiguration config, IEmailService emailService)
        {
            _context = context;
            _config = config;
            _emailService = emailService;
        }

        [HttpPost("register")]
        public async Task<IActionResult> Register([FromBody] RegisterRequest request)
        {
            if (await _context.Users.AnyAsync(u => u.Email == request.Email))
                return BadRequest("Acest email este deja utilizat.");

            string hashedPassword = BCrypt.Net.BCrypt.HashPassword(request.Password);

            var user = new User
            {
                Email = request.Email,
                PasswordHash = hashedPassword,
                Role = "User"
            };

            _context.Users.Add(user);
            await _context.SaveChangesAsync();

            return Ok(new { message = "Cont creat cu succes!" });
        }

        [HttpPost("google-login")]
public async Task<IActionResult> GoogleLogin([FromBody] string accessToken)
{
    try
    {
        // Validăm access_token cu Google UserInfo endpoint
        using var httpClient = new HttpClient();
        httpClient.DefaultRequestHeaders.Authorization =
            new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", accessToken);

        var response = await httpClient.GetAsync("https://www.googleapis.com/oauth2/v3/userinfo");

        if (!response.IsSuccessStatusCode)
            return BadRequest("Token Google invalid.");

        var json = await response.Content.ReadAsStringAsync();
        var userInfo = System.Text.Json.JsonSerializer.Deserialize<GoogleUserInfo>(json);

        if (userInfo == null || string.IsNullOrEmpty(userInfo.Email))
            return BadRequest("Nu s-au putut obține datele utilizatorului Google.");

        var user = await _context.Users.FirstOrDefaultAsync(u => u.Email == userInfo.Email);

        if (user == null)
        {
            user = new User
            {
                Email = userInfo.Email,
                Role = "User",
                PasswordHash = "EXTERNAL_AUTH_GOOGLE",
                CreatedAt = DateTime.Now
            };
            _context.Users.Add(user);
            await _context.SaveChangesAsync();
        }

        var token = GenerateJwtToken(user);

        return Ok(new {
            Token = token,
            Email = user.Email,
            Role = user.Role,
            Message = "Logare cu Google reușită!"
        });
    }
    catch (Exception ex)
    {
        return StatusCode(500, $"Eroare internă: {ex.Message}");
    }
}

        [HttpPost("login")]
        public async Task<IActionResult> Login([FromBody] LoginRequest request)
        {
            var user = await _context.Users.FirstOrDefaultAsync(u => u.Email == request.Email);

            if (user == null)
                return Unauthorized("Email sau parolă incorectă.");

            if (user.PasswordHash == "EXTERNAL_AUTH_GOOGLE")
                return BadRequest("Acest cont a fost creat cu Google. Te rugăm să folosești butonul 'Login cu Google'.");

            if (!BCrypt.Net.BCrypt.Verify(request.Password, user.PasswordHash))
                return Unauthorized("Email sau parolă incorectă.");

            var tokenString = GenerateJwtToken(user);

            return Ok(new
            {
                Token = tokenString,
                Email = user.Email,
                Role = user.Role,
                Message = "Autentificare reușită!"
            });
        }

        [Authorize]
        [HttpPut("profile")]
        public async Task<IActionResult> UpdateProfile([FromBody] UpdateProfileRequest request)
        {
            var userIdClaim = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userIdClaim)) return Unauthorized();
            var userId = int.Parse(userIdClaim);

            var allowedSkinTypes = new[] { "oily", "dry", "combination", "sensitive", "normal" };
            var allowedConcerns = new[] { "acne", "dehydration", "anti_aging", "dark_spots", "redness", "dullness" };
            var allowedBudgets = new[] { "low", "medium", "high" };

            if (!allowedSkinTypes.Contains(request.SkinType))
                return BadRequest($"skin_type invalid. Valori permise: {string.Join(", ", allowedSkinTypes)}");

            if (!allowedConcerns.Contains(request.MainConcern))
                return BadRequest($"main_concern invalid. Valori permise: {string.Join(", ", allowedConcerns)}");

            if (!allowedBudgets.Contains(request.BudgetLevel))
                return BadRequest($"budget_level invalid. Valori permise: {string.Join(", ", allowedBudgets)}");

            var user = await _context.Users.FindAsync(userId);
            if (user == null) return NotFound();

            user.SkinType = request.SkinType;
            user.MainConcern = request.MainConcern;
            user.BudgetLevel = request.BudgetLevel;

            await _context.SaveChangesAsync();

            return Ok(new { message = "Profil actualizat cu succes!" });
        }

        [Authorize]
        [HttpGet("profile")]
        public async Task<IActionResult> GetProfile()
        {
            var userIdClaim = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userIdClaim)) return Unauthorized();
            var userId = int.Parse(userIdClaim);

            var user = await _context.Users.FindAsync(userId);
            if (user == null) return NotFound();

            return Ok(new
            {
                user.Email,
                user.Role, 
                user.SkinType,
                user.MainConcern,
                user.BudgetLevel,
                user.CreatedAt
            });
        }

        [HttpPost("forgot-password")]
        public async Task<IActionResult> ForgotPassword([FromBody] ForgotPasswordRequest request)
        {
            var user = await _context.Users.FirstOrDefaultAsync(u => u.Email == request.Email);

            if (user == null)
                return Ok(new { message = "Dacă emailul există, vei primi un link de resetare." });

            if (user.PasswordHash == "EXTERNAL_AUTH_GOOGLE")
                return BadRequest("Acest cont folosește autentificarea Google. Parola se gestionează din contul Google.");

            var token = Convert.ToBase64String(Guid.NewGuid().ToByteArray())
                .Replace("+", "-").Replace("/", "_").Replace("=", "");

            user.PasswordResetToken = token;
            user.PasswordResetTokenExpiry = DateTime.UtcNow.AddHours(1);
            await _context.SaveChangesAsync();

            var baseUrl = _config["AppBaseUrl"] ?? "http://localhost:5173";
            var resetLink = $"{baseUrl}/reset-password?token={token}&email={Uri.EscapeDataString(request.Email)}";

            try
{
    try
{
    await _emailService.SendPasswordResetEmailAsync(request.Email, resetLink);
    Console.WriteLine("EMAIL TRIMIS CU SUCCES!");
}
catch (Exception ex)
{
    Console.WriteLine($"EROARE EMAIL: {ex.Message}");
}
    Console.WriteLine($"Email trimis cu succes către {request.Email}");
}
catch (Exception ex)
{
    Console.WriteLine($"EROARE trimitere email: {ex.Message}");
    Console.WriteLine($"Stack trace: {ex.StackTrace}");
    // Nu returna eroare utilizatorului, dar logăm
}

            return Ok(new { message = "Dacă emailul există, vei primi un link de resetare." });
        }

        [HttpPost("reset-password")]
        public async Task<IActionResult> ResetPassword([FromBody] ResetPasswordRequest request)
        {
            var user = await _context.Users.FirstOrDefaultAsync(u => u.Email == request.Email);

            if (user == null)
                return BadRequest("Cerere invalidă.");

            if (user.PasswordResetToken != request.Token)
                return BadRequest("Token invalid.");

            if (user.PasswordResetTokenExpiry == null || user.PasswordResetTokenExpiry < DateTime.UtcNow)
                return BadRequest("Token-ul a expirat. Solicită un nou link de resetare.");

            user.PasswordHash = BCrypt.Net.BCrypt.HashPassword(request.NewPassword);
            user.PasswordResetToken = null;
            user.PasswordResetTokenExpiry = null;

            await _context.SaveChangesAsync();

            return Ok(new { message = "Parola a fost resetată cu succes!" });
        }

        private string GenerateJwtToken(User user)
        {
            var tokenHandler = new JwtSecurityTokenHandler();
            var key = Encoding.UTF8.GetBytes(_config["Jwt:Key"]!);
            var tokenDescriptor = new SecurityTokenDescriptor
            {
                Subject = new ClaimsIdentity(new[]
                {
                    new Claim(ClaimTypes.Name, user.Email),
                    new Claim(ClaimTypes.Role, user.Role),
                    new Claim(ClaimTypes.NameIdentifier, user.Id.ToString()),
                }),
                Expires = DateTime.UtcNow.AddDays(7),
                Issuer = _config["Jwt:Issuer"],
                Audience = _config["Jwt:Audience"],
                SigningCredentials = new SigningCredentials(
                    new SymmetricSecurityKey(key),
                    SecurityAlgorithms.HmacSha256Signature)
            };
            var token = tokenHandler.CreateToken(tokenDescriptor);
            return tokenHandler.WriteToken(token);
        }
    }

    public class GoogleUserInfo
{
    [System.Text.Json.Serialization.JsonPropertyName("email")]
    public string Email { get; set; } = string.Empty;

    [System.Text.Json.Serialization.JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [System.Text.Json.Serialization.JsonPropertyName("picture")]
    public string Picture { get; set; } = string.Empty;
}
}