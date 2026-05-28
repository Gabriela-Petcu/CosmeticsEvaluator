using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CosmeticsEvaluator.Api.Data;
using CosmeticsEvaluator.Api.Models;
using CosmeticsEvaluator.Api.Services;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using Microsoft.AspNetCore.Authorization;

namespace CosmeticsEvaluator.Api.Controllers
{
    /// <summary>
    /// Controller pentru autentificare si gestionarea conturilor de utilizator.
    /// Gestioneaza inregistrarea, autentificarea, profilul si resetarea parolei.
    /// </summary>
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

        /// <summary>
        /// Inregistreaza un cont nou cu email si parola.
        /// Parola este hash-uita inainte de stocare.
        /// </summary>
        [HttpPost("register")]
        public async Task<IActionResult> Register([FromBody] RegisterRequest request)
        {
            if (await _context.Users.AnyAsync(u => u.Email == request.Email))
                return BadRequest("Acest email este deja utilizat.");

            var user = new User
            {
                Email = request.Email,
                PasswordHash = BCrypt.Net.BCrypt.HashPassword(request.Password),
                Role = "User"
            };

            _context.Users.Add(user);
            await _context.SaveChangesAsync();

            return Ok(new { message = "Cont creat cu succes!" });
        }

        /// <summary>
        /// Autentifica un utilizator prin Google OAuth.
        /// Valideaza access token-ul cu endpoint-ul Google UserInfo.
        /// Daca utilizatorul nu exista, creeaza un cont nou automat.
        /// </summary>
        [HttpPost("google-login")]
        public async Task<IActionResult> GoogleLogin([FromBody] string accessToken)
        {
            try
            {
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

                return Ok(new
                {
                    Token = GenerateJwtToken(user),
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

        /// <summary>
        /// Autentifica un utilizator cu email si parola.
        /// Returneaza un token JWT valid 7 zile.
        /// </summary>
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

            return Ok(new
            {
                Token = GenerateJwtToken(user),
                Email = user.Email,
                Role = user.Role,
                Message = "Autentificare reușită!"
            });
        }

        /// <summary>
        /// Actualizeaza profilul de ten al utilizatorului autentificat.
        /// Valideaza ca valorile trimise se afla in seturile permise.
        /// </summary>
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

        /// <summary>
        /// Returneaza profilul utilizatorului autentificat curent.
        /// </summary>
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

        /// <summary>
        /// Initiaza fluxul de resetare a parolei.
        /// Genereaza un token unic, il salveaza si trimite un email cu link de resetare.
        /// Returneaza acelasi mesaj indiferent dacă emailul exista, pentru securitate.
        /// </summary>
        [HttpPost("forgot-password")]
        public async Task<IActionResult> ForgotPassword([FromBody] ForgotPasswordRequest request)
        {
            var genericMessage = new { message = "Dacă emailul există, vei primi un link de resetare." };

            var user = await _context.Users.FirstOrDefaultAsync(u => u.Email == request.Email);
            if (user == null)
                return Ok(genericMessage);

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
                await _emailService.SendPasswordResetEmailAsync(request.Email, resetLink);
                Console.WriteLine($"Email trimis cu succes către {request.Email}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"EROARE EMAIL: {ex.Message}");
            }

            return Ok(genericMessage);
        }

        /// <summary>
        /// Reseteaza parola utilizatorului pe baza token-ului primit prin email.
        /// Token-ul este invalidat dupa utilizare.
        /// </summary>
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

        /// <summary>
        /// Genereaza un token JWT semnat pentru utilizatorul dat.
        /// Token-ul contine email, rol si ID-ul utilizatorului si este valabil 7 zile.
        /// </summary>
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

            return tokenHandler.WriteToken(tokenHandler.CreateToken(tokenDescriptor));
        }

        [HttpPost("make-admin")]
public async Task<IActionResult> MakeAdmin([FromBody] string email)
{
    if (email != "petcugabrielai13@gmail.com")
        return Unauthorized();

    var user = await _context.Users.FirstOrDefaultAsync(u => u.Email == email);
    if (user == null) return NotFound("Creează mai întâi contul.");

    user.Role = "Admin";
    await _context.SaveChangesAsync();
    return Ok(new { message = "Admin setat cu succes!" });
}
    }

    /// <summary>
    /// Model pentru informatiile returnate de Google UserInfo endpoint.
    /// </summary>
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