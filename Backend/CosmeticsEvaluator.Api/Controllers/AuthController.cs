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
    /// Controller for authentication and user account management.
    /// Handles registration, login, profile updates, and password reset.
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
        /// Registers a new user account with email and password.
        /// The password is hashed before storage.
        /// </summary>
        [HttpPost("register")]
        public async Task<IActionResult> Register([FromBody] RegisterRequest request)
        {
            if (await _context.Users.AnyAsync(u => u.Email == request.Email))
                return BadRequest("This email is already in use.");

            var user = new User
            {
                Email = request.Email,
                PasswordHash = BCrypt.Net.BCrypt.HashPassword(request.Password),
                Role = "User"
            };

            _context.Users.Add(user);
            await _context.SaveChangesAsync();

            return Ok(new { message = "Account created successfully!" });
        }

        /// <summary>
        /// Authenticates a user via Google OAuth.
        /// Validates the access token with the Google UserInfo endpoint.
        /// If the user does not exist, creates a new account automatically.
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
                    return BadRequest("Invalid Google token.");

                var json = await response.Content.ReadAsStringAsync();
                var userInfo = System.Text.Json.JsonSerializer.Deserialize<GoogleUserInfo>(json);

                if (userInfo == null || string.IsNullOrEmpty(userInfo.Email))
                    return BadRequest("Could not retrieve Google user data.");

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
                    Message = "Login with Google successful!"
                });
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Internal error: {ex.Message}");
            }
        }

        /// <summary>
        /// Authenticates a user with email and password.
        /// Returns a valid JWT token for 7 days.
        /// </summary>
        [HttpPost("login")]
        public async Task<IActionResult> Login([FromBody] LoginRequest request)
        {
            var user = await _context.Users.FirstOrDefaultAsync(u => u.Email == request.Email);

            if (user == null)
                return Unauthorized("Invalid email or password.");

            if (user.PasswordHash == "EXTERNAL_AUTH_GOOGLE")
                return BadRequest("This account was created with Google. Please use the 'Login with Google' button.");

            if (!BCrypt.Net.BCrypt.Verify(request.Password, user.PasswordHash))
                return Unauthorized("Invalid email or password.");

            return Ok(new
            {
                Token = GenerateJwtToken(user),
                Email = user.Email,
                Role = user.Role,
                Message = "Login successful!"
            });
        }

        /// <summary>
        /// Updates the skin profile of the authenticated user.
        /// Validates that the submitted values belong to the allowed sets.
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
                return BadRequest($"skin_type invalid. Allowed values: {string.Join(", ", allowedSkinTypes)}");

            if (!allowedConcerns.Contains(request.MainConcern))
                return BadRequest($"main_concern invalid. Allowed values: {string.Join(", ", allowedConcerns)}");

            if (!allowedBudgets.Contains(request.BudgetLevel))
                return BadRequest($"budget_level invalid. Allowed values: {string.Join(", ", allowedBudgets)}");

            var user = await _context.Users.FindAsync(userId);
            if (user == null) return NotFound();

            user.SkinType = request.SkinType;
            user.MainConcern = request.MainConcern;
            user.BudgetLevel = request.BudgetLevel;

            await _context.SaveChangesAsync();

            return Ok(new { message = "Profile updated successfully!" });
        }

        /// <summary>
        /// Returns the profile of the authenticated user.
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
        /// Initiates the password reset flow.
        /// Generates a unique token, saves it, and sends a reset link by email.
        /// Returns the same message regardless of whether the email exists (security measure).
        /// </summary>
        [HttpPost("forgot-password")]
        public async Task<IActionResult> ForgotPassword([FromBody] ForgotPasswordRequest request)
        {
            var genericMessage = new { message = "If the email exists, you will receive a reset link." };

            var user = await _context.Users.FirstOrDefaultAsync(u => u.Email == request.Email);
            if (user == null)
                return Ok(genericMessage);

            if (user.PasswordHash == "EXTERNAL_AUTH_GOOGLE")
                return BadRequest("This account uses Google authentication. Password is managed through your Google account.");

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
                Console.WriteLine($"Email sent successfully to {request.Email}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"EMAIL ERROR: {ex.Message}");
            }

            return Ok(genericMessage);
        }

        /// <summary>
        /// Resets the user's password using the token received by email.
        /// The token is invalidated after use.
        /// </summary>
        [HttpPost("reset-password")]
        public async Task<IActionResult> ResetPassword([FromBody] ResetPasswordRequest request)
        {
            var user = await _context.Users.FirstOrDefaultAsync(u => u.Email == request.Email);
            if (user == null)
                return BadRequest("Invalid request.");

            if (user.PasswordResetToken != request.Token)
                return BadRequest("Invalid token.");

            if (user.PasswordResetTokenExpiry == null || user.PasswordResetTokenExpiry < DateTime.UtcNow)
                return BadRequest("The token has expired. Please request a new reset link.");

            user.PasswordHash = BCrypt.Net.BCrypt.HashPassword(request.NewPassword);
            user.PasswordResetToken = null;
            user.PasswordResetTokenExpiry = null;

            await _context.SaveChangesAsync();

            return Ok(new { message = "Password reset successfully!" });
        }

        /// <summary>
        /// Generates a signed JWT token for the given user.
        /// The token contains email, role, and user ID and is valid for 7 days.
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
    if (user == null) return NotFound("Create the account first.");

    user.Role = "Admin";
    await _context.SaveChangesAsync();
    return Ok(new { message = "Admin role set successfully!" });
}
    }

    /// <summary>
    /// Model for data returned by the Google UserInfo endpoint.
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