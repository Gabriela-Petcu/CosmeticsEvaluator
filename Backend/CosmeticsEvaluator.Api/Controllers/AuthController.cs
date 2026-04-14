using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CosmeticsEvaluator.Api.Data;
using CosmeticsEvaluator.Api.Models;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using Google.Apis.Auth;

namespace CosmeticsEvaluator.Api.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class AuthController : ControllerBase
    {
        private readonly AppDbContext _context;
        private readonly IConfiguration _config; // Adăugat pentru acces la appsettings.json

        public AuthController(AppDbContext context, IConfiguration config)
        {
            _context = context;
            _config = config;
        }

        [HttpPost("register")]
        public async Task<IActionResult> Register([FromBody] RegisterRequest request)
        {
            if (await _context.Users.AnyAsync(u => u.Email == request.Email))
            {
                return BadRequest("Acest email este deja utilizat.");
            }

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
public async Task<IActionResult> GoogleLogin([FromBody] string idToken)
{
    try
    {
        // 1. Validăm token-ul cu serverele Google
        var settings = new GoogleJsonWebSignature.ValidationSettings()
        {
            Audience = new List<string>() { _config["Google:ClientId"]! }
        };

        var payload = await GoogleJsonWebSignature.ValidateAsync(idToken, settings);

        // 2. Căutăm utilizatorul în baza noastră de date după email
        var user = await _context.Users.FirstOrDefaultAsync(u => u.Email == payload.Email);

        if (user == null)
        {
            // 3. Dacă nu există, îl înregistrăm automat
            user = new User
            {
                Email = payload.Email,
                Role = "User", // Putem pune Admin dacă ești tu
                PasswordHash = "EXTERNAL_AUTH_GOOGLE", // Nu avem parolă pentru Google
                CreatedAt = DateTime.Now
            };
            _context.Users.Add(user);
            await _context.SaveChangesAsync();
        }

        // 4. Generăm token-ul nostru JWT (refolosim logica de la Login-ul clasic)
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
        return BadRequest("Autentificare Google eșuată: " + ex.Message);
    }
}

// 💡 Mic truc: Mută logica de generare token într-o metodă separată 
// ca să nu repeți codul de la Login-ul normal
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
            new Claim(ClaimTypes.NameIdentifier, user.Id.ToString()), // Adaugă asta lângă Email și Role
        }),
        Expires = DateTime.UtcNow.AddDays(7),
        Issuer = _config["Jwt:Issuer"],
        Audience = _config["Jwt:Audience"],
        SigningCredentials = new SigningCredentials(new SymmetricSecurityKey(key), SecurityAlgorithms.HmacSha256Signature)
    };
    var token = tokenHandler.CreateToken(tokenDescriptor);
    return tokenHandler.WriteToken(token);
}

        [HttpPost("login")]
public async Task<IActionResult> Login([FromBody] LoginRequest request)
{
    var user = await _context.Users.FirstOrDefaultAsync(u => u.Email == request.Email);

    // 1. Verificăm dacă user-ul există
    if (user == null)
    {
        return Unauthorized("Email sau parolă incorectă.");
    }

    // 2. PROTECȚIE: Dacă user-ul e făcut prin Google, nu îl lăsăm să se logheze cu parolă
    if (user.PasswordHash == "EXTERNAL_AUTH_GOOGLE")
    {
        return BadRequest("Acest cont a fost creat cu Google. Te rugăm să folosești butonul 'Login cu Google'.");
    }

    // 3. Verificăm parola cu BCrypt
    if (!BCrypt.Net.BCrypt.Verify(request.Password, user.PasswordHash))
    {
        return Unauthorized("Email sau parolă incorectă.");
    }

    // 4. Folosim metoda noastră curată pentru a genera token-ul!
    var tokenString = GenerateJwtToken(user);

    return Ok(new 
    { 
        Token = tokenString, 
        Email = user.Email, 
        Role = user.Role,
        Message = "Autentificare reușită!"
    });
}
        }
}