using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CosmeticsEvaluator.Api.Data;
using CosmeticsEvaluator.Api.Models;
using BCrypt.Net;

namespace CosmeticsEvaluator.Api.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class AuthController : ControllerBase
    {
        private readonly AppDbContext _context;

        public AuthController(AppDbContext context)
        {
            _context = context;
        }

        [HttpPost("register")]
        public async Task<IActionResult> Register([FromBody] RegisterRequest request)
        {
            // Verificăm dacă utilizatorul există deja
            if (await _context.Users.AnyAsync(u => u.Email == request.Email))
            {
                return BadRequest("Acest email este deja utilizat.");
            }

            // CRIPTARE PAROLĂ: Transformăm "parola123" în ceva de tip "$2a$11$..."
            string hashedPassword = BCrypt.Net.BCrypt.HashPassword(request.Password);

            var user = new User
            {
                Email = request.Email,
                PasswordHash = hashedPassword,
                Role = "User" // Primul cont creat va fi User. Putem schimba manual în DB în "Admin"
            };

            _context.Users.Add(user);
            await _context.SaveChangesAsync();

            return Ok(new { message = "Cont creat cu succes!" });
        }

        [HttpPost("login")]
        public async Task<IActionResult> Login([FromBody] LoginRequest request)
        {
            var user = await _context.Users.FirstOrDefaultAsync(u => u.Email == request.Email);

            if (user == null || !BCrypt.Net.BCrypt.Verify(request.Password, user.PasswordHash))
            {
                return Unauthorized("Email sau parolă incorectă.");
            }

            // Momentan returnăm datele utilizatorului. 
            // Pasul următor va fi să generăm un Token JWT aici.
            return Ok(new { 
                id = user.Id, 
                email = user.Email, 
                role = user.Role,
                message = "Autentificare reușită!" 
            });
        }
    }
}