using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CosmeticsEvaluator.Api.Data;
using CosmeticsEvaluator.Api.Models;
using System.Security.Claims;

namespace CosmeticsEvaluator.Api.Controllers
{
    /// <summary>
    /// Controller pt operatiunile administrative.
    /// Accesibil doar utilizatorilor cu rolul Admin.
    /// </summary>
    [Authorize(Roles = "Admin")]
    [ApiController]
    [Route("[controller]")]
    public class AdminController : ControllerBase
    {
        private readonly AppDbContext _context;

        public AdminController(AppDbContext context)
        {
            _context = context;
        }


        /// <summary>
        /// Returneaza statistici globale: nr utilizatori, evaluari, produse,
        /// distributia verdictelor, evaluarile recente si produsele cel mai evaluate.
        /// </summary>
        [HttpGet("stats")]
        public async Task<IActionResult> GetStats()
        {
            var totalUsers = await _context.Users.CountAsync();
            var totalEvaluations = await _context.EvaluationHistory.CountAsync();
            var totalProducts = await _context.ProductCatalog.CountAsync();

            var verdictCounts = await _context.EvaluationHistory
                .GroupBy(e => e.FinalVerdict)
                .Select(g => new { Verdict = g.Key, Count = g.Count() })
                .ToListAsync();

            var recentEvaluations = await _context.EvaluationHistory
                .OrderByDescending(e => e.CreatedAt)
                .Take(5)
                .ToListAsync();

            var topProducts = await _context.EvaluationHistory
                .GroupBy(e => new { e.Name, e.Brand })
                .Select(g => new
                {
                    g.Key.Name,
                    g.Key.Brand,
                    Count = g.Count(),
                    AvgProbability = g.Average(e => e.MlProbability)
                })
                .OrderByDescending(x => x.Count)
                .Take(5)
                .ToListAsync();

            return Ok(new
            {
                TotalUsers = totalUsers,
                TotalEvaluations = totalEvaluations,
                TotalProducts = totalProducts,
                VerdictCounts = verdictCounts,
                RecentEvaluations = recentEvaluations,
                TopProducts = topProducts
            });
        }


        /// <summary>
        /// Returneaza lista tuturor utilizatorilor cu informatii despre profil si nr de evaluari.
        /// </summary>
        [HttpGet("users")]
        public async Task<IActionResult> GetUsers()
        {
            var users = await _context.Users
                .Select(u => new
                {
                    u.Id,
                    u.Email,
                    u.Role,
                    u.SkinType,
                    u.MainConcern,
                    u.BudgetLevel,
                    u.CreatedAt,
                    EvaluationCount = _context.EvaluationHistory.Count(e => e.UserId == u.Id)
                })
                .OrderByDescending(u => u.CreatedAt)
                .ToListAsync();

            return Ok(users);
        }

        /// <summary>
        /// Actualizeaza rolul unui utilizator. Un admin nu isi poate schimba propriul rol.
        /// </summary>
        [HttpPut("users/{id}/role")]
        public async Task<IActionResult> UpdateUserRole(int id, [FromBody] UpdateRoleRequest request)
        {
            var allowedRoles = new[] { "User", "Admin" };
            if (!allowedRoles.Contains(request.Role))
                return BadRequest("Rol invalid. Valori permise: User, Admin");

            var user = await _context.Users.FindAsync(id);
            if (user == null)
                return NotFound("Utilizatorul nu a fost găsit.");

            var currentUserId = int.Parse(User.FindFirst(ClaimTypes.NameIdentifier)?.Value ?? "0");
            if (user.Id == currentUserId)
                return BadRequest("Nu îți poți schimba propriul rol.");

            user.Role = request.Role;
            await _context.SaveChangesAsync();

            return Ok(new { message = $"Rolul utilizatorului {user.Email} a fost schimbat în {request.Role}." });
        }

        /// <summary>
        /// Sterge un utilizator si toate evaluarile asociate acestuia.
        /// Un admin nu isi poate sterge propriul cont.
        /// </summary>
        [HttpDelete("users/{id}")]
        public async Task<IActionResult> DeleteUser(int id)
        {
            var currentUserId = int.Parse(User.FindFirst(ClaimTypes.NameIdentifier)?.Value ?? "0");
            if (id == currentUserId)
                return BadRequest("Nu îți poți șterge propriul cont din panoul de admin.");

            var user = await _context.Users.FindAsync(id);
            if (user == null)
                return NotFound("Utilizatorul nu a fost găsit.");

            var evaluations = _context.EvaluationHistory.Where(e => e.UserId == id);
            _context.EvaluationHistory.RemoveRange(evaluations);
            _context.Users.Remove(user);
            await _context.SaveChangesAsync();

            return Ok(new { message = $"Utilizatorul {user.Email} a fost șters." });
        }


        /// <summary>
        /// Returneaza produsele din catalog cu suport pentru paginare si cautare dupa nume sau brand.
        /// </summary>
        [HttpGet("products")]
        public async Task<IActionResult> GetProducts(
            [FromQuery] int page = 1,
            [FromQuery] int pageSize = 20,
            [FromQuery] string? search = null)
        {
            var query = _context.ProductCatalog.AsQueryable();

            if (!string.IsNullOrEmpty(search))
                query = query.Where(p => p.Name.Contains(search) || p.Brand.Contains(search));

            var total = await query.CountAsync();
            var products = await query
                .OrderBy(p => p.Brand)
                .Skip((page - 1) * pageSize)
                .Take(pageSize)
                .ToListAsync();

            return Ok(new { Total = total, Page = page, PageSize = pageSize, Products = products });
        }

        /// <summary>
        /// Adauga un produs nou in catalog.
        /// </summary>
        [HttpPost("products")]
        public async Task<IActionResult> AddProduct([FromBody] ProductCatalog product)
        {
            _context.ProductCatalog.Add(product);
            await _context.SaveChangesAsync();
            return Ok(new { message = "Produs adăugat cu succes!", id = product.Id });
        }

        /// <summary>
        /// Actualizeaza informatiile de baza ale unui produs existent.
        /// </summary>
        [HttpPut("products/{id}")]
        public async Task<IActionResult> UpdateProduct(int id, [FromBody] ProductCatalog updated)
        {
            var product = await _context.ProductCatalog.FindAsync(id);
            if (product == null)
                return NotFound("Produsul nu a fost găsit.");

            product.Brand = updated.Brand;
            product.Name = updated.Name;
            product.Price = updated.Price;
            product.NOfReviews = updated.NOfReviews;
            product.NOfLoves = updated.NOfLoves;
            product.ReviewScore = updated.ReviewScore;
            product.PricePerOunce = updated.PricePerOunce;

            await _context.SaveChangesAsync();
            return Ok(new { message = "Produs actualizat cu succes!" });
        }

        /// <summary>
        /// Sterge un produs din catalog.
        /// </summary>
        [HttpDelete("products/{id}")]
        public async Task<IActionResult> DeleteProduct(int id)
        {
            var product = await _context.ProductCatalog.FindAsync(id);
            if (product == null)
                return NotFound("Produsul nu a fost găsit.");

            _context.ProductCatalog.Remove(product);
            await _context.SaveChangesAsync();
            return Ok(new { message = "Produs șters din catalog." });
        }
    }

    /// <summary>
    /// Model pentru cererea de actualizare a rolului unui utilizator.
    /// </summary>
    public class UpdateRoleRequest
    {
        public string Role { get; set; } = string.Empty;
    }
}