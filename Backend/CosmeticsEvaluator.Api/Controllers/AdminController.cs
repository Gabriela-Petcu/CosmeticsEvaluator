using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CosmeticsEvaluator.Api.Data;
using CosmeticsEvaluator.Api.Models;
using System.Security.Claims;

namespace CosmeticsEvaluator.Api.Controllers
{
    /// <summary>
    /// Controller for administrative operations.
    /// Accessible only to users with the Admin role.
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
        /// Returns global statistics: user count, evaluation count, product count,
        /// verdict distribution, recent evaluations, and most-evaluated products.
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
        /// Returns a list of all users with profile information and evaluation count.
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
        /// Updates the role of a user. An admin cannot change their own role.
        /// </summary>
        [HttpPut("users/{id}/role")]
        public async Task<IActionResult> UpdateUserRole(int id, [FromBody] UpdateRoleRequest request)
        {
            var allowedRoles = new[] { "User", "Admin" };
            if (!allowedRoles.Contains(request.Role))
                return BadRequest("Invalid role. Allowed values: User, Admin");

            var user = await _context.Users.FindAsync(id);
            if (user == null)
                return NotFound("User not found.");

            var currentUserId = int.Parse(User.FindFirst(ClaimTypes.NameIdentifier)?.Value ?? "0");
            if (user.Id == currentUserId)
                return BadRequest("You cannot change your own role.");

            user.Role = request.Role;
            await _context.SaveChangesAsync();

            return Ok(new { message = $"Role of user {user.Email} has been changed to {request.Role}." });
        }

        /// <summary>
        /// Deletes a user and all evaluations associated with them.
        /// An admin cannot delete their own account.
        /// </summary>
        [HttpDelete("users/{id}")]
        public async Task<IActionResult> DeleteUser(int id)
        {
            var currentUserId = int.Parse(User.FindFirst(ClaimTypes.NameIdentifier)?.Value ?? "0");
            if (id == currentUserId)
                return BadRequest("You cannot delete your own account.");

            var user = await _context.Users.FindAsync(id);
            if (user == null)
                return NotFound("User not found.");

            var evaluations = _context.EvaluationHistory.Where(e => e.UserId == id);
            _context.EvaluationHistory.RemoveRange(evaluations);
            _context.Users.Remove(user);
            await _context.SaveChangesAsync();

            return Ok(new { message = $"User {user.Email} has been deleted." });
        }


        /// <summary>
        ///  Returns products from the catalog with pagination and search by name or brand.
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
        /// Adds a new product to the catalog.
        /// </summary>
        [HttpPost("products")]
        public async Task<IActionResult> AddProduct([FromBody] ProductCatalog product)
        {
            _context.ProductCatalog.Add(product);
            await _context.SaveChangesAsync();
            return Ok(new { message = "Product added successfully!", id = product.Id });
        }

        /// <summary>
        /// Updates the basic information of an existing product.
        /// </summary>
        [HttpPut("products/{id}")]
        public async Task<IActionResult> UpdateProduct(int id, [FromBody] ProductCatalog updated)
        {
            var product = await _context.ProductCatalog.FindAsync(id);
            if (product == null)
                return NotFound("Product not found.");

            product.Brand = updated.Brand;
            product.Name = updated.Name;
            product.Price = updated.Price;
            product.NOfReviews = updated.NOfReviews;
            product.NOfLoves = updated.NOfLoves;
            product.ReviewScore = updated.ReviewScore;
            product.PricePerOunce = updated.PricePerOunce;

            await _context.SaveChangesAsync();
            return Ok(new { message = "Product updated successfully!" });
        }

        /// <summary>
        /// Deletes a product from the catalog.
        /// </summary>
        [HttpDelete("products/{id}")]
        public async Task<IActionResult> DeleteProduct(int id)
        {
            var product = await _context.ProductCatalog.FindAsync(id);
            if (product == null)
                return NotFound("Product not found.");

            _context.ProductCatalog.Remove(product);
            await _context.SaveChangesAsync();
            return Ok(new { message = "Product deleted from catalog." });
        }
    }

    /// <summary>
    /// Request model for updating a user's role. The role must be either "User" or "Admin".
    /// </summary>
    public class UpdateRoleRequest
    {
        public string Role { get; set; } = string.Empty;
    }
}