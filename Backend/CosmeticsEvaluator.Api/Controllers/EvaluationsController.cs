using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CosmeticsEvaluator.Api.Data;
using CosmeticsEvaluator.Api.Models;
using System.Security.Claims;

namespace CosmeticsEvaluator.Api.Controllers
{
    [Authorize]
    [ApiController]
    [Route("[controller]")]
    public class EvaluationsController : ControllerBase
    {
        private readonly AppDbContext _context;

        public EvaluationsController(AppDbContext context)
        {
            _context = context;
        }

        [HttpGet]
        public async Task<IActionResult> GetAll()
        {
            var userIdClaim = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userIdClaim)) return Unauthorized();
            var userId = int.Parse(userIdClaim);

            var role = User.FindFirst(ClaimTypes.Role)?.Value;

            if (role == "Admin")
                return Ok(await _context.EvaluationHistory.ToListAsync());

            var userEvaluations = await _context.EvaluationHistory
                .Where(x => x.UserId == userId)
                .ToListAsync();

            return Ok(userEvaluations);
        }

        [HttpPost]
        public async Task<IActionResult> Create([FromBody] CreateEvaluationRequest request)
        {
            var userIdClaim = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userIdClaim)) return Unauthorized();
            var userId = int.Parse(userIdClaim);

            var evaluation = new EvaluationEntry
            {
                UserId = userId,
                ProductId = request.ProductId,
                Name = request.Name,
                Brand = request.Brand,
                ReviewScore = request.ReviewScore,
                NOfReviews = request.NOfReviews,
                NOfLoves = request.NOfLoves,
                Price = request.Price,
                PricePerOunce = request.PricePerOunce,
                MlProbability = request.MlProbability,
                FinalVerdict = request.FinalVerdict,
                CreatedAt = DateTime.Now
            };

            _context.EvaluationHistory.Add(evaluation);
            await _context.SaveChangesAsync();

            return Ok(new { message = "Evaluare salvată cu succes!", data = evaluation });
        }

        [HttpDelete("{id}")]
        public async Task<IActionResult> Delete(int id)
        {
            var userIdClaim = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userIdClaim)) return Unauthorized();
            var userId = int.Parse(userIdClaim);

            var evaluation = await _context.EvaluationHistory.FindAsync(id);
            if (evaluation == null) return NotFound("Evaluarea nu a fost găsită.");

            if (evaluation.UserId != userId) 
                return Forbid("Nu ai permisiunea să ștergi această evaluare.");

            _context.EvaluationHistory.Remove(evaluation);
            await _context.SaveChangesAsync();

            return Ok(new { message = "Evaluare ștearsă cu succes!" });
        }
    }
}