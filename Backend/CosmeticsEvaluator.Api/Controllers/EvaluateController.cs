using CosmeticsEvaluator.Api.Models;
using CosmeticsEvaluator.Api.Services;
using CosmeticsEvaluator.Api.Data;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Authorization;
using System.Security.Claims;

namespace CosmeticsEvaluator.Api.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class EvaluateController : ControllerBase
    {
        private readonly IMlService _mlService;
        private readonly AppDbContext _context;

        public EvaluateController(IMlService mlService, AppDbContext context)
        {
            _mlService = mlService;
            _context = context;
        }

        [Authorize]
        [HttpPost]
        public async Task<IActionResult> EvaluateProduct([FromBody] ProductEvaluationRequest request)
        {
            // 1. Extrage ID-ul userului
            var userIdClaim = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userIdClaim)) return Unauthorized();
            int userId = int.Parse(userIdClaim);

            // 2. Apelează serviciul ML
            var result = await _mlService.GetPredictionAsync(request);
            
            if (result == null) 
                return StatusCode(500, "Nu s-a putut contacta serviciul ML Python.");

            // 3. Folosește verdictul venit direct din noul Pipeline Python
            string verdict = result.VerdictFinal;

            // 4. Salvează în istoric
            var entry = new EvaluationEntry
            {
                UserId = userId,
                ProductId = request.ProductId,
                Name = request.ProductId, 
                Brand = "Manual Entry",    
                ReviewScore = request.Data.review_score,
                NOfReviews = request.Data.n_of_reviews,
                NOfLoves = request.Data.n_of_loves,
                PricePerOunce = request.Data.price_per_ounce,
                MlProbability = result.ProbabilitateML,
                FinalVerdict = verdict,
                CreatedAt = DateTime.Now
            };

            _context.EvaluationHistory.Add(entry);
            await _context.SaveChangesAsync();

            return Ok(new { 
                OriginalResult = result, 
                FinalVerdict = verdict,
                SavedAt = entry.CreatedAt 
            });
        }

        [Authorize]
        [HttpGet("history")]
        public async Task<IActionResult> GetHistory()
        {
            var userIdClaim = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userIdClaim)) return Unauthorized();
            
            var userId = int.Parse(userIdClaim);

            var history = await _context.EvaluationHistory
                .Where(x => x.UserId == userId) 
                .OrderByDescending(x => x.CreatedAt)
                .ToListAsync();
                
            return Ok(history);
        }

        [Authorize]
        [HttpGet("products")]
        public async Task<IActionResult> GetProducts()
        {
            var products = await _context.ProductCatalog
                .Select(p => new { p.Id, p.Brand, p.Name })
                .ToListAsync();
            return Ok(products);
        }

        [Authorize]
        [HttpPost("evaluate-by-id/{id}")]
        public async Task<IActionResult> EvaluateById(int id)
        {
            var userIdClaim = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userIdClaim)) return Unauthorized();
            int userId = int.Parse(userIdClaim);

            var product = await _context.ProductCatalog.FindAsync(id);
            if (product == null) 
                return NotFound("Produsul nu a fost găsit în catalog.");

            var mlRequest = new ProductEvaluationRequest {
                ProductId = product.Name,
                Data = new ProductData {
                    review_score = product.ReviewScore,
                    n_of_reviews = product.NOfReviews,
                    n_of_loves = product.NOfLoves,
                    price_per_ounce = product.PricePerOunce
                },
                UserProfile = new UserProfileData {
                    skin_type = "combination", 
                    main_concern = "acne",
                    budget_level = "low"
                }
            };

            var result = await _mlService.GetPredictionAsync(mlRequest);
            if (result == null) 
                return StatusCode(500, "Serviciul ML nu răspunde.");

            string verdict = result.VerdictFinal;

            var entry = new EvaluationEntry
            {
                UserId = userId,
                ProductId = product.Name,
                Name = product.Name,
                Brand = product.Brand,
                ReviewScore = product.ReviewScore,
                NOfReviews = product.NOfReviews,
                NOfLoves = product.NOfLoves,
                Price = product.Price,
                PricePerOunce = product.PricePerOunce,
                MlProbability = result.ProbabilitateML,
                FinalVerdict = verdict,
                CreatedAt = DateTime.Now
            };

            _context.EvaluationHistory.Add(entry);
            await _context.SaveChangesAsync();

            return Ok(new { 
                OriginalResult = result, 
                FinalVerdict = verdict,
                ProductInfo = new { product.Brand, product.Name, product.Price }
            });
        }
    }
}