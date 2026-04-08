using CosmeticsEvaluator.Api.Models;
using CosmeticsEvaluator.Api.Services;
using CosmeticsEvaluator.Api.Data;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

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

        // 1. Evaluare manuală (folosește datele trimise de utilizator în JSON)
        [HttpPost]
        public async Task<IActionResult> EvaluateProduct([FromBody] ProductEvaluationRequest request)
        {
            var result = await _mlService.GetPredictionAsync(request);
            
            if (result == null) 
                return StatusCode(500, "Nu s-a putut contacta serviciul ML Python.");

            string verdict = result.ml.merita_ml ? "Produs Recomandat" : "Nu este recomandat";

            // SALVĂM ÎN ISTORIC folosind datele din 'request'
            var entry = new EvaluationEntry
            {
                ProductId = request.ProductId,
                Name = request.ProductId, // În modul manual, punem ID-ul ca nume
                Brand = "Manual Entry",    // Nu avem brand în request-ul manual
                ReviewScore = request.Data.review_score,
                NOfReviews = request.Data.n_of_reviews,
                NOfLoves = request.Data.n_of_loves,
                PricePerOunce = request.Data.price_per_ounce,
                MlProbability = result.ml.probability,
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

        // 2. Istoric
        [HttpGet("history")]
        public async Task<IActionResult> GetHistory()
        {
            var history = await _context.EvaluationHistory
                .OrderByDescending(x => x.CreatedAt)
                .ToListAsync();
            return Ok(history);
        }

        // 3. Produse din Catalog
        [HttpGet("products")]
        public async Task<IActionResult> GetProducts()
        {
            var products = await _context.ProductCatalog
                .Select(p => new { p.Id, p.Brand, p.Name })
                .ToListAsync();
            return Ok(products);
        }

        // 4. Evaluare inteligentă (folosește datele din baza de date)
        [HttpPost("evaluate-by-id/{id}")]
        public async Task<IActionResult> EvaluateById(int id)
        {
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
                }
            };

            var result = await _mlService.GetPredictionAsync(mlRequest);
            if (result == null) 
                return StatusCode(500, "Serviciul ML nu răspunde.");

            string verdict = result.ml.merita_ml ? "Produs Recomandat" : "Nu este recomandat";

            // SALVĂM ÎN ISTORIC folosind datele din 'product' găsit în DB
            var entry = new EvaluationEntry
            {
                ProductId = product.Name,
                Name = product.Name,
                Brand = product.Brand,
                ReviewScore = product.ReviewScore,
                NOfReviews = product.NOfReviews,
                NOfLoves = product.NOfLoves,
                Price = product.Price,
                PricePerOunce = product.PricePerOunce,
                MlProbability = result.ml.probability,
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