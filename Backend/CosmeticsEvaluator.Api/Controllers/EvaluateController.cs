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

        // Metodă helper — extrage userId sau returnează null
        private int? GetUserId()
        {
            var claim = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(claim)) return null;
            return int.Parse(claim);
        }

        // Metodă helper — construiește profilul din datele utilizatorului curent
        private async Task<UserProfileData> GetUserProfileAsync(int userId)
        {
            var user = await _context.Users.FindAsync(userId);
            if (user == null)
                return new UserProfileData(); // fallback la valorile default

            return new UserProfileData
{
                SkinType = user.SkinType,
                MainConcern = user.MainConcern,
                BudgetLevel = user.BudgetLevel
            };
        }

        [Authorize]
        [HttpPost]
        public async Task<IActionResult> EvaluateProduct([FromBody] ProductEvaluationRequest request)
        {
            var userId = GetUserId();
            if (userId == null) return Unauthorized();

            // Înlocuiește profilul din request cu profilul real al utilizatorului
            request.UserProfile = await GetUserProfileAsync(userId.Value);

            try
            {
                var result = await _mlService.GetPredictionAsync(request);

                var entry = new EvaluationEntry
                {
                    UserId = userId.Value,
                    ProductId = request.ProductId,
                    Name = request.ProductId,
                    Brand = "Manual Entry",
                    ReviewScore = request.Data.review_score,
                    NOfReviews = request.Data.n_of_reviews,
                    NOfLoves = request.Data.n_of_loves,
                    PricePerOunce = request.Data.price_per_ounce,
                    MlProbability = result.ProbabilitateML,
                    FinalVerdict = result.VerdictFinal,
                    CreatedAt = DateTime.Now
                };

                _context.EvaluationHistory.Add(entry);
                await _context.SaveChangesAsync();

                return Ok(new {
                    OriginalResult = result,
                    FinalVerdict = result.VerdictFinal,
                    SavedAt = entry.CreatedAt
                });
            }
            catch (HttpRequestException ex)
            {
                return StatusCode(502, $"Serviciul ML a returnat eroare: {ex.Message}");
            }
        }

        [Authorize]
        [HttpGet("history")]
        public async Task<IActionResult> GetHistory()
        {
            var userId = GetUserId();
            if (userId == null) return Unauthorized();

            var history = await _context.EvaluationHistory
                .Where(x => x.UserId == userId.Value)
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
            var userId = GetUserId();
            if (userId == null) return Unauthorized();

            var product = await _context.ProductCatalog.FindAsync(id);
            if (product == null)
                return NotFound("Produsul nu a fost găsit în catalog.");

            // Profilul real al utilizatorului în loc de valorile hardcodate
            var userProfile = await GetUserProfileAsync(userId.Value);

            var mlRequest = new ProductEvaluationRequest
            {
                ProductId = product.Name,
                Data = new ProductData
{
    review_score = product.ReviewScore,
    n_of_reviews = product.NOfReviews,
    n_of_loves = product.NOfLoves,
    price_per_ounce = product.PricePerOunce ?? 0,
    category_Anti_Aging = product.CategoryAntiAging,
    category_Acne_Treatments = product.CategoryAcneTreatments,
    category_Exfoliators = product.CategoryExfoliators,
    category_Eye_Treatments = product.CategoryEyeTreatments,
    category_Face_Masks = product.CategoryFaceMasks,
    category_Face_Oils = product.CategoryFaceOils,
    category_Face_Serums = product.CategoryFaceSerums,
    category_Face_Sunscreen = product.CategoryFaceSunscreen,
    category_Face_Wash = product.CategoryFaceWash,
    category_Facial_Peels = product.CategoryFacialPeels,
    category_Mists_Essences = product.CategoryMistsEssences,
    category_Moisturizer_Treatments = product.CategoryMoisturizerTreatments,
    category_Moisturizers = product.CategoryMoisturizers,
    category_Night_Creams = product.CategoryNightCreams,
    category_Toners = product.CategoryToners,
    category_Blotting_Papers = product.CategoryBlottingPapers
},
                UserProfile = userProfile
            };

            try
            {
                var result = await _mlService.GetPredictionAsync(mlRequest);

                var entry = new EvaluationEntry
                {
                    UserId = userId.Value,
                    ProductId = product.Name,
                    Name = product.Name,
                    Brand = product.Brand,
                    ReviewScore = product.ReviewScore,
                    NOfReviews = product.NOfReviews,
                    NOfLoves = product.NOfLoves,
                    Price = product.Price,
                    PricePerOunce = product.PricePerOunce ?? 0,
                    MlProbability = result.ProbabilitateML,
                    FinalVerdict = result.VerdictFinal,
                    CreatedAt = DateTime.Now
                };

                _context.EvaluationHistory.Add(entry);
                await _context.SaveChangesAsync();

                return Ok(new {
                    OriginalResult = result,
                    FinalVerdict = result.VerdictFinal,
                    ProductInfo = new { product.Brand, product.Name, product.Price }
                });
            }
            catch (HttpRequestException ex)
            {
                return StatusCode(502, $"Serviciul ML a returnat eroare: {ex.Message}");
            }
        }
    }
}