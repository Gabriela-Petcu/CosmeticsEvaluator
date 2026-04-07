using CosmeticsEvaluator.Api.Models;
using CosmeticsEvaluator.Api.Services; 
using Microsoft.AspNetCore.Mvc;
namespace CosmeticsEvaluator.Api.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class EvaluateController : ControllerBase
    {
        private readonly IMlService _mlService;

        public EvaluateController(IMlService mlService)
        {
            _mlService = mlService;
        }

        [HttpPost]
public async Task<IActionResult> EvaluateProduct([FromBody] ProductEvaluationRequest request)
{
    var result = await _mlService.GetPredictionAsync(request);
    
    if (result == null) return StatusCode(500, "ML Service unreachable");

    // Adăugăm o logică de interpretare (Verdict)
    string verdict;
    string recommendationColor;

    if (result.ml.merita_ml && result.baseline.score > 70)
    {
        verdict = "Produs Excelent! Atât statisticile cât și AI-ul îl recomandă.";
        recommendationColor = "Green";
    }
    else if (result.ml.merita_ml)
    {
        verdict = "Recomandat de AI. Modelul a găsit potențial în acest produs peste media pieței.";
        recommendationColor = "Blue";
    }
    else
    {
        verdict = "Nu este recomandat momentan pe baza analizei datelor.";
        recommendationColor = "Red";
    }

    // Putem returna un obiect nou care include și verdictul nostru "uman"
    return Ok(new { 
        OriginalResult = result, 
        FinalVerdict = verdict,
        Color = recommendationColor
    });
}
    }
}