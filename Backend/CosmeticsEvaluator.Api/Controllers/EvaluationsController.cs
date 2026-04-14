using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CosmeticsEvaluator.Api.Data;
using CosmeticsEvaluator.Api.Models;

namespace CosmeticsEvaluator.Api.Controllers
{
    [Authorize] // 🔒 ACESTA ESTE LACĂTUL!
    [ApiController]
    [Route("[controller]")]
    public class EvaluationsController : ControllerBase
    {
        private readonly AppDbContext _context;

        public EvaluationsController(AppDbContext context)
        {
            _context = context;
        }

        // Șterge liniile duplicate de [HttpGet] și parantezele rătăcite
[HttpGet]
public async Task<IActionResult> GetAll()
{
    var userIdClaim = User.FindFirst(System.Security.Claims.ClaimTypes.NameIdentifier)?.Value;
    if (string.IsNullOrEmpty(userIdClaim)) return Unauthorized();

    var userId = int.Parse(userIdClaim);
    var role = User.FindFirst(System.Security.Claims.ClaimTypes.Role)?.Value;

    if (role == "Admin")
    {
        return Ok(await _context.EvaluationHistory.ToListAsync());
    }

    var userEvaluations = await _context.EvaluationHistory
        .Where(x => x.UserId == userId)
        .ToListAsync();

    return Ok(userEvaluations);
}

        // POST: /Evaluations
        [HttpPost]
public async Task<IActionResult> Create([FromBody] EvaluationEntry evaluation)
{
    var userId = int.Parse(User.FindFirst(System.Security.Claims.ClaimTypes.NameIdentifier)?.Value);
    
    evaluation.UserId = userId; // Altfel crapă la SaveChanges fiindcă e Required
    evaluation.CreatedAt = DateTime.Now;

    _context.EvaluationHistory.Add(evaluation);
    await _context.SaveChangesAsync();

    return Ok(new { message = "Evaluare salvată cu succes!", data = evaluation });
}
        // DELETE: /Evaluations/5
        [HttpDelete("{id}")]
public async Task<IActionResult> Delete(int id)
{
    var userId = int.Parse(User.FindFirst(System.Security.Claims.ClaimTypes.NameIdentifier)?.Value);
    
    var evaluation = await _context.EvaluationHistory.FindAsync(id);
    
    if (evaluation == null) return NotFound("Evaluarea nu a fost găsită.");

    // Verificăm dacă evaluarea aparține utilizatorului care vrea să o șteargă
    if (evaluation.UserId != userId) return Forbid("Nu ai permisiunea să ștergi această evaluare.");

    _context.EvaluationHistory.Remove(evaluation);
    await _context.SaveChangesAsync();

    return Ok(new { message = "Evaluare ștearsă cu succes!" });
}
    }
}