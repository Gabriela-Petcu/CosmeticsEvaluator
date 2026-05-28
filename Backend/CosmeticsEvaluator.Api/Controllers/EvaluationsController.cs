using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CosmeticsEvaluator.Api.Data;
using System.Security.Claims;

namespace CosmeticsEvaluator.Api.Controllers
{
    /// <summary>
    /// Controller pentru gestionarea istoricului de evaluari.
    /// Utilizatorii pot sterge doar evaluarile proprii.
    /// </summary>
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

        /// <summary>
        /// Sterge o evaluare din istoricul utilizatorului autentificat.
        /// Un utilizator nu poate sterge evaluarile altui utilizator.
        /// </summary>
        [HttpDelete("{id}")]
        public async Task<IActionResult> Delete(int id)
        {
            var userIdClaim = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userIdClaim)) return Unauthorized();
            var userId = int.Parse(userIdClaim);

            var evaluation = await _context.EvaluationHistory.FindAsync(id);
            if (evaluation == null)
                return NotFound("Evaluarea nu a fost găsită.");

            if (evaluation.UserId != userId)
                return Forbid("Nu ai permisiunea să ștergi această evaluare.");

            _context.EvaluationHistory.Remove(evaluation);
            await _context.SaveChangesAsync();

            return Ok(new { message = "Evaluare ștearsă cu succes!" });
        }
    }
}