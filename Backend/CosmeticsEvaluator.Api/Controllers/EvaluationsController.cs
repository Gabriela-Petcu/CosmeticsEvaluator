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

        // GET: /Evaluations
        [HttpGet]
        public async Task<IActionResult> GetAll()
        {
            // Am modificat _context.Evaluations în _context.EvaluationHistory
            var evaluations = await _context.EvaluationHistory.ToListAsync();
            return Ok(evaluations);
        }

        // POST: /Evaluations
        [HttpPost]
        public async Task<IActionResult> Create([FromBody] EvaluationEntry evaluation)
        {
            evaluation.CreatedAt = DateTime.Now;

            // Am modificat aici
            _context.EvaluationHistory.Add(evaluation);
            await _context.SaveChangesAsync();

            return Ok(new { message = "Evaluare salvată cu succes!", data = evaluation });
        }

        // DELETE: /Evaluations/5
        [HttpDelete("{id}")]
        public async Task<IActionResult> Delete(int id)
        {
            // Am modificat aici
            var evaluation = await _context.EvaluationHistory.FindAsync(id);
            if (evaluation == null) return NotFound("Evaluarea nu a fost găsită.");

            _context.EvaluationHistory.Remove(evaluation);
            await _context.SaveChangesAsync();

            return Ok(new { message = "Evaluare ștearsă cu succes!" });
        }
    }
}