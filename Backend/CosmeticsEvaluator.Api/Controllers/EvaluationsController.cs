using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CosmeticsEvaluator.Api.Data;
using System.Security.Claims;

namespace CosmeticsEvaluator.Api.Controllers
{
    /// <summary>
    /// Controller for managing evaluation history.
    /// Users can only delete their own evaluations.
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
        /// Deletes an evaluation from the authenticated user's history.
        /// A user cannot delete evaluations of another user.
        /// </summary>
        [HttpDelete("{id}")]
        public async Task<IActionResult> Delete(int id)
        {
            var userIdClaim = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userIdClaim)) return Unauthorized();
            var userId = int.Parse(userIdClaim);

            var evaluation = await _context.EvaluationHistory.FindAsync(id);
            if (evaluation == null)
                return NotFound("The evaluation was not found.");

            if (evaluation.UserId != userId)
                return Forbid("You do not have permission to delete this evaluation.");

            _context.EvaluationHistory.Remove(evaluation);
            await _context.SaveChangesAsync();

            return Ok(new { message = "Evaluation deleted successfully!" });
        }
    }
}