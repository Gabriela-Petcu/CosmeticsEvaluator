using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authorization;
using CosmeticsEvaluator.Api.Data;
using CosmeticsEvaluator.Api.Models;
using Microsoft.EntityFrameworkCore;

namespace CosmeticsEvaluator.Api.Controllers
{
    [ApiController]
    [Route("[controller]")]
    [Authorize(Roles = "Admin")] // Doar Adminul are acces la TOT controllerul
    public class ProductController : ControllerBase
    {
        private readonly AppDbContext _context;

        public ProductController(AppDbContext context)
        {
            _context = context;
        }

        // ADĂUGARE PRODUS NOU
        [HttpPost]
        public async Task<IActionResult> AddProduct([FromBody] ProductCatalog product)
        {
            _context.ProductCatalog.Add(product);
            await _context.SaveChangesAsync();
            return Ok(new { message = "Produs adăugat cu succes!", id = product.Id });
        }

        // ȘTERGERE PRODUS
        [HttpDelete("{id}")]
        public async Task<IActionResult> DeleteProduct(int id)
        {
            var product = await _context.ProductCatalog.FindAsync(id);
            if (product == null) return NotFound("Produsul nu există.");

            _context.ProductCatalog.Remove(product);
            await _context.SaveChangesAsync();
            return Ok(new { message = "Produs șters din catalog." });
        }
    }
}