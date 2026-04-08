using Microsoft.EntityFrameworkCore;
using CosmeticsEvaluator.Api.Models;

namespace CosmeticsEvaluator.Api.Data
{
    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

        public DbSet<EvaluationEntry> EvaluationHistory { get; set; }
        public DbSet<ProductCatalog> ProductCatalog { get; set; }
        public DbSet<User> Users { get; set; }
    }
}