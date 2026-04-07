using CosmeticsEvaluator.Api.Services;
using Microsoft.EntityFrameworkCore;
using CosmeticsEvaluator.Api.Data;
using CosmeticsEvaluator.Api.Models; // Necesar pentru ProductCatalog

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
builder.Services.AddHttpClient<IMlService, MlService>();

builder.Services.AddCors(options => {
    options.AddDefaultPolicy(p => p.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());
});

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseSqlite("Data Source=cosmetics.db"));

var app = builder.Build();

// --- LOGICA DE SEEDING (TREBUIE SĂ FIE AICI) ---
using (var scope = app.Services.CreateScope())
{
    var services = scope.ServiceProvider;
    var context = services.GetRequiredService<AppDbContext>();
    // Ne asigurăm că baza de date e creată
    context.Database.EnsureCreated();
    SeedDatabase(context);
}

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.UseCors(); // Mutat înainte de Authorization
app.UseAuthorization();
app.MapControllers();

app.Run();

// METODA DE SEEDING (la finalul fișierului, în afara fluxului principal)
void SeedDatabase(AppDbContext context)
{
    if (context.ProductCatalog.Any()) return;

    // Navigăm din Backend/CosmeticsEvaluator.Api către rădăcina proiectului, apoi în Data/Raw
    var basePath = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName;
    var path = Path.Combine(basePath, "Data", "Raw", "skincare_df.csv");
    
    // Dacă prima variantă nu merge (depinde de unde rulezi terminalul), încercăm calea directă
    if (!File.Exists(path)) {
        path = Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "Data", "Raw", "skincare_df.csv");
    }

    if (!File.Exists(path)) {
        Console.WriteLine($"Eroare: Nu am găsit fișierul la calea: {path}");
        return;
    }

    using var reader = new StreamReader(path);
    reader.ReadLine(); // Skip header

    while (!reader.EndOfStream)
    {
        var line = reader.ReadLine();
        if (string.IsNullOrWhiteSpace(line)) continue;
        
        var values = line.Split(',');
        try {
            var product = new ProductCatalog
            {
                // Atenție: indexul depinde de CSV-ul tău. 
                // Din exemplul tău anterior: 0=id, 1=brand, 2=name, 3=price...
                Brand = values[1].Trim(),
                Name = values[2].Trim(),
                Price = double.Parse(values[3]),
                NOfReviews = (int)double.Parse(values[4]), // Folosim double.Parse pentru siguranță dacă sunt virgule
                NOfLoves = (int)double.Parse(values[5]),
                ReviewScore = double.Parse(values[6]),
                PricePerOunce = values.Length > 45 ? double.Parse(values[45]) : 0
            };
            context.ProductCatalog.Add(product);
        } catch { continue; }
    }
    context.SaveChanges();
    Console.WriteLine("Import finalizat cu succes pentru cele 1784 de produse!");
}