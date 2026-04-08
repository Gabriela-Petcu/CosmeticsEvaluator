using CosmeticsEvaluator.Api.Services;
using Microsoft.EntityFrameworkCore;
using CosmeticsEvaluator.Api.Data;
using CosmeticsEvaluator.Api.Models;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using Microsoft.OpenApi.Models;

var builder = WebApplication.CreateBuilder(args);

// 1. SERVICII
builder.Services.AddControllers();
builder.Services.AddHttpClient<IMlService, MlService>();

builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new OpenApiInfo { Title = "Cosmetics AI API", Version = "v1" });

    // Această parte configurează câmpul de introducere pentru Token
    c.AddSecurityDefinition("Bearer", new OpenApiSecurityScheme
    {
        Description = "Introduceți token-ul JWT astfel: Bearer {tokenul_tau}",
        Name = "Authorization",
        In = ParameterLocation.Header,
        Type = SecuritySchemeType.ApiKey,
        Scheme = "Bearer"
    });

    c.AddSecurityRequirement(new OpenApiSecurityRequirement
    {
        {
            new OpenApiSecurityScheme
            {
                Reference = new OpenApiReference
                {
                    Type = ReferenceType.SecurityScheme,
                    Id = "Bearer"
                }
            },
            new string[] {}
        }
    });
});

// Configurare Autentificare JWT
builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
    {
        options.TokenValidationParameters = new TokenValidationParameters
        {
            ValidateIssuer = true,
            ValidateAudience = true,
            ValidateLifetime = true,
            ValidateIssuerSigningKey = true,
            ValidIssuer = builder.Configuration["Jwt:Issuer"],
            ValidAudience = builder.Configuration["Jwt:Audience"],
            IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(builder.Configuration["Jwt:Key"]))
        };
    });

builder.Services.AddCors(options => {
    options.AddDefaultPolicy(p => p.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());
});

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseSqlite("Data Source=cosmetics.db"));

var app = builder.Build();

// 2. LOGICA DE SEEDING
using (var scope = app.Services.CreateScope())
{
    var services = scope.ServiceProvider;
    var context = services.GetRequiredService<AppDbContext>();
    context.Database.EnsureCreated();
    SeedDatabase(context);
}

// 3. MIDDLEWARE (Ordinea este critică!)
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.UseCors();

app.UseAuthentication(); // Activează verificarea token-ului
app.UseAuthorization();  // Verifică permisiunile (rolurile)

app.MapControllers();

app.Run();

// 4. METODA DE SEEDING
void SeedDatabase(AppDbContext context)
{
    if (context.ProductCatalog.Any()) return;

    var basePath = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName;
    var path = Path.Combine(basePath, "Data", "Raw", "skincare_df.csv");
    
    if (!File.Exists(path)) {
        path = Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "Data", "Raw", "skincare_df.csv");
    }

    if (!File.Exists(path)) return;

    using var reader = new StreamReader(path);
    reader.ReadLine(); 

    while (!reader.EndOfStream)
    {
        var line = reader.ReadLine();
        if (string.IsNullOrWhiteSpace(line)) continue;
        
        var values = line.Split(',');
        try {
            var product = new ProductCatalog
            {
                Brand = values[1].Trim(),
                Name = values[2].Trim(),
                Price = double.Parse(values[3]),
                NOfReviews = (int)double.Parse(values[4]),
                NOfLoves = (int)double.Parse(values[5]),
                ReviewScore = double.Parse(values[6]),
                PricePerOunce = values.Length > 45 ? double.Parse(values[45]) : 0
            };
            context.ProductCatalog.Add(product);
        } catch { continue; }
    }
    context.SaveChanges();
    Console.WriteLine("Import finalizat cu succes!");
}