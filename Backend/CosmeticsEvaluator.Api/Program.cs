using CosmeticsEvaluator.Api.Services;
using Microsoft.EntityFrameworkCore;
using CosmeticsEvaluator.Api.Data;
using CosmeticsEvaluator.Api.Models;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using Microsoft.OpenApi.Models;
using CsvHelper;
using System.Globalization;

var builder = WebApplication.CreateBuilder(args);

// 1. SERVICII
builder.Services.AddControllers()
    .AddJsonOptions(options =>
    {
        options.JsonSerializerOptions.PropertyNameCaseInsensitive = true;
    });

builder.Services.AddHttpClient<IMlService, MlService>();
builder.Services.AddScoped<IEmailService, EmailService>();

builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new OpenApiInfo { Title = "Cosmetics AI API", Version = "v1" });

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
            IssuerSigningKey = new SymmetricSecurityKey(
                Encoding.UTF8.GetBytes(builder.Configuration["Jwt:Key"]!))
        };
    });

builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(p => p
        .WithOrigins(
            "https://skiniq-gamma.vercel.app",
            "https://skiniq-9ornfc8h0-gabriela-petcus-projects.vercel.app",
            "http://localhost:5173"
        )
        .AllowAnyMethod()
        .AllowAnyHeader());
});

builder.Services.AddEndpointsApiExplorer();

builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseSqlite("Data Source=cosmetics.db"));

var app = builder.Build();

// 2. SEEDING
using (var scope = app.Services.CreateScope())
{
    var services = scope.ServiceProvider;
    var context = services.GetRequiredService<AppDbContext>();
    context.Database.Migrate();
    SeedDatabase(context, builder.Environment.ContentRootPath);
}

// 3. MIDDLEWARE
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.UseCors();
app.UseAuthentication();
app.UseAuthorization();
app.MapControllers();
app.Run();

// 4. SEEDING
void SeedDatabase(AppDbContext context, string contentRootPath)
{
    if (context.ProductCatalog.Any()) return;

    var possiblePaths = new[]
    {
        Path.Combine(contentRootPath, "Data", "skincare_df.csv"),
        Path.Combine(contentRootPath, "..", "..", "Data", "Raw", "skincare_df.csv"),
        "/app/Data/skincare_df.csv"
    };

    var path = possiblePaths.FirstOrDefault(File.Exists);
    if (path == null)
    {
        Console.WriteLine("CSV nu a fost găsit în nicio cale.");
        return;
    }

    Console.WriteLine($"Cale CSV găsită: {path}");

    try
    {
        using var reader = new StreamReader(path);
        using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);

        csv.Context.RegisterClassMap<ProductCatalogMap>();
        var records = csv.GetRecords<ProductCatalog>().ToList();

        context.ProductCatalog.AddRange(records);
        context.SaveChanges();

        Console.WriteLine($"Import finalizat: {records.Count} produse importate.");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Eroare la import CSV: {ex.Message}");
    }
}