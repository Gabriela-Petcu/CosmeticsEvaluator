using CosmeticsEvaluator.Api.Services;

var builder = WebApplication.CreateBuilder(args);

// Adaugă suportul pentru Controllere
builder.Services.AddControllers();

// Serviciul de comunicare cu Python
builder.Services.AddHttpClient<IMlService, MlService>();

// Configurare Swagger (fără AddOpenApi)
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Activăm Swagger în interfața web
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.UseAuthorization();

app.MapControllers();

app.Run();