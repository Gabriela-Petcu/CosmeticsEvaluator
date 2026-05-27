# ─── Stage 1: Build ───────────────────────────────────────────────────────────
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copiază fișierele de proiect și restaurează dependențele
COPY ["Backend/CosmeticsEvaluator.Api/CosmeticsEvaluator.Api.csproj", "Backend/CosmeticsEvaluator.Api/"]

# Dacă există alte proiecte referențiate (ex: librării din soluție), adaugă-le mai jos:
# COPY ["Backend/CosmeticsEvaluator.Core/CosmeticsEvaluator.Core.csproj", "Backend/CosmeticsEvaluator.Core/"]

RUN dotnet restore "Backend/CosmeticsEvaluator.Api/CosmeticsEvaluator.Api.csproj"

# Copiază tot codul sursă
COPY . .

# Publish în Release
WORKDIR /src/Backend/CosmeticsEvaluator.Api
RUN dotnet publish "CosmeticsEvaluator.Api.csproj" -c Release -o /app/publish --no-restore

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS final
WORKDIR /app

# Copiază aplicația publicată
COPY --from=build /app/publish .

# Railway injectează PORT dinamic — ASP.NET trebuie să asculte pe el
ENV ASPNETCORE_URLS=http://+:${PORT:-8080}
ENV ASPNETCORE_ENVIRONMENT=Production

EXPOSE 8080

ENTRYPOINT ["dotnet", "CosmeticsEvaluator.Api.dll"]