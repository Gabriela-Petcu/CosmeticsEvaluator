FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

COPY ["Backend/CosmeticsEvaluator.Api/CosmeticsEvaluator.Api.csproj", "Backend/CosmeticsEvaluator.Api/"]
RUN dotnet restore "Backend/CosmeticsEvaluator.Api/CosmeticsEvaluator.Api.csproj"

COPY . .

WORKDIR /src/Backend/CosmeticsEvaluator.Api
RUN dotnet publish "CosmeticsEvaluator.Api.csproj" -c Release -o /app/publish --no-restore

FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS final
WORKDIR /app

COPY --from=build /app/publish .

COPY --from=build /src/Data/Raw/skincare_df.csv /app/Data/skincare_df.csv

ENV ASPNETCORE_URLS=http://+:${PORT:-8080}
ENV ASPNETCORE_ENVIRONMENT=Production

EXPOSE 8080

ENTRYPOINT ["dotnet", "CosmeticsEvaluator.Api.dll"]