using CosmeticsEvaluator.Api.Models;

namespace CosmeticsEvaluator.Api.Services
{
    public interface IMlService
    {
        Task<EvaluationResponse?> GetPredictionAsync(ProductEvaluationRequest request);
    }

    public class MlService : IMlService
    {
        private readonly HttpClient _httpClient;

        public MlService(HttpClient httpClient, IConfiguration config)
        {
            _httpClient = httpClient;

            var baseUrl = config["MlService:BaseUrl"]
                ?? throw new InvalidOperationException(
                    "MlService:BaseUrl nu este configurat în appsettings."
                );

            _httpClient.BaseAddress = new Uri(baseUrl);
            _httpClient.Timeout = TimeSpan.FromSeconds(30);
        }

        public async Task<EvaluationResponse?> GetPredictionAsync(ProductEvaluationRequest request)
        {
            var response = await _httpClient.PostAsJsonAsync("/evaluate", request);

            if (response.IsSuccessStatusCode)
                return await response.Content.ReadFromJsonAsync<EvaluationResponse>();

            var errorBody = await response.Content.ReadAsStringAsync();
            throw new HttpRequestException(
                $"ML service returned {(int)response.StatusCode}: {errorBody}"
            );
        }
    }
}