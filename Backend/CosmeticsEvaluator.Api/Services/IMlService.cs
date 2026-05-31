using CosmeticsEvaluator.Api.Models;

namespace CosmeticsEvaluator.Api.Services
{
    /// <summary>
    /// Defines the contract for communicating with the Machine Learning component.
    /// Sends product data and receives the evaluation prediction.
    /// </summary>
    public interface IMlService
    {
        /// <summary>
        /// Sends an evaluation request to the ML service and returns the response.
        /// </summary>
        Task<EvaluationResponse?> GetPredictionAsync(ProductEvaluationRequest request);
    }

    /// <summary>
    /// Implements the ML service that communicates with the FastAPI endpoint via HTTP.
    /// The base URL is configured through MlService:BaseUrl in appsettings or environment variables.
    /// </summary>
    public class MlService : IMlService
    {
        private readonly HttpClient _httpClient;

        public MlService(HttpClient httpClient, IConfiguration config)
        {
            _httpClient = httpClient;

            var baseUrl = config["MlService:BaseUrl"]
                ?? throw new InvalidOperationException(
                    "MlService:BaseUrl is not configured in appsettings."
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