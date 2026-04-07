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

        public MlService(HttpClient httpClient)
        {
            _httpClient = httpClient;
            // Adresa unde rulează FastAPI-ul tău
            _httpClient.BaseAddress = new Uri("http://127.0.0.1:8000");
        }

        public async Task<EvaluationResponse?> GetPredictionAsync(ProductEvaluationRequest request)
        {
            var response = await _httpClient.PostAsJsonAsync("/evaluate", request);
            
            if (response.IsSuccessStatusCode)
            {
                return await response.Content.ReadFromJsonAsync<EvaluationResponse>();
            }

            return null;
        }
    }
}