using CosmeticsEvaluator.Api.Models;

namespace CosmeticsEvaluator.Api.Services
{
    /// <summary>
    /// Defineste contractul pentru comunicarea cu componenta de Machine Learning.
    /// Trimite datele produsului si primeste predictia de evaluare.
    /// </summary>
    public interface IMlService
    {
        /// <summary>
        /// Trimite un request de evaluare catre serviciul ML si returneaza raspunsul.
        /// </summary>
        Task<EvaluationResponse?> GetPredictionAsync(ProductEvaluationRequest request);
    }

    /// <summary>
    /// Implementare a serviciului ML care comunica cu endpoint-ul FastAPI prin HTTP.
    /// URL-ul de baza este configurat prin MlService:BaseUrl in appsettings / variabile de mediu.
    /// </summary>
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