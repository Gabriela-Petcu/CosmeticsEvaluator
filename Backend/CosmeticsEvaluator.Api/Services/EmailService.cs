using SendGrid;
using SendGrid.Helpers.Mail;

namespace CosmeticsEvaluator.Api.Services
{
    public interface IEmailService
    {
        Task SendPasswordResetEmailAsync(string toEmail, string resetLink);
    }

    public class EmailService : IEmailService
    {
        private readonly IConfiguration _config;

        public EmailService(IConfiguration config)
        {
            _config = config;
        }

        public async Task SendPasswordResetEmailAsync(string toEmail, string resetLink)
        {
            var apiKey = _config["SendGrid__ApiKey"]!;
            var senderEmail = _config["Email:SenderEmail"]!;
            var senderName = _config["Email:SenderName"]!;

            var client = new SendGridClient(apiKey);
            var from = new EmailAddress(senderEmail, senderName);
            var to = new EmailAddress(toEmail);
            var subject = "Resetare parolă SkinIQ";

            var htmlContent = $@"
                <div style='font-family: Georgia, serif; max-width: 500px; margin: 0 auto; padding: 40px 20px;'>
                    <h1 style='font-size: 28px; font-weight: 300; color: #2C2C2A;'>
                        Skin<em style='color: #D4537E;'>IQ</em>
                    </h1>
                    <h2 style='font-size: 20px; font-weight: 300; color: #2C2C2A; margin-top: 32px;'>
                        Resetare parolă
                    </h2>
                    <p style='color: #888780; font-size: 14px; line-height: 1.6; margin-top: 16px;'>
                        Am primit o cerere de resetare a parolei pentru contul tău SkinIQ.
                        Apasă butonul de mai jos pentru a seta o parolă nouă.
                    </p>
                    <a href='{resetLink}' 
                       style='display: inline-block; margin-top: 24px; padding: 12px 32px;
                              background-color: #D4537E; color: white; text-decoration: none;
                              border-radius: 6px; font-size: 13px; letter-spacing: 1px;'>
                        RESETEAZĂ PAROLA
                    </a>
                    <p style='color: #B4B2A9; font-size: 12px; margin-top: 24px;'>
                        Link-ul este valabil 1 oră. Dacă nu ai solicitat resetarea parolei, ignoră acest email.
                    </p>
                    <hr style='border: none; border-top: 1px solid #F4C0D1; margin-top: 32px;' />
                    <p style='color: #B4B2A9; font-size: 11px; margin-top: 16px;'>
                        SkinIQ — evaluare inteligentă de produse skincare
                    </p>
                </div>";

            var msg = MailHelper.CreateSingleEmail(from, to, subject, "", htmlContent);
            var response = await client.SendEmailAsync(msg);

            Console.WriteLine($"SendGrid status: {response.StatusCode}");
        }
    }
}