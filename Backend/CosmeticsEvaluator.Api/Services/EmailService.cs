using MailKit.Net.Smtp;
using MailKit.Security;
using MimeKit;

public async Task SendPasswordResetEmailAsync(string toEmail, string resetLink)
{
    var message = new MimeMessage();
    message.From.Add(new MailboxAddress(
        _config["Email:SenderName"], 
        _config["Email:SenderEmail"]));
    message.To.Add(MailboxAddress.Parse(toEmail));
    message.Subject = "Resetare parolă SkinIQ";
    
    message.Body = new TextPart("html") { Text = $"..." }; // același HTML ca înainte

    using var client = new SmtpClient();
    await client.ConnectAsync(
        _config["Email:SmtpHost"], 
        587, 
        SecureSocketOptions.StartTls);
    await client.AuthenticateAsync(
        _config["Email:SenderEmail"], 
        _config["Email:AppPassword"]);
    await client.SendAsync(message);
    await client.DisconnectAsync(true);
}