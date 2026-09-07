# Brevo password-reset delivery

OccupAI sends password-reset mail through Brevo's transactional email API and
falls back to the existing SMTP configuration if Brevo is unavailable.

Never commit a real API key. Revoke any key pasted into chat, screenshots, or
source files and create a replacement in Brevo.

Configure these environment variables locally or in Render:

```text
BREVO_API_KEY=<new Brevo transactional API key>
BREVO_SENDER_EMAIL=<verified sender address in Brevo>
BREVO_SENDER_NAME=OccupAI Support
PASSWORD_RESET_URL_BASE=https://<your-public-backend-host>
```

`BREVO_SENDER_EMAIL` must be a sender or domain verified in the Brevo account.
Restart the backend after changing environment variables. The public reset URL
must use HTTPS so Gmail recipients can safely open it from another device.

Test with an existing non-admin account from the app's **Forgot password**
screen. The API intentionally returns the same message for existing and unknown
addresses, so confirm delivery in Brevo's transactional logs and the recipient's
Spam folder without exposing reset tokens in application logs.
