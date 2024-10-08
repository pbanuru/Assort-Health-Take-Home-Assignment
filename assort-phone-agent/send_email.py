import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os


def send_email(recipient_email, subject, body):
    # Email configuration
    sender_email = os.getenv("GMAIL_USER")
    sender_password = os.getenv("GMAIL_PASSWORD")
    smtp_server = "smtp.gmail.com"
    smtp_port = 465  # Changed to 465 for SSL

    # Create the email message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        # Create SMTP_SSL session
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.set_debuglevel(1)  # Keep debug output
            print(f"Attempting to login with email: {sender_email}")
            server.login(sender_email, sender_password)

            # Send email
            server.send_message(message)

        print(f"Email sent successfully to {recipient_email}")
        return True
    except Exception as e:
        print(f"Failed to send email. Error: {str(e)}")
        print(f"Sender email: {sender_email}")
        print(f"Recipient email: {recipient_email}")
        return False


# Example usage:
if __name__ == "__main__":
    recipient = "pbanuru10@gmail.com"
    subject = "Test Email"
    body = "This is a test email sent from Python using Gmail SMTP."
    send_email(recipient, subject, body)
