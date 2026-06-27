from twilio.rest import Client
from src.config import settings

class WhatsAppService:
    def __init__(self):
        self.client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        self.from_number = settings.TWILIO_WHATSAPP_NUMBER

    def send_message(self, to_number: str, message_body: str):
        try:
            message = self.client.messages.create(
                from_=self.from_number,
                body=message_body,
                to=to_number
            )
            return message.sid
        except Exception as e:
            print(f"❌ خطأ أثناء إرسال الرسالة إلى الواتساب: {e}")
            return None