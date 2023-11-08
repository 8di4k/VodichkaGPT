import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, MessageHandler, Filters, CallbackContext

# Replace these with your bot token and YooMoney keys
TELEGRAM_TOKEN = '5994300602:AAGPg3hcSnEEdNhElh98S8mOmB0YGz_ROv0'
YOO_PROVIDER_TOKEN = 'live_5n_gCH6MLN54HMRTQcl97IQ0_shVSljvmaTfbz7OuJw'

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

def start(update: Update, context: CallbackContext):
    update.message.reply_text("Welcome to the bot! Type /subscribe to start a subscription.")

def subscribe(update: Update, context: CallbackContext):
    chat_id = update.message.chat_id
    send_invoice(chat_id, context.bot)

def send_invoice(chat_id, bot):
    price_rub = 29900  # Price in kopecks

    payload = {
        "chat_id": chat_id,
        "title": "Your Subscription",
        "description": "Subscribe to access premium content",
        "payload": "subscription",
        "provider_token": YOO_PROVIDER_TOKEN,
        "start_parameter": "subscription",
        "currency": "RUB",
        "prices": [{"label": "Subscription", "amount": price_rub}],
    }

    bot.send_invoice(**payload)

def successful_payment(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()

    chat_id = update.effective_chat.id
    logger.info(f"Payment from user {chat_id} was successful")
    context.bot.send_message(chat_id, "Thank you for your payment! You now have access to premium content.")

def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # Command and callback handlers
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("subscribe", subscribe))
    dp.add_handler(MessageHandler(Filters.successful_payment, successful_payment))

    # Start the bot
    updater.start_polling()

    # Run the bot until the user presses Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT
    updater.idle()

if __name__ == '__main__':
    main()
