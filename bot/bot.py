import io
import logging
import asyncio
import traceback
import html
import json
import tempfile
import requests
import openai
import base64

from pathlib import Path
from datetime import datetime, timedelta
from pytesseract import image_to_string
from telegram.constants import ParseMode
from openai.error import InvalidRequestError
import telegram
from telegram import (
    Update,
    User,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
    LabeledPrice
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    AIORateLimiter,
    filters,
    ConversationHandler,
    PreCheckoutQueryHandler
)
from telegram.constants import ParseMode, ChatAction

import config
import database
import openai_utils


# setup
db = database.Database()
logger = logging.getLogger(__name__)

user_semaphores = {}
user_tasks = {}

HELP_MESSAGE = """Commands:
🔄 /retry – Regenerate last bot answer
🆕 /new – Start new dialog
🔀 /mode – Select chat mod
*️⃣ /settings - Show available models\n
🔔 /subscribe – Renew subscription
⏳ /balance – Show remaining days
❓ /help – Show help

🎨 Generate images from text in:\n <b>Artist</b> /mode
👥 Add bot to <b>group chat</b>:\n /help_group_chat
"""

HELP_GROUP_CHAT_MESSAGE = """
✨ <b>Add this bot to a Group Chat!</b> ✨

Ready to take your group chat from 'meh' to 'wow'?

Just do this:

🚀 <b>Step 1: Invite Our Bot</b>
- Add our friendly bot to your group chat.

🔧 <b>Step 2: Make it an Admin</b>
- Don't worry, it only needs to see the messages.

🎉 <b>Step 3: Watch the Fun Unfold!</b>

🤖 Wanna chat with the bot? 
Tag <b>{bot_username}</b> in a message or hit 'reply' on one of its messages.

📝 Looking for a demo? Try:
"{bot_username} write a poem about Telegram"
And watch the creativity flow!
"""

CHANNEL_USERNAME = 'VodichkaGPT'

def ocr_image(img_path, lang='eng+rus'):
    img = cv2.imread(img_path, 0)
    text = image_to_string(img, lang=lang)
    return text

def split_text_into_chunks(text, chunk_size):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

async def subscribe_handle(update: Update, context: CallbackContext):
    # Send an invoice to the user for the subscription
    title = "Monthly Subscription"
    description = "Your subscription to VodichkaGPT"
    payload = "Custom-Payload"
    provider_token = config.YOOKASSA_PROVIDER_TOKEN  # Add this to your config file
    currency = "RUB"
    # The price must be in the smallest currency unit (e.g., cents for USD)
    price = 34900  # For example, 100.00 RUB should be written as 10000
    prices = [LabeledPrice("Monthly Subscription", price)]

    await context.bot.send_invoice(
        chat_id=update.effective_chat.id,
        title=title,
        description=description,
        payload=payload,
        provider_token=provider_token,
        currency=currency,
        prices=prices
    )

async def precheckout_callback(update: Update, context: CallbackContext):
    query = update.pre_checkout_query
    if query.invoice_payload != 'Custom-Payload':
        # Answer pre-checkout query with error if the payload doesn't match
        await context.bot.answer_pre_checkout_query(pre_checkout_query_id=query.id, ok=False,
                                                    error_message="Something went wrong...")
    else:
        # Answer pre-checkout query with success
        await context.bot.answer_pre_checkout_query(pre_checkout_query_id=query.id, ok=True)


async def successful_payment_callback(update: Update, context: CallbackContext):
    # Payment is successful, so let's cheerfully update the user's subscription
    user_id = update.message.chat.id
    # Add 30 delightful days to the user's subscription as a sign of our gratitude
    end_date = datetime.now() + timedelta(days=30)
    db.update_user_subscription(user_id, end_date, is_trial=False)  # Our database gets the good news too

    # Craft a warm and welcoming confirmation message
    confirmation_message = (
        "🌟Hooray! We received your payment!🌟\n\n"
        "We're absolutely thrilled to have you continue this journey with us. \n"
        "Your subscription has been extended! \n"
        "You can check the remaining days at any time by clicking /balance \n\n"
        "Thank you for your support, and enjoy your enhanced experience with us! 💫"
    )

    # Log right before sending the message
    logger.info(f"Sending confirmation message: {confirmation_message}")

    # Send the crafted message to the user's chat
    await update.message.reply_text(confirmation_message, parse_mode=ParseMode.HTML)




async def register_user_if_not_exists(update: Update, context: CallbackContext, user: User):
    if not db.check_if_user_exists(user.id):
        db.add_new_user(
            user.id,
            update.message.chat_id,
            username=user.username,
            first_name=user.first_name,
            last_name= user.last_name
        )
        db.start_new_dialog(user.id)

    if db.get_user_attribute(user.id, "current_dialog_id") is None:
        db.start_new_dialog(user.id)

    if user.id not in user_semaphores:
        user_semaphores[user.id] = asyncio.Semaphore(1)

    if db.get_user_attribute(user.id, "current_model") is None:
        db.set_user_attribute(user.id, "current_model", config.models["available_text_models"][0])

    # back compatibility for n_used_tokens field
    n_used_tokens = db.get_user_attribute(user.id, "n_used_tokens")
    if isinstance(n_used_tokens, int) or isinstance(n_used_tokens, float):  # old format
        new_n_used_tokens = {
            "gpt-4-mini": {
                "n_input_tokens": 0,
                "n_output_tokens": n_used_tokens
            }
        }
        db.set_user_attribute(user.id, "n_used_tokens", new_n_used_tokens)

    # voice message transcription
    if db.get_user_attribute(user.id, "n_transcribed_seconds") is None:
        db.set_user_attribute(user.id, "n_transcribed_seconds", 0.0)

    # image generation
    if db.get_user_attribute(user.id, "n_generated_images") is None:
        db.set_user_attribute(user.id, "n_generated_images", 0)

    # Update subscription for a new user
    if db.get_user_subscription(user.id) is None:
        trial_end_date = datetime.now() + timedelta(days=1)
        db.update_user_subscription(user.id, trial_end_date, is_trial=True)


async def is_bot_mentioned(update: Update, context: CallbackContext):
     try:
         message = update.message

         if message.chat.type == "private":
             return True

         if message.text is not None and ("@" + context.bot.username) in message.text:
             return True

         if message.reply_to_message is not None:
             if message.reply_to_message.from_user.id == context.bot.id:
                 return True
     except:
         return True
     else:
         return False


async def start_handle(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id

    # Check if the user is new or returning
    is_new_user = not db.check_if_user_exists(user_id)

    # Register user and start their free trial if they're new
    await register_user_if_not_exists(update, context, update.message.from_user)

    # Check if the user is already a subscriber to the channel
    try:
        status = await context.bot.get_chat_member(chat_id=f"@{CHANNEL_USERNAME}", user_id=user_id)
        is_subscriber = status.status in ["member", "creator", "administrator"]
    except Exception as e:
        logging.error("Could not fetch the chat member status.", exc_info=e)
        is_subscriber = False

    if not is_subscriber:
        message = (
            "🌟 Welcome to VodichkaGPT! 🌟\n\n"
            "Please join our channel first. "
            "It's just a click away!\n\n"
            "👉 [Join Our Channel](https://t.me/VodichkaGPT)\n\n"
            "After joining, simply hit /start to begin exploring all the amazing features.\n\n"
        )
        if is_new_user:
            message += "🎉 You have *one-day free trial*! 🎉"
        keyboard = [[InlineKeyboardButton("🔗 Join Channel 🔗", url="https://t.me/VodichkaGPT")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
        return

    # Greeting new or returning user
    greeting_text = "Hi! I'm <b>VodichkaGPT!</b>\nBot powered by GPT 4o.\n\n"
    greeting_text += HELP_MESSAGE
    greeting_text += "\nFeel free to ask me anything!"

    
    await update.message.reply_text(greeting_text, parse_mode=ParseMode.HTML)
    await show_chat_modes_handle(update, context)



async def help_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.HTML)


async def help_group_chat_handle(update: Update, context: CallbackContext):
     await register_user_if_not_exists(update, context, update.message.from_user)
     user_id = update.message.from_user.id
     db.set_user_attribute(user_id, "last_interaction", datetime.now())

     text = HELP_GROUP_CHAT_MESSAGE.format(bot_username="@" + context.bot.username)

     await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def retry_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
    if len(dialog_messages) == 0:
        await update.message.reply_text("No message to retry 🤷‍♂️")
        return

    last_dialog_message = dialog_messages.pop()
    db.set_dialog_messages(user_id, dialog_messages, dialog_id=None)  # last message was removed from the context

    await message_handle(update, context, message=last_dialog_message["user"], use_new_dialog_timeout=False)

async def check_user_subscription(user_id, bot):
    try:
        status = await bot.get_chat_member(chat_id=f"@{CHANNEL_USERNAME}", user_id=user_id)
        return status.status in ["member", "creator", "administrator"]
    except Exception as e:
        logger.error("Error checking user subscription status:", exc_info=e)
        return False

async def enforce_subscription(update, context):
    user_id = update.effective_user.id
    is_subscribed = await check_user_subscription(user_id, context.bot)
    if not is_subscribed:
        # User is not subscribed, send the subscription message.
        message = (
            "To use this bot, please join our channel first.\n\n"
            f"[Join @{CHANNEL_USERNAME}](https://t.me/{CHANNEL_USERNAME})"
        )
        keyboard = [[InlineKeyboardButton("Join Channel", url=f"https://t.me/{CHANNEL_USERNAME}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True
        )
        return False
    return True

async def message_handle(update: Update, context: CallbackContext, message=None, use_new_dialog_timeout=True):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    # check if message is edited
    if update.edited_message is not None:
        await edited_message_handle(update, context)
        return

    _message = message or update.message.text

    # remove bot mention (in group chats)
    if update.message.chat.type != "private":
        _message = _message.replace("@" + context.bot.username, "").strip()

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context):
        return

    user_id = update.message.from_user.id

    # Check if the user has an active subscription
    subscription = db.get_user_subscription(user_id)
    if subscription is None or subscription['end_date'] < datetime.now():
        subscription_message = (
            "🚫 <b>Access Denied:</b> It looks like you don't have an active subscription.\n\n"
            "💡 To continue enjoying our services, please subscribe.\n"
            "🔗 Just tap on /subscribe to get started!"
        )
        await update.message.reply_text(subscription_message, parse_mode=ParseMode.HTML)
        return

    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")

    if chat_mode == "artist":
        await generate_image_handle(update, context, message=message)
        return

    async def message_handle_fn():
        # new dialog timeout
        if use_new_dialog_timeout:
            if (datetime.now() - db.get_user_attribute(user_id, "last_interaction")).seconds > config.new_dialog_timeout and len(db.get_dialog_messages(user_id)) > 0:
                db.start_new_dialog(user_id)
                await update.message.reply_text(f"Starting new dialog due to timeout (<b>{config.chat_modes[chat_mode]['name']}</b> mode) ✅", parse_mode=ParseMode.HTML)
        db.set_user_attribute(user_id, "last_interaction", datetime.now())

        # in case of CancelledError
        n_input_tokens, n_output_tokens = 0, 0
        current_model = db.get_user_attribute(user_id, "current_model")

        # Initialize n_first_dialog_messages_removed
        n_first_dialog_messages_removed = 0

        try:
            # send placeholder message to user
            placeholder_message = await update.message.reply_text("...")

            # send typing action
            await update.message.chat.send_action(action="typing")

            if _message is None or len(_message) == 0:
                await update.message.reply_text("🥲 You sent <b>empty message</b>. Please, try again!", parse_mode=ParseMode.HTML)
                return

            dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
            parse_mode = {
                "html": ParseMode.HTML,
                "markdown": ParseMode.MARKDOWN
            }[config.chat_modes[chat_mode]["parse_mode"]]

            chatgpt_instance = openai_utils.ChatGPT(model=current_model)
            if config.enable_message_streaming:
                gen = chatgpt_instance.send_message_stream(_message, dialog_messages=dialog_messages, chat_mode=chat_mode)
            else:
                response = await chatgpt_instance.send_message(
                    _message,
                    dialog_messages=dialog_messages,
                    chat_mode=chat_mode
                )

                # Ensure response is in the correct format
                if isinstance(response, tuple) and len(response) == 3 and isinstance(response[1], tuple) and len(response[1]) == 2:
                    answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = response
                else:
                    raise ValueError(f"Unexpected response format: {response}")

                gen = async_generator(answer, n_input_tokens, n_output_tokens, n_first_dialog_messages_removed)

            prev_answer = ""
            async for gen_item in gen:
                status, answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = gen_item

                answer = answer[:4096]  # telegram message limit

                # update only when 100 new symbols are ready
                if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
                    continue

                try:
                    await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id, parse_mode=parse_mode)
                except telegram.error.BadRequest as e:
                    if str(e).startswith("Message is not modified"):
                        continue
                    else:
                        await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)

                await asyncio.sleep(0.01)  # wait a bit to avoid flooding

                prev_answer = answer

            # update user data
            new_dialog_message = {"user": _message, "bot": answer, "date": datetime.now()}
            db.set_dialog_messages(
                user_id,
                db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
                dialog_id=None
            )

        except ValueError as ve:
            logger.error(f"ValueError: {ve}")
            await update.message.reply_text("Error: Unexpected response format. Please try again.")

        except asyncio.CancelledError:
            raise

        except Exception as e:
            error_text = f"Something went wrong during completion. Reason: {e}"
            logger.error(error_text)
            await update.message.reply_text(error_text)

        # send message if some messages were removed from the context
        if n_first_dialog_messages_removed > 0:
            if n_first_dialog_messages_removed == 1:
                text = "✍️ <i>Note:</i> Your current dialog is too long, so your <b>first message</b> was removed from the context.\n Send /new command to start new dialog"
            else:
                text = f"✍️ <i>Note:</i> Your current dialog is too long, so <b>{n_first_dialog_messages_removed} first messages</b> were removed from the context.\n Send /new command to start new dialog"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async with user_semaphores[user_id]:
        task = asyncio.create_task(message_handle_fn())
        user_tasks[user_id] = task

        try:
            await task
        except asyncio.CancelledError:
            await update.message.reply_text("✅ Canceled", parse_mode=ParseMode.HTML)
        else:
            pass
        finally:
            if user_id in user_tasks:
                del user_tasks[user_id]



async def is_previous_message_not_answered_yet(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    if user_semaphores[user_id].locked():
        text = "⏳ Please <b>wait</b> for a reply to the previous message\n"
        text += "Or you can /cancel it"
        await update.message.reply_text(text, reply_to_message_id=update.message.id, parse_mode=ParseMode.HTML)
        return True
    else:
        return False

async def unsupport_message_handle(update: Update, context: CallbackContext, message=None):
    error_text = f"I don't know how to read files or videos. Send the picture in normal mode (Quick Mode)."
    logger.error(error_text)
    await update.message.reply_text(error_text)
    return
    
async def image_message_handle(update: Update, context: CallbackContext):
    # Register user if not already registered
    await register_user_if_not_exists(update, context, update.message.from_user)
    
    # Check if the previous message is still being processed
    if await is_previous_message_not_answered_yet(update, context):
        return

    user_id = update.message.from_user.id
    current_model = db.get_user_attribute(user_id, "current_model")  # Ensure this contains the correct model
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")

    # If the user is in "Artist" chat mode, notify them to switch and return
    if chat_mode == "artist":
        await update.message.reply_text("Switch to VodichkaGPT for image recognition", reply_to_message_id=update.message.message_id)
        return

    # Handle image messages and store the image temporarily
    photo = update.message.photo[-1]  # Get the highest resolution photo
    image_file = await context.bot.get_file(photo.file_id)

    # Extract caption if provided
    image_caption = update.message.caption if update.message.caption else "Describe the contents of this image clearly and concisely. Focus on identifying key objects, text, or visual elements. If there is any math task or a problem solve it providing correct answer shoving few breef steps in plain text. Always double check before answering! Remember to NEVER use special symbols like LaTeX-style math notation. Remember to never use Do not use any bold text, headings, or formatting. Instead, use words like 'percent', 'multiplied by', and 'divided by'. Or %, ×, ÷. Make your answer as short as u can, whole answer must always fit one telegram message containing not more than 37 lines and 3400 characters. Always make it easily readable. Never brake these rules!"


    # Use a temporary directory to store the image
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        image_path = tmp_dir / "image.jpg"
        
        # Download the image to the temporary directory
        await image_file.download_to_drive(image_path)

        # Function to encode the image to base64
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # Encode the image to base64
        base64_image = encode_image(image_path)

        # Prepare headers and payload for OpenAI API, using the API key from config directly
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.openai_api_key}"  # Refer to the API key in config directly
        }

        # Prepare the payload to include the caption as part of the prompt
        payload = {
            "model": current_model,  # Use the selected model (GPT-4-Mini or GPT-4o)
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": image_caption  # Use the image caption as part of the prompt
                        },
                        {
                            "type": "image_url",  # Use 'image_url' instead of 'image'
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        # Make the request to OpenAI API
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        # Check if the response is successful
        if response.status_code == 200:
            description = response.json()["choices"][0]["message"]["content"]
            await update.message.reply_text(f"🖼️: {description}", parse_mode=ParseMode.HTML, reply_to_message_id=update.message.message_id)
        else:
            await update.message.reply_text(f"Error processing the image: {response.status_code}. {response.json()}", parse_mode=ParseMode.HTML, reply_to_message_id=update.message.message_id)


async def voice_message_handle(update: Update, context: CallbackContext):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    voice = update.message.voice
    voice_file = await context.bot.get_file(voice.file_id)
    
    # store file in memory, not on disk
    buf = io.BytesIO()
    await voice_file.download_to_memory(buf)
    buf.name = "voice.oga"  # file extension is required
    buf.seek(0)  # move cursor to the beginning of the buffer

    transcribed_text = await openai_utils.transcribe_audio(buf)
    text = f"🎤: <i>{transcribed_text}</i>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    # update n_transcribed_seconds
    db.set_user_attribute(user_id, "n_transcribed_seconds", voice.duration + db.get_user_attribute(user_id, "n_transcribed_seconds"))

    await message_handle(update, context, message=transcribed_text)


async def generate_image_handle(update: Update, context: CallbackContext, message=None):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    # Check if the user is on trial
    subscription = db.get_user_subscription(user_id)
    is_on_trial = subscription.get("is_trial", False)

    if is_on_trial:
        # If on trial, calculate remaining image generation limit
        trial_image_limit = 3
        current_images_generated = db.get_user_attribute(user_id, "n_generated_images")
        remaining_limit = trial_image_limit - current_images_generated

        if remaining_limit <= 0:
            # User has reached the limit of images that can be generated during the trial period
            text = "You've reached your image generation limit of <b>3 images</b> during the trial.\nTo continue creating, please consider our subscription options."
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return
        else:
            # Limit the number of images to the remaining trial limit
            config.return_n_generated_images = min(config.return_n_generated_images, remaining_limit)
    else:
        # Non-trial users have no limit, so use the configured default
        config.return_n_generated_images = config.return_n_generated_images

    # token usage
    db.set_user_attribute(user_id, "n_generated_images", config.return_n_generated_images + db.get_user_attribute(user_id, "n_generated_images"))

    await update.message.chat.send_action(action="upload_photo")
    message = message or update.message.text

    try:
        # Attempt to generate images with the adjusted limit for trial users, or no limit for non-trial users
        image_urls = await openai_utils.generate_images(message, n_images=config.return_n_generated_images, size=config.image_size)
    except openai.error.InvalidRequestError as e:
        if str(e).startswith("Your request was rejected as a result of our safety system"):
            text = "🥲 Your request <b>doesn't comply</b> with our usage policies.\nPlease adjust your request and try again."
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return
        else:
            raise

    # Update the number of generated images in the database for trial users
    if is_on_trial:
        db.set_user_attribute(user_id, "n_generated_images", current_images_generated + len(image_urls))

    # Send the images to the user
    for image_url in image_urls:
        await update.message.chat.send_action(action="upload_photo")
        await update.message.reply_photo(image_url, parse_mode=ParseMode.HTML)


async def new_dialog_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    db.start_new_dialog(user_id)
    await update.message.reply_text("Starting new dialog ✅")

    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    await update.message.reply_text(f"{config.chat_modes[chat_mode]['welcome_message']}", parse_mode=ParseMode.HTML)


async def cancel_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    if user_id in user_tasks:
        task = user_tasks[user_id]
        task.cancel()
    else:
        await update.message.reply_text("<i>Nothing to cancel...</i>", parse_mode=ParseMode.HTML)


def get_chat_mode_menu(page_index: int):
    n_chat_modes_per_page = config.n_chat_modes_per_page
    text = f"Select <b>chat mode</b> ({len(config.chat_modes)} modes available):"

    # buttons
    chat_mode_keys = list(config.chat_modes.keys())
    page_chat_mode_keys = chat_mode_keys[page_index * n_chat_modes_per_page:(page_index + 1) * n_chat_modes_per_page]

    keyboard = []
    for chat_mode_key in page_chat_mode_keys:
        name = config.chat_modes[chat_mode_key]["name"]
        keyboard.append([InlineKeyboardButton(name, callback_data=f"set_chat_mode|{chat_mode_key}")])

    # pagination
    if len(chat_mode_keys) > n_chat_modes_per_page:
        is_first_page = (page_index == 0)
        is_last_page = ((page_index + 1) * n_chat_modes_per_page >= len(chat_mode_keys))

        if is_first_page:
            keyboard.append([
                InlineKeyboardButton("»", callback_data=f"show_chat_modes|{page_index + 1}")
            ])
        elif is_last_page:
            keyboard.append([
                InlineKeyboardButton("«", callback_data=f"show_chat_modes|{page_index - 1}"),
            ])
        else:
            keyboard.append([
                InlineKeyboardButton("«", callback_data=f"show_chat_modes|{page_index - 1}"),
                InlineKeyboardButton("»", callback_data=f"show_chat_modes|{page_index + 1}")
            ])

    reply_markup = InlineKeyboardMarkup(keyboard)

    return text, reply_markup


async def show_chat_modes_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    text, reply_markup = get_chat_mode_menu(0)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def show_chat_modes_callback_handle(update: Update, context: CallbackContext):
     await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
     if await is_previous_message_not_answered_yet(update.callback_query, context): return

     user_id = update.callback_query.from_user.id
     db.set_user_attribute(user_id, "last_interaction", datetime.now())

     query = update.callback_query
     await query.answer()

     page_index = int(query.data.split("|")[1])
     if page_index < 0:
         return

     text, reply_markup = get_chat_mode_menu(page_index)
     try:
         await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
     except telegram.error.BadRequest as e:
         if str(e).startswith("Message is not modified"):
             pass


async def set_chat_mode_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    chat_mode = query.data.split("|")[1]

    db.set_user_attribute(user_id, "current_chat_mode", chat_mode)
    db.start_new_dialog(user_id)

    await context.bot.send_message(
        update.callback_query.message.chat.id,
        f"{config.chat_modes[chat_mode]['welcome_message']}",
        parse_mode=ParseMode.HTML
    )


def get_settings_menu(user_id: int):
    current_model = db.get_user_attribute(user_id, "current_model")

    # Fallback to a default model if the current_model is not available in the config
    if current_model not in config.models["info"]:
        current_model = "default_model_key"  # Replace with the key of your default model

    text = config.models["info"][current_model]["description"]

    text += "\n\n"
    score_dict = config.models["info"][current_model]["scores"]
    for score_key, score_value in score_dict.items():
        text += "🟣" * score_value + "⚪" * (5 - score_value) + f" – {score_key}\n\n"

    text += "\nSelect <b>model</b>:"

    # buttons to choose models
    buttons = []
    for model_key in config.models["available_text_models"]:
        title = config.models["info"][model_key]["name"]
        if model_key == current_model:
            title = "✅ " + title

        buttons.append(
            InlineKeyboardButton(title, callback_data=f"set_settings|{model_key}")
        )
    reply_markup = InlineKeyboardMarkup([buttons])

    return text, reply_markup

async def settings_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    text, reply_markup = get_settings_menu(user_id)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def set_settings_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    _, model_key = query.data.split("|")
    db.set_user_attribute(user_id, "current_model", model_key)
    db.start_new_dialog(user_id)

    text, reply_markup = get_settings_menu(user_id)
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass


async def show_subscription_status(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    try:
        subscription_info = db.get_user_attribute(user_id, 'subscription')

        if not subscription_info:
            text = "🔔 <b>Subscription Status:</b> None 🔘\nℹ️ You do not have an active subscription."
        else:
            end_date = subscription_info.get('end_date')
            is_trial = subscription_info.get('is_trial', False)

            # Convert end_date to a datetime object if it's not already one
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%S.%fZ')

            now = datetime.now()
            days_left = (end_date - now).days

            if days_left >= 0 and is_trial:
                
                # Special handling for trial subscriptions on their last day
                text = "🔔 <b>Subscription Status:</b> Trial 🟢\n📅 <b>Days Remaining:</b> 1"
            elif days_left > 0:
                text = f"🔔 <b>Subscription Status:</b> Active 🟢\n📅 <b>Days Remaining:</b> {days_left}"
            else:
                text = "🔔 <b>Subscription Status:</b> Expired 🔴\n⚠️ Please renew your subscription."
    except Exception as e:
        text = f"🔔 Error while retrieving subscription status: {str(e)}"

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def edited_message_handle(update: Update, context: CallbackContext):
    if update.edited_message.chat.type == "private":
        text = "🥲 Unfortunately, message <b>editing</b> is not supported"
        await update.edited_message.reply_text(text, parse_mode=ParseMode.HTML)


async def error_handle(update: Update, context: CallbackContext) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    try:
        # collect error message
        tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
        tb_string = "".join(tb_list)
        update_str = update.to_dict() if isinstance(update, Update) else str(update)
        message = (
            f"An exception was raised while handling an update\n"
            f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
            "</pre>\n\n"
            f"<pre>{html.escape(tb_string)}</pre>"
        )

        # split text into multiple messages due to 4096 character limit
        for message_chunk in split_text_into_chunks(message, 4096):
            try:
                await context.bot.send_message(update.effective_chat.id, message_chunk, parse_mode=ParseMode.HTML)
            except telegram.error.BadRequest:
                # answer has invalid characters, so we send it without parse_mode
                await context.bot.send_message(update.effective_chat.id, message_chunk)
    except:
        await context.bot.send_message(update.effective_chat.id, "Some error in error handler")

async def post_init(application: Application):
    await application.bot.set_my_commands([
        BotCommand("/new", "Start new dialog"),
        BotCommand("/mode", "Select chat mode"),
	BotCommand("/settings", "Show available models"),
        BotCommand("/retry", "Re-generate response for previous query"),
        BotCommand("/subscribe", "Renew subscription"),
        BotCommand("/balance", "Show remaining days"),
        BotCommand("/help", "Show help message"),
    ])

def run_bot() -> None:
    application = (
        ApplicationBuilder()
        .token(config.telegram_token)
        .concurrent_updates(True)
        .rate_limiter(AIORateLimiter(max_retries=5))
        .http_version("1.1")
        .get_updates_http_version("1.1")
        .post_init(post_init)
        .build()
    )

    # add handlers
    user_filter = filters.ALL
    if len(config.allowed_telegram_usernames) > 0:
        usernames = [x for x in config.allowed_telegram_usernames if isinstance(x, str)]
        any_ids = [x for x in config.allowed_telegram_usernames if isinstance(x, int)]
        user_ids = [x for x in any_ids if x > 0]
        group_ids = [x for x in any_ids if x < 0]
        user_filter = filters.User(username=usernames) | filters.User(user_id=user_ids) | filters.Chat(chat_id=group_ids)

    application.add_handler(CommandHandler("start", start_handle, filters=user_filter))
    application.add_handler(CommandHandler("help", help_handle, filters=user_filter))
    application.add_handler(CommandHandler("help_group_chat", help_group_chat_handle, filters=user_filter))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, message_handle))
    application.add_handler(CommandHandler("retry", retry_handle, filters=user_filter))
    application.add_handler(CommandHandler("new", new_dialog_handle, filters=user_filter))
    application.add_handler(CommandHandler("cancel", cancel_handle, filters=user_filter))

    application.add_handler(MessageHandler(filters.VOICE & user_filter, voice_message_handle))
    # application.add_handler(MessageHandler(filters.PHOTO & user_filter, image_message_handle))

    application.add_handler(CommandHandler("mode", show_chat_modes_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(show_chat_modes_callback_handle, pattern="^show_chat_modes"))
    application.add_handler(CallbackQueryHandler(set_chat_mode_handle, pattern="^set_chat_mode"))

    application.add_handler(CommandHandler("settings", settings_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(set_settings_handle, pattern="^set_settings"))

    application.add_handler(CommandHandler("balance", show_subscription_status, filters=user_filter))
    application.add_handler(CommandHandler("subscribe", subscribe_handle, filters=user_filter))
    application.add_handler(PreCheckoutQueryHandler(precheckout_callback))
    application.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT & user_filter, successful_payment_callback))
    application.add_handler(MessageHandler(filters.PHOTO & user_filter, image_message_handle))
    application.add_error_handler(error_handle)

    # start the bot
    application.run_polling()


if __name__ == "__main__":
    run_bot()
