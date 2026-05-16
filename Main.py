import asyncio
import io
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import anthropic
from telegram import Update, ChatPermissions
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import BadRequest, Forbidden

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

TRANSLATE_DELETE_SECONDS = 120

FRENCH_WARN_THRESHOLD = 3
FRENCH_MUTE1_THRESHOLD = 5
FRENCH_MUTE2_THRESHOLD = 5
MUTE1_SECONDS = 15 * 60
MUTE2_SECONDS = 60 * 60

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

french_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
mute_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))


ANALYSE_PROMPT = """You are an expert English language editor. Analyse the user's message and respond with a JSON object only — no extra text, no markdown fences.

The JSON must have this exact shape:
{
  "language": "english" | "french" | "other",
  "has_mistakes": true | false,
  "corrected": "<corrected text, or empty string if no mistakes or not English>",
  "explanations": [
    {"en": "<explanation in English>", "fr": "<explication en français>"},
    ...
  ]
}

Rules:
- Detect the primary language of the message.
- If the language is "english":
    - Check for grammar, spelling, punctuation, and style mistakes.
    - If there are mistakes, set has_mistakes=true, fill corrected with the fixed text, and fill explanations with one entry per change.
    - If there are NO mistakes, set has_mistakes=false, leave corrected as "" and explanations as [].
- If the language is "french" or "other":
    - Set has_mistakes=false, corrected="", explanations=[].
- Each explanation entry must have both "en" (English) and "fr" (French) keys.
- Keep the original meaning and tone. Be concise in explanations.
- Output ONLY the JSON object."""

TRANSLATE_PROMPT = """You are a professional translator between English and French.
Detect the language of the given text (English or French), then translate it to the other language.
Respond with a JSON object only — no extra text, no markdown fences.

Shape:
{
  "source_language": "english" | "french" | "other",
  "translation": "<translated text, or empty string if source is neither English nor French>"
}

Output ONLY the JSON object."""


def analyse_message(text: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=ANALYSE_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    raw = response.content[0].text.strip()
    return json.loads(raw)


def translate_message(text: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=TRANSLATE_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    raw = response.content[0].text.strip()
    return json.loads(raw)


def escape_markdown(text: str) -> str:
    special = r"\_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{c}" if c in special else c for c in text)


def format_correction(corrected: str, explanations: list[dict]) -> str:
    lines = [f"✏️ *Corrected:*\n_{escape_markdown(corrected)}_\n"]
    if explanations:
        lines.append("📝 *Changes:*")
        for item in explanations:
            en = item.get("en", "")
            fr = item.get("fr", "")
            lines.append(f"• {escape_markdown(en)}\n  _🇫🇷 {escape_markdown(fr)}_")
    return "\n".join(lines)


def is_bot_mentioned(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    bot_username = context.bot.username
    if not bot_username:
        return False
    entities = update.message.entities or []
    text = update.message.text or ""
    for entity in entities:
        if entity.type == "mention":
            mention = text[entity.offset: entity.offset + entity.length]
            if mention.lstrip("@").lower() == bot_username.lower():
                return True
    return False


def is_group(update: Update) -> bool:
    return update.message.chat.type in ("group", "supergroup")


async def delete_after(chat_id: int, message_id: int, bot, delay: int) -> None:
    await asyncio.sleep(delay)
    try:
        await bot.delete_message(chat_id=chat_id, message_id=message_id)
    except (BadRequest, Exception) as e:
        logger.warning("Could not delete message %s: %s", message_id, e)


async def midnight_reset_loop() -> None:
    while True:
        now = datetime.now(timezone.utc)
        next_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        sleep_seconds = (next_midnight - now).total_seconds()
        await asyncio.sleep(sleep_seconds)
        french_counts.clear()
        mute_counts.clear()
        logger.info("French-message counters reset at midnight UTC")


async def handle_french_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    chat_id = message.chat_id
    user_id = message.from_user.id
    user = message.from_user
    name = user.first_name or "User"

    french_counts[chat_id][user_id] += 1
    count = french_counts[chat_id][user_id]
    already_muted = mute_counts[chat_id][user_id]

    if count == FRENCH_WARN_THRESHOLD:
        mention = f"[{escape_markdown(name)}](tg://user?id={user_id})"
        await message.reply_text(
            f"⚠️ {mention}\n\n"
            f"🇫🇷 Merci d'écrire en anglais dans ce groupe\\. "
            f"C'est la {escape_markdown(str(count))}ème fois aujourd'hui — "
            f"après {escape_markdown(str(FRENCH_MUTE1_THRESHOLD))} messages en français vous serez mis\\(e\\) en sourdine\\.\n\n"
            f"🇬🇧 Please write in English in this group\\. "
            f"This is the {escape_markdown(str(count))}rd time today — "
            f"after {escape_markdown(str(FRENCH_MUTE1_THRESHOLD))} French messages you will be muted\\.",
            parse_mode="MarkdownV2",
        )

    elif count >= FRENCH_MUTE1_THRESHOLD:
        if already_muted == 0:
            mute_seconds = MUTE1_SECONDS
            duration_en = "15 minutes"
            duration_fr = "15 minutes"
        else:
            mute_seconds = MUTE2_SECONDS
            duration_en = "1 hour"
            duration_fr = "1 heure"

        until = datetime.now(timezone.utc) + timedelta(seconds=mute_seconds)

        try:
            await context.bot.restrict_chat_member(
                chat_id=chat_id,
                user_id=user_id,
                permissions=ChatPermissions(can_send_messages=False),
                until_date=until,
            )
            mute_counts[chat_id][user_id] += 1
            french_counts[chat_id][user_id] = 0

            mention = f"[{escape_markdown(name)}](tg://user?id={user_id})"
            await message.reply_text(
                f"🔇 {mention}\n\n"
                f"🇫🇷 Vous avez été mis\\(e\\) en sourdine pendant *{escape_markdown(duration_fr)}* "
                f"pour avoir écrit en français à plusieurs reprises\\.\n\n"
                f"🇬🇧 You have been muted for *{escape_markdown(duration_en)}* "
                f"for repeatedly writing in French\\.",
                parse_mode="MarkdownV2",
            )

        except Forbidden:
            await message.reply_text(
                "⚠️ I need admin rights to mute members\\. "
                "Please promote me to admin with the 'Restrict Members' permission\\.",
                parse_mode="MarkdownV2",
            )
        except BadRequest as e:
            logger.error("Failed to mute user %s: %s", user_id, e)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Hello! I'm your English correction bot.\n\n"
        "Send me any English text and I'll fix grammar, spelling, "
        "punctuation, and style mistakes — with explanations in both "
        "English 🇬🇧 and French 🇫🇷.\n\n"
        "• I only correct English messages.\n"
        "• I stay silent when your text is already correct.\n"
        "• French messages in the group are tracked — write in English!\n"
        "• Reply to any message with /translate to translate it.\n"
        "• Send a voice message and I'll transcribe and correct it.\n\n"
        "Just send a message to get started!"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📖 *How to use this bot:*\n\n"
        "*Text correction*\n"
        "Send any English text — I fix grammar, spelling, punctuation, "
        "and style\\. Each correction is explained in English and French\\.\n"
        "I stay silent if your text is already correct\\.\n\n"
        "*Translation* `/translate`\n"
        "Reply to any message with /translate to get it translated\\.\n"
        "The translation disappears after 2 minutes\\.\n\n"
        "*Voice messages*\n"
        "Send a voice message — I transcribe and correct it if it's in English\\.\n\n"
        "*French language policy* 🇫🇷\n"
        "• 3 French messages → warning\n"
        "• 5 messages → muted 15 minutes\n"
        "• 5 more → muted 1 hour\n"
        "Counters reset at midnight UTC\\.",
        parse_mode="MarkdownV2",
    )


async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message:
        return

    replied = message.reply_to_message
    if not replied:
        await message.reply_text(
            "↩️ Please *reply to a message* with /translate to translate it\\.",
            parse_mode="MarkdownV2",
        )
        return

    source_text = replied.text or replied.caption
    if not source_text:
        await message.reply_text("⚠️ I can only translate text messages.")
        return

    await message.chat.send_action("typing")

    try:
        result = translate_message(source_text)
    except Exception as e:
        logger.error("Translation error: %s", e)
        await message.reply_text("⚠️ Translation failed. Please try again.")
        return

    source_lang = result.get("source_language", "other")
    translation = result.get("translation", "").strip()

    if source_lang == "other" or not translation:
        await message.reply_text("⚠️ I can only translate between English and French.")
        return

    flag_from = "🇬🇧" if source_lang == "english" else "🇫🇷"
    flag_to = "🇫🇷" if source_lang == "english" else "🇬🇧"
    lang_to = "French" if source_lang == "english" else "English"

    reply_text = (
        f"{flag_from} → {flag_to} *{lang_to} translation:*\n\n"
        f"{escape_markdown(translation)}\n\n"
        f"_🕐 This message will be deleted in 2 minutes\\._"
    )

    sent = await message.reply_text(reply_text, parse_mode="MarkdownV2")
    asyncio.create_task(
        delete_after(sent.chat_id, sent.message_id, context.bot, TRANSLATE_DELETE_SECONDS)
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message or not message.text:
        return

    user_text = message.text.strip()
    if not user_text:
        return

    mentioned = is_bot_mentioned(update, context)

    await message.chat.send_action("typing")

    try:
        result = analyse_message(user_text)
    except Exception as e:
        logger.error("Error calling Claude or parsing response: %s", e)
        return

    language = result.get("language", "other")
    has_mistakes = result.get("has_mistakes", False)
    corrected = result.get("corrected", "")
    explanations = result.get("explanations", [])

    if language == "english":
        if not has_mistakes:
            return
        await message.reply_text(
            format_correction(corrected, explanations), parse_mode="MarkdownV2"
        )

    elif language == "french":
        if mentioned:
            await message.reply_text(
                "👋 Bonjour ! Je suis un correcteur d'anglais.\n"
                "Hello! I only correct English text. Send me a message in English "
                "and I'll fix any mistakes — with explanations in both English 🇬🇧 and French 🇫🇷!"
            )
        elif is_group(update):
            await handle_french_message(update, context)

    else:
        if mentioned:
            await message.reply_text(
                "👋 I only correct English text. Send me a message in English and I'll help!"
            )


async def post_init(app: Application) -> None:
    asyncio.create_task(midnight_reset_loop())
    logger.info("Midnight reset loop started")


def main() -> None:
    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("translate", translate_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    logger.info("Bot is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message or not message.voice:
        return

    await message.chat.send_action("typing")

    try:
        voice_file = await message.voice.get_file()
        audio_bytes = bytes(await voice_file.download_as_bytearray())
    except Exception as e:
        logger.error("Failed to download voice message: %s", e)
        return

    # For voice, we use Claude to transcribe via base64
    import base64
    audio_b64 = base64.standard_b64encode(audio_bytes).decode("utf-8")

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Transcribe this voice message, then analyse it for English mistakes. Respond with JSON only: {\"transcript\": \"...\", \"language\": \"english|french|other\", \"has_mistakes\": true|false, \"corrected\": \"...\", \"explanations\": [{\"en\": \"...\", \"fr\": \"...\"}]}"
                    }
                ]
            }]
        )
        result = json.loads(response.content[0].text.strip())
    except Exception as e:
        logger.error("Voice analysis error: %s", e)
        await message.reply_text("⚠️ I couldn't process that voice message. Please try again.")
        return

    language = result.get("language", "other")
    has_mistakes = result.get("has_mistakes", False)
    transcript = result.get("transcript", "")
    corrected = result.get("corrected", "")
    explanations = result.get("explanations", [])

    if language != "english" or not has_mistakes:
        return

    reply = (
        f"🎙️ *Transcription:*\n_{escape_markdown(transcript)}_\n\n"
        + format_correction(corrected, explanations)
    )
    await message.reply_text(reply, parse_mode="MarkdownV2")


if __name__ == "__main__":
    main()
