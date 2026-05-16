import asyncio
import io
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from openai import OpenAI
from telegram import Update, ChatPermissions
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import BadRequest, Forbidden

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_BASE_URL = os.environ["AI_INTEGRATIONS_OPENAI_BASE_URL"]
OPENAI_API_KEY = os.environ["AI_INTEGRATIONS_OPENAI_API_KEY"]

TRANSLATE_DELETE_SECONDS = 120

FRENCH_WARN_THRESHOLD = 3
FRENCH_MUTE1_THRESHOLD = 5   # first mute: 15 minutes
FRENCH_MUTE2_THRESHOLD = 5   # second mute (after reset): 1 hour
MUTE1_SECONDS = 15 * 60
MUTE2_SECONDS = 60 * 60

client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)

# --- In-memory French-message counters (reset at midnight) ---
# french_counts[chat_id][user_id] = int
# mute_counts[chat_id][user_id]   = int (0 = never muted today, 1 = muted once today)
french_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
mute_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))


# ── Prompts ──────────────────────────────────────────────────────────────────

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


# ── AI helpers ────────────────────────────────────────────────────────────────

def analyse_message(text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-5-mini",
        max_completion_tokens=1024,
        messages=[
            {"role": "system", "content": ANALYSE_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


def translate_message(text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-5-mini",
        max_completion_tokens=1024,
        messages=[
            {"role": "system", "content": TRANSLATE_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


def transcribe_audio(audio_bytes: bytes, filename: str = "voice.ogg") -> str:
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename
    result = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_file,
        response_format="json",
    )
    return result.text.strip()


# ── Formatting helpers ────────────────────────────────────────────────────────

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


# ── Utility ───────────────────────────────────────────────────────────────────

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
    """Reset all French-message counters every day at midnight UTC."""
    while True:
        now = datetime.now(timezone.utc)
        next_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        sleep_seconds = (next_midnight - now).total_seconds()
        logger.info("Next counter reset in %.0f seconds (at midnight UTC)", sleep_seconds)
        await asyncio.sleep(sleep_seconds)
        french_counts.clear()
        mute_counts.clear()
        logger.info("French-message counters reset at midnight UTC")


# ── French warning / mute logic ───────────────────────────────────────────────

async def handle_french_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Called when a French message is detected in a group and the bot is not mentioned."""
    message = update.message
    chat_id = message.chat_id
    user_id = message.from_user.id
    user = message.from_user
    name = user.first_name or "User"

    french_counts[chat_id][user_id] += 1
    count = french_counts[chat_id][user_id]
    already_muted = mute_counts[chat_id][user_id]

    logger.info(
        "French message from %s (id=%s) in chat %s — count=%s, muted_today=%s",
        name, user_id, chat_id, count, already_muted,
    )

    if count == FRENCH_WARN_THRESHOLD:
        mention = f"[{escape_markdown(name)}](tg://user?id={user_id})"
        await message.reply_text(
            f"⚠️ {mention}\n\n"
            f"🇫🇷 Merci d'écrire en anglais dans ce groupe\\. "
            f"C'est la {escape_markdown(str(count))}ème fois aujourd'hui — "
            f"après {escape_markdown(str(FRENCH_MUTE1_THRESHOLD))} messages en français, "
            f"vous serez temporairement mis\\(e\\) en sourdine\\.\n\n"
            f"🇬🇧 Please write in English in this group\\. "
            f"This is the {escape_markdown(str(count))}rd time today — "
            f"after {escape_markdown(str(FRENCH_MUTE1_THRESHOLD))} French messages you will be muted temporarily\\.",
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
                f"pour avoir écrit en français à plusieurs reprises\\. "
                f"Veuillez utiliser l'anglais dans ce groupe\\.\n\n"
                f"🇬🇧 You have been muted for *{escape_markdown(duration_en)}* "
                f"for repeatedly writing in French\\. "
                f"Please use English in this group\\.",
                parse_mode="MarkdownV2",
            )

        except Forbidden:
            logger.warning(
                "Cannot mute user %s in chat %s — bot lacks admin rights", user_id, chat_id
            )
            await message.reply_text(
                "⚠️ I need admin rights to mute members\\. "
                "Please promote me to admin with the 'Restrict Members' permission\\.",
                parse_mode="MarkdownV2",
            )
        except BadRequest as e:
            logger.error("Failed to mute user %s: %s", user_id, e)


# ── Command handlers ──────────────────────────────────────────────────────────

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
        "Reply to any message with /translate to get it translated:\n"
        "• English → French\n"
        "• French → English\n"
        "The translation disappears after 2 minutes\\.\n\n"
        "*Voice messages*\n"
        "Send a voice message — I transcribe it and, if it's in English, "
        "correct any mistakes with bilingual explanations\\.\n\n"
        "*French language policy* 🇫🇷\n"
        "French messages in the group are counted per user per day:\n"
        "• 3 messages → warning\n"
        "• 5 messages → muted for 15 minutes\n"
        "• 5 more after unmute → muted for 1 hour\n"
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


# ── Message handlers ──────────────────────────────────────────────────────────

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
        logger.error("Error calling OpenAI or parsing response: %s", e)
        await message.reply_text(
            "⚠️ Something went wrong while processing your text. Please try again."
        )
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

    try:
        transcript = transcribe_audio(audio_bytes, filename="voice.ogg")
    except Exception as e:
        logger.error("Transcription error: %s", e)
        await message.reply_text("⚠️ I couldn't transcribe that voice message. Please try again.")
        return

    if not transcript:
        return

    logger.info("Transcribed voice: %s", transcript)

    try:
        result = analyse_message(transcript)
    except Exception as e:
        logger.error("Analysis error after transcription: %s", e)
        return

    language = result.get("language", "other")
    has_mistakes = result.get("has_mistakes", False)
    corrected = result.get("corrected", "")
    explanations = result.get("explanations", [])

    if language != "english" or not has_mistakes:
        return

    reply = (
        f"🎙️ *Transcription:*\n_{escape_markdown(transcript)}_\n\n"
        + format_correction(corrected, explanations)
    )
    await message.reply_text(reply, parse_mode="MarkdownV2")


# ── Entry point ───────────────────────────────────────────────────────────────

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


if __name__ == "__main__":
    main()
