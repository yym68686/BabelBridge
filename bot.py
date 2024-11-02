from ModelMerge.src.ModelMerge import chatgpt, claude3, gemini
from md2tgmd.src.md2tgmd import escape

from telegram.ext import MessageHandler, ApplicationBuilder, filters, Application, AIORateLimiter, CommandHandler
from telegram import BotCommand

from sqlalchemy import create_engine, Column, String, Integer, select, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

import time

time_out = 600

async def start(update, context):
    user = update.effective_user
    message = (
        f"Hi `{user.username}` !"
    )
    await update.message.reply_text(message)

async def get_file_url(file, context):
    file_id = file.file_id
    new_file = await context.bot.get_file(file_id, read_timeout=time_out, write_timeout=time_out, connect_timeout=time_out, pool_timeout=time_out)
    file_url = new_file.file_path
    return file_url

async def GetMesage(update_message, context):
    image_url = None

    chatid = str(update_message.chat_id)
    if update_message.is_topic_message:
        message_thread_id = update_message.message_thread_id
    else:
        message_thread_id = None
    if message_thread_id:
        convo_id = str(chatid) + "_" + str(message_thread_id)
    else:
        convo_id = str(chatid)

    messageid = update_message.message_id

    if update_message.photo:
        photo = update_message.photo[-1]

        image_url = await get_file_url(photo, context)

    return image_url, chatid, messageid, message_thread_id, convo_id

async def GetMesageInfo(update, context):
    if update.edited_message:
        image_url, chatid, messageid, message_thread_id, convo_id = await GetMesage(update.edited_message, context)
        update_message = update.edited_message
    elif update.message:
        image_url, chatid, messageid, message_thread_id, convo_id = await GetMesage(update.message, context)
        update_message = update.message
    else:
        return None, None, None, None, None
    # 添加获取用户语言的逻辑
    user_lang = None
    if update.effective_user:
        user_lang = update.effective_user.language_code
    return image_url, chatid, messageid, message_thread_id, convo_id, user_lang

from dotenv import load_dotenv
load_dotenv()
import os
CLAUDE_API = os.environ.get('CLAUDE_API_KEY', None)
GOOGLE_AI_API_KEY = os.environ.get('GOOGLE_AI_API_KEY', None)
temperature = float(os.environ.get('TEMPERATURE', '0'))
api_key = os.environ.get('API')
api_url = os.environ.get('API_URL', 'https://api.openai.com/v1/chat/completions')
engine = os.environ.get('ENGINE', 'gpt-4-turbo')
whitelist = os.environ.get('WHITELIST', "")
if whitelist == "":
    whitelist = None
if whitelist:
    whitelist = [id for id in whitelist.split(",")]

ChatGPTbot, claude3Bot, gemini_Bot = None, None, None
def InitEngine(api_key = None):
    global ChatGPTbot, claude3Bot, gemini_Bot
    if api_key:
        ChatGPTbot = chatgpt(temperature=temperature, print_log=True, use_plugins=False)
    if CLAUDE_API:
        claude3Bot = claude3(temperature=temperature, print_log=True, use_plugins=False)
    if GOOGLE_AI_API_KEY:
        gemini_Bot = gemini(temperature=temperature, print_log=True, use_plugins=False)

InitEngine(api_key)

def get_robot(engine, api_key = None, api_url = None):
    global ChatGPTbot, claude3Bot, gemini_Bot
    if CLAUDE_API and "claude-3" in engine:
        robot = claude3Bot
        api_key = CLAUDE_API
        api_url = "https://api.anthropic.com/v1/messages"
    elif GOOGLE_AI_API_KEY and "gemini" in engine:
        robot = gemini_Bot
        api_key = GOOGLE_AI_API_KEY
        api_url = "https://generativelanguage.googleapis.com/v1beta/models/{model}:{stream}?key={api_key}"
    elif ChatGPTbot:
        robot = ChatGPTbot
        api_key = api_key
        api_url = api_url

    return robot, api_key, api_url

import imghdr
import base64
import urllib.parse
import requests

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        file_content = image_file.read()
        file_type = imghdr.what(None, file_content)
        base64_encoded = base64.b64encode(file_content).decode('utf-8')

        if file_type == 'png':
            return f"data:image/png;base64,{base64_encoded}"
        elif file_type in ['jpeg', 'jpg']:
            return f"data:image/jpeg;base64,{base64_encoded}"
        else:
            raise ValueError(f"不支持的图片格式: {file_type}")

def get_doc_from_url(url):
    filename = urllib.parse.unquote(url.split("/")[-1])
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    return filename

def get_encode_image(image_url):
    filename = get_doc_from_url(image_url)
    image_path = os.getcwd() + "/" + filename
    base64_image = encode_image(image_path)
    # os.remove(image_path)
    return base64_image

def get_image_message(image_url, message, engine = None):
    if image_url:
        base64_image = get_encode_image(image_url)
        colon_index = base64_image.index(":")
        semicolon_index = base64_image.index(";")
        image_type = base64_image[colon_index + 1:semicolon_index]

        if "gpt-4" in engine \
        or (os.environ.get('CLAUDE_API_KEY', None) is None and "claude-3" in engine) \
        or (os.environ.get('GOOGLE_AI_API_KEY', None) is None and "gemini" in engine) \
        or (os.environ.get('GOOGLE_AI_API_KEY', None) is None and os.environ.get('VERTEX_CLIENT_EMAIL', None) is None and os.environ.get('VERTEX_PRIVATE_KEY', None) is None and os.environ.get("VERTEX_PROJECT_ID", None) is None and "gemini" in engine):
            message.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image
                    }
                }
            )
        if os.environ.get('CLAUDE_API_KEY', None) and "claude-3" in engine:
            message.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_type,
                        "data": base64_image.split(",")[1],
                    }
                }
            )
        if (
            os.environ.get('GOOGLE_AI_API_KEY', None) \
            or (os.environ.get('VERTEX_CLIENT_EMAIL', None) and os.environ.get('VERTEX_PRIVATE_KEY', None) and os.environ.get("VERTEX_PROJECT_ID", None))
        ) \
        and "gemini" in engine:
            message.append(
                {
                    "text": "You are an orc expert. Return each card number and PIN code. The card number is usually 16-21 digits, and the PIN generally needs to scratch off the label, so the background color of the PIN may be slightly different. The PIN may also be inside a rectangular boundary box, and it is usually near the card number. If multiple sequences exist at the same time, the string sequence containing uppercase letters is more likely to be the PIN. If there is no string with a mix of uppercase letters and numbers, please prioritize using a four-digit pure number as the PIN. If there is no four-digit pure number, please select the longest numeric sequence as the PIN. Please identify all card numbers and PIN codes. If there are multiple card numbers and PIN codes, please arrange them in numerical order, and the card numbers should not contain spaces. Some images are oriented incorrectly, and you need to rotate them yourself to recognize the text in the correct direction. Return format: 1. Card: [card number]\nPIN: [PIN code]..."
                }
            )
            message.append(
                {
                    "inlineData": {
                        "mimeType": image_type,
                        "data": base64_image.split(",")[1],
                    }
                }
            )
    return message

def Authorization(func):
    async def wrapper(*args, **kwargs):
        update, context = args[:2]
        if whitelist == None:
            return await func(*args, **kwargs)
        if whitelist and str(update.effective_user.id) not in whitelist:
            print(f"User {update.effective_user.id} is not in whitelist")
            return
        return await func(*args, **kwargs)
    return wrapper

# 创建SQLAlchemy基础类
Base = declarative_base()

# 在现有的 Base 和 CardPin 表定义后添加新表
class UserTopicMapping(Base):
    __tablename__ = 'user_topic_mappings'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_chat_id = Column(String(255), nullable=False)  # 用户的chat id
    topic_id = Column(Integer, nullable=False)          # 话题id
    group_id = Column(String(255), nullable=False)      # 群组id
    created_at = Column(Integer, nullable=False)        # 创建时间戳

# 修改数据库配置部分
db_path = os.getenv('DB_PATH', './data/stats.db')
data_dir = os.path.dirname(db_path)
os.makedirs(data_dir, exist_ok=True)

# 替换原有的数据库引擎创建
database_engine = create_async_engine(f'sqlite+aiosqlite:///{db_path}', echo=False)
async_session = sessionmaker(database_engine, class_=AsyncSession, expire_on_commit=False)

@Authorization
async def handle_file(update, context):
    image_url, chatid, messageid, message_thread_id, convo_id = await GetMesageInfo(update, context)
    user_id = str(update.effective_user.id)
    is_admin = user_id in whitelist if whitelist else False

    # global engine, api_key, api_url
    # robot, api_key, api_url = get_robot(engine, api_key, api_url)

    # # 处理图片消息
    # message = get_image_message(image_url, [], engine)
    # system_prompt = "You are an orc expert..."  # 保持原有的 system_prompt
    # result = await robot.ask_async(message, convo_id=convo_id, system_prompt=system_prompt,
    #                              api_key=api_key, api_url=api_url, model=engine, pass_history=2)

    if is_admin and message_thread_id:
        # 管理员在话题中发送图片，查找对应的用户并发送到私聊
        async with async_session() as session:
            mapping = await session.execute(
                select(UserTopicMapping).filter_by(
                    topic_id=message_thread_id,
                    group_id=str(chatid)
                )
            )
            mapping = mapping.scalar_one_or_none()

            if mapping:
                # 转发图片到用户私聊
                await context.bot.send_photo(
                    chat_id=mapping.user_chat_id,
                    photo=update.message.photo[-1].file_id,
                    # caption=result
                )
    else:
        # 普通用户发送图片
        if not message_thread_id:
            # 如果不在话题中，创建新话题
            user_lang = update.effective_user.language_code if update.effective_user else None
            message_thread_id = await create_translation_thread(update, context, user_lang)

        # 在话题中发送图片和结果
        await context.bot.send_photo(
            chat_id=chatid,
            message_thread_id=message_thread_id,
            photo=update.message.photo[-1].file_id,
            # caption=result,
            reply_to_message_id=messageid
        )

async def post_init(application: Application) -> None:
    await application.bot.set_my_commands([
        BotCommand('start', 'Start'),
    ])
BOT_TOKEN = os.getenv("BOT_TOKEN")

# 修改 create_translation_thread 函数
async def create_translation_thread(update, context, target_lang):
    chat = update.effective_chat
    user_chat_id = str(update.effective_user.id)
    title = f"Translation to {target_lang}"

    thread = await context.bot.create_forum_topic(
        chat_id=chat.id,
        name=title
    )

    # 使用异步会话
    async with async_session() as session:
        async with session.begin():
            mapping = UserTopicMapping(
                user_chat_id=user_chat_id,
                topic_id=thread.message_thread_id,
                group_id=str(chat.id),
                created_at=int(time.time())
            )
            session.add(mapping)
            await session.commit()

    return thread.message_thread_id

# 修改 handle_message 函数
async def handle_message(update, context):
    image_url, chatid, messageid, message_thread_id, convo_id, user_lang = await GetMesageInfo(update, context)

    user_id = str(update.effective_user.id)
    is_admin = user_id in whitelist if whitelist else False
    message = update.message.text

    robot, role, api_key, api_url = get_robot(convo_id)

    if is_admin and message_thread_id:
        async with async_session() as session:
            mapping = await session.execute(
                select(UserTopicMapping).filter_by(
                    topic_id=message_thread_id,
                    group_id=str(chatid)
                )
            )
            mapping = mapping.scalar_one_or_none()

            if mapping:
                translated = await robot.ask_async(
                    f"Translate the following text to {user_lang}:\n{message}",
                    convo_id=convo_id
                )
                await context.bot.send_message(
                    chat_id=mapping.user_chat_id,
                    text=translated
                )
    else:
        # 普通用户的处理逻辑保持不变
        if not message_thread_id:
            message_thread_id = await create_translation_thread(update, context, user_lang)

        translated = await robot.ask_async(
            f"Translate the following text to {user_lang}:\n{message}",
            convo_id=convo_id
        )
        await context.bot.send_message(
            chat_id=chatid,
            message_thread_id=message_thread_id,
            text=f"Original:\n{message}\n\nTranslated:\n{translated}",
            reply_to_message_id=messageid
        )

if __name__ == '__main__':
    application = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .concurrent_updates(True)
        .connection_pool_size(65536)
        .get_updates_connection_pool_size(65536)
        .read_timeout(time_out)
        .write_timeout(time_out)
        .connect_timeout(time_out)
        .pool_timeout(time_out)
        .get_updates_read_timeout(time_out)
        .get_updates_write_timeout(time_out)
        .get_updates_connect_timeout(time_out)
        .get_updates_pool_timeout(time_out)
        .rate_limiter(AIORateLimiter(max_retries=5))
        .post_init(post_init)
        .build()
    )

    application.add_handler(MessageHandler(
        (filters.PHOTO & ~filters.COMMAND) |
        (
            filters.Document.FileExtension("jpg") |
            filters.Document.FileExtension("jpeg") |
            filters.Document.FileExtension("png")
        ), handle_file
    ))

    application.add_handler(CommandHandler("start", start))

    # 添加消息处理器
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_message
    ))

    application.run_polling(timeout=time_out)
