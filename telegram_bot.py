#!/usr/bin/env python3
"""
Telegram Bot for Automated Face Swapping
Monitors Telegram channels for face images and videos, processes them using roop,
and manages output through Google Sheets and forum topics.

Requirements:
    pip install -r requirements.txt  # or requirements-colab.txt for Colab
    pip install -r requirements-telegram-bot.txt

Dependencies:
    - telethon: Telegram client library
    - gspread: Google Sheets API wrapper
    - roop: Face swapping engine (from main requirements)
    
Environment:
    - Requires /content/ssclient.session file for Telegram authentication
    - Requires Google Service Account credentials (embedded in script)
    - Designed to run on Google Colab with CUDA support
"""

import asyncio
import os
import sys
from time import sleep
import logging
from datetime import datetime

from telethon import TelegramClient, events
from telethon.tl.types import InputMessagesFilterPhotos, InputMessagesFilterVideo
from telethon.tl.functions.channels import CreateChannelRequest
from telethon.tl.functions.messages import CreateForumTopicRequest
from telethon.events import MessageDeleted
import gspread

# Roop modules will be imported lazily when needed
# This prevents import errors when roop is not available
_roop_imported = False
_roop_globals = None
_roop_start = None
_pre_check = None
_limit_resources = None
_update_status = None
_get_frame_processors_modules = None

def _ensure_roop_imported():
    """Lazy import roop modules only when needed"""
    global _roop_imported, _roop_globals, _roop_start, _pre_check, _limit_resources, _update_status, _get_frame_processors_modules
    
    if not _roop_imported:
        logger.info("📦 Loading roop modules...")
        try:
            # Add roop to path if not already there
            roop_path = os.getenv('ROOP_PATH', '/content/myroop')
            if roop_path not in sys.path:
                sys.path.insert(0, roop_path)
            
            import roop.globals as rg
            from roop.core import start as roop_start_func, pre_check as pre_check_func, limit_resources as limit_resources_func, update_status as update_status_func
            from roop.processors.frame.core import get_frame_processors_modules as get_frame_processors_func
            
            _roop_globals = rg
            _roop_start = roop_start_func
            _pre_check = pre_check_func
            _limit_resources = limit_resources_func
            _update_status = update_status_func
            _get_frame_processors_modules = get_frame_processors_func
            
            _roop_imported = True
            logger.info("✅ Roop modules loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to import roop modules: {e}")
            logger.error(f"   Make sure roop is installed and ROOP_PATH is set correctly")
            logger.error(f"   Current ROOP_PATH: {os.getenv('ROOP_PATH', '/content/myroop')}")
            raise

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True,
)
logger = logging.getLogger(__name__)

def log_separator(title=""):
    """Create a visual separator in logs"""
    if title:
        logger.info(f"{'='*20} {title} {'='*20}")
    else:
        logger.info("="*50)

# ============================================================================
# TELEGRAM CONFIGURATION
# ============================================================================
api_id = 25324831
api_hash = '3ae696f7cf3fcb591da4b8c9cda9c41e'

# Session file path - use local path when not on Colab
SESSION_PATH = os.getenv('TELEGRAM_SESSION_PATH', './ssclient.session')
client = TelegramClient(SESSION_PATH, api_id, api_hash)

# Personal session for folder management (user account, not bot)
PERSONAL_SESSION_PATH = os.getenv('TELEGRAM_PERSONAL_SESSION_PATH', './sspersonal.session')
personal_client = TelegramClient(PERSONAL_SESSION_PATH, api_id, api_hash)

entity_map = {
    "group": {
        "id": 2192019759,
        "threads": {
            "inputimage": 2,
            "inputvideo": 3,
            "output": 4,
            "etcd": 375
        }
    },
    "output_chat_id": {
        "id": 2326968720,
        "threads": {}
    },
    "user_id": {
        "id": 5177338966,
        "threads": {}
    },
    "input_chat_id": {
        "id": 2354814383,
        "threads": {
            "faces": 2,
            "vid": 3,
            "etcd": 7
        }
    }
}

def get_entity_id(entity_name):
    return entity_map[entity_name]["id"]

thread_map = {
    "inputimage": 2,
    "inputvideo": 3,
    "output": 4,
    "etcd": 375
}

entity_list = {}

# Directory paths - use local paths when not on Colab
BASE_DIR = os.getenv('BOT_BASE_DIR', '.')
os_dir = {
    "input_image": os.path.join(BASE_DIR, "input_image"),
    "input_video": os.path.join(BASE_DIR, "input_video"),
    "output": os.path.join(BASE_DIR, "output")
}

# Create directories if they don't exist
for dir_path in os_dir.values():
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# GOOGLE SHEETS CONFIGURATION
# ============================================================================
SHEET_ID = "1ic3oeMukAoQKVRkaJAs55f9bTpAACvzXmBLKc_1Kq2A"

service_account = {
    "type": "service_account",
    "project_id": "psychic-valve-379909",
    "private_key_id": "1acfb016be67af8bcf9844c6573f4330380df57f",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQCzk0ta5IkIiFsK\ng1RQAWPqtOSPOsPQEigXw8id0TFYuMR6zfm4sWQN4Ybtzhu4Km9DcpNV3Gyab7C/\n1lSxh0rd+Pmse5IWut2kXLogE6r8Q48ZYAJVniFR5uh9Ja4RcaR8Bep93XS2ZQoC\nPsKDiaJir31RPSluqPbbWmZFuRnbsrw1IceUCSxCg+L4Hu1B0qNhWsac5Cn2nWhi\nN/ikGvtd879OnzPOlOWJUEKb1XK6hBgDQvWehxR2wlXzEYxxtLAVH/xY8TAAMTYS\n+qslosoYX/D2HWgS7i24c+SFfImXZfQ4GGEqQL5ZhNIoQtU7BL/bGS6JLu+yenBf\n+04y+629AgMBAAECggEAA4hIvqWpRyywtcU7HFizRSYd7+4Z6zHz7fHl5apex0ft\nAmvdvoQ/ZV7QGTHkVO1QJefi1oVOe3Bzxp1Yx8Or65JXXH5fn9UlVP1bMSgIClE4\n+GPP71P7+h1OwobV6acB7nM79Oqjb3teGH7nFgGdAoFUwYF/ViU6aPW8zvk5EsMC\nYKi75sXWz/H2shP2trcetvLfTJxTJmlIQKDUZFD5D4Bn8Ggc6auoseRVzYbb0I2f\nJK+zptGYw/tQv7Ov7csetMwPKnSFSbRP98ADcM+QXploQdjTiVMGEDpgnkW3fwm8\nvtWbt0eEmafsY2WDjq5ABYrJQSro5Q2XxiDJhOQVtQKBgQD7iXyral0uI4Tljce7\nF/BN+tVlcGUrmYMV1MtBDp6GkRfdGi4wfsG8RjWbEzPSVNTtDW64sbzC1JFUS8qy\nom1l8HYs/rOv/kR1VKuFNpMZIL3FyoDzXupsS2Xm44jjap4WMppBHejBwfC6ao71\ncmj2DF3m/NJ4jqSDnQgeIr3N3wKBgQC2wvLD/+Y+Ya/fqn0LOrRsDjRXHk2zoLuT\nM0ieD3LdZ0bUpZ8kaSUSUOg283kzgkjl+gPE2Uuhj8inHOMNwecfog9C1s/jno++\nwq6BSPDQglUWSMVBswhiJjoTURvqRoEAgbPwHwJ0BrmdlNhcZF0sLSgBD8Xt8aae\n7YSWdQP/4wKBgQDv8yngg6i7+Nov+V5NFfsBCIVFLUD1pI2t5761IvMKv6KaIv1o\n+aJqrjc45cqLnK06LZQv0C2XVhwE5ABLBDOECS1Sou5atx4rJ0gv3e3ZDKi74/+7\nKhJKmGpvJSNSfkSsPD12XMfK/e1ch4Y0kp2aChYR4aANrVU20DgMN2wE1QKBgQCQ\ntnqt0dHJMWambwPGQXydVv4AwYD7K9l67A+jAE6Om3lNZbtGhG8TOrc008Lzk0UP\nvNEBLKIjUouqL0hmG1RcCuGUpi1EacgeX174Bf6Qy7J+C68Nn6qStPwlUYMOFTeO\nZCVLi/JQVwbgjPxo5BuWoEbd2FR8XWz2yNHEpPE+ywKBgQC0xGqdSPS+w/yeP/cL\nCWSBJQZ1tosnAHPmsneAJ0z2OYI0ALWUpbBZofTKznxyWOBRR99QPVZ4+4h3UOJi\nWBIVIhgfs4V/Br1A81NzaCYicEjFOZVzhRkuHNYZVfQAR298Jo0jeNaV4/FjMghu\njgigAqymis9eARjMapihe+eZ8w==\n-----END PRIVATE KEY-----\n",
    "client_email": "sheet-etcd@psychic-valve-379909.iam.gserviceaccount.com",
    "client_id": "107637598360658622635",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/sheet-etcd%40psychic-valve-379909.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
}

gc = gspread.service_account_from_dict(service_account)
sh = gc.open_by_key(SHEET_ID)

# ============================================================================
# GOOGLE SHEETS FUNCTIONS
# ============================================================================
def get_data(worksheet_name):
    return sh.worksheet(worksheet_name).get_all_values()[1:]

def check_topic_exist(image_name, data=None):
    if data is None:
        data = get_data("list_image")
    for i in data:
        if i[1] == image_name and i[3] != "":
            return True, i[4]
    return False, None

def check_image_by_message_id(message_id, data=None):
    """Check if an image entry exists by message_id and return its data"""
    if data is None:
        data = get_data("list_image")
    for i in range(len(data)):
        if data[i][0] == str(message_id):
            # Old structure: [message_id, message_text, output_chat_id, topic_id, old_name]
            # New structure: [message_id, message_text, output_chat_id, topic_id, old_name, channel_id]
            # Return: (exists, row_index, message_text, output_chat_id, topic_id, channel_id, old_name)
            channel_id = data[i][5] if len(data[i]) > 5 else ""  # Column 6 (index 5)
            old_name = data[i][4] if len(data[i]) > 4 else data[i][1]  # Column 5 (index 4)
            return True, i, data[i][1], data[i][2], data[i][3], channel_id, old_name
    return False, None, None, None, None, None, None

def check_video_exist(video_id, data=None):
    if data is None:
        data = get_data("list_video")
    for i in data:
        if i[0] == video_id:
            return True, i[1]
    return False, None

def get_topic_id(image_id, data=None):
    if data is None:
        data = get_data("list_image")
    for i in data:
        if i[0] == image_id:
            return i[3]
    return None

def check_output_exist(output_id, data=None):
    if data is None:
        data = get_data("list_output")
    for i in data:
        if i[0] == output_id:
            return True, i[1]
    return False, None

def update_topic_id(image_id, image_name, data=None):
    if data is None:
        data = get_data("list_image")
    logger.info(f"📝 Updating topic ID for image: {image_name} → {image_id}")
    for i in range(len(data)):
        if data[i][1] == image_name:
            try:
                sh.worksheet("list_image").update_acell(f"A{i+2}", image_id)
                logger.info(f"✅ Successfully updated topic ID for {image_name}")
                return True
            except Exception as e:
                logger.error(f"❌ Error updating topic ID for {image_name}: {e}")
                return False
    logger.warning(f"⚠️ Image name {image_name} not found in data")
    return True

def delete_image_by_message_id(message_id):
    """Delete image entry from Google Sheet by message ID"""
    logger.info(f"🔍 Checking for image with message ID: {message_id}")
    try:
        data = get_data("list_image")
        for i in range(len(data)):
            if data[i][0] == str(message_id):
                row_number = i + 2
                sh.worksheet("list_image").delete_rows(row_number)
                logger.info(f"✅ Successfully deleted image '{data[i][1]}' (ID: {message_id}) from sheet")
                return True, data[i][1]
        logger.info(f"ℹ️ Image with message ID {message_id} not found in sheet")
        return False, None
    except Exception as e:
        logger.error(f"❌ Error deleting image with message ID {message_id}: {e}")
        return False, None

def delete_video_by_message_id(message_id):
    """Delete video entry from Google Sheet by message ID"""
    logger.info(f"🔍 Checking for video with message ID: {message_id}")
    try:
        data = get_data("list_video")
        for i in range(len(data)):
            if data[i][0] == str(message_id):
                row_number = i + 2
                sh.worksheet("list_video").delete_rows(row_number)
                logger.info(f"✅ Successfully deleted video '{data[i][1]}' (ID: {message_id}) from sheet")
                return True, data[i][1]
        logger.info(f"ℹ️ Video with message ID {message_id} not found in sheet")
        return False, None
    except Exception as e:
        logger.error(f"❌ Error deleting video with message ID {message_id}: {e}")
        return False, None

def delete_outputs_by_image_id(image_id):
    """Delete all output entries that contain this image ID"""
    logger.info(f"🗑️ Deleting all outputs containing image ID: {image_id}")
    try:
        data = get_data("list_output")
        deleted_count = 0
        deleted_outputs = []

        for i in range(len(data) - 1, -1, -1):
            output_id = data[i][0]
            if output_id.startswith(f"{image_id}_"):
                row_number = i + 2
                sh.worksheet("list_output").delete_rows(row_number)
                deleted_outputs.append(data[i][1])
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"✅ Successfully deleted {deleted_count} output(s) containing image ID {image_id}")
            logger.info(f"   📝 Deleted outputs: {', '.join(deleted_outputs)}")
        else:
            logger.info(f"ℹ️ No outputs found containing image ID {image_id}")

        return deleted_count, deleted_outputs
    except Exception as e:
        logger.error(f"❌ Error deleting outputs for image ID {image_id}: {e}")
        return 0, []

def create_map_user_image():
    data = get_data("list_image")
    map_user_image = {}
    for i in data:
        if i[1] not in map_user_image:
            map_user_image[i[1]] = {
                "chat_id": i[2],
                "topic_id": i[3],
                "message_id": i[0]
            }
    return map_user_image

def create_map_video():
    data = get_data("list_video")
    map_video = {}
    for i in data:
        map_video[i[1]] = {
            "message_id": i[0]
        }
    return map_video

# ============================================================================
# TELEGRAM HELPER FUNCTIONS
# ============================================================================
def entity(entity_name):
    if entity_name not in entity_map:
        return None
    if entity_name not in entity_list:
        entity_list[entity_name] = client.get_entity(get_entity_id(entity_name))
    return entity_list[entity_name]

async def download_file(message_id, sub_path=".", ext="jpg"):
    logger.info(f"📥 Downloading file {message_id} to {sub_path}")
    message_id = int(message_id)
    final_path = f"{sub_path}/{message_id}.{ext}"
    
    if os.path.exists(final_path):
        logger.info(f"⚡ File {final_path} already exists, skipping download")
        return final_path

    try:
        message = await client.get_messages(get_entity_id("input_chat_id"), limit=1, ids=message_id)
        logger.info(f"📨 Retrieved message: {message}")
        result = await client.download_media(message, final_path)
        logger.info(f"✅ Successfully downloaded file to {final_path}")
        return result
    except Exception as e:
        logger.error(f"❌ Failed to download file {message_id}: {e}")
        raise

async def render_video(input_image, input_video, output_path):
    """
    Render video using roop directly (not subprocess)
    This calls roop's internal functions instead of running as a separate process
    """
    logger.info(f"🎬 Starting video render: {input_video} with image {input_image} → {output_path}")
    
    try:
        # Lazy load roop modules only when needed
        _ensure_roop_imported()
        
        # Configure roop globals
        _roop_globals.source_path = input_image
        _roop_globals.target_path = input_video
        _roop_globals.output_path = output_path
        _roop_globals.frame_processors = ['face_swapper']
        _roop_globals.keep_fps = True
        _roop_globals.keep_audio = True
        _roop_globals.keep_frames = True
        _roop_globals.many_faces = True
        _roop_globals.video_encoder = 'libx264'
        _roop_globals.video_quality = 18
        _roop_globals.max_memory = 14
        _roop_globals.execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        _roop_globals.execution_threads = 18
        _roop_globals.headless = True
        
        logger.info(f"🔧 Roop configuration:")
        logger.info(f"   Source: {_roop_globals.source_path}")
        logger.info(f"   Target: {_roop_globals.target_path}")
        logger.info(f"   Output: {_roop_globals.output_path}")
        logger.info(f"   Many faces: {_roop_globals.many_faces}")
        logger.info(f"   Execution providers: {_roop_globals.execution_providers}")
        
        # Pre-check
        if not _pre_check():
            raise Exception("Roop pre-check failed")
        
        # Check frame processors
        for frame_processor in _get_frame_processors_modules(_roop_globals.frame_processors):
            if not frame_processor.pre_check():
                raise Exception(f"Frame processor {frame_processor.NAME} pre-check failed")
        
        # Limit resources
        _limit_resources()
        
        # Start processing
        logger.info(f"🚀 Starting roop processing...")
        _roop_start()
        
        logger.info(f"✅ Video render completed successfully: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Video render failed: {e}")
        raise

async def create_forum_topic(topic_name, chat_id):
    logger.info(f"🆕 Creating forum topic: '{topic_name}' in chat {chat_id}")
    try:
        # For Telethon 1.42+, CreateForumTopicRequest uses 'peer' instead of 'channel'
        topic = await client(CreateForumTopicRequest(
            peer=chat_id,
            title=topic_name,
            icon_color=None,  # Optional: can set a color
            icon_emoji_id=None  # Optional: can set an emoji
        ))
        
        # Extract topic ID from response
        # The response structure may vary between versions
        if hasattr(topic, 'updates') and len(topic.updates) > 0:
            topic_id = topic.updates[0].id
        else:
            logger.warning(f"⚠️ Unexpected response structure, trying to extract topic ID")
            # Try to find the topic ID in the response
            topic_id = getattr(topic.updates[0], 'id', None) if hasattr(topic, 'updates') else None
            
        if topic_id:
            logger.info(f"✅ Forum topic created successfully: '{topic_name}' (ID: {topic_id})")
            return topic_id
        else:
            logger.error(f"❌ Could not extract topic ID from response: {topic}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to create forum topic '{topic_name}': {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return None

async def create_separate_chat(chat_name, photo_message=None):
    """
    Create a separate channel for an image
    This allows grouping multiple chats into folders later
    Optionally sets the image as the channel avatar
    """
    logger.info(f"📢 Creating separate channel: '{chat_name}'")
    try:
        # Create a new channel (can be private or public)
        result = await client(CreateChannelRequest(
            title=chat_name,
            about=f"Chat for {chat_name}",
            megagroup=False  # False = channel, True = supergroup
        ))
        
        # Extract channel ID from response
        if hasattr(result, 'chats') and len(result.chats) > 0:
            channel = result.chats[0]
            channel_id = channel.id
            logger.info(f"✅ Channel created successfully: '{chat_name}' (ID: {channel_id})")
            
            # Set the photo as channel avatar if provided
            if photo_message and photo_message.photo:
                try:
                    logger.info(f"🖼️ Setting channel avatar from image...")
                    
                    # Download the photo to a temporary file with proper extension
                    import tempfile
                    temp_photo = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    photo_path = temp_photo.name
                    temp_photo.close()
                    
                    await client.download_media(photo_message, file=photo_path)
                    logger.info(f"📥 Photo downloaded to {photo_path}")
                    
                    # Upload and set as channel photo using InputChatUploadedPhoto
                    from telethon.tl.functions.channels import EditPhotoRequest
                    from telethon.tl.types import InputChatUploadedPhoto
                    
                    uploaded_file = await client.upload_file(photo_path)
                    await client(EditPhotoRequest(
                        channel=channel,
                        photo=InputChatUploadedPhoto(file=uploaded_file)
                    ))
                    
                    # Clean up temp file
                    import os as temp_os
                    temp_os.unlink(photo_path)
                    
                    logger.info(f"✅ Channel avatar set successfully for '{chat_name}'")
                except Exception as avatar_error:
                    logger.warning(f"⚠️ Failed to set channel avatar: {avatar_error}")
                    import traceback
                    logger.warning(f"   Traceback: {traceback.format_exc()}")
                    # Don't fail the whole operation if avatar setting fails
            
            # Add channel to folder
            try:
                await add_channel_to_folder(channel, "ODF")
            except Exception as folder_error:
                logger.warning(f"⚠️ Failed to add channel to folder: {folder_error}")
                # Don't fail the whole operation if folder addition fails
            
            return channel_id
        else:
            logger.error(f"❌ Could not extract channel ID from response: {result}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to create channel '{chat_name}': {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return None

async def update_channel_info(channel_id, new_name, new_photo_message=None, old_name=None):
    """
    Update an existing channel's name and optionally its photo
    
    Args:
        channel_id: The channel ID to update (can be string or int)
        new_name: The new name for the channel
        new_photo_message: Optional message with photo to update channel avatar
        old_name: The old name to compare against (skip title update if same)
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"🔄 Updating channel {channel_id}: name='{new_name}'")
    try:
        from telethon.tl.functions.channels import EditTitleRequest, EditPhotoRequest
        from telethon.tl.types import InputChatUploadedPhoto
        
        # Convert channel_id to int if it's a string, then get the entity
        channel_id_int = int(channel_id) if isinstance(channel_id, str) else channel_id
        channel_entity = await client.get_input_entity(channel_id_int)
        
        title_updated = False
        photo_updated = False
        
        # Only update channel title if it actually changed
        if old_name is None or new_name != old_name:
            await client(EditTitleRequest(
                channel=channel_entity,
                title=new_name
            ))
            logger.info(f"✅ Channel name updated to '{new_name}'")
            title_updated = True
        else:
            logger.info(f"ℹ️ Channel name unchanged ('{new_name}'), skipping title update")
        
        # Update channel photo if provided
        if new_photo_message and new_photo_message.photo:
            import tempfile
            logger.info("🖼️ Updating channel avatar from new image...")
            
            # Download photo to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                photo_path = temp_file.name
                await client.download_media(new_photo_message.photo, photo_path)
                logger.info(f"📥 Photo downloaded to {photo_path}")
            
            # Upload and set as channel photo
            uploaded_file = await client.upload_file(photo_path)
            await client(EditPhotoRequest(
                channel=channel_entity,
                photo=InputChatUploadedPhoto(uploaded_file)
            ))
            
            # Clean up temp file
            os.unlink(photo_path)
            logger.info(f"✅ Channel avatar updated successfully for '{new_name}'")
            photo_updated = True
        
        # Return success if either title or photo was updated
        if title_updated or photo_updated:
            return True
        else:
            logger.info(f"ℹ️ No changes made to channel {channel_id}")
            return True  # Still return True since no error occurred
        
    except Exception as e:
        logger.error(f"❌ Failed to update channel {channel_id}: {e}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return False

async def add_channel_to_folder(channel, folder_name):
    """
    Add a channel to a Telegram folder using client.edit_folder()
    """
    logger.info(f"📁 Adding channel to folder: '{folder_name}'")
    try:
        from telethon.tl.functions.messages import GetDialogFiltersRequest
        
        filters_result = await personal_client(GetDialogFiltersRequest())
        filters = filters_result.filters if hasattr(filters_result, 'filters') else []
        
        target_folder_id = None
        for f in filters:
            if hasattr(f, 'title'):
                title_text = f.title.text if hasattr(f.title, 'text') else str(f.title)
                if title_text == folder_name:
                    target_folder_id = f.id
                    logger.info(f"� Found folder '{folder_name}' with ID {target_folder_id}")
                    break
        
        if target_folder_id is None:
            logger.warning(f"⚠️ Folder '{folder_name}' not found - create it in Telegram first")
            return
        
        # Use EditPeerFoldersRequest with InputFolderPeer
        from telethon.tl.functions.messages import UpdateDialogFilterRequest
        from telethon.tl.types import DialogFilter
        
        # Get the full filter object
        target_filter = next((f for f in filters if hasattr(f, 'title') and 
                             (f.title.text if hasattr(f.title, 'text') else str(f.title)) == folder_name), None)
        
        if not target_filter:
            logger.warning(f"⚠️ Could not get filter object")
            return
        
        # Convert channel to proper peer format
        peer = await personal_client.get_input_entity(channel)
        
        # Add to include_peers
        existing_peers = list(target_filter.include_peers) if target_filter.include_peers else []
        existing_peers.append(peer)
        
        await personal_client(UpdateDialogFilterRequest(
            id=target_filter.id,
            filter=DialogFilter(
                id=target_filter.id,
                title=target_filter.title,
                pinned_peers=getattr(target_filter, 'pinned_peers', []),
                include_peers=existing_peers,
                exclude_peers=getattr(target_filter, 'exclude_peers', []),
                emoticon=getattr(target_filter, 'emoticon', None),
                contacts=getattr(target_filter, 'contacts', False),
                non_contacts=getattr(target_filter, 'non_contacts', False),
                groups=getattr(target_filter, 'groups', False),
                broadcasts=getattr(target_filter, 'broadcasts', False),
                bots=getattr(target_filter, 'bots', False),
                exclude_muted=getattr(target_filter, 'exclude_muted', False),
                exclude_read=getattr(target_filter, 'exclude_read', False),
                exclude_archived=getattr(target_filter, 'exclude_archived', False)
            )
        ))
        logger.info(f"✅ Channel added to folder '{folder_name}' successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to add channel to folder: {e}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")

async def send_message(chat_name, thread_id, message):
    return await client.send_message(get_entity_id(chat_name), message, reply_to=thread_id)

async def edit_message(message, new_message):
    try:
        await message.edit(new_message)
        logger.info(f"✏️ Message edited successfully")
    except Exception as e:
        logger.error(f"❌ Error editing message: {e}")
    return message

async def send_video(chat_name, thread_id, video_path, caption=""):
    return await client.send_file(get_entity_id(chat_name), video_path, reply_to=int(thread_id), caption=caption)

# ============================================================================
# MESSAGE HANDLERS
# ============================================================================
async def handle_image_no_text(event):
    reply_message = await event.reply("Please update the caption for the image")
    sleep(2)
    await reply_message.delete()

async def handle_image_has_text(event):
    message_id = event.message.id
    message_text = event.message.text.lower()
    logger.info(f"🖼️ Processing image with text: '{message_text}' (ID: {message_id})")

    # First check if this message_id already exists (this is an edit, not a new message)
    exists, row_idx, old_text, output_chat, topic_id, channel_id, old_name = check_image_by_message_id(message_id)
    
    if exists:
        logger.info(f"✏️ Message ID {message_id} already exists - this is an edit from '{old_text}' to '{message_text}'")
        
        # Update the channel name and photo if channel_id exists
        if channel_id:
            update_success = await update_channel_info(channel_id, message_text, event.message, old_text)
            if update_success:
                # Update the sheet with new name
                # Column B (index 1) = message_text
                # Column E (index 4) = old_name (keep it the same as message_text for consistency)
                sh.worksheet("list_image").update_acell(f"B{row_idx+2}", message_text)  # Column B = message_text
                sh.worksheet("list_image").update_acell(f"E{row_idx+2}", message_text)  # Column E = old_name
                logger.info(f"✅ Updated sheet entry for message {message_id}")
                
                # Customize message based on what changed
                if old_text != message_text:
                    reply_message = await event.reply(f"✅ Channel '{old_text}' renamed to '{message_text}' and photo updated")
                else:
                    reply_message = await event.reply(f"✅ Channel '{message_text}' photo updated")
            else:
                reply_message = await event.reply(f"⚠️ Failed to update channel '{old_text}'")
        else:
            logger.warning(f"⚠️ No channel_id found for message {message_id}, cannot update channel")
            reply_message = await event.reply(f"⚠️ No channel associated with '{old_text}'")
        
        sleep(2)
        await reply_message.delete()
        return

    # Not an edit - check if topic with this name already exists (different message)
    exist_topic = check_topic_exist(message_text)

    if exist_topic[0]:
        logger.info(f"📂 Topic already exists: {exist_topic[1]}")
        reply_message = await event.reply(f"Topic already exist with name {exist_topic[1]}")
        update_topic_id(message_id, message_text)
        await reply_message.edit(f"Topic already exist with name {exist_topic[1]} and updated the id")
        sleep(2)
        await reply_message.delete()
        return

    # New message with new name - create everything
    logger.info(f"🆕 Creating forum topic and separate channel: {message_text}")
    
    # Run both operations in parallel using asyncio.gather
    topic_task = create_forum_topic(message_text, get_entity_id("output_chat_id"))
    channel_task = create_separate_chat(message_text, event.message)  # Pass the message with photo
    
    # Wait for both to complete
    topic_id, channel_id = await asyncio.gather(topic_task, channel_task)

    if topic_id is not None:
        # Old structure: [message_id, message_text, output_chat_id, topic_id, message_text]
        # New structure: [message_id, message_text, output_chat_id, topic_id, message_text, channel_id]
        # Append channel_id at the END to not break old records
        sh.worksheet("list_image").append_row([message_id, message_text, get_entity_id("output_chat_id"), topic_id, message_text, channel_id or ""])
        logger.info(f"✅ Topic '{message_text}' created successfully with ID: {topic_id}")
        
        if channel_id is not None:
            logger.info(f"✅ Separate channel '{message_text}' created with ID: {channel_id}")
            reply_message = await event.reply(f"✅ Topic and channel created for {message_text}\n📂 Topic ID: {topic_id}\n📢 Channel ID: {channel_id}")
        else:
            logger.warning(f"⚠️ Channel creation failed but topic succeeded")
            reply_message = await event.reply(f"⚠️ Topic {message_text} created (channel failed)")
        
        sleep(2)
        await reply_message.delete()
    else:
        logger.error(f"❌ Failed to create topic: {message_text}")
        if channel_id is not None:
            logger.info(f"⚠️ Channel created but topic failed for {message_text}")

    logger.info(f"✅ Completed handling image with text: {message_text}")

async def handle_input_video_no_text(event):
    reply_message = await event.reply("Please update the caption for the video")
    sleep(2)
    await reply_message.delete()

async def handle_input_video_has_text(event):
    message_id = event.message.id
    message_text = event.message.text.lower()
    logger.info(f"🎥 Processing video with text: '{message_text}' (ID: {message_id})")

    video_exist = check_video_exist(message_id)
    if video_exist[0]:
        logger.info(f"⚠️ Video already exists: {video_exist[1]}")
        reply_message = await event.reply(f"Video already exist with name {video_exist[1]}")
        sleep(2)
        await reply_message.delete()
        return

    sh.worksheet("list_video").append_row([message_id, message_text])
    logger.info(f"✅ Video '{message_text}' added successfully")
    reply_message = await event.reply(f"Video {message_text} is added")
    sleep(2)
    await reply_message.delete()

async def handle_message_deleted(event):
    """Handle when messages are deleted from input_chat_id"""
    logger.info(f"🗑️ Message deletion detected")

    deleted_count = 0
    for message_id in event.deleted_ids:
        logger.info(f"🔍 Checking deleted message ID: {message_id}")

        success, image_name = delete_image_by_message_id(message_id)
        if success:
            deleted_count += 1
            output_count, deleted_outputs = delete_outputs_by_image_id(message_id)
            if output_count > 0:
                logger.info(f"🧹 Cleaned up {output_count} related output(s) for deleted image '{image_name}'")
            continue

        success, video_name = delete_video_by_message_id(message_id)
        if success:
            deleted_count += 1
            continue

        logger.info(f"ℹ️ Message ID {message_id} not found in any sheet (might be untracked message)")

    if deleted_count > 0:
        logger.info(f"✅ Successfully processed {deleted_count} deleted message(s)")
    else:
        logger.info(f"ℹ️ No matching entries found in sheets for deleted messages")

async def domany(event):
    message = event.message.text.split(" ")
    image_names = message[1].split(",")
    video_names = message[2].split(",")
    total = len(image_names) * len(video_names)

    logger.info(f"🚀 Starting batch processing:")
    logger.info(f"   📸 Images ({len(image_names)}): {', '.join(image_names)}")
    logger.info(f"   🎥 Videos ({len(video_names)}): {', '.join(video_names)}")
    logger.info(f"   📊 Total combinations: {total}")

    progress = 0
    success = 0
    skip = 0
    fail_list = []

    tracking_message = await send_message("group", thread_map["output"], 
                                         f"🎬 Starting batch render: {len(image_names)} images × {len(video_names)} videos = {total} total combinations")
    image_map = create_map_user_image()
    video_map = create_map_video()
    current_data = get_data("list_output")

    for image_name in image_names:
        for video_name in video_names:
            progress += 1
            combo_name = f"{image_name}_{video_name}"
            logger.info(f"🔄 Processing [{progress}/{total}]: {combo_name}")

            image_id = image_map[image_name].get("message_id")
            video_id = video_map[video_name].get("message_id")
            output_exist = check_output_exist(f"{image_id}_{video_id}", current_data)

            try:
                if output_exist[0]:
                    logger.info(f"⚡ Output {combo_name} already exists, skipping")
                    skip += 1
                    continue

                tracking_message_content = f"📊 Progress: {progress}/{total}\n✅ Success: {success} | ⚡ Skip: {skip} | ❌ Fail: {len(fail_list)}\n🔄 Current: {combo_name}"
                await edit_message(tracking_message, tracking_message_content)

                logger.info(f"📥 Downloading files for {combo_name}")
                input_image = await download_file(message_id=image_id, sub_path="./image", ext="jpg")
                input_video = await download_file(message_id=video_id, sub_path="./video", ext="mp4")
                output_path = f"{os_dir['output']}/{image_id}_{video_id}.mp4"

                if not os.path.exists(input_image) or not os.path.exists(input_video):
                    raise Exception("Input files not found after download")

                if os.path.exists(output_path):
                    logger.info(f"📤 Output file exists, sending to Telegram: {combo_name}")
                    output_thread = get_topic_id(image_id)
                    await send_video("output_chat_id", output_thread, output_path, combo_name)
                    logger.info(f"✅ Video sent successfully: {combo_name}")
                else:
                    logger.info(f"🎬 Starting video rendering: {combo_name}")
                    output_thread = get_topic_id(image_id)
                    await render_video(input_image, input_video, output_path)
                    logger.info(f"🎬 Video rendering completed: {combo_name}")

                    logger.info(f"📤 Sending rendered video to Telegram: {combo_name}")
                    await send_video("output_chat_id", output_thread, output_path, combo_name)
                    logger.info(f"✅ Video sent successfully: {combo_name}")

                await send_message("input_chat_id", thread_map["etcd"], f"✅ {combo_name} completed successfully")
                sh.worksheet("list_output").append_row([f"{image_id}_{video_id}", combo_name])
                success += 1
                logger.info(f"✅ Successfully completed: {combo_name}")

            except Exception as e:
                logger.error(f"❌ FAILED processing {combo_name}: {str(e)}")
                await send_message("group", thread_map["etcd"], f"❌ {combo_name} failed: {str(e)}")
                fail_list.append(combo_name)

            finally:
                tracking_message_content = f"📊 Progress: {progress}/{total}\n✅ Success: {success} | ⚡ Skip: {skip} | ❌ Fail: {len(fail_list)}"
                if fail_list:
                    tracking_message_content += f"\n\n❌ Failed items:\n{chr(10).join(fail_list)}"
                await edit_message(tracking_message, tracking_message_content)

    logger.info(f"🏁 Batch processing completed!")
    logger.info(f"   📊 Total processed: {progress}/{total}")
    logger.info(f"   ✅ Successful: {success}")
    logger.info(f"   ⚡ Skipped: {skip}")
    logger.info(f"   ❌ Failed: {len(fail_list)}")
    if fail_list:
        logger.info(f"   📝 Failed items: {', '.join(fail_list)}")

def is_topic_reply(topic, event):
    if event.reply_to.reply_to_msg_id == entity_map["input_chat_id"]["threads"][topic]:
        return True
    return False

# ============================================================================
# EVENT HANDLERS
# ============================================================================
@client.on(events.NewMessage(chats=get_entity_id("input_chat_id"), from_users=get_entity_id("user_id")))
async def new_photo_handler(event):
    logger.info(f"📨 New message received from user {get_entity_id('user_id')}")

    if is_topic_reply("faces", event):
        if not event.photo:
            logger.info("❌ Message in faces topic but no photo found")
            return
        logger.info("🖼️ Processing photo in faces topic")
        if event.message.text:
            await handle_image_has_text(event)
        else:
            await handle_image_no_text(event)

    if is_topic_reply("vid", event):
        if not event.video:
            logger.info("❌ Message in vid topic but no video found")
            return
        logger.info("🎥 Processing video in vid topic")
        if event.message.text:
            await handle_input_video_has_text(event)
        else:
            await handle_input_video_no_text(event)

@client.on(events.MessageEdited(chats=get_entity_id("input_chat_id"), from_users=get_entity_id("user_id")))
async def message_edited_handler(event):
    logger.info(f"✏️ Message edited by user {get_entity_id('user_id')}")

    if is_topic_reply("faces", event):
        if event.photo and event.message.text:
            logger.info("🖼️ Processing edited photo with text in faces topic")
            await handle_image_has_text(event)
        return

    if is_topic_reply("vid", event):
        if event.video and event.message.text:
            logger.info("🎥 Processing edited video with text in vid topic")
            await handle_input_video_has_text(event)
        return

@client.on(events.NewMessage(chats=get_entity_id("group"), from_users=get_entity_id("user_id"), pattern='/domany .*'))
async def domany_handler(event):
    logger.info(f"🎯 Received /domany command from user")
    await domany(event)

@client.on(events.NewMessage(chats=get_entity_id("group"), pattern='/get_chat_id'))
async def get_chat_id(event):
    logger.info(f"🆔 Chat ID requested")
    logger.info(f"Event details: {event.stringify()}")
    await event.reply(f"💬 Chat ID: {event.chat_id}")

@client.on(events.NewMessage(chats=get_entity_id("group"), pattern='/getres'))
async def get_resource(event):
    logger.info(f"📋 Resource list requested")
    image_list = create_map_user_image()
    video_list = create_map_video()
    image_list_key = ",".join(image_list.keys())
    video_list_key = ",".join(video_list.keys())
    logger.info(f"📸 Available images: {len(image_list)} items")
    logger.info(f"🎥 Available videos: {len(video_list)} items")
    await event.reply(f"📋 **Available Resources:**\n\n📸 **Images:** {image_list_key}\n\n🎥 **Videos:** {video_list_key}")

@client.on(MessageDeleted(chats=get_entity_id("input_chat_id")))
async def message_deleted_handler(event):
    logger.info(f"🗑️ Message deletion event detected in input_chat_id")
    await handle_message_deleted(event)

@client.on(events.NewMessage(chats=get_entity_id("group"), from_users=get_entity_id("user_id"), pattern='/delete_image .*'))
async def delete_image_command(event):
    """Handle manual image deletion by name"""
    try:
        command_parts = event.message.text.split(" ", 1)
        if len(command_parts) != 2:
            await event.reply("❌ Usage: /delete_image <image_name>")
            return

        image_name = command_parts[1].lower().strip()
        logger.info(f"🗑️ Manual image deletion requested: {image_name}")

        data = get_data("list_image")
        found = False
        for i in range(len(data)):
            if data[i][1] == image_name:
                message_id = data[i][0]
                row_number = i + 2
                sh.worksheet("list_image").delete_rows(row_number)

                output_count, deleted_outputs = delete_outputs_by_image_id(message_id)

                success_msg = f"✅ Deleted image '{image_name}' (ID: {message_id})"
                if output_count > 0:
                    success_msg += f"\n🧹 Also deleted {output_count} related output(s)"

                await event.reply(success_msg)
                logger.info(f"✅ Successfully deleted image '{image_name}' manually")
                found = True
                break

        if not found:
            await event.reply(f"❌ Image '{image_name}' not found")
            logger.warning(f"⚠️ Manual deletion failed: image '{image_name}' not found")

    except Exception as e:
        logger.error(f"❌ Error in manual image deletion: {e}")
        await event.reply(f"❌ Error deleting image: {str(e)}")

@client.on(events.NewMessage(chats=get_entity_id("group"), from_users=get_entity_id("user_id"), pattern='/delete_video .*'))
async def delete_video_command(event):
    """Handle manual video deletion by name"""
    try:
        command_parts = event.message.text.split(" ", 1)
        if len(command_parts) != 2:
            await event.reply("❌ Usage: /delete_video <video_name>")
            return

        video_name = command_parts[1].lower().strip()
        logger.info(f"🗑️ Manual video deletion requested: {video_name}")

        data = get_data("list_video")
        found = False
        for i in range(len(data)):
            if data[i][1] == video_name:
                message_id = data[i][0]
                row_number = i + 2
                sh.worksheet("list_video").delete_rows(row_number)

                await event.reply(f"✅ Deleted video '{video_name}' (ID: {message_id})")
                logger.info(f"✅ Successfully deleted video '{video_name}' manually")
                found = True
                break

        if not found:
            await event.reply(f"❌ Video '{video_name}' not found")
            logger.warning(f"⚠️ Manual deletion failed: video '{video_name}' not found")

    except Exception as e:
        logger.error(f"❌ Error in manual video deletion: {e}")
        await event.reply(f"❌ Error deleting video: {str(e)}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================
async def main():
    """Main function to run the Telegram bot"""
    log_separator("STARTING TELEGRAM CLIENT")
    logger.info("🚀 Starting Telegram clients...")

    async with client, personal_client:
        await client.start()
        await personal_client.start()
        logger.info("✅ Both clients started successfully")
        logger.info("👂 Bot is now listening for events...")
        logger.info("🔧 Available commands:")
        logger.info("   • /domany <images> <videos> - Start batch processing")
        logger.info("   • /get_chat_id - Get current chat ID")
        logger.info("   • /getres - List available resources")
        logger.info("   • /delete_image <name> - Manually delete image by name")
        logger.info("   • /delete_video <name> - Manually delete video by name")
        logger.info("🗑️ Auto-deletion enabled: Deleting messages will remove entries from sheets")
        log_separator()
        print("Start Polling")

        try:
            await asyncio.sleep(6000000)
        except asyncio.CancelledError:
            logger.info("⚠️ Received cancellation signal")
        except KeyboardInterrupt:
            logger.info("⚠️ Received keyboard interrupt")
        finally:
            logger.info("🛑 Stopping bot...")
            print("Stopping")
            await client.disconnect()
            logger.info("✅ Bot stopped successfully")
            print("Stopped")

if __name__ == "__main__":
    asyncio.run(main())
