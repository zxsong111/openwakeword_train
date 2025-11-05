import requests
import os
import time
from dotenv import load_dotenv
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm   # ✅ 新增：进度条支持
import logging           # ✅ 新增：日志记录

# ========= 初始化配置 =========
load_dotenv()

SAVE_PATH = "./cartesia_hi_adela_test"
USED_VOICES_FILE = os.path.join(SAVE_PATH, "used_voices.txt")
LOG_FILE = os.path.join(SAVE_PATH, "generate_log.txt")
MAX_WORKERS = 10          # 并发线程数
RETRY_TIMES = 3           # 请求失败重试次数
GENERATE_COUNT = 10       # 每个 voice 生成的音频数

# ========= 日志配置 =========
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ========= Session 复用 =========
session = requests.Session()
session.headers.update({
    "Cartesia-Version": "2025-04-16",
    "Authorization": f"Bearer {os.getenv('CARTESIA_API_KEY')}",
    "Content-Type": "application/json"
})


# ========= 基础函数 =========
def load_used_voices() -> set:
    """加载已使用 voice id"""
    if os.path.exists(USED_VOICES_FILE):
        with open(USED_VOICES_FILE, "r") as f:
            return set(line.strip() for line in f)
    return set()


def save_used_voice(voice_id: str):
    """追加保存 voice id"""
    with open(USED_VOICES_FILE, "a") as f:
        f.write(f"{voice_id}\n")


def get_voices(last_voice_id: str = None, limit: int = 10):
    """从 Cartesia API 获取 voice 列表"""
    url = "https://api.cartesia.ai/voices/"
    params = {"limit": limit}
    if last_voice_id:
        params["starting_after"] = last_voice_id

    for _ in range(RETRY_TIMES):
        try:
            response = session.get(url, params=params, timeout=15)
            response.raise_for_status()
            voices = response.json()["data"]
            return voices, voices[-1]["id"]
        except Exception as e:
            logging.warning(f"Fetch voices failed ({_+1}/{RETRY_TIMES}): {e}")
            time.sleep(2)
    return [], last_voice_id


def generate_wav(voice_id: str, target_phrase: str = "hi, adela."):
    """生成单个语音文件"""
    url = "https://api.cartesia.ai/tts/bytes"
    payload = {
        "model_id": "sonic-2",
        "transcript": target_phrase,
        "voice": {"mode": "id", "id": voice_id},
        "output_format": {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": 16000
        },
        "language": "en",
        "save": True
    }

    for attempt in range(RETRY_TIMES):
        try:
            response = session.post(url, json=payload, timeout=20)
            response.raise_for_status()
            file_name = f"{voice_id}_{uuid4()}.wav"
            file_path = os.path.join(SAVE_PATH, file_name)
            with open(file_path, "wb") as f:
                f.write(response.content)
            logging.info(f"SUCCESS: {voice_id} -> {file_name}")
            return file_path
        except Exception as e:
            logging.warning(f"Generate failed for {voice_id} ({attempt+1}/{RETRY_TIMES}): {e}")
            time.sleep(1)

    logging.error(f"FAILED: {voice_id}")
    return None


# ========= 主流程 =========
def main():
    used_voices = load_used_voices()
    last_voice_id = None

    print("🚀 Start generating voices...")
    for _ in range(1, 2000, 100):  # 分页
        voices, last_voice_id = get_voices(last_voice_id, 100)
        print(f"Fetched {len(voices)} voices...")
        if not voices:
            break

        english_voices = [
            v for v in voices if v["language"] == "en" and v["id"] not in used_voices
        ]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for voice in english_voices:
                voice_id = voice["id"]
                used_voices.add(voice_id)
                save_used_voice(voice_id)
                for _ in range(GENERATE_COUNT):
                    futures.append(executor.submit(generate_wav, voice_id))

            # ✅ tqdm 进度条显示
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating WAVs"):
                path = future.result()
                if path:
                    print(f"[OK] Saved: {path}")

    print("✅ All tasks completed. Check logs for details:", LOG_FILE)


if __name__ == "__main__":
    main()