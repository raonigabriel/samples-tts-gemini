import os
import re
import sys
import json
import time
import base64
import requests

# ---------------------------------------------------------------------------
# 1. Configuration & Rate Limit Settings
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    sys.exit("Error: GEMINI_API_KEY environment variable is not set.")

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash-tts:generateContent"

INPUT_JSON_PATH = "sentences.json"
OUTPUT_DIR = "output_wavs"

RPM_LIMIT = 10
MIN_REQUEST_INTERVAL = (60.0 / RPM_LIMIT) + 0.5  # 6.5 seconds
MAX_RETRIES = 5

# ---------------------------------------------------------------------------
# 2. Voice Lookup Map: Dict[locale, Dict[gender, voice_name]]
# ---------------------------------------------------------------------------
VOICE_MAP: dict[str, dict[str, str]] = {
    "pt-BR": {
        "male": "pt-br-advisor-10",
        "female": "pt-br-advisor-1",
    },
    "en-US": {
        "male": "Arlo",
        "female": "Enya",
    },
    "es-ES": {
        "male": "es-es-advisor-2",
        "female": "es-es-advisor-8",
    },
    "fr-FR": {
        "male": "fr-fr-concierge-8",
        "female": "fr-fr-podcaster-10",
    },
    "de-DE": {
        "male": "de-de-podcaster-1",
        "female": "de-de-assistant-2",
    },
    "hi-IN": {
        "male": "hi-in-concierge-10",
        "female": "hi-in-commercial-3",
    },
    "ja-JP": {
        "male": "ja-jp-commercial-5",
        "female": "ja-jp-csagent-4",
    },
    "zh-CN": {
        "male": "zh-cn-concierge-1",
        "female": "zh-cn-techagent-1",
    },
}


# ---------------------------------------------------------------------------
# 3. Helper Functions
# ---------------------------------------------------------------------------
def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def inspect_429(response_text: str) -> tuple[bool, float, str]:
    """
    Parses the 429 JSON error.
    Returns: (is_retriable, wait_seconds, human_summary)
    """
    try:
        err_json = json.loads(response_text).get("error", {})
        msg = err_json.get("message", response_text)
    except Exception:
        msg = response_text

    # Check if this is a Daily quota exhaustion (RPD)
    if "PerDay" in msg or "per_day" in msg or "per day" in msg.lower():
        return False, 0.0, msg.strip()

    # Extract "Please retry in Xs" if present
    match = re.search(r"Please retry in ([0-9.]+)s", msg)
    if match:
        return True, float(match.group(1)) + 2.0, msg.strip()

    # If no retry timer is given, still return the message so we can see why
    return True, 60.0, msg.strip()


def synthesize_speech(
    session: requests.Session, text: str, voice_name: str
) -> bytes:
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": f"## Transcript:\n{text}"
                    }
                ],
            }
        ],
        "generationConfig": {
            "responseModalities": ["audio"],
            "temperature": 1,
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": voice_name
                    }
                }
            },
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        response = session.post(
            API_URL,
            params={"key": GEMINI_API_KEY},
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )

        if response.status_code == 429:
            is_retriable, wait_sec, summary = inspect_429(response.text)
            print(f"\n[429 DETAILS] {summary}\n")

            if not is_retriable:
                sys.exit("[FATAL] Daily quota (RPD) exhausted for this API key/project. Exiting.")

            print(
                f"[429 ] Quota hit (attempt {attempt}/{MAX_RETRIES}). "
                f"Sleeping {wait_sec:.1f}s before retry..."
            )
            time.sleep(wait_sec)
            continue

        response.raise_for_status()

        data = response.json()
        b64_audio = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        return base64.b64decode(b64_audio)

    raise RuntimeError(f"Exceeded {MAX_RETRIES} retries due to HTTP 429 rate limits.")


# ---------------------------------------------------------------------------
# 4. Main Batch Loop
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    session = requests.Session()
    last_request_ts = 0.0

    for cat_idx, item in enumerate(dataset, start=1):
        category = item["category"]
        cat_slug = slugify(category)

        for sentence_obj in item["sentences"]:
            locale = sentence_obj["locale"]
            text = sentence_obj["text"]

            voices_for_locale = VOICE_MAP.get(locale)
            if not voices_for_locale:
                print(f"[SKIP] No voice mapping configured for locale '{locale}'")
                continue

            for gender, voice_name in voices_for_locale.items():
                filename = f"{cat_idx:02d}_{cat_slug}_{locale}_{gender}_{voice_name}.wav"
                filepath = os.path.join(OUTPUT_DIR, filename)

                if os.path.exists(filepath):
                    print(f"[SKIP] File already exists: {filename}")
                    continue

                elapsed = time.monotonic() - last_request_ts
                if last_request_ts > 0 and elapsed < MIN_REQUEST_INTERVAL:
                    sleep_time = MIN_REQUEST_INTERVAL - elapsed
                    time.sleep(sleep_time)

                print(f"[RUN ] {category} | {locale} | {gender} ({voice_name}) -> {filename}")
                last_request_ts = time.monotonic()

                try:
                    wav_bytes = synthesize_speech(session, text, voice_name)
                    with open(filepath, "wb") as out_file:
                        out_file.write(wav_bytes)
                    duration_sec = len(wav_bytes) / 48000
                    print(f"[ OK ] Saved {len(wav_bytes):,} bytes (~{duration_sec:.1f}s) to {filepath}")
                except requests.HTTPError as http_err:
                    print(f"[ERR ] HTTP {http_err.response.status_code}: {http_err.response.text}")
                except Exception as err:
                    print(f"[ERR ] Failed on {filename}: {err}")


if __name__ == "__main__":
    main()
