# samples-tts-gemini

A multilingual, dual-gender audio dataset synthesized via Google's [`gemini-3.8-flash-tts`](https://ai.google.dev/gemini-api/docs/speech-generation) API, designed for benchmarking multimodal classification and zero-token decision models (such as [Jev / System One Models](https://typesafe.ai/blog/introducing-system-one-models-and-jev)).

---

## Overview & Dataset Structure

The dataset consists of **160 `.wav` audio clips** (`24kHz`, `16-bit`, `mono` PCM) stored in [`output_wavs/`](./output_wavs), generated from [`sentences.json`](./sentences.json) using [`generate_tts_batch.py`](./generate_tts_batch.py).

* **10 Universal Categories:** Travel & Booking, Retail & E-Commerce, Weather & Environment, Business & Finance, Medical & Appointment, Construction & Architecture, Agriculture & Food Supply, Library & Education, Energy & Utilities, and Event & Conference.
* **8 Locales:** `en-US`, `pt-BR`, `es-ES`, `fr-FR`, `de-DE`, `hi-IN`, `ja-JP`, and `zh-CN`.
* **2 Speaker Genders per Locale:** 1 male voice and 1 female voice per language (16 distinct prebuilt voices total).
* **Duration & Content:** Each clip is **10 to 20 seconds** long (~25–35 written words) and embeds cardinal numbers, dates, timestamps, and currency amounts to test phonetic variety and text normalization across languages without relying on cultural idioms.

### Voice Matrix

| Locale | Language | Male Voice ID | Female Voice ID |
| :--- | :--- | :--- | :--- |
| `en-US` | English (US) | `Arlo` | `Enya` |
| `pt-BR` | Portuguese (Brazil) | `pt-br-advisor-10` | `pt-br-advisor-1` |
| `es-ES` | Spanish (Spain) | `es-es-advisor-2` | `es-es-advisor-8` |
| `fr-FR` | French (France) | `fr-fr-concierge-8` | `fr-fr-podcaster-10` |
| `de-DE` | German (Germany) | `de-de-podcaster-1` | `de-de-assistant-2` |
| `hi-IN` | Hindi (India) | `hi-in-concierge-10` | `hi-in-commercial-3` |
| `ja-JP` | Japanese (Japan) | `ja-jp-commercial-5` | `ja-jp-csagent-4` |
| `zh-CN` | Chinese (Simplified) | `zh-cn-concierge-1` | `zh-cn-techagent-1` |

### Filename Convention

All generated files follow a deterministic naming pattern so ground-truth labels (`category`, `locale`, `gender`, `voice_name`) can be parsed directly from the filename:

```text
{category_idx}_{category_slug}_{locale}_{gender}_{voice_name}.wav
```

*Example:* `09_energy_utilities_hi-IN_female_hi-in-commercial-3.wav`

---

## Reproducing or Extending the Audio Dataset

To regenerate missing files or add new categories/voices:

```bash
export GEMINI_API_KEY="your-api-key"
python generate_tts_batch.py
```

* Uses the unary `:generateContent` endpoint to download complete RIFF `.wav` blobs in a single call.
* Automatically skips files that already exist in [`output_wavs/`](./output_wavs).
* Includes built-in rate-limit pacing (`10 RPM` Tier 1 sliding window) and automatic `429` retry handling.

---

## Running the Multimodal Decision Benchmark

For this evaluation, we test zero-output-token audio classification (`output_tokens: 0`) using the [LiquidAI/d1-Omni-2B-GGUF](https://huggingface.co/LiquidAI/d1-Omni-2B-GGUF) model served via [`llama.cpp`](https://github.com/ggml-org/llama.cpp) / [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

Start the server:

```bash
./build/bin/llama-server \
  -hf LiquidAI/d1-Omni-2B-GGUF:Q8_0 \
  -b 8192 \
  -ub 8192
```

Each request sends a Base64-encoded `.wav` clip to `/v1/systemone` and evaluates two simultaneous tasks in a single forward pass:
1. **`male_speaker` (`noul`):** Continuous probability score (`0.0` = Female, `1.0` = Male).
2. **`spoken_language` (`choice`):** Softmax probability distribution across the 8 candidate locales.

---

## Example 1 (`hi-IN`, Female)

* **Audio Sample:** [`output_wavs/09_energy_utilities_hi-IN_female_hi-in-commercial-3.wav`](./output_wavs/09_energy_utilities_hi-IN_female_hi-in-commercial-3.wav)
* **Ground Truth:** Hindi (`hi-IN`), Female speaker (`male_speaker` expected `< 0.5`)

### Request

```bash
curl http://127.0.0.1:9931/v1/systemone -H "Content-Type: application/json" -d @- <<JSON
{
  "files": [
    "data:audio/wav;base64,$(base64 < output_wavs/09_energy_utilities_hi-IN_female_hi-in-commercial-3.wav | tr -d '\n')"
  ],
  "questions": {
    "male_speaker": {
      "criteria": {
        "true": "Speaker is male",
        "false": "Speaker is female"
      },
      "instructions": "Which is the most likely speaker gender?",
      "type": "noul"
    },
    "spoken_language": {
      "criteria": {
        "en-US": "English (US)",
        "es-ES": "Spanish (ES)",
        "ja-JP": "Japanese (JP)",
        "pt-BR": "Portuguese (BR)",
        "fr-FR": "French (FR)",
        "de-DE": "German (DE)",
        "hi-IN": "Hindi (IN)",
        "zh-CN": "Chinese (CN)"
      },
      "instructions": "What language is this?",
      "type": "choice"
    }
  },
  "state": "Audio recording from a single person speaking"
}
JSON
```

### Response

The model predicts `hi-IN` with **94.53% confidence** and scores `male_speaker` at **0.2307** (correctly classifying the voice as female) with `0` generated output tokens:

```json
{
  "model": "LiquidAI/d1-Omni-2B-GGUF:Q8_0",
  "answers": {
    "male_speaker": {
      "type": "noul",
      "noul": 0.23067223795637065
    },
    "spoken_language": {
      "type": "choice",
      "choice": "hi-IN",
      "probabilities": {
        "en-US": 0.015097171007290444,
        "es-ES": 0.00856913693035878,
        "ja-JP": 0.006051132003147758,
        "pt-BR": 0.007291487200858344,
        "fr-FR": 0.003672923601022524,
        "de-DE": 0.005694933698612155,
        "hi-IN": 0.9452862100663423,
        "zh-CN": 0.008337005492367745
      },
      "confidence": 0.9452862100663423
    }
  },
  "usage": {
    "input_tokens": 494,
    "output_tokens": 0
  }
}
```

---

## Example 2 (`fr-FR`, Male)

* **Audio Sample:** [`output_wavs/02_retail_e_commerce_fr-FR_male_fr-fr-concierge-8.wav`](./output_wavs/02_retail_e_commerce_fr-FR_male_fr-fr-concierge-8.wav)
* **Ground Truth:** French (`fr-FR`), Male speaker (`male_speaker` expected `> 0.5`)

### Request

```bash
curl http://127.0.0.1:9931/v1/systemone -H "Content-Type: application/json" -d @- <<JSON
{
  "files": [
    "data:audio/wav;base64,$(base64 < output_wavs/02_retail_e_commerce_fr-FR_male_fr-fr-concierge-8.wav | tr -d '\n')"
  ],
  "questions": {
    "male_speaker": {
      "criteria": {
        "true": "Speaker is male",
        "false": "Speaker is female"
      },
      "instructions": "Which is the most likely speaker gender?",
      "type": "noul"
    },
    "spoken_language": {
      "criteria": {
        "en-US": "English (US)",
        "es-ES": "Spanish (ES)",
        "ja-JP": "Japanese (JP)",
        "pt-BR": "Portuguese (BR)",
        "fr-FR": "French (FR)",
        "de-DE": "German (DE)",
        "hi-IN": "Hindi (IN)",
        "zh-CN": "Chinese (CN)"
      },
      "instructions": "What language is this?",
      "type": "choice"
    }
  },
  "state": "Audio recording from a single person speaking"
}
JSON
```

### Response

The model predicts `fr-FR` with **94.58% confidence** and scores `male_speaker` at **0.6581** (correctly classifying the voice as male):

```json
{
  "answers": {
    "male_speaker": {
      "noul": 0.6581467066511919,
      "type": "noul"
    },
    "spoken_language": {
      "choice": "fr-FR",
      "confidence": 0.9458058774324632,
      "probabilities": {
        "de-DE": 0.008348496537373485,
        "en-US": 0.01146728321137657,
        "es-ES": 0.008435167990366683,
        "fr-FR": 0.9458058774324632,
        "hi-IN": 0.006495068289679596,
        "ja-JP": 0.0058435602510993785,
        "pt-BR": 0.007813059497121412,
        "zh-CN": 0.005791486790519572
      },
      "type": "choice"
    }
  },
  "model": "LiquidAI/d1-Omni-2B-GGUF:Q8_0",
  "usage": {
    "input_tokens": 440,
    "output_tokens": 0
  }
}
```

## 👨‍💻 Author

Code and audio files by [Raoni Gabriel](https://github.com/raonigabriel)  
🔗 [LinkedIn](https://www.linkedin.com/in/raonigabriel/)

---

## 📜 License

[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
