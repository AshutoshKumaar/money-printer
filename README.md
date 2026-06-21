# AI Hindi Shorts Automation System

Production-ready pipeline for faceless Hindi Shorts:

`Topic -> Gemini script -> AI image prompts -> AI images -> Edge TTS voiceover -> captions -> vertical video -> YouTube upload`

## Features

- Gemini 2.5 Flash script and metadata generation.
- 55-60 second Hindi Shorts with 10-12 structured scenes by default.
- Complete fallback script if Gemini script generation fails.
- AI image generation per segment with vertical 9:16 enforcement.
- Relevance-first hybrid visual matching with per-scene category, keywords, provider, and confidence score.
- Prompt-hash visual cache and a configurable Hugging Face image budget per video.
- Relevant Pexels photo fallback and category-aware local visuals when external providers fail.
- Cinematic zoom, pan, caption overlays, transitions, and background music mix.
- Cached reusable background music.
- Per-run metadata saved in `storage/metadata/`.
- Topic history tracking to avoid repeating recent automated topics.
- Final videos saved only in `final_shorts/`.
- YouTube Data API v3 OAuth upload with thumbnail support.
- Daily scheduler at 6:00 PM by default.
- Docker, Docker Compose, Railway, and Render deployment files.

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Create your environment file.

```powershell
Copy-Item .env.example .env
```

3. Fill `.env`.

Required for generation:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Required for upload/default/scheduled runs:

```env
YOUTUBE_CLIENT_SECRETS_FILE=storage/credentials/client_secret.json
YOUTUBE_CLIENT_SECRETS_JSON=
YOUTUBE_CLIENT_SECRETS_BASE64=
YOUTUBE_TOKEN_FILE=storage/credentials/youtube_token.json
```

Download OAuth client secrets from Google Cloud Console, enable YouTube Data API v3, and place the file at `storage/credentials/client_secret.json`. The first authentication opens the OAuth consent flow and stores the refresh token in `storage/credentials/youtube_token.json`, which is gitignored.

## YouTube Authentication

First-time local authentication:

```powershell
python main.py --auth-only
```

What happens:

- The app detects `storage/credentials/client_secret.json`.
- If `storage/credentials/youtube_token.json` already exists and is valid, it is reused.
- If the access token is expired and has a refresh token, it is refreshed automatically.
- If the token is missing, your browser opens for Google OAuth consent.
- After consent, `storage/credentials/youtube_token.json` is created.

After `youtube_token.json` exists, scheduled uploads do not need browser authentication again.

For Railway, Render, VPS, or any headless server:

- Generate `youtube_token.json` locally once with `python main.py --auth-only`.
- Deploy the OAuth client secret and token via persistent storage, mounted secret file, or environment secrets.
- Supported environment secret methods:

```env
YOUTUBE_CLIENT_SECRETS_JSON={"installed":{...}}
YOUTUBE_TOKEN_JSON={"token":"...","refresh_token":"..."}
```

or base64:

```env
YOUTUBE_CLIENT_SECRETS_BASE64=base64_encoded_client_secret_json
YOUTUBE_TOKEN_BASE64=base64_encoded_youtube_token_json
```

If the app detects Railway, Render, CI, or another headless environment and no valid token is available, it fails with a clear error instead of trying to open a browser.

## Commands

Generate script and metadata only:

```powershell
python main.py "3 Creepy Space Mysteries" --dry-run
```

Generate video locally without upload:

```powershell
python main.py "3 Creepy Space Mysteries" --generate-only
```

Generate and upload:

```powershell
python main.py "3 Creepy Space Mysteries"
```

Upload an existing render:

```powershell
python main.py --upload-only --video-path final_shorts/video.mp4 --metadata-path storage/metadata/video.metadata.json
```

Authenticate YouTube only:

```powershell
python main.py --auth-only
```

Run daily at 6:00 PM locally:

```powershell
python main.py --schedule
```

Use existing assets for a run folder when files already exist:

```powershell
python main.py "Topic" --generate-only --use-existing-assets
```

## Project Layout

```text
config/                 environment-backed settings and validation
core/                   typed models, retry, logging, orchestration pipeline
services/               Gemini, image, voice, caption, video, YouTube, notifications
scheduler/              daily scheduler
storage/credentials/    OAuth token and client secrets, gitignored
storage/metadata/       per-run metadata JSON
logs/                   rotating application logs
final_shorts/           final MP4 videos and thumbnails
assets/                 generated per-run audio/images and cached music
```

## Hybrid Visual System

The visual pipeline prioritizes narration relevance over image quality:

1. Analyze narration, subtitles, topic, and image prompt.
2. Extract keywords and classify the scene as space, ocean, history, person, technology, nature, or horror.
3. Build a category-specific search query and generation prompt.
4. Check the prompt/category cache.
5. Use a limited number of Hugging Face images.
6. Search Pexels with the relevant query when `PEXELS_API_KEY` is configured.
7. Try Pollinations.
8. Generate a category-aware local fallback instead of an unrelated random image.

Each run writes `visual_manifest.json` inside its image directory. Final metadata also records:

- `visual_category`
- `visual_provider`
- `visual_confidence`

Relevant configuration:

```env
IMAGE_PROVIDER=hybrid
HF_MAX_IMAGES_PER_VIDEO=3
VISUAL_MIN_CONFIDENCE=0.65
VISUAL_CACHE_DIR=storage/visual_cache
PEXELS_MAX_RESULTS=8
```

## Deployment

Docker:

```powershell
docker compose up --build
```

Railway uses `railway.json`; Render uses `render.yaml`. Configure secrets in the platform dashboard:

- `GEMINI_API_KEY`
- `YOUTUBE_CLIENT_SECRETS_FILE=storage/credentials/client_secret.json`
- `YOUTUBE_TOKEN_FILE=storage/credentials/youtube_token.json`
- `YOUTUBE_TOKEN_JSON` or `YOUTUBE_TOKEN_BASE64`, unless using persistent disk/file storage

For Railway automation:

1. Push this repo to GitHub.
2. Create a Railway project from the GitHub repo.
3. Add the environment variables from `.env.example` in the service `Variables` tab. Use Railway's raw editor if you want to paste many variables at once.
4. Generate `youtube_token.json` locally first with `python main.py --auth-only`.
5. Put `client_secret.json` into Railway as `YOUTUBE_CLIENT_SECRETS_BASE64` or `YOUTUBE_CLIENT_SECRETS_JSON`.
6. Put `youtube_token.json` into Railway as `YOUTUBE_TOKEN_BASE64` or `YOUTUBE_TOKEN_JSON`.
7. Keep `YOUTUBE_CLIENT_SECRETS_FILE=storage/credentials/client_secret.json` and `YOUTUBE_TOKEN_FILE=storage/credentials/youtube_token.json`.
8. Add a Railway volume mounted at `/app/storage` if you want credentials, visual cache, metadata, and generated outputs to persist between deployments.
9. In Railway service settings, set Cron Schedule to `30 12 * * *` for 6:00 PM India time.

Railway cron schedules use UTC, so 6:00 PM IST is 12:30 PM UTC. The Railway start command is intentionally `python main.py`, so each cron run generates one video, uploads it, sends notifications, and exits. Do not use `python main.py --schedule` for Railway cron, because Railway cron services should finish and exit after their task.

For cloud upload, generate `youtube_token.json` locally first and provide it through a secure secret or persistent volume. The scheduler/cron run can then generate, authenticate, refresh tokens, upload, log, and notify without manual intervention.

## Notes

- Default runs require YouTube credentials because the production workflow includes upload.
- Use `--generate-only` when you only want the local MP4.
- Logs are written to `logs/automation.log`.
- Generated credentials, outputs, logs, and media assets are intentionally gitignored.
