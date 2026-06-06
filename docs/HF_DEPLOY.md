# Deploying ReguLumen to a HuggingFace Space (Docker SDK)

This runbook deploys the Gradio UI as a single-container HuggingFace **Docker**
Space. The Space builds its Chroma index from the bundled corpus on first boot
(ingest-on-startup), then serves the UI on port 7860.

> The UI imports the ReguLumen core **in-process**, so one container is a
> complete app. We do NOT expose the FastAPI service on a Space — HF Docker
> Spaces serve a single port.

---

## 1. Prerequisites

- A [HuggingFace account](https://huggingface.co/join).
- Either the `huggingface_hub` CLI (`pip install -U huggingface_hub`) **or** git
  with an HF access token configured as the password for `https` pushes.
- The two provider tokens (kept OUT of the repo — they go in as Space secrets):
  - `REGULUMEN_GROQ_API_KEY` — Groq LLM key (<https://console.groq.com/keys>)
  - `REGULUMEN_HF_API_TOKEN` — HuggingFace read token for embeddings
    (<https://huggingface.co/settings/tokens>). **Required at startup** for the
    ingest step to build the real index.

---

## 2. Create the Space

**Web UI (recommended):** New Space → choose **SDK: Docker** → pick a blank/empty
template → name it (e.g. `regulumen`). This creates
`https://huggingface.co/spaces/<user>/regulumen`.

**Or via CLI:**

```bash
huggingface-cli login
huggingface-cli repo create regulumen --type space --space_sdk docker
```

---

## 3. The Dockerfile-naming gotcha (read this)

**HF Docker Spaces build the file literally named `Dockerfile` at the repo
root** — there is no Space setting to point HF at a differently-named Dockerfile.
This repo intentionally keeps **two** Dockerfiles:

- `Dockerfile` — the LOCAL multi-service image (API + UI + ingest via
  docker-compose). **Do not change it.**
- `Dockerfile.hf` — the HF Space image (non-root uid 1000, single port 7860,
  ingest-on-startup).

So the Space needs `Dockerfile.hf`'s contents to live at `Dockerfile`. The
**least-error-prone** approach (and the one below) is a dedicated **`space`
deploy branch**: on that branch only, `Dockerfile` is the HF variant. Your local
`master`/`main` working tree is never touched, and you can re-deploy by
re-running the same steps.

---

## 4. Push to the Space (deploy branch approach)

Your work currently lives on the **`master`** branch. HF Spaces serve from
**`main`**. We build a throwaway `space` branch whose `Dockerfile` is the HF one,
and push it to the Space's `main`.

```bash
# From the repo root, with master checked out and clean.

# 1. Add the Space as a remote (one time).
git remote add space https://huggingface.co/spaces/<user>/regulumen

# 2. Create a deploy branch from your current work.
git checkout -b space

# 3. On THIS branch only, make Dockerfile.hf the active Dockerfile.
#    (master's Dockerfile is unaffected — this commit lives only on `space`.)
cp Dockerfile.hf Dockerfile
git add Dockerfile
git commit -m "deploy: use HF Dockerfile for the Space"

# 4. Push the deploy branch to the Space's main branch.
git push space space:main

# 5. Go back to your normal branch.
git checkout master
```

To re-deploy after changes: `git checkout space && git merge master && cp
Dockerfile.hf Dockerfile && git commit -am "redeploy" && git push space
space:main && git checkout master`.

> Alternative (manual): if you prefer not to keep a deploy branch, you can simply
> rename `Dockerfile.hf` → `Dockerfile` in a clone you push to the Space. The
> deploy-branch approach is preferred because it keeps the local Docker setup
> intact and makes re-deploys repeatable.

---

## 5. Set the Space SECRETS (never committed)

In the Space → **Settings → Variables and secrets → New secret**, add:

| Name | Value | Kind |
|---|---|---|
| `REGULUMEN_GROQ_API_KEY` | your Groq key | Secret |
| `REGULUMEN_HF_API_TOKEN` | your HF read token | Secret |

These are injected as environment variables at runtime; the app reads them via
pydantic-settings (`REGULUMEN_` prefix). **Do not** put them in the repo, the
Dockerfile, or any committed file. Any non-secret config (e.g.
`REGULUMEN_ANSWERER_MODE`) can be added as a plain **Variable** here too.

---

## 6. First-boot expectation

- The Space builds the image, then runs `scripts/hf_startup.sh`: it ingests the
  bundled corpus into the Chroma index, then launches the UI.
- Expect roughly a **1–2 min cold start** while it ingests, then the UI appears
  at the Space URL.
- **Graceful degradation:** if `REGULUMEN_HF_API_TOKEN` is missing or the
  provider is unreachable, ingest fails but the Space **still boots** — the UI
  loads and shows a "backend unavailable / set credentials and restart" message
  per request. Fix: add the secrets in Settings, then **Restart** the Space (or
  Factory reboot) to rebuild the index.

---

## 7. Troubleshooting

- **Space won't start / crash-loops:** confirm the front-matter in `README.md`
  has `sdk: docker` and `app_port: 7860`, and that `Dockerfile` on the Space is
  the HF variant (step 3/4).
- **UI loads but every answer says "backend unavailable":** the secrets are
  missing or wrong — add them (step 5) and restart.
- **Permission/`Read-only file system` errors:** the index path must be
  user-writable; `Dockerfile.hf` already sets
  `REGULUMEN_CHROMA_PATH=/home/user/app/.chroma` under the uid-1000 home. Don't
  override it to a root-owned path.
