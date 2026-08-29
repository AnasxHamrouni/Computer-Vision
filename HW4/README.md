# HW4 — Autonomous 3D Gaussian Splatting Scene Exploration

Interactive viewer and Python pipeline for exploring 3D Gaussian Splatting (3DGS) scenes: cinematic camera paths, collision-aware indoor navigation, and 2D/3D object detection.

The web viewer (`index.html`) loads the two scene PLYs that GitHub can host. The unpacked conference-hall cloud, the extra outdoor PLY, and the demo recordings are too large for git and are linked below.

## What's in this folder

| Path | Purpose |
|------|---------|
| `index.html` | Spark.js / Three.js viewer: scene switcher, trajectory playback, detection overlay |
| `core/` | Path generation, PLY unpacking, YOLO 2D detection, geometric 3D detection |
| `scenes/outdoor-drone/` | Outdoor drone PLY (~49 MB) + trajectories |
| `scenes/conference-hall/` | Packed indoor PLY (~95 MB) + indoor trajectory |
| `detections/` | 2D/3D detection JSON and preview images |
| `TASK_COMPLETION_REPORT.md` | Feature-by-feature writeup |

## Quick start — viewer

The viewer needs a local HTTP server (browsers block PLY loads from `file://`):

```bash
cd HW4
python -m http.server 8000
```

Open [http://localhost:8000](http://localhost:8000), pick **Outdoor Drone** or **Conference Hall**, and play the trajectory.

## Python setup

```bash
cd HW4
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Indoor path example (needs the unpacked PLY from Drive — see below):

```bash
python core/generate_indoor_path.py \
  --ply scenes/conference-hall/ConferenceHall_unpacked.ply \
  --output scenes/conference-hall/trajectory_indoor.json
```

See `core/README_indoor_navigation.md` and `core/README_detection.md` for the rest of the CLI.

## Large files (not in git)

GitHub rejects any **single file over 100 MB**. These stay out of the repo:

| File | Size | Why you need it |
|------|------|-----------------|
| `scenes/conference-hall/ConferenceHall_unpacked.ply` | ~1.4 GB | Standard XYZ cloud for the indoor path generator |
| `outdoor-standard.ply` | ~170 MB | Extra outdoor splat (viewer uses `scenes/outdoor-drone/` instead) |
| Demo screen recordings | ~45–85 MB each | Walkthrough videos for the report / portfolio |

### 1. Google Drive (already uploaded)

Shared folder (archive + both demo videos):

**[CV — Autonomous 3DGS Scene Exploration](https://drive.google.com/drive/folders/1u__uYojc0tdCILr5VsDkYwwH3azczJ1z?usp=share_link)**

| Drive item | Size | Use |
|------------|------|-----|
| `3DGSexploration_01.zip` | 1.07 GB | Full project archive — unzip and copy the large PLYs into this `HW4/` tree |
| `Screen Recording 2025-12-01 at 21.05.26.mov` | 85 MB | Demo video |
| `Screen Recording 2025-12-01 at 23.55.18.mov` | 45 MB | Demo video |

After unzipping the archive, copy at least:

```text
ConferenceHall_unpacked.ply  →  HW4/scenes/conference-hall/
outdoor-standard.ply         →  HW4/          (optional)
```

`CodeBase` in this folder is the same Drive link, kept for the original homework hand-in.

### 2. GitHub Releases (good alternative)

If you want everything on GitHub without putting blobs in git history:

1. Open the repo → **Releases** → **Create a new release** (e.g. tag `hw4-assets`).
2. Attach files up to **2 GB each** (`ConferenceHall_unpacked.ply` fits; the Drive zip also fits).
3. Link the release from this README.

Anyone can then download with:

```bash
# after installing GitHub CLI: brew install gh
gh release download hw4-assets --repo AnasxHamrouni/Computer-Vision --dir ./
```

### 3. Hugging Face Hub (best for ML / 3D assets)

Create a dataset repo (free, designed for large binaries), then:

```bash
huggingface-cli upload <you>/3dgs-exploration ConferenceHall_unpacked.ply
```

Download:

```bash
huggingface-cli download <you>/3dgs-exploration ConferenceHall_unpacked.ply \
  --local-dir scenes/conference-hall
```

### 4. YouTube / Streamable (recordings only)

Unlisted YouTube links are the usual portfolio pattern: they play in the README, do not count against git size, and recruiters do not have to download 80 MB zips.

---

**Recommendation:** keep using the [existing Drive folder](https://drive.google.com/drive/folders/1u__uYojc0tdCILr5VsDkYwwH3azczJ1z?usp=share_link) for the 1.4 GB PLY and videos. Use GitHub Releases or Hugging Face only if you want a second, GitHub-adjacent copy.
