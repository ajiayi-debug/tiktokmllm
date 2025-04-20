#!/usr/bin/env python3
"""
Delete every Gemini‑API file whose display name matches a *.mp4
inside Benchmark-AllVideos-HQ-Encoded-challenge.
"""

import os
import pathlib
from google import genai
from dotenv import load_dotenv

# ---------- 1. local video basenames --------------------------------------
VID_DIR = pathlib.Path("Benchmark-AllVideos-HQ-Encoded-challenge")
local_videos = {p.name.lower() for p in VID_DIR.glob("*.mp4")}
print(f"Local video files found: {len(local_videos)}")

# ---------- 2. Gemini client (sync) ---------------------------------------
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------- 3. list ALL uploaded files (handle pagination) ----------------
files = []
token = ""
while True:
    resp = client.files.list(page_token=token)
    files.extend(resp.files)
    token = getattr(resp, "next_page_token", "")
    if not token:
        break
print(f"Uploaded files found: {len(files)}")

# ---------- 4. filter matches ---------------------------------------------
matches = []
for f in files:
    remote_name = (
        getattr(f, "display_name", None)
        or getattr(f, "displayName", None)      # older SDK field
        or f.uri.split("/")[-1]
    ).lower()
    if remote_name in local_videos:
        matches.append(f)

if not matches:
    print("No matching uploaded videos to delete. ✅")
    quit()

# ---------- 5. delete ------------------------------------------------------
print(f"Deleting {len(matches)} matching files …")
for f in matches:
    client.files.delete(name=f.name)
    print("–", getattr(f, "display_name", getattr(f, "displayName", f.name)))

print("✅ Finished. Storage quota will refresh in < 1 min.")