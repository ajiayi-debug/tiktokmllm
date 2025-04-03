from huggingface_hub import hf_hub_download

# Replace with the actual filename if different
repo_id = "Fahad-S/videobench_cotv01"
filename = "videos.zip"

# Download the file
file_path = hf_hub_download(repo_id=repo_id, filename=filename)

print("Downloaded to:", file_path)
