from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="jhboyo/ppe-dataset",
    local_dir="ppe-dataset"
)
