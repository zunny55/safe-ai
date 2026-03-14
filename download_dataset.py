from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="keremberke/ppe-detection",
    repo_type="dataset",
    local_dir="ppe-dataset"
)
