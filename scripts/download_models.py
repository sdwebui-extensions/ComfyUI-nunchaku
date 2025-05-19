import os

from huggingface_hub import hf_hub_download, snapshot_download

from nunchaku.utils import get_precision


def download_file(
    repo_id: str,
    filename: str,
    sub_folder: str,
    new_filename: str | None = None,
) -> str:
    assert os.path.isdir(os.path.join("models", sub_folder))
    target_folder = os.path.join("models", sub_folder)
    target_file = os.path.join(target_folder, filename if new_filename is None else new_filename)
    if not os.path.exists(target_file):
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=target_folder)
        if new_filename is not None:
            os.rename(os.path.join(target_folder, filename), os.path.join(target_folder, new_filename))
    return target_file


def download_original_models():
    download_file(
        repo_id="comfyanonymous/flux_text_encoders", filename="clip_l.safetensors", sub_folder="text_encoders"
    )
    download_file(
        repo_id="comfyanonymous/flux_text_encoders", filename="t5xxl_fp16.safetensors", sub_folder="text_encoders"
    )
    download_file(repo_id="black-forest-labs/FLUX.1-dev", filename="ae.safetensors", sub_folder="vae")
    download_file(
        repo_id="black-forest-labs/FLUX.1-dev", filename="flux1-dev.safetensors", sub_folder="diffusion_models"
    )
    download_file(
        repo_id="black-forest-labs/FLUX.1-schnell", filename="flux1-schnell.safetensors", sub_folder="diffusion_models"
    )


def download_svdquant_models():
    precision = get_precision()
    svdquant_models = [
        f"mit-han-lab/svdq-{precision}-shuttle-jaguar",
        f"mit-han-lab/svdq-{precision}-flux.1-schnell",
        f"mit-han-lab/svdq-{precision}-flux.1-dev",
        f"mit-han-lab/svdq-{precision}-flux.1-schnell",
        f"mit-han-lab/svdq-{precision}-flux.1-canny-dev",
        f"mit-han-lab/svdq-{precision}-flux.1-depth-dev",
        f"mit-han-lab/svdq-{precision}-flux.1-fill-dev",
    ]
    for model_path in svdquant_models:
        snapshot_download(
            model_path, local_dir=os.path.join("models", "diffusion_models", os.path.basename(model_path))
        )


def download_loras():
    download_file(
        repo_id="alimama-creative/FLUX.1-Turbo-Alpha",
        filename="diffusion_pytorch_model.safetensors",
        sub_folder="loras",
        new_filename="flux.1-turbo-alpha.safetensors",
    )

    download_file(
        repo_id="aleksa-codes/flux-ghibsky-illustration",
        filename="lora.safetensors",
        sub_folder="loras",
        new_filename="flux.1-dev-ghibsky.safetensors",
    )

    download_file(
        repo_id="black-forest-labs/FLUX.1-Depth-dev-lora",
        filename="flux1-depth-dev-lora.safetensors",
        sub_folder="loras",
    )
    download_file(
        repo_id="black-forest-labs/FLUX.1-Canny-dev-lora",
        filename="flux1-canny-dev-lora.safetensors",
        sub_folder="loras",
    )


if __name__ == "__main__":
    download_original_models()
    download_svdquant_models()
    download_loras()
