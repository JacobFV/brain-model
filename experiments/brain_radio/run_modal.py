#!/usr/bin/env python3
"""
Run Brain Radio on Modal with GPU.

Creates a sandbox in the "test-20260327" environment, installs TRIBE v2,
runs brain_radio.py, and pulls the output images back locally.
"""

import modal
modal.enable_output()
import base64
import sys
import os
from pathlib import Path

ENV_NAME = "test-20260327"
OUTPUT_DIR = Path("./output")

# ── Build the image ───────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1", "git", "libgl1", "libglib2.0-0", "libxrender1")
    .pip_install("torch", "torchaudio")
    .run_commands(
        "pip install git+https://github.com/facebookresearch/tribev2.git"
    )
    .pip_install(
        "gtts", "langdetect", "soundfile", "matplotlib",
        "nilearn", "pyvista", "vtk", "colorcet", "seaborn", "scikit-image",
    )
)


def run_in_sandbox(presets: list[str], hf_token: str | None = None):
    """Create a Modal sandbox, upload brain_radio.py, run it, and download results."""

    secrets = [modal.Secret.from_name("huggingface-secret", environment_name=ENV_NAME)]
    env = {}
    if hf_token:
        env["HF_TOKEN"] = hf_token

    app = modal.App.lookup("brain-radio", create_if_missing=True, environment_name=ENV_NAME)

    print(f"Creating sandbox in environment '{ENV_NAME}' with A10G GPU...")
    sb = modal.Sandbox.create(
        app=app,
        image=image,
        gpu="A10G",
        timeout=1800,  # 30 min
        cpu=4,
        memory=32768,  # 32 GB
        secrets=secrets,
        env=env,
    )
    print(f"Sandbox created: {sb.object_id}")

    try:
        # Upload brain_radio.py into the sandbox
        local_script = Path(__file__).parent / "brain_radio.py"
        script_content = local_script.read_text()

        sb.exec("bash", "-c", "mkdir -p /work/output /work/cache").wait()

        # Write script via base64 to avoid heredoc quoting issues
        encoded = base64.b64encode(script_content.encode()).decode()
        sb.exec("bash", "-c", f"echo '{encoded}' | base64 -d > /work/brain_radio.py").wait()

        # Run it
        preset_args = " ".join(presets)
        cmd = f"cd /work && python brain_radio.py {preset_args} --output-dir /work/output"
        print(f"Running: {cmd}")
        print("--- sandbox output ---")

        p = sb.exec("bash", "-c", cmd)
        # Stream stdout and stderr together
        for line in p.stdout:
            print(line, end="")
        for line in p.stderr:
            print(line, end="", file=sys.stderr)
        p.wait()

        if p.returncode != 0:
            print(f"\nScript exited with code {p.returncode}")
            # Try to get stderr
            return

        print("--- end sandbox output ---")

        # List outputs
        p = sb.exec("ls", "-la", "/work/output/")
        for line in p.stdout:
            print(line, end="")
        p.wait()

        # Download output files via tar + base64 (single fast transfer)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print("Downloading output files...")
        p = sb.exec("bash", "-c", "cd /work/output && tar cf - *.png 2>/dev/null | base64")
        b64_chunks = list(p.stdout)
        p.wait()

        if b64_chunks:
            import io, tarfile
            tar_bytes = base64.b64decode("".join(c.strip() for c in b64_chunks))
            tar = tarfile.open(fileobj=io.BytesIO(tar_bytes))
            tar.extractall(path=str(OUTPUT_DIR))
            for member in tar.getmembers():
                size_kb = member.size / 1024
                print(f"  {OUTPUT_DIR / member.name} ({size_kb:.0f} KB)")
            tar.close()

        print(f"\nAll outputs saved to {OUTPUT_DIR}/")

    finally:
        sb.terminate()
        print("Sandbox terminated.")


def main():
    presets = sys.argv[1:] if len(sys.argv) > 1 else ["poetry", "code", "love", "fear"]
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    run_in_sandbox(presets, hf_token)


if __name__ == "__main__":
    main()
