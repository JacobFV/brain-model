"""Modal sandbox utilities for running experiments on GPU."""

import base64
import io
import sys
import tarfile
from pathlib import Path

import modal

ENV_NAME = "test-20260327"

# Standard image with all dependencies
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1", "git", "libgl1", "libglib2.0-0", "libxrender1")
    .pip_install("torch", "torchaudio")
    .run_commands("pip install git+https://github.com/facebookresearch/tribev2.git")
    .pip_install(
        "gtts", "langdetect", "soundfile", "matplotlib",
        "nilearn", "pyvista", "vtk", "colorcet", "seaborn", "scikit-image",
    )
)


def create_sandbox(gpu: str = "A10G", timeout: int = 1800, memory: int = 32768):
    """Create a Modal sandbox with GPU in the test environment."""
    modal.enable_output()
    secrets = [modal.Secret.from_name("huggingface-secret", environment_name=ENV_NAME)]
    app = modal.App.lookup("brain-radio", create_if_missing=True, environment_name=ENV_NAME)

    return modal.Sandbox.create(
        app=app,
        image=IMAGE,
        gpu=gpu,
        timeout=timeout,
        cpu=4,
        memory=memory,
        secrets=secrets,
    )


def upload_and_run(sb, script_path: str, args: str = "", work_dir: str = "/work"):
    """Upload a script to a sandbox, run it, and stream output."""
    script_content = Path(script_path).read_text()
    encoded = base64.b64encode(script_content.encode()).decode()

    sb.exec("bash", "-c", f"mkdir -p {work_dir}/output {work_dir}/cache").wait()
    sb.exec("bash", "-c", f"echo '{encoded}' | base64 -d > {work_dir}/script.py").wait()

    cmd = f"cd {work_dir} && python script.py {args}"
    print(f"Running: {cmd}")
    p = sb.exec("bash", "-c", cmd)
    for line in p.stdout:
        print(line, end="")
    for line in p.stderr:
        print(line, end="", file=sys.stderr)
    p.wait()
    return p.returncode


def download_outputs(sb, remote_dir: str = "/work/output", local_dir: str = "./output", pattern: str = "*.png *.npy *.json"):
    """Download output files from sandbox via tar."""
    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)

    p = sb.exec("bash", "-c", f"cd {remote_dir} && tar cf - {pattern} 2>/dev/null | base64")
    b64_chunks = list(p.stdout)
    p.wait()

    if not b64_chunks:
        print("No output files found.")
        return []

    tar_bytes = base64.b64decode("".join(c.strip() for c in b64_chunks))
    tar = tarfile.open(fileobj=io.BytesIO(tar_bytes))
    tar.extractall(path=str(local), filter="data")
    downloaded = []
    for member in tar.getmembers():
        size_kb = member.size / 1024
        print(f"  {local / member.name} ({size_kb:.0f} KB)")
        downloaded.append(local / member.name)
    tar.close()
    return downloaded


def run_in_sandbox(script_path: str, args: str = "", gpu: str = "A10G", output_dir: str = "./output"):
    """Full pipeline: create sandbox, upload script, run, download results."""
    sb = create_sandbox(gpu=gpu)
    print(f"Sandbox created: {sb.object_id}")
    try:
        rc = upload_and_run(sb, script_path, args)
        if rc != 0:
            print(f"Script exited with code {rc}")
            return
        download_outputs(sb, local_dir=output_dir)
    finally:
        sb.terminate()
        print("Sandbox terminated.")
