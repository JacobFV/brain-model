#!/usr/bin/env python3
"""
Run any experiment on Modal with GPU.

Usage:
    python run_experiment.py emotion_stimuli
    python run_experiment.py brain_ux
    python run_experiment.py virtual_eeg
    python run_experiment.py consciousness
    python run_experiment.py affective_intervention
"""

import base64
import io
import sys
import tarfile
from pathlib import Path

import modal
modal.enable_output()

ENV_NAME = "test-20260327"

EXPERIMENTS = {
    "emotion_stimuli": "experiments/emotion_stimuli/run.py",
    "brain_ux": "experiments/brain_ux/run.py",
    "virtual_eeg": "experiments/virtual_eeg/run.py",
    "consciousness": "experiments/consciousness/run.py",
    "affective_intervention": "experiments/affective_intervention/run.py",
    "brain_radio": "experiments/brain_radio/brain_radio.py",
    "brain_ux_render": "experiments/brain_ux/render_brains.py",
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1", "git", "libgl1", "libglib2.0-0", "libxrender1")
    .pip_install("torch", "torchaudio")
    .run_commands("pip install git+https://github.com/facebookresearch/tribev2.git")
    .run_commands("python -m spacy download en_core_web_lg")  # pre-bake spacy model
    .pip_install(
        "gtts", "langdetect", "soundfile", "matplotlib",
        "nilearn", "pyvista", "vtk", "colorcet", "seaborn",
        "scikit-image", "scikit-learn", "Pillow",
        "transformers>=4.45,<4.50",  # pin for LLaMA 3.2 compat
    )
)


def run(experiment_name: str, extra_args: str = "", gpu: str = "A10G"):
    if experiment_name not in EXPERIMENTS:
        print(f"Unknown experiment: {experiment_name}")
        print(f"Available: {', '.join(EXPERIMENTS.keys())}")
        sys.exit(1)

    script_path = Path(EXPERIMENTS[experiment_name])
    core_dir = Path("core")

    secrets = [modal.Secret.from_name("huggingface-secret", environment_name=ENV_NAME)]
    app = modal.App.lookup("brain-experiments", create_if_missing=True, environment_name=ENV_NAME)

    print(f"Launching experiment '{experiment_name}' on Modal ({gpu})...")
    sb = modal.Sandbox.create(
        app=app,
        image=image,
        gpu=gpu,
        timeout=3600,  # 1 hour for larger experiments
        cpu=4,
        memory=32768,
        secrets=secrets,
    )
    print(f"Sandbox: {sb.object_id}")

    try:
        sb.exec("bash", "-c", "mkdir -p /work/output /work/cache /work/core").wait()

        # Upload core module
        for core_file in core_dir.glob("*.py"):
            encoded = base64.b64encode(core_file.read_bytes()).decode()
            sb.exec("bash", "-c", f"echo '{encoded}' | base64 -d > /work/core/{core_file.name}").wait()

        # Upload experiment script
        encoded = base64.b64encode(script_path.read_bytes()).decode()
        sb.exec("bash", "-c", f"echo '{encoded}' | base64 -d > /work/run.py").wait()

        # Upload any pre-existing output data the script might need (e.g. render scripts)
        local_output = Path(f"output/{experiment_name.split('_render')[0]}")
        if local_output.exists():
            for f in local_output.glob("*.npz"):
                print(f"Uploading {f.name} ({f.stat().st_size // 1024} KB)...")
                remote_path = f"/work/output/{f.name}"
                fh = sb.open(remote_path, "wb")
                fh.write(f.read_bytes())
                fh.close()
                print(f"  Uploaded {f.name}")

        # Run
        cmd = f"cd /work && python run.py {extra_args}"
        print(f"Running: {cmd}")
        print("=" * 60)

        p = sb.exec("bash", "-c", cmd)
        for line in p.stdout:
            print(line, end="")
        for line in p.stderr:
            print(line, end="", file=sys.stderr)
        p.wait()

        if p.returncode != 0:
            print(f"\nExperiment exited with code {p.returncode}")
            return

        print("=" * 60)

        # Download outputs
        output_dir = Path(f"output/{experiment_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        p = sb.exec("bash", "-c", "cd /work/output && tar cf - . 2>/dev/null | base64")
        b64_chunks = list(p.stdout)
        p.wait()

        if b64_chunks:
            tar_bytes = base64.b64decode("".join(c.strip() for c in b64_chunks))
            tar = tarfile.open(fileobj=io.BytesIO(tar_bytes))
            tar.extractall(path=str(output_dir), filter="data")
            print(f"\nDownloaded outputs to {output_dir}/:")
            for member in tar.getmembers():
                if member.isfile():
                    print(f"  {member.name} ({member.size / 1024:.0f} KB)")
            tar.close()

    finally:
        sb.terminate()
        print("Sandbox terminated.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <experiment_name> [args...]")
        print(f"Available experiments: {', '.join(EXPERIMENTS.keys())}")
        sys.exit(1)

    experiment = sys.argv[1]
    extra_args = " ".join(sys.argv[2:])
    run(experiment, extra_args)


if __name__ == "__main__":
    main()
