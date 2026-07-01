import os
import sys
import time
import uuid
import shutil
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

AI_ROOT = Path(os.environ.get("AI_ROOT", "/home/md_dz6/AI")).resolve()

FASTSPEECH_ROOT = AI_ROOT / "FastSpeech2"
FASTSPEECH_RUNS = AI_ROOT / "FastSpeech2_runs"

COQUI_ROOT = AI_ROOT / "coqui-ai-TTS"
COQUI_RUNS = AI_ROOT / "coqui_runs"

WEB_ROOT = AI_ROOT / "web"
FE_ROOT = WEB_ROOT / "fe"
OUTPUT_ROOT = WEB_ROOT / "output"
RUNTIME_ROOT = WEB_ROOT / "runtime"
FASTSPEECH_PYTHON = os.environ.get("FASTSPEECH_PYTHON", "python3")
COQUI_PYTHON = os.environ.get("COQUI_PYTHON", "python3")

USE_CUDA = os.environ.get("USE_CUDA", "0") == "1"


FASTSPEECH = {
    "en": {
        "dataset": "en_ljspeech",
        "restore_step": 85000,
        "template_dir": FASTSPEECH_ROOT / "configs" / "templates" / "en_ljspeech",
        "ckpt_dir": FASTSPEECH_RUNS / "checkpoints" / "en_ljspeech",
    },
    "vi": {
        "dataset": "vi_ljspeech",
        "restore_step": 220000,
        "template_dir": FASTSPEECH_ROOT / "configs" / "templates" / "vi_ljspeech",
        "ckpt_dir": FASTSPEECH_RUNS / "checkpoints" / "vi_ljspeech",
    },
}

app = FastAPI(title="Local TTS Web Demo")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

app.mount("/output", StaticFiles(directory=str(OUTPUT_ROOT)), name="output")


class GenerateRequest(BaseModel):
    model: str
    lang: str
    text: str

    duration_control: float = 1.0
    pitch_control: float = 1.0
    energy_control: float = 1.0

    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_scale_dp: float = 0.8


JOBS: Dict[str, Dict[str, Any]] = {}

def make_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def set_job(job_id: str, **kwargs):
    if job_id in JOBS:
        JOBS[job_id].update(kwargs)


def add_log(job_id: str, line: str):
    job = JOBS.get(job_id)
    if not job:
        return

    logs = job.setdefault("logs", [])
    logs.append(line.rstrip())

    if len(logs) > 120:
        del logs[:-120]


def run_cmd(job_id: str, cmd: List[str], cwd: Path, env_extra: Optional[Dict[str, str]] = None) -> int:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    add_log(job_id, "CWD: " + str(cwd))
    add_log(job_id, "CMD: " + " ".join(str(x) for x in cmd))

    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    progress = 5
    set_job(job_id, progress=progress)

    while True:
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            add_log(job_id, line)
            progress = min(progress + 2, 95)
            set_job(job_id, progress=progress)
            continue

        if proc.poll() is not None:
            break

        progress = min(progress + 1, 90)
        set_job(job_id, progress=progress)
        time.sleep(1)

    if proc.stdout:
        for line in proc.stdout:
            add_log(job_id, line)

    return proc.returncode


def public_audio_url(path: Path) -> str:
    return "/output/" + path.relative_to(OUTPUT_ROOT).as_posix()


def ensure_file(path: Path, message: str):
    if not path.is_file():
        raise RuntimeError(f"{message}: {path}")


def ensure_dir(path: Path, message: str):
    if not path.is_dir():
        raise RuntimeError(f"{message}: {path}")


def prepare_fastspeech_runtime(lang: str, out_dir: Path) -> Dict[str, Path]:
    cfg = FASTSPEECH[lang]
    template_dir = cfg["template_dir"]
    ckpt_dir = cfg["ckpt_dir"]
    dataset = cfg["dataset"]
    restore_step = cfg["restore_step"]

    ensure_dir(template_dir, "Không thấy config template FastSpeech2")
    ensure_dir(ckpt_dir, "Không thấy checkpoint folder FastSpeech2")
    ensure_file(ckpt_dir / f"{restore_step}.pth.tar", "Không thấy checkpoint FastSpeech2")

    runtime_dir = RUNTIME_ROOT / "fastspeech2" / dataset
    runtime_dir.mkdir(parents=True, exist_ok=True)

    preprocess_path = runtime_dir / "preprocess.yaml"
    model_path = runtime_dir / "model.yaml"
    train_path = runtime_dir / "train.yaml"

    shutil.copy2(template_dir / "preprocess.yaml", preprocess_path)
    shutil.copy2(template_dir / "model.yaml", model_path)
    shutil.copy2(template_dir / "train.yaml", train_path)

    with open(train_path, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    train_cfg["path"]["ckpt_path"] = str(ckpt_dir)
    train_cfg["path"]["result_path"] = str(out_dir)
    train_cfg["path"]["log_path"] = str(RUNTIME_ROOT / "fastspeech2_logs" / dataset)

    with open(train_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(train_cfg, f, sort_keys=False, allow_unicode=True)

    return {
        "preprocess": preprocess_path,
        "model": model_path,
        "train": train_path,
    }



def generate_fastspeech2(job_id: str, req: GenerateRequest) -> Path:
    lang = req.lang.strip().lower()
    text = req.text.strip()

    cfg = FASTSPEECH[lang]

    out_dir = OUTPUT_ROOT / "fastspeech2"
    out_dir.mkdir(parents=True, exist_ok=True)

    name = f"{lang}_{make_id()}"
    final_wav = out_dir / f"{name}.wav"

    runtime_cfg = prepare_fastspeech_runtime(lang, out_dir)

    cmd = [
        FASTSPEECH_PYTHON,
        "synthesize.py",
        "--restore_step",
        str(cfg["restore_step"]),
        "--mode",
        "single",
        "--text",
        text,
        "--output_name",
        name,
        "-p",
        str(runtime_cfg["preprocess"]),
        "-m",
        str(runtime_cfg["model"]),
        "-t",
        str(runtime_cfg["train"]),
        "--duration_control",
        str(req.duration_control),
        "--pitch_control",
        str(req.pitch_control),
        "--energy_control",
        str(req.energy_control),
    ]

    env = {
        "PYTHONPATH": str(FASTSPEECH_ROOT / "src"),
    }

    code = run_cmd(job_id, cmd, FASTSPEECH_ROOT, env)

    if code != 0:
        raise RuntimeError(f"FastSpeech2 lỗi, exit code = {code}")

    ensure_file(final_wav, "FastSpeech2 chạy xong nhưng không thấy wav output")
    return final_wav


def find_vits_run(lang: str) -> Dict[str, Path]:

    VITS_FIXED_RUNS = {
        "vi": "vi_ljspeech_vits_real/vi_ljspeech_vits_real-May-05-2026_02+39AM-cd47a1e",
        "en": "en_ljspeech_vits_real/en_ljspeech_vits_real-April-08-2026_11+57PM-0000000",
    }

    if lang not in VITS_FIXED_RUNS:
        raise RuntimeError(f"Chưa cấu hình VITS checkpoint cho lang={lang}")

    run_dir = COQUI_RUNS / lang / VITS_FIXED_RUNS[lang]

    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best_model.pth"

    if not run_dir.is_dir():
        raise RuntimeError(f"Không thấy folder VITS run: {run_dir}")

    if not config_path.is_file():
        raise RuntimeError(f"Không thấy config.json: {config_path}")

    if not checkpoint_path.is_file():
        raise RuntimeError(f"Không thấy best_model.pth: {checkpoint_path}")

    return {
        "run_dir": run_dir,
        "config": config_path,
        "checkpoint": checkpoint_path,
    }



def generate_vits(job_id: str, req: GenerateRequest) -> Path:
    lang = req.lang.strip().lower()
    text = req.text.strip()

    out_dir = OUTPUT_ROOT / "vits"
    out_dir.mkdir(parents=True, exist_ok=True)

    name = f"{lang}_{make_id()}"
    final_wav = out_dir / f"{name}.wav"

    vits = find_vits_run(lang)

    cmd = [
        COQUI_PYTHON,
        "-m",
        "TTS.bin.synthesize",
        "--text",
        text,
        "--model_path",
        str(vits["checkpoint"]),
        "--config_path",
        str(vits["config"]),
        "--out_path",
        str(final_wav),
        "--length_scale",
        str(req.length_scale),
        "--noise_scale",
        str(req.noise_scale),
        "--noise_scale_dp",
        str(req.noise_scale_dp),
    ]

    device = "cuda" if USE_CUDA else "cpu"
    cmd += ["--device", device]

    env = {
        "PYTHONPATH": str(COQUI_ROOT),
    }

    code = run_cmd(job_id, cmd, COQUI_ROOT, env)

    if code != 0:
        raise RuntimeError(f"Coqui VITS lỗi, exit code = {code}")

    ensure_file(final_wav, "Coqui VITS chạy xong nhưng không thấy wav output")
    return final_wav

# ============================================================
# API
# ============================================================

@app.get("/")
def index():
    return FileResponse(str(FE_ROOT / "index.html"))


@app.get("/style.css")
def style():
    return FileResponse(str(FE_ROOT / "style.css"))


@app.get("/app.js")
def frontend_js():
    return FileResponse(str(FE_ROOT / "app.js"))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "ai_root": str(AI_ROOT),
        "fastspeech_root": str(FASTSPEECH_ROOT),
        "coqui_root": str(COQUI_ROOT),
        "output_root": str(OUTPUT_ROOT),
    }


@app.post("/api/generate")
def generate(req: GenerateRequest):
    model = req.model.strip().lower()
    lang = req.lang.strip().lower()
    text = req.text.strip()

    if model not in {"fastspeech2", "vits"}:
        raise HTTPException(status_code=400, detail="model phải là fastspeech2 hoặc vits")

    if lang not in {"en", "vi"}:
        raise HTTPException(status_code=400, detail="lang phải là en hoặc vi")

    if not text:
        raise HTTPException(status_code=400, detail="text không được rỗng")

    job_id = uuid.uuid4().hex

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "model": model,
        "lang": lang,
        "text": text,
        "audio_url": None,
        "file_path": None,
        "error": None,
        "logs": [],
    }

    thread = threading.Thread(
        target=generate_worker,
        args=(job_id, req),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Không tìm thấy job")

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "model": job.get("model"),
        "lang": job.get("lang"),
        "audio_url": job.get("audio_url"),
        "file_path": job.get("file_path"),
        "error": job.get("error"),
        "logs": job.get("logs", [])[-80:],
    }



def generate_worker(job_id: str, req: GenerateRequest):
    try:
        model = req.model.strip().lower()
        lang = req.lang.strip().lower()
        text = req.text.strip()

        set_job(job_id, status="running", progress=3)
        add_log(job_id, f"Model: {model}")
        add_log(job_id, f"Lang: {lang}")
        add_log(job_id, f"Text: {text}")

        if model == "fastspeech2":
            add_log(job_id, f"duration_control: {req.duration_control}")
            add_log(job_id, f"pitch_control: {req.pitch_control}")
            add_log(job_id, f"energy_control: {req.energy_control}")
            wav_path = generate_fastspeech2(job_id, req)

        elif model == "vits":
            add_log(job_id, f"length_scale: {req.length_scale}")
            add_log(job_id, f"noise_scale: {req.noise_scale}")
            add_log(job_id, f"noise_scale_dp: {req.noise_scale_dp}")
            wav_path = generate_vits(job_id, req)

        else:
            raise RuntimeError("model không hợp lệ")

        set_job(
            job_id,
            status="done",
            progress=100,
            audio_url=public_audio_url(wav_path),
            file_path=str(wav_path),
        )
        add_log(job_id, f"DONE: {wav_path}")

    except Exception as e:
        add_log(job_id, "ERROR: " + str(e))
        set_job(
            job_id,
            status="error",
            progress=100,
            error=str(e),
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
