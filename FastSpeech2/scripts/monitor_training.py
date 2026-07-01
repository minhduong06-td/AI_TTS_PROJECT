import argparse
import os
import re
import subprocess
import time
from datetime import datetime

DEFAULT_TRAIN_LOG = "./outputs/logs/vi_ljspeech/train/log.txt"
DEFAULT_OUT_TXT   = "./output/log/vi_ljspeech/train/metrics_log.txt"
DEFAULT_INTERVAL  = 60  # giây

LOG_PATTERN = re.compile(
    r"Step (\d+)/(\d+), "
    r"Total Loss: ([\d.]+), "
    r"Mel Loss: ([\d.]+), "
    r"Mel PostNet Loss: ([\d.]+), "
    r"Pitch Loss: ([\d.]+), "
    r"Energy Loss: ([\d.]+), "
    r"Duration Loss: ([\d.]+)"
)


def parse_last_log_entry(log_path):
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in reversed(lines):
            m = LOG_PATTERN.search(line)
            if m:
                return {
                    "step":             int(m.group(1)),
                    "total_steps":      int(m.group(2)),
                    "total_loss":       float(m.group(3)),
                    "mel_loss":         float(m.group(4)),
                    "mel_postnet_loss": float(m.group(5)),
                    "pitch_loss":       float(m.group(6)),
                    "energy_loss":      float(m.group(7)),
                    "duration_loss":    float(m.group(8)),
                }
    except Exception as e:
        print(f"[WARN] Không đọc được train log: {e}")
    return None


def get_gpu_stats():
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            p = [x.strip() for x in result.stdout.strip().split(",")]
            return {
                "util":  p[0] + "%",
                "vram":  f"{p[1]}/{p[2]} MiB",
                "temp":  p[3] + "°C",
                "power": (p[4] + "W") if p[4] != "N/A" else "N/A",
            }
    except Exception:
        pass
    return {"util": "N/A", "vram": "N/A", "temp": "N/A", "power": "N/A"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",      default=DEFAULT_TRAIN_LOG)
    parser.add_argument("--out",      default=DEFAULT_OUT_TXT)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[Monitor] Ghi log mỗi {args.interval}s → {args.out}")
    print("[Monitor] Ctrl+C để dừng.\n")

    prev_step = None
    prev_time = None

    # Ghi header 1 lần
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"  TRAINING SESSION — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*70}\n")
        f.write(f"{'Timestamp':<22} {'Step':>10} {'%':>6} {'Speed':>9} {'TotalLoss':>10} {'MelLoss':>8} {'PitchLoss':>10} {'EnerLoss':>9} {'DurLoss':>8} {'GPU':>5} {'VRAM':>16} {'Temp':>6}\n")
        f.write(f"{'-'*130}\n")

    try:
        while True:
            now     = datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            entry   = parse_last_log_entry(args.log)
            gpu     = get_gpu_stats()

            if entry is None:
                print(f"[{now_str}] Chưa có log, đang chờ...")
                time.sleep(args.interval)
                continue

            cur_step = entry["step"]
            cur_time = time.time()

            # Tính tốc độ
            if prev_step is not None and cur_step > prev_step:
                elapsed = cur_time - prev_time
                speed_str = f"{(cur_step - prev_step) / elapsed:.2f} it/s"
            else:
                speed_str = "—"
            prev_step = cur_step
            prev_time = cur_time

            pct = f"{cur_step / entry['total_steps'] * 100:.2f}%"

            line = (
                f"{now_str:<22} "
                f"{cur_step:>10} "
                f"{pct:>6} "
                f"{speed_str:>9} "
                f"{entry['total_loss']:>10.4f} "
                f"{entry['mel_loss']:>8.4f} "
                f"{entry['pitch_loss']:>10.4f} "
                f"{entry['energy_loss']:>9.4f} "
                f"{entry['duration_loss']:>8.4f} "
                f"{gpu['util']:>5} "
                f"{gpu['vram']:>16} "
                f"{gpu['temp']:>6}"
            )

            with open(args.out, "a", encoding="utf-8") as f:
                f.write(line + "\n")

            print(line)
            time.sleep(args.interval)

    except KeyboardInterrupt:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(f"{'-'*130}\n")
            f.write(f"  Dừng lúc {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"\n[Monitor] Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
