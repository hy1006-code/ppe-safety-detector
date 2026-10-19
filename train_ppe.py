"""Train the PPE detector; logs and checkpoints live in runs/ppe/yolov8n_baseline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parent
STATUS = ROOT / "training_status.json"


def status(state, **extra):
    payload = {"state": state, "updated_at": datetime.now().astimezone().isoformat(), **extra}
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(STATUS)


def epoch_done(trainer):
    offset = getattr(trainer, "_ppe_epoch_offset", 0)
    status("training", epoch=offset + trainer.epoch + 1, epochs=offset + trainer.epochs,
           metrics=trainer.metrics, output_dir=str(trainer.save_dir))


def batch_done(trainer):
    import torch
    batch_index = getattr(trainer, "_ppe_batch_index", 0)
    trainer._ppe_batch_index = batch_index + 1
    if not torch.isfinite(trainer.loss).all():
        raise RuntimeError("Non-finite training loss; stopped to preserve the last valid checkpoint")
    if batch_index % 25 == 0:
        losses = torch.stack(list(trainer.tloss.values())).detach().cpu()
        if not torch.isfinite(losses).all():
            raise RuntimeError("Non-finite training loss; stopped to preserve the last valid checkpoint")
        offset = getattr(trainer, "_ppe_epoch_offset", 0)
        status("training", epoch_completed=offset + trainer.epoch, current_epoch=offset + trainer.epoch + 1,
               epochs=offset + trainer.epochs, batch=batch_index + 1,
               batches=len(trainer.train_loader), losses=losses.tolist(),
               device=str(trainer.device), pid=os.getpid(), output_dir=str(trainer.save_dir))


def main():
    import torch
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--finetune", type=Path, help="Continue weights with a fresh, low-rate optimizer")
    parser.add_argument("--completed-epochs", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or CUDA index")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else (0 if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        torch.set_num_threads(min(6, os.cpu_count() or 1))
    status("starting", device=device, pid=os.getpid())
    if args.resume and args.finetune:
        parser.error("Choose --resume or --finetune")
    model = YOLO(str(args.resume or args.finetune or ROOT / "yolov8n.pt"))
    model.add_callback("on_fit_epoch_end", epoch_done)
    model.add_callback("on_train_batch_end", batch_done)
    model.add_callback("on_train_epoch_start", lambda trainer: setattr(trainer, "_ppe_batch_index", 0))
    model.add_callback("on_train_start", lambda trainer: setattr(trainer, "_ppe_epoch_offset", args.completed_epochs))
    if args.resume:
        model.train(resume=True, device=device)
    else:
        stable = dict(optimizer="AdamW", lr0=0.0001, warmup_epochs=1.0,
                      warmup_bias_lr=0.001, mosaic=0.5) if args.finetune else {}
        model.train(data=str(ROOT / "construction_safety_class9_data/data.yaml"),
                    epochs=args.epochs, imgsz=640, batch=args.batch, device=device,
                    workers=2, project=str(ROOT / "runs/ppe"),
                    name="yolov8n_stable" if args.finetune else "yolov8n_baseline",
                    patience=20, seed=42, deterministic=True, cache=False,
                    amp=False, save=True, save_period=10, plots=True, close_mosaic=10, **stable)
    best = Path(model.trainer.best)
    status("generating_submission", best_checkpoint=str(best))
    subprocess.run([sys.executable, str(ROOT / "submit.py"), "--checkpoint", str(best)], check=True)
    status("complete", best_checkpoint=str(best), submission=str(ROOT / "submission.csv"),
           output_dir=str(model.trainer.save_dir))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error=repr(error))
        raise
