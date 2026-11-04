"""Create two synthetic frames and C's upstream input fixture, without external data."""

from pathlib import Path

from PIL import Image, ImageDraw

from embedding.contracts import SCHEMA, file_sha256, write_json, write_jsonl


def generate():
    root = Path(__file__).parent
    for stamp, color in ((0, "red"), (200, "blue")):
        image = Image.new("RGB", (320, 180), "#d8e2ea")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 120, 320, 180), fill="#555555")
        draw.rectangle((72, 72, 216, 144), fill=color)
        draw.ellipse((88, 128, 112, 152), fill="black")
        draw.ellipse((176, 128, 200, 152), fill="black")
        image.save(root / f"frame_{stamp}.png")
    write_json(root / "frames_source.json", {"frames": [
        {"timestamp_ms": stamp, "path": f"frame_{stamp}.png",
         "sha256": file_sha256(root / f"frame_{stamp}.png")} for stamp in (0, 200)]})
    write_jsonl(root / "manifest.jsonl", [{
        "schema_version": SCHEMA, "video_id": "fixture01", "dataset": "synthetic",
        "camera_id": None, "relative_path": "frames_source.json", "width": 320, "height": 180,
        "duration_ms": 400, "fps": 5, "time_base": None, "source_offset_ms": 0,
        "available_start_ms": 0, "available_end_ms": 400, "split": "dev",
        "file_sha256": file_sha256(root / "frames_source.json")}])
    samples = []
    for stamp in (0, 200):
        for representation in ("scene", "object"):
            samples.append({"schema_version": SCHEMA, "item_id": f"fixture-{representation}-{stamp}",
                            "video_id": "fixture01", "frame_index": stamp // 200,
                            "timestamp_ms": stamp, "representation": representation,
                            "track_uid": "fixture:fixture01:t1" if representation == "object" else None,
                            "bbox": [72.4, 72.2, 216.1, 152.0] if representation == "object" else None,
                            "event_ids": [], "reasons": ["fixture"], "policy": "fixture",
                            "source_run_id": "fixture-a", "track_source_id": "fixture"})
    write_jsonl(root / "samples.jsonl", samples)
    (root / "empty_samples.jsonl").write_text("", encoding="utf-8")


if __name__ == "__main__":
    generate()
