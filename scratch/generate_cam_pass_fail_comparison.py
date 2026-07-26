#!/usr/bin/env python3
"""Render representative gate-passing and gate-failing native CAMs side by side."""

import base64
import csv
import io
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/viet/Capstone/ml")
AUDIT_DIR = ROOT / "docs/report/dense_net_121/assets/2026-07-25_15-32-33_UTC_api_cam_localization_audit"
API_URL = "http://127.0.0.1:8010/api/v1/predict"
OUTPUT = AUDIT_DIR / "passing_vs_failing_cam_comparison.jpg"

# Each pair uses the same predicted grade. The failure examples cover distinct
# spatial errors rather than repeatedly showing the same shortcut.
PAIRS = [
    {
        "grade": 0,
        "pass": ("9023617_20050606_00829903_png.rf.MgcYqK7oyB7fEp3X166B.png", 1),
        "fail": ("9003175_20050511_00771504_png.rf.IoxlFMVl0YwSkDAlE75n.png", 1),
        "failure": "upper femur / top edge",
    },
    {
        "grade": 1,
        "pass": ("9069117_20050628_00907403_png.rf.IJ2Oenzw0AV8h8252JDt.png", 1),
        "fail": ("9086407_20050725_00954404_png.rf.IKDcVHr9lQLozCRYjGmE.png", 1),
        "failure": "diffuse / off-joint",
    },
    {
        "grade": 2,
        "pass": ("9078486_20060207_01363303_png.rf.GKIYVOY4lgIEy5TnSfeh.png", 1),
        "fail": ("9007422_20041130_00406404_png.rf.LqRaNgiaqvX8VQ2wKGey.png", 2),
        "failure": "far lateral edge",
    },
    {
        "grade": 3,
        "pass": ("9211011_20050412_00706803_png.rf.LxaEv53I6v3WTqarjWr5.png", 1),
        "fail": ("9109062_20051108_01265903_png.rf.J5v8ffaCxfDkphk4XoyP.png", 1),
        "failure": "lower tibia / bottom",
    },
    {
        "grade": 4,
        "pass": ("9173792_20050914_01149903_png.rf.MKrb6lxzAohnPIfdzKhH.png", 1),
        "fail": ("9035779_20041028_00368804_png.rf.MoNZGRxe0kpuGqGDdNXC.png", 1),
        "failure": "diffuse / border",
    },
]


def data_url_image(value: str) -> Image.Image:
    encoded = value.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def get_prediction(filename: str, knee_index: int) -> dict:
    with (ROOT / "test_images" / filename).open("rb") as handle:
        response = requests.post(
            API_URL,
            files={"file": (filename, handle, "image/png")},
            timeout=180,
        )
    response.raise_for_status()
    return response.json()["predictions"][knee_index - 1]


def make_tile(prediction: dict, row: dict) -> Image.Image:
    roi = data_url_image(prediction["roi_image"]).resize((384, 384))
    cam = data_url_image(prediction["gradcam_image"]).resize((384, 384))
    tile = Image.new("RGB", (768, 450), "black")
    tile.paste(roi, (0, 66))
    tile.paste(cam, (384, 66))
    draw = ImageDraw.Draw(tile)
    draw.text(
        (8, 10),
        f"{row['filename']} | {row['knee_side']} | G{row['predicted_grade']} p={float(row['confidence']):.3f}",
        fill="white",
    )
    draw.text(
        (8, 34),
        f"joint={float(row['joint_energy']):.3f} border={float(row['border_energy']):.3f} "
        f"lower={float(row['lower_tibia_energy']):.3f} peak=({float(row['peak_x']):.2f},{float(row['peak_y']):.2f})",
        fill="white",
    )
    draw.text((8, 72), "ROI", fill="white")
    draw.text((392, 72), "API native-CAM overlay", fill="white")
    return tile


def main() -> None:
    with (AUDIT_DIR / "all_knees_cam_audit.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(row["filename"], int(row["knee_index"])): row for row in rows}

    font = ImageFont.load_default()
    row_height = 498
    canvas = Image.new("RGB", (1536, 80 + row_height * len(PAIRS)), (16, 18, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), "Native CAM localization: PASS versus FAIL", fill="white", font=font)
    draw.text(
        (20, 40),
        "PASS means the frozen engineering anatomy gate passed; it does not prove the KL grade is correct.",
        fill=(205, 210, 220),
        font=font,
    )

    for index, pair in enumerate(PAIRS):
        y = 80 + index * row_height
        pass_row = lookup[pair["pass"]]
        fail_row = lookup[pair["fail"]]
        if pass_row["gate_pass"] != "True" or fail_row["gate_pass"] != "False":
            raise RuntimeError(f"Unexpected gate result in pair {index + 1}")
        pass_prediction = get_prediction(*pair["pass"])
        pass_tile = make_tile(pass_prediction, pass_row)
        fail_tile = Image.open(AUDIT_DIR / fail_row["failure_image"]).convert("RGB")

        draw.rectangle((0, y, 767, y + 47), fill=(20, 110, 62))
        draw.rectangle((768, y, 1535, y + 47), fill=(152, 42, 42))
        draw.text(
            (16, y + 15),
            f"PASS | predicted Grade {pair['grade']} | activation concentrated in joint band",
            fill="white",
            font=font,
        )
        draw.text(
            (784, y + 15),
            f"FAIL | predicted Grade {pair['grade']} | {pair['failure']}",
            fill="white",
            font=font,
        )
        canvas.paste(pass_tile, (0, y + 48))
        canvas.paste(fail_tile, (768, y + 48))

    canvas.save(OUTPUT, quality=92, subsampling=0)
    print(OUTPUT)


if __name__ == "__main__":
    main()
