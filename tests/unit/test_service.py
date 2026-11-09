"""Unit tests for FastAPI service endpoint handlers (safe for Python 3.14 on Windows)."""

from __future__ import annotations

import pytest

from ecg_digitization.service.app import (
    AnalyzeRequest,
    DigitizeRequest,
    SyntheticRequest,
    analyze_ecg,
    digitize_image,
    encode_img_b64,
    generate_synthetic,
    get_kaggle_status,
    get_report_html,
    get_report_json,
    get_sample_image,
    get_status,
    list_samples,
    serve_index,
)
import cv2
import numpy as np
from fastapi import HTTPException


def test_get_status_handler() -> None:
    data = get_status()
    assert data["status"] == "online"
    assert data["modules"]["synthetic_generator"] == "ready"
    assert data["modules"]["ai_classifier"] == "model_not_trained"
    assert "safety_disclaimer" in data


def test_list_samples_handler() -> None:
    data = list_samples()
    assert "ptbxl_records" in data
    assert "00001_lr" in data["ptbxl_records"]


def test_synthetic_generate_handler() -> None:
    req = SyntheticRequest(
        record_id="00001_lr",
        rotation_deg=0.5,
        gaussian_noise_sigma=1.0,
        seed=42,
    )
    data = generate_synthetic(req)
    assert data["record_id"] == "00001_lr"
    assert "image_b64" in data
    assert len(data["image_b64"]) > 1000
    assert "lead_bounding_boxes" in data
    assert "I" in data["lead_bounding_boxes"]


def test_digitize_and_analyze_handlers() -> None:
    # 1. Digitize
    digit_req = DigitizeRequest(record_id="00001_lr", layout_hint="3x4_1")
    digit_data = digitize_image(digit_req)
    assert digit_data["success"] is True
    assert "leads" in digit_data
    assert "I" in digit_data["leads"]
    assert "plot_b64" in digit_data

    # 2. Analyze
    analyze_req = AnalyzeRequest(record_id="00001_lr")
    analysis_data = analyze_ecg(analyze_req)
    assert "rhythm_analysis" in analysis_data
    assert analysis_data["ai_screening"]["model_status"] == "model_not_trained"
    assert analysis_data["risk_level"].startswith("Not assessed")

    # 3. JSON Report
    rep_json = get_report_json("00001_lr")
    assert rep_json["metadata"]["record_id"] == "00001_lr"
    assert "findings" in rep_json

    # 4. HTML Report
    rep_html = get_report_html("00001_lr")
    assert "ECG DigiAI" in rep_html
    assert "model_not_trained" in rep_html


def test_kaggle_status_handler() -> None:
    data = get_kaggle_status()
    assert "auth" in data
    assert "job_card" in data
    assert "kaggle kernels push" in data["job_card"]["dry_run_command"]


def test_serve_index_handler() -> None:
    resp = serve_index()
    assert resp.status_code == 200
    assert "ECG DigiAI Platform" in resp.body.decode("utf-8")


def test_get_sample_image_handler() -> None:
    # Test valid sample image from real_ecg_eval/samples
    data = get_sample_image("N10007.png")
    assert data["filename"] == "N10007.png"
    assert "image_b64" in data
    assert len(data["image_b64"]) > 100
    assert data["width"] > 50
    assert data["height"] > 50
    assert data["suggested_layout"] == "single_lead"

    # Test non-existent image
    with pytest.raises(HTTPException) as exc:
        get_sample_image("non_existent_file_xyz.png")
    assert exc.value.status_code == 404


def test_digitize_uploaded_image_base64_handler() -> None:
    # Create valid synthetic ECG-like crop (100x100)
    img = np.full((128, 256, 3), 255, dtype=np.uint8)
    cv2.line(img, (0, 64), (256, 64), (0, 0, 0), 2)  # trace
    b64 = encode_img_b64(img)

    req = DigitizeRequest(
        record_id="uploaded_test_lead",
        image_base64=b64,
        layout_hint="single_lead",
        extractor_mode="classical",
    )
    res = digitize_image(req)
    assert res["success"] is True
    assert "leads" in res
    assert "II" in res["leads"]
    assert res["leads"]["II"]["is_trained_lead"] is True


def test_digitize_rejects_undersized_or_blank_image() -> None:
    # Undersized image (30x30 < 50x50)
    small_img = np.full((30, 30, 3), 255, dtype=np.uint8)
    small_b64 = encode_img_b64(small_img)
    req_small = DigitizeRequest(image_base64=small_b64)
    with pytest.raises(HTTPException) as exc1:
        digitize_image(req_small)
    assert exc1.value.status_code == 400
    assert "dimensions too small" in exc1.value.detail

    # Blank image (all white 100x100)
    blank_img = np.full((100, 100, 3), 255, dtype=np.uint8)
    blank_b64 = encode_img_b64(blank_img)
    req_blank = DigitizeRequest(image_base64=blank_b64)
    with pytest.raises(HTTPException) as exc2:
        digitize_image(req_blank)
    assert exc2.value.status_code == 400
    assert "blank or empty" in exc2.value.detail
