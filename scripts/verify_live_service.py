"""Verification script for live HTTP service, real ECG upload, CSV export, and 12-lead inspection."""

import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def main():
    print("=" * 70)
    print("ECG DigiAI — LIVE HTTP SERVICE & 12-LEAD VERIFICATION")
    print("=" * 70)

    # 1. Start uvicorn server as a subprocess using documented startup command
    port = 8000
    host = "127.0.0.1"
    base_url = f"http://{host}:{port}"

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "ecg_digitization.service.app:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    print(f"1. Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        # Wait for server to become responsive
        max_retries = 20
        server_ready = False
        for i in range(max_retries):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(f"{base_url}/api/status", timeout=2) as resp:
                    if resp.status == 200:
                        server_ready = True
                        break
            except Exception:
                pass

        if not server_ready:
            print("ERROR: Server failed to start or probe failed within timeout.")
            proc.terminate()
            try:
                stdout, stderr = proc.communicate(timeout=3)
            except Exception:
                proc.kill()
                stdout, stderr = proc.communicate(timeout=2)
            print("STDOUT:", stdout.decode(errors="replace") if stdout else "")
            print("STDERR:", stderr.decode(errors="replace") if stderr else "")
            sys.exit(1)

        print(f"Server is LIVE at {base_url} (PID: {proc.pid})")

        # 2. Test GET /
        print("\n2. Testing GET / (HTML Index)")
        with urllib.request.urlopen(f"{base_url}/", timeout=5) as resp:
            assert resp.status == 200, f"Expected 200, got {resp.status}"
            html_text = resp.read().decode("utf-8")
            assert "ECG DigiAI Platform" in html_text
            assert "drop-zone" in html_text
            print(f"   [PASS] HTTP {resp.status} OK — HTML payload {len(html_text)} bytes")

        # 3. Test GET /api/status
        print("\n3. Testing GET /api/status")
        with urllib.request.urlopen(f"{base_url}/api/status", timeout=5) as resp:
            assert resp.status == 200
            status_data = json.loads(resp.read().decode("utf-8"))
            print(f"   [PASS] Status: {status_data.get('status')}")
            print(f"   Project: {status_data.get('project')}")
            ckpt = status_data.get("deep_digitizer_checkpoint", {})
            print(f"   Checkpoint Loaded: {ckpt.get('available')}")
            print(f"   Best Val Pearson r: {ckpt.get('best_val_pearson_r')}")
            print(f"   Best Val RMSE: {ckpt.get('val_rmse_mv')} mV")
            assert ckpt.get("available") is True
            assert ckpt.get("best_val_pearson_r") == 0.9971

        # 4. Upload & Process Real ECG Image (N10007.png)
        sample_path = Path("data/real_ecg_eval/samples/N10007.png")
        print(f"\n4. Uploading Real ECG Image: {sample_path}")
        assert sample_path.is_file(), f"File {sample_path} missing!"

        img_b64 = base64.b64encode(sample_path.read_bytes()).decode("utf-8")
        digitize_payload = json.dumps({
            "record_id": "real_N10007_test",
            "image_base64": img_b64,
            "layout_hint": "single_lead",
            "extractor_mode": "deep",
            "is_calibration_reliable": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{base_url}/api/digitize",
            data=digitize_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            assert resp.status == 200
            dig_res = json.loads(resp.read().decode("utf-8"))
            print(f"   [PASS] Digitize Success: {dig_res.get('success')}")
            print(f"   Deskew Angle: {dig_res.get('deskew_angle_deg')} deg")
            print(f"   Extracted Leads: {list(dig_res.get('leads', {}).keys())}")
            lead_ii = dig_res.get("leads", {}).get("II", {})
            print(f"   Lead II Confidence: {lead_ii.get('confidence') * 100:.2f}%")
            print(f"   Lead II Trained Status: {lead_ii.get('is_trained_lead')}")
            print(f"   Lead II Validation Scope: {lead_ii.get('validation_status')}")
            assert dig_res.get("success") is True
            assert "II" in dig_res.get("leads", {})

        # 5. Verify CSV Download
        print("\n5. Verifying Reconstructed CSV Download")
        csv_url = f"{base_url}/api/export/csv/real_N10007_test"
        with urllib.request.urlopen(csv_url, timeout=5) as resp:
            assert resp.status == 200
            assert "text/csv" in resp.headers.get("Content-Type", "")
            csv_content = resp.read().decode("utf-8")
            csv_lines = csv_content.strip().splitlines()
            print(f"   [PASS] CSV downloaded ({len(csv_lines)} lines, {len(csv_content)} bytes)")
            # Print header metadata lines
            for line in csv_lines[:5]:
                print(f"   | {line}")
            assert "time_s" in csv_content
            assert "II" in csv_content
            assert len(csv_lines) > 200

        # 6. Test Standard 3x4 ECG Sheet & Inspect All 12 Leads Individually
        print("\n6. Testing Standard 3x4 Sheet Digitisation & 12-Lead Inspection")
        sheet_payload = json.dumps({
            "record_id": "00001_lr",
            "layout_hint": "3x4_1",
            "extractor_mode": "deep",
            "is_calibration_reliable": True,
        }).encode("utf-8")

        req_sheet = urllib.request.Request(
            f"{base_url}/api/digitize",
            data=sheet_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_sheet, timeout=25) as resp:
            assert resp.status == 200
            sheet_res = json.loads(resp.read().decode("utf-8"))
            print(f"   [PASS] Sheet Digitize Success: {sheet_res.get('success')}")

            all_leads = sheet_res.get("leads", {})
            print(f"   Detected Total Leads: {len(all_leads)}")

            standard_12_leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6", "II_RHYTHM"]

            print("\n   --- Individual Lead Inspection Table ---")
            print(f"   {'Lead':<10} {'Trained / Status':<28} {'Confidence':<12} {'Samples':<10} {'Scope Category'}")
            print("   " + "-" * 75)

            validated_count = 0
            experimental_count = 0

            for lname in standard_12_leads:
                if lname in all_leads:
                    info = all_leads[lname]
                    is_trained = info.get("is_trained_lead", False)
                    vstatus = info.get("validation_status", "unknown")
                    conf = info.get("confidence", 0.0)
                    nsamples = len(info.get("voltage_mv", []))

                    category = "VALIDATED (TRAINED)" if is_trained else "EXPERIMENTAL (ZERO-SHOT)"
                    if is_trained:
                        validated_count += 1
                    else:
                        experimental_count += 1

                    print(f"   {lname:<10} {vstatus:<28} {conf * 100:>8.2f}%   {nsamples:<10} {category}")
                else:
                    print(f"   {lname:<10} MISSING FROM DETECTION")

            print("   " + "-" * 75)
            print(f"   Summary: {validated_count} Formally Validated Leads | {experimental_count} Experimental Zero-Shot Leads")
            assert validated_count == 3  # II, V1, II_RHYTHM
            assert experimental_count == 10  # I, III, AVR, AVL, AVF, V2, V3, V4, V5, V6

        # 7. Check 12-lead sheet overlay image
        assert "sheet_overlay_b64" in sheet_res
        overlay_len = len(sheet_res["sheet_overlay_b64"])
        print(f"\n7. Diagnostic Sheet Overlay Generated: {overlay_len} bytes base64")
        assert overlay_len > 1000

        print("\n" + "=" * 70)
        print("ALL LIVE HTTP AND 12-LEAD CHECKS COMPLETED SUCCESSFULLY!")
        print("=" * 70)

    finally:
        print("\nShutting down uvicorn test server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Server shutdown complete.")


if __name__ == "__main__":
    main()
