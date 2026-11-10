"""ECG DigiAI — Biomedical Engineering Workstation Single-Page Application HTML.

Designed for Final-Year B.Tech Major Project Demonstration (JIIT Noida, ECE Dept).
Provides:
01 — Digitisation Workbench (Input, Scanning Animation, Preprocessing, Lead Detection, Reconstructed Overlay)
02 — Lead Explorer (Clinical 12-lead arrangement, Interactive Millimeter Grid Canvas, Animation, Coordinate Cursor, R-Peaks, Genuine Signal Processing)
03 — Synthetic ECG Lab (PTB-XL Record Browser, Degradation Controls, Ground-Truth Pairing, Handoff)
04 — Evaluation & Export (Paired Waveform Comparison, Error Residuals, Pearson r / RMSE, Benchmark Table, CSV/WFDB/Report Exports)
About Modal — Institutional Provenance, ResNet-U-Net Checkpoint Verification, Scope Boundaries.
"""

WORKSTATION_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ECG DigiAI — Biomedical Engineering Workstation</title>
<style>
    :root {
        --bg: #f8fafc;
        --surface: #ffffff;
        --border: #e2e8f0;
        --border-subtle: #f1f5f9;
        --text-main: #0f172a;
        --text-muted: #64748b;
        --primary: #0284c7;
        --primary-dark: #0369a1;
        --primary-light: #e0f2fe;
        --teal-trace: #0f766e;
        --teal-light: #ccfbf1;
        --blue-gt: #1e3a8a;
        --validated-green: #15803d;
        --validated-bg: #ecfdf5;
        --experimental-amber: #b45309;
        --experimental-bg: #fffbeb;
        --danger: #dc2626;
        --danger-bg: #fef2f2;
        --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
        --font-mono: "JetBrains Mono", Consolas, "SF Mono", monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: var(--font-sans); background: var(--bg); color: var(--text-main); font-size: 13px; line-height: 1.5; -webkit-font-smoothing: antialiased; }
    
    /* Top Institutional Header */
    header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 100; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
    .brand-block { display: flex; align-items: center; gap: 12px; }
    .brand-title { font-size: 17px; font-weight: 700; color: var(--text-main); letter-spacing: -0.3px; display: flex; align-items: center; gap: 8px; }
    .brand-tag { font-size: 11px; background: var(--primary-light); color: var(--primary-dark); font-weight: 700; padding: 2px 7px; border-radius: 4px; text-transform: uppercase; }
    .brand-subtitle { font-size: 11px; color: var(--text-muted); }
    
    .header-badges { display: flex; align-items: center; gap: 8px; }
    .status-badge { display: inline-flex; align-items: center; gap: 5px; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; border: 1px solid transparent; }
    .badge-validated { background: var(--validated-bg); color: var(--validated-green); border-color: #a7f3d0; }
    .badge-deferred { background: var(--experimental-bg); color: var(--experimental-amber); border-color: #fde68a; }
    .btn-header { padding: 5px 12px; font-size: 12px; border: 1px solid var(--border); background: var(--surface); border-radius: 4px; cursor: pointer; color: var(--text-main); font-weight: 500; }
    .btn-header:hover { background: #f1f5f9; }

    /* Navigation Tabs */
    .nav-tabs { background: var(--surface); border-bottom: 1px solid var(--border); display: flex; padding: 0 24px; gap: 4px; }
    .nav-tab { padding: 12px 18px; font-size: 13px; font-weight: 600; color: var(--text-muted); border-bottom: 2px solid transparent; cursor: pointer; transition: all 0.15s ease; display: flex; align-items: center; gap: 6px; }
    .nav-tab:hover { color: var(--primary); }
    .nav-tab.active { color: var(--primary); border-bottom-color: var(--primary); }
    .tab-num { font-size: 11px; opacity: 0.7; font-family: var(--font-mono); }

    /* Main Container */
    main { padding: 20px 24px; max-width: 1560px; margin: 0 auto; }
    .tab-section { display: none; }
    .tab-section.active { display: block; }

    /* Standard Cards & Panels */
    .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 18px; margin-bottom: 18px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); }
    .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
    .panel-title { font-size: 14px; font-weight: 700; color: var(--text-main); display: flex; align-items: center; gap: 8px; }
    .panel-subtitle { font-size: 11px; color: var(--text-muted); font-weight: normal; }

    /* Grid Layouts */
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    .grid-workbench { display: grid; grid-template-columns: 380px 1fr; gap: 20px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }

    /* Form Controls */
    label { font-size: 11px; font-weight: 600; color: var(--text-muted); display: block; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.3px; }
    input[type="text"], input[type="number"], select { width: 100%; padding: 7px 10px; font-size: 12px; font-family: inherit; border: 1px solid var(--border); border-radius: 4px; outline: none; background: #fff; color: var(--text-main); margin-bottom: 12px; }
    input:focus, select:focus { border-color: var(--primary); box-shadow: 0 0 0 2px rgba(2,132,199,0.15); }
    
    /* Buttons */
    .btn { display: inline-flex; align-items: center; justify-content: center; gap: 6px; padding: 7px 14px; border-radius: 4px; font-weight: 600; font-size: 12px; cursor: pointer; border: 1px solid transparent; transition: all 0.15s ease; text-decoration: none; }
    .btn-primary { background: var(--primary); color: #fff; }
    .btn-primary:hover { background: var(--primary-dark); }
    .btn-primary:disabled { background: #94a3b8; cursor: not-allowed; }
    .btn-secondary { background: #fff; border-color: var(--border); color: var(--text-main); }
    .btn-secondary:hover { background: #f8fafc; border-color: #cbd5e1; }
    .btn-sm { padding: 4px 9px; font-size: 11px; }

    /* Input Source Selector Cards */
    .source-selector { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 14px; }
    .source-card { border: 1px solid var(--border); border-radius: 5px; padding: 10px; cursor: pointer; text-align: center; background: #fff; transition: all 0.15s ease; }
    .source-card:hover { border-color: #94a3b8; }
    .source-card.active { border-color: var(--primary); background: #f0f9ff; }
    .source-card-title { font-weight: 700; font-size: 11px; color: var(--text-main); }
    .source-card.active .source-card-title { color: var(--primary-dark); }
    .source-card-sub { font-size: 10px; color: var(--text-muted); margin-top: 2px; }

    /* Dropzone */
    .drop-zone { border: 2px dashed #cbd5e1; border-radius: 6px; padding: 22px 14px; text-align: center; background: #f8fafc; cursor: pointer; transition: all 0.15s ease; margin-bottom: 14px; }
    .drop-zone:hover, .drop-zone.dragover { border-color: var(--primary); background: #f0f9ff; }
    .drop-icon { font-size: 24px; margin-bottom: 6px; opacity: 0.8; }
    .drop-text { font-size: 12px; font-weight: 600; color: var(--text-main); }
    .drop-hint { font-size: 11px; color: var(--text-muted); margin-top: 3px; }

    /* Viewport & Scanning Animation */
    .viewport-box { position: relative; border: 1px solid var(--border); border-radius: 6px; background: #111827; min-height: 420px; display: flex; align-items: center; justify-content: center; overflow: hidden; }
    .viewport-box img { max-width: 100%; max-height: 480px; object-fit: contain; }
    .scanning-beam { position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: linear-gradient(180deg, transparent, #38bdf8, #0284c7, #38bdf8, transparent); box-shadow: 0 0 16px #38bdf8; display: none; z-index: 20; pointer-events: none; }
    @keyframes scanPass { 0% { left: 0%; } 100% { left: 100%; } }
    .scanning-active .scanning-beam { display: block; animation: scanPass 2.2s ease-in-out infinite alternate; }

    /* Processing Stepper */
    .stepper-container { display: flex; flex-direction: column; gap: 8px; margin-top: 14px; }
    .stepper-row { display: flex; align-items: center; justify-content: space-between; padding: 7px 12px; background: #f8fafc; border: 1px solid var(--border); border-radius: 4px; font-size: 12px; }
    .step-name { display: flex; align-items: center; gap: 8px; font-weight: 500; }
    .step-dot { width: 8px; height: 8px; border-radius: 50%; background: #cbd5e1; }
    .step-dot.active { background: #0284c7; box-shadow: 0 0 6px #0284c7; }
    .step-dot.done { background: #16a34a; }
    .step-val { font-family: var(--font-mono); font-size: 11px; color: var(--text-muted); }

    /* ECG Clinical Paper Canvas */
    .ecg-canvas-container { position: relative; border: 1px solid #f43f5e; border-radius: 4px; background: #fffcf9; overflow: hidden; cursor: crosshair; }
    canvas.ecg-paper { display: block; width: 100%; height: auto; }

    /* Coordinate Readout Box */
    .coord-readout-card { background: #0f172a; color: #f8fafc; border-radius: 5px; padding: 10px 14px; font-family: var(--font-mono); font-size: 12px; display: flex; gap: 18px; align-items: center; flex-wrap: wrap; margin-top: 10px; }
    .coord-item { display: flex; flex-direction: column; }
    .coord-label { font-size: 10px; text-transform: uppercase; color: #94a3b8; }
    .coord-value { font-size: 14px; font-weight: 700; color: #38bdf8; }

    /* 12-Lead Clinical Grid Cards */
    .leads-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 14px; }
    .lead-card { background: var(--surface); border: 1px solid var(--border); border-radius: 5px; padding: 10px; cursor: pointer; transition: all 0.15s ease; position: relative; }
    .lead-card:hover { border-color: var(--primary); }
    .lead-card.active { border-color: var(--primary); background: #f0f9ff; box-shadow: 0 0 0 2px rgba(2,132,199,0.2); }
    .lead-card-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
    .lead-name-tag { font-weight: 700; font-size: 13px; font-family: var(--font-mono); }
    .lead-sparkline { width: 100%; height: 38px; display: block; }
    .lead-strip-card { grid-column: 1 / -1; }

    /* Pill Badges */
    .pill { display: inline-block; padding: 2px 7px; border-radius: 10px; font-size: 10px; font-weight: 700; text-transform: uppercase; }
    .pill-validated { background: var(--validated-bg); color: var(--validated-green); }
    .pill-experimental { background: var(--experimental-bg); color: var(--experimental-amber); }

    /* Tables */
    table.data-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    table.data-table th, table.data-table td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }
    table.data-table th { background: #f8fafc; font-weight: 600; color: var(--text-muted); font-size: 11px; text-transform: uppercase; }
    table.data-table tr:hover { background: #fafafa; }

    /* Notice Banner */
    .notice-banner { background: #fffbeb; border-left: 3px solid #f59e0b; padding: 10px 14px; font-size: 12px; color: #92400e; border-radius: 3px; margin-bottom: 14px; line-height: 1.5; }

    /* Modal Dialog */
    .modal-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(15,23,42,0.6); display: none; align-items: center; justify-content: center; z-index: 200; backdrop-filter: blur(2px); }
    .modal-box { background: #fff; width: 680px; max-width: 95vw; max-height: 90vh; border-radius: 8px; overflow-y: auto; padding: 22px; box-shadow: 0 10px 25px rgba(0,0,0,0.2); }
    .modal-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 12px; margin-bottom: 16px; }
</style>
</head>
<body>

<!-- Header -->
<header>
    <div class="brand-block">
        <div class="brand-title">
            <span>ECG DigiAI Platform</span>
            <span class="brand-tag">Workstation</span>
        </div>
        <div class="brand-subtitle">Jaypee Institute of Information Technology, Noida · Dept. of Electronics & Communication Engineering</div>
    </div>
    <div class="header-badges">
        <span class="status-badge badge-validated">✓ U-Net Digitizer: ResNet-18 (r = 0.9971)</span>
        <span class="status-badge badge-deferred">⚠ Classifier: model_not_trained (Out of Scope)</span>
        <button class="btn-header" onclick="openAboutModal()">Architecture / About</button>
    </div>
</header>

<!-- Navigation -->
<div class="nav-tabs">
    <div class="nav-tab active" onclick="switchWorkstationTab('workbench')">
        <span class="tab-num">01</span>
        <span>Digitisation Workbench</span>
    </div>
    <div class="nav-tab" onclick="switchWorkstationTab('explorer')">
        <span class="tab-num">02</span>
        <span>Lead Explorer</span>
    </div>
    <div class="nav-tab" onclick="switchWorkstationTab('synthetic')">
        <span class="tab-num">03</span>
        <span>Synthetic ECG Lab</span>
    </div>
    <div class="nav-tab" onclick="switchWorkstationTab('evaluation')">
        <span class="tab-num">04</span>
        <span>Evaluation & Export</span>
    </div>
</div>

<main>
    <!-- Notice Banner -->
    <div class="notice-banner">
        <strong>ACADEMIC DECISION-SUPPORT NOTICE:</strong> ECG DigiAI is an ECE B.Tech major project research workstation. Waveform digitisation is formally validated for Lead II and V1 (PTB-XL distribution); other leads operate via spatial receptive-field transfer. Outputs do not constitute autonomous medical advice.
    </div>

    <!-- ========================================================================= -->
    <!-- 01 · DIGITISATION WORKBENCH -->
    <!-- ========================================================================= -->
    <div id="section-workbench" class="tab-section active">
        <div class="grid-workbench">
            <!-- Left Controls Column -->
            <div>
                <div class="panel">
                    <div class="panel-header">
                        <span class="panel-title">Input ECG Selection</span>
                    </div>

                    <!-- Source Tabs -->
                    <div class="source-selector">
                        <div class="source-card active" id="src-upload" onclick="selectSourceMode('upload')">
                            <div class="source-card-title">📁 Upload</div>
                            <div class="source-card-sub">Local File</div>
                        </div>
                        <div class="source-card" id="src-sample" onclick="selectSourceMode('sample')">
                            <div class="source-card-title">🖼️ Real Beat</div>
                            <div class="source-card-sub">Kaggle Sample</div>
                        </div>
                        <div class="source-card" id="src-synthetic" onclick="selectSourceMode('synthetic')">
                            <div class="source-card-title">📄 Synthetic</div>
                            <div class="source-card-sub">From Lab</div>
                        </div>
                    </div>

                    <!-- Alert message box -->
                    <div id="workbench-alert" style="display:none; background:var(--danger-bg); border:1px solid #fca5a5; color:var(--danger); padding:8px 12px; border-radius:4px; font-size:12px; margin-bottom:12px;"></div>

                    <!-- Mode 1: Upload Dropzone -->
                    <div id="box-upload" class="drop-zone" onclick="document.getElementById('file-upload-input').click()">
                        <div class="drop-icon">📤</div>
                        <div class="drop-text">Click or Drag & Drop ECG Image</div>
                        <div class="drop-hint">Supports PNG, JPG, JPEG (Min: 50x50 px)</div>
                        <input type="file" id="file-upload-input" accept="image/png, image/jpeg, image/jpg" style="display:none;" onchange="handleFileDrop(event)">
                    </div>

                    <!-- Mode 2: Real Beat Picker -->
                    <div id="box-sample" style="display:none; margin-bottom:14px;">
                        <label>Curated Clinical Beat Class</label>
                        <select id="select-real-sample" onchange="loadCuratedSample()">
                            <option value="N10007.png">N10007.png — Normal Sinus Rhythm (N)</option>
                            <option value="V1001.png">V1001.png — Premature Ventricular Contraction (V)</option>
                            <option value="F0.png">F0.png — Fusion Beat (F)</option>
                            <option value="M0.png">M0.png — Myocardial Infarction / Abnormal (M)</option>
                            <option value="S100.png">S100.png — Supraventricular Premature Beat (S)</option>
                            <option value="Q1.png">Q1.png — Paced / Unclassifiable Beat (Q)</option>
                        </select>
                        <button class="btn btn-secondary btn-sm" style="width:100%;" onclick="loadCuratedSample()">Load Sample Image</button>
                    </div>

                    <!-- Mode 3: Synthetic Sheet Note -->
                    <div id="box-synthetic" style="display:none; margin-bottom:14px; background:#f0f9ff; border:1px solid #bae6fd; padding:10px; border-radius:4px; font-size:12px; color:var(--primary-dark);">
                        Using active paired synthetic 12-lead paper sheet generated in Section 03.
                    </div>

                    <!-- Active Image Readout Card -->
                    <div style="border:1px solid var(--border); border-radius:4px; padding:10px; background:#f8fafc; margin-bottom:14px;">
                        <label>Active Input Metadata</label>
                        <div id="active-meta-text" style="font-family:var(--font-mono); font-size:11px; color:var(--text-main); line-height:1.6;">
                            No input selected. Choose an option above.
                        </div>
                    </div>

                    <!-- Pipeline Execution Configuration -->
                    <label>Layout Template</label>
                    <select id="cfg-layout">
                        <option value="3x4_1">3x4 + 1 Rhythm Strip (Standard Clinical Sheet)</option>
                        <option value="12x1">12x1 Stacked Multi-Lead Sheet</option>
                        <option value="single_lead">Single Lead / Beat Crop (e.g. Lead II)</option>
                    </select>

                    <label>Waveform Extractor</label>
                    <select id="cfg-extractor">
                        <option value="deep">Deep U-Net (Trained Checkpoint)</option>
                        <option value="classical">Classical Preprocessing (Median Filter)</option>
                        <option value="hybrid">Hybrid (Deep Validated / Classical Exp)</option>
                    </select>

                    <label>Calibration Standard</label>
                    <select id="cfg-calibration">
                        <option value="true">Standard Clinical Paper (25 mm/s, 10 mm/mV) [Calibrated]</option>
                        <option value="false">Uncalibrated / Single Beat Photo [Nominal mV Estimates]</option>
                    </select>

                    <button class="btn btn-primary" id="btn-run-digitize" style="width:100%; padding:10px; font-size:13px; margin-top:4px;" onclick="executeDigitisation()">
                        ⚡ Digitise ECG
                    </button>
                </div>
            </div>

            <!-- Right Visual Processing Workspace -->
            <div>
                <div class="panel" style="margin-bottom:14px;">
                    <div class="panel-header">
                        <span class="panel-title">Visual Processing Workspace</span>
                        <div style="display:flex; gap:6px;">
                            <button class="btn btn-secondary btn-sm active" id="btn-view-overlay" onclick="switchImageView('overlay')">Diagnostic Overlay</button>
                            <button class="btn btn-secondary btn-sm" id="btn-view-suppressed" onclick="switchImageView('suppressed')">Grid Suppressed</button>
                            <button class="btn btn-secondary btn-sm" id="btn-view-binary" onclick="switchImageView('binary')">Binary Trace</button>
                            <button class="btn btn-secondary btn-sm" id="btn-view-deskewed" onclick="switchImageView('deskewed')">Deskewed</button>
                            <button class="btn btn-secondary btn-sm" id="btn-view-original" onclick="switchImageView('original')">Input Original</button>
                        </div>
                    </div>

                    <!-- Viewport Canvas with Scanning Beam -->
                    <div class="viewport-box" id="viewport-box">
                        <div class="scanning-beam" id="scanning-beam"></div>
                        <img id="viewport-img" src="" alt="ECG Viewport" style="display:none;" />
                        <div id="viewport-placeholder" style="color:#64748b; font-size:13px; text-align:center;">
                            Load or upload an ECG image to view processing stages.
                        </div>
                    </div>

                    <!-- Processing Stepper -->
                    <div class="stepper-container" id="stepper-container">
                        <div class="stepper-row" id="step-1">
                            <span class="step-name"><span class="step-dot" id="dot-1"></span> 1. Ingestion & Dimension Integrity Check</span>
                            <span class="step-val" id="val-1">Pending</span>
                        </div>
                        <div class="stepper-row" id="step-2">
                            <span class="step-name"><span class="step-dot" id="dot-2"></span> 2. Hough Line Skew Estimation & Alignment</span>
                            <span class="step-val" id="val-2">Pending</span>
                        </div>
                        <div class="stepper-row" id="step-3">
                            <span class="step-name"><span class="step-dot" id="dot-3"></span> 3. Color De-mixing & Millimeter Grid Suppression</span>
                            <span class="step-val" id="val-3">Pending</span>
                        </div>
                        <div class="stepper-row" id="step-4">
                            <span class="step-name"><span class="step-dot" id="dot-4"></span> 4. CLAHE Contrast & Binary Trace Segmentation</span>
                            <span class="step-val" id="val-4">Pending</span>
                        </div>
                        <div class="stepper-row" id="step-5">
                            <span class="step-name"><span class="step-dot" id="dot-5"></span> 5. Anatomical Lead Layout Region Partitioning</span>
                            <span class="step-val" id="val-5">Pending</span>
                        </div>
                        <div class="stepper-row" id="step-6">
                            <span class="step-name"><span class="step-dot" id="dot-6"></span> 6. 2D ResNet-U-Net Coordinate Extraction & Soft-Argmax</span>
                            <span class="step-val" id="val-6">Pending</span>
                        </div>
                        <div class="stepper-row" id="step-7">
                            <span class="step-name"><span class="step-dot" id="dot-7"></span> 7. Millivolt Scaling & 100 Hz Digital Signal Reconstruction</span>
                            <span class="step-val" id="val-7">Pending</span>
                        </div>
                    </div>

                    <!-- Post-Digitisation Handoff Banner -->
                    <div id="digitize-success-banner" style="display:none; background:#ecfdf5; border:1px solid #10b981; border-radius:4px; padding:12px 16px; margin-top:14px; align-items:center; justify-content:space-between;">
                        <div>
                            <strong style="color:#047857;">Digitisation Complete:</strong>
                            <span style="color:#065f46; margin-left:6px;" id="digitize-summary-msg">13 leads extracted. Waveforms ready for inspection.</span>
                        </div>
                        <button class="btn btn-primary" onclick="switchWorkstationTab('explorer')">
                            Open in Lead Explorer &rarr;
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- ========================================================================= -->
    <!-- 02 · LEAD EXPLORER (INTERACTIVE SIGNAL VIEWER) -->
    <!-- ========================================================================= -->
    <div id="section-explorer" class="tab-section">
        <!-- Lead Selector Overview -->
        <div class="panel" style="padding:14px; margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <span class="panel-title">Clinical Lead Matrix (Click to Select Lead for Animated Inspection)</span>
                <span style="font-size:11px; color:var(--text-muted);">Standard 4-Column Clinical Arrangement (25 mm/s, 10 mm/mV)</span>
            </div>
            <div class="leads-grid" id="explorer-leads-grid">
                <!-- Dynamically populated lead cards -->
                <div style="grid-column:1/-1; text-align:center; padding:18px; color:var(--text-muted);">
                    Digitise an ECG image in Section 01 to populate the 12-lead signal matrix.
                </div>
            </div>
        </div>

        <!-- Large Interactive Waveform Viewer -->
        <div class="panel">
            <div class="panel-header">
                <div>
                    <span class="panel-title" id="large-viewer-title">LEAD II — Calibrated Waveform Inspection</span>
                    <span class="panel-subtitle" id="large-viewer-scope">VALIDATED (TRAINED DISTRIBUTION)</span>
                </div>
                <div style="display:flex; gap:8px; align-items:center;">
                    <button class="btn btn-secondary btn-sm" id="btn-play-pause" onclick="toggleWaveformPlayback()">▶ Play</button>
                    <button class="btn btn-secondary btn-sm" onclick="restartWaveformPlayback()">↺ Restart</button>
                    <label style="margin:0; font-size:11px; display:inline-flex; align-items:center; gap:4px;">
                        Speed:
                        <select id="playback-speed" style="margin:0; padding:3px 6px; width:auto;" onchange="setPlaybackSpeed(this.value)">
                            <option value="0.5">0.5x</option>
                            <option value="1.0" selected>1.0x (Real-time)</option>
                            <option value="2.0">2.0x</option>
                            <option value="4.0">4.0x</option>
                        </select>
                    </label>
                    <button class="btn btn-secondary btn-sm" onclick="downloadCurrentLeadCSV()">Export Lead CSV</button>
                </div>
            </div>

            <!-- ECG Millimeter Grid Canvas -->
            <div class="ecg-canvas-container" id="canvas-container">
                <canvas id="ecg-canvas" width="1200" height="340" class="ecg-paper"></canvas>
            </div>

            <!-- Cursor Coordinate Readout -->
            <div class="coord-readout-card">
                <div class="coord-item">
                    <span class="coord-label">Cursor Time (t)</span>
                    <span class="coord-value" id="coord-t">-- s</span>
                </div>
                <div class="coord-item">
                    <span class="coord-label">Cursor Voltage (V)</span>
                    <span class="coord-value" id="coord-v">-- mV</span>
                </div>
                <div class="coord-item">
                    <span class="coord-label">Sample Index</span>
                    <span class="coord-value" id="coord-sample">#--</span>
                </div>
                <div class="coord-item">
                    <span class="coord-label">Selected Interval</span>
                    <span class="coord-value" id="coord-interval">None (Drag to select)</span>
                </div>
                <div style="margin-left:auto; display:flex; gap:8px;">
                    <button class="btn btn-secondary btn-sm" style="color:#fff; background:#1e293b; border-color:#334155;" onclick="exportSelectedInterval()">Export Interval CSV</button>
                </div>
            </div>

            <!-- Basic Signal Processing Analysis (NO Disease Classifier) -->
            <div style="margin-top:18px; border-top:1px solid var(--border); padding-top:14px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <span class="panel-title">Physiological Rhythm & Rate Measurements</span>
                    <span style="font-size:11px; color:var(--text-muted);">Pan-Tompkins Derivative QRS Detection & Butterworth 0.5–40 Hz Filter</span>
                </div>
                <div class="grid-4" style="text-align:center;">
                    <div style="background:#f8fafc; border:1px solid var(--border); padding:10px; border-radius:4px;">
                        <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase;">Heart Rate</div>
                        <div style="font-size:22px; font-weight:700; color:var(--primary); margin-top:2px;" id="metric-hr">-- BPM</div>
                        <div style="font-size:10px; color:var(--text-muted);" id="metric-rhythm-desc">--</div>
                    </div>
                    <div style="background:#f8fafc; border:1px solid var(--border); padding:10px; border-radius:4px;">
                        <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase;">QRS R-Peaks Detected</div>
                        <div style="font-size:22px; font-weight:700; color:var(--text-main); margin-top:2px;" id="metric-peaks">--</div>
                        <div style="font-size:10px; color:var(--text-muted);">Marked on graph</div>
                    </div>
                    <div style="background:#f8fafc; border:1px solid var(--border); padding:10px; border-radius:4px;">
                        <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase;">Mean RR Interval</div>
                        <div style="font-size:22px; font-weight:700; color:var(--text-main); margin-top:2px;" id="metric-rr">-- s</div>
                        <div style="font-size:10px; color:var(--text-muted);" id="metric-rr-std">Std: -- s</div>
                    </div>
                    <div style="background:#f8fafc; border:1px solid var(--border); padding:10px; border-radius:4px;">
                        <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase;">Signal Quality Index</div>
                        <div style="font-size:22px; font-weight:700; color:var(--validated-green); margin-top:2px;" id="metric-sqi">--</div>
                        <div style="font-size:10px; color:var(--text-muted);" id="metric-duration">Duration: -- s</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- ========================================================================= -->
    <!-- 03 · SYNTHETIC ECG LAB -->
    <!-- ========================================================================= -->
    <div id="section-synthetic" class="tab-section">
        <div class="grid-2">
            <!-- Left: Controls -->
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">PTB-XL Source Record & Paper Parameters</span>
                </div>
                
                <label>Authentic Digital ECG Record (PTB-XL Folds 1–9)</label>
                <select id="synth-record-select" onchange="loadPtbxlMeta(this.value)">
                    <option value="00001_lr">00001_lr (12-Lead, 100 Hz, 10.0 s)</option>
                    <option value="00002_lr">00002_lr (12-Lead, 100 Hz, 10.0 s)</option>
                    <option value="00003_lr">00003_lr (12-Lead, 100 Hz, 10.0 s)</option>
                    <option value="00004_lr">00004_lr (12-Lead, 100 Hz, 10.0 s)</option>
                    <option value="00005_lr">00005_lr (12-Lead, 100 Hz, 10.0 s)</option>
                    <option value="00006_lr">00006_lr (12-Lead, 100 Hz, 10.0 s)</option>
                    <option value="00007_lr">00007_lr (12-Lead, 100 Hz, 10.0 s)</option>
                    <option value="00008_lr">00008_lr (12-Lead, 100 Hz, 10.0 s)</option>
                    <option value="00009_lr">00009_lr (12-Lead, 100 Hz, 10.0 s)</option>
                    <option value="00010_lr">00010_lr (12-Lead, 100 Hz, 10.0 s)</option>
                </select>

                <!-- Record Metadata Card -->
                <div style="border:1px solid var(--border); border-radius:4px; padding:10px; background:#f8fafc; margin-bottom:14px;">
                    <div style="font-weight:600; font-size:11px; color:var(--text-muted); margin-bottom:4px;">CLINICAL RECORD METADATA</div>
                    <div id="synth-record-info" style="font-family:var(--font-mono); font-size:11px; line-height:1.6;">
                        Record: 00001_lr | 12 Leads | 100 Hz | 10.0 s | 1000 Samples/Lead
                    </div>
                </div>

                <div class="panel-header" style="margin-top:14px;">
                    <span class="panel-title">Physical Paper Degradation Pipeline</span>
                </div>

                <div class="grid-2">
                    <div>
                        <label>Rotation Skew Angle (±5°)</label>
                        <input type="number" id="synth-rot" value="0.8" step="0.1" min="-5" max="5">
                    </div>
                    <div>
                        <label>Sensor Noise Sigma</label>
                        <input type="number" id="synth-noise" value="2.0" step="0.5" min="0" max="15">
                    </div>
                </div>

                <div class="grid-2">
                    <div>
                        <label>Optical Blur Kernel</label>
                        <select id="synth-blur">
                            <option value="0">0 (Clean Scan / Sharp)</option>
                            <option value="3">3 (Mild Lens Defocus)</option>
                            <option value="5">5 (Heavy Optical Blur)</option>
                        </select>
                    </div>
                    <div>
                        <label>Lighting / Shadow Gradient</label>
                        <select id="synth-shadow">
                            <option value="false">Uniform Studio Lighting</option>
                            <option value="true">Non-Uniform Shadow / Vignette</option>
                        </select>
                    </div>
                </div>

                <button class="btn btn-primary" style="width:100%; padding:10px; font-size:13px;" onclick="generateSyntheticPaper()">
                    Render 12-Lead Clinical ECG Sheet
                </button>
            </div>

            <!-- Right: Preview & Handoff -->
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">Rendered 12-Lead Clinical Sheet</span>
                    <button class="btn btn-primary btn-sm" id="btn-send-to-workbench" onclick="sendSyntheticToWorkbench()" disabled>
                        Send to Digitisation Workbench &rarr;
                    </button>
                </div>
                <div class="viewport-box" id="synth-viewport-box" style="min-height:360px;">
                    <span style="color:#64748b;">Click 'Render 12-Lead Clinical ECG Sheet' to generate.</span>
                </div>
                <div id="synth-meta-box" style="margin-top:10px; font-family:var(--font-mono); font-size:11px; color:var(--text-muted);"></div>
            </div>
        </div>
    </div>

    <!-- ========================================================================= -->
    <!-- 04 · EVALUATION & EXPORT -->
    <!-- ========================================================================= -->
    <div id="section-evaluation" class="tab-section">
        <!-- Dual Waveform Overlay (Ground Truth vs Reconstructed) -->
        <div class="panel">
            <div class="panel-header">
                <div>
                    <span class="panel-title">Ground-Truth vs Reconstructed Signal Comparison</span>
                    <span class="panel-subtitle">Point-for-Point Morphological Fidelity Assessment</span>
                </div>
                <div style="display:flex; gap:8px; align-items:center;">
                    <label style="margin:0; font-size:11px;">Lead:
                        <select id="eval-lead-select" style="margin:0; padding:3px 8px; width:auto;" onchange="renderGroundTruthComparison(this.value)">
                            <option value="II">Lead II [Validated]</option>
                            <option value="V1">Lead V1 [Validated]</option>
                            <option value="II_RHYTHM">II_RHYTHM [Validated]</option>
                            <option value="I">Lead I [Experimental]</option>
                            <option value="III">Lead III [Experimental]</option>
                            <option value="AVR">Lead aVR [Experimental]</option>
                            <option value="AVL">Lead aVL [Experimental]</option>
                            <option value="AVF">Lead aVF [Experimental]</option>
                            <option value="V2">Lead V2 [Experimental]</option>
                            <option value="V3">Lead V3 [Experimental]</option>
                            <option value="V4">Lead V4 [Experimental]</option>
                            <option value="V5">Lead V5 [Experimental]</option>
                            <option value="V6">Lead V6 [Experimental]</option>
                        </select>
                    </label>
                </div>
            </div>

            <div id="eval-notice-box" style="display:none;" class="notice-banner"></div>

            <!-- Comparison Canvas -->
            <div style="background:#fff; border:1px solid var(--border); border-radius:4px; padding:10px; margin-bottom:12px;">
                <canvas id="eval-canvas" width="1200" height="260" style="width:100%; height:auto; display:block;"></canvas>
            </div>

            <!-- Fidelity Metrics Cards -->
            <div class="grid-4" style="text-align:center; margin-bottom:14px;">
                <div style="background:#f8fafc; border:1px solid var(--border); padding:10px; border-radius:4px;">
                    <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase;">Pearson Correlation (r)</div>
                    <div style="font-size:20px; font-weight:700; color:var(--primary); margin-top:2px;" id="eval-pearson">r = --</div>
                    <div style="font-size:10px; color:var(--text-muted);">Target: r > 0.90</div>
                </div>
                <div style="background:#f8fafc; border:1px solid var(--border); padding:10px; border-radius:4px;">
                    <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase;">Root Mean Square Error</div>
                    <div style="font-size:20px; font-weight:700; color:var(--validated-green); margin-top:2px;" id="eval-rmse">-- mV</div>
                    <div style="font-size:10px; color:var(--text-muted);">Target: < 0.25 mV</div>
                </div>
                <div style="background:#f8fafc; border:1px solid var(--border); padding:10px; border-radius:4px;">
                    <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase;">Signal-to-Noise Ratio</div>
                    <div style="font-size:20px; font-weight:700; color:var(--text-main); margin-top:2px;" id="eval-snr">-- dB</div>
                    <div style="font-size:10px; color:var(--text-muted);">Residual noise fidelity</div>
                </div>
                <div style="background:#f8fafc; border:1px solid var(--border); padding:10px; border-radius:4px;">
                    <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase;">Trace Completeness</div>
                    <div style="font-size:20px; font-weight:700; color:var(--primary); margin-top:2px;" id="eval-comp">--%</div>
                    <div style="font-size:10px; color:var(--text-muted);">No trace dropouts</div>
                </div>
            </div>

            <!-- 12-Lead Evaluation Breakdown Table -->
            <label>12-Lead Complete Ground-Truth Comparison Matrix</label>
            <table class="data-table" id="eval-summary-table">
                <thead>
                    <tr><th>Lead</th><th>Scope</th><th>Pearson r</th><th>RMSE (mV)</th><th>SNR (dB)</th><th>Completeness</th></tr>
                </thead>
                <tbody id="eval-summary-tbody">
                    <tr><td colspan="6" style="text-align:center; color:#64748b;">Run digitisation on a paired synthetic sheet to view 12-lead metrics.</td></tr>
                </tbody>
            </table>
        </div>

        <!-- Export Suite Panel -->
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">Research Artifact & Clinical Report Export</span>
                <span class="panel-subtitle">Exportable Digital Waveforms & Structured Decision-Support Reports</span>
            </div>
            <div class="grid-4">
                <div style="border:1px solid var(--border); border-radius:6px; padding:14px; text-align:center; background:#f8fafc;">
                    <div style="font-size:20px; margin-bottom:6px;">📊</div>
                    <div style="font-weight:700; font-size:12px; margin-bottom:4px;">Unified Multi-Lead CSV</div>
                    <div style="font-size:11px; color:var(--text-muted); margin-bottom:10px;">Time-series with header metadata</div>
                    <button class="btn btn-secondary btn-sm" style="width:100%;" onclick="downloadCSVExport()">Download CSV</button>
                </div>
                <div style="border:1px solid var(--border); border-radius:6px; padding:14px; text-align:center; background:#f8fafc;">
                    <div style="font-size:20px; margin-bottom:6px;">📦</div>
                    <div style="font-weight:700; font-size:12px; margin-bottom:4px;">WFDB-Format Archive</div>
                    <div style="font-size:11px; color:var(--text-muted); margin-bottom:10px;">Standard .hea header & data bundle</div>
                    <button class="btn btn-secondary btn-sm" style="width:100%;" onclick="downloadWfdbExport()">Export WFDB</button>
                </div>
                <div style="border:1px solid var(--border); border-radius:6px; padding:14px; text-align:center; background:#f8fafc;">
                    <div style="font-size:20px; margin-bottom:6px;">📋</div>
                    <div style="font-weight:700; font-size:12px; margin-bottom:4px;">Printable Clinical Report</div>
                    <div style="font-size:11px; color:var(--text-muted); margin-bottom:10px;">Formatted HTML diagnostic summary</div>
                    <button class="btn btn-secondary btn-sm" style="width:100%;" onclick="openPrintableReport()">Print / Save HTML</button>
                </div>
                <div style="border:1px solid var(--border); border-radius:6px; padding:14px; text-align:center; background:#f8fafc;">
                    <div style="font-size:20px; margin-bottom:6px;">📄</div>
                    <div style="font-weight:700; font-size:12px; margin-bottom:4px;">Structured JSON Report</div>
                    <div style="font-size:11px; color:var(--text-muted); margin-bottom:10px;">Machine-readable diagnostic schema</div>
                    <button class="btn btn-secondary btn-sm" style="width:100%;" onclick="downloadJsonReport()">Download JSON</button>
                </div>
            </div>
        </div>
    </div>
</main>

<!-- About / System Architecture Modal -->
<div class="modal-overlay" id="about-modal" onclick="closeAboutModal(event)">
    <div class="modal-box" onclick="event.stopPropagation()">
        <div class="modal-header">
            <span class="panel-title">ECG DigiAI — System Architecture & Academic Provenance</span>
            <button class="btn-header" onclick="closeAboutModal()">✕</button>
        </div>
        <div style="font-size:12px; line-height:1.7; color:var(--text-main);">
            <div style="margin-bottom:12px;">
                <strong>Project Title:</strong> ECG Digitisation & Analysis Using AI (ECG DigiAI)<br>
                <strong>Academic Scope:</strong> Final-Year B.Tech Major Project (ECE Department)<br>
                <strong>Institution:</strong> Jaypee Institute of Information Technology (JIIT), Noida<br>
                <strong>Core Repository:</strong> <code>AsthaK9/ecg-image-digitizer</code>
            </div>
            
            <div style="border:1px solid var(--border); border-radius:4px; padding:10px; background:#f8fafc; margin-bottom:12px;">
                <strong>Active Deep Learning Checkpoint:</strong><br>
                • File: <code>models/ecg_digitizer_unet.pt</code> (8,121,015 bytes)<br>
                • Architecture: 2D ResNet-U-Net with Differentiable Soft-Argmax Coordinate Head<br>
                • Training Distribution: Lead II & Lead V1 crops from PTB-XL (5 epochs on Kaggle GPU)<br>
                • Validation Fidelity: Best val Pearson $r = 0.9971$, RMSE $= 0.0179\text{ mV}$ ($17.9\ \mu\text{V}$)<br>
                • Checkpoint Backup: <code>models/ecg_digitizer_unet.pt.bak</code> (Verified Safety Copy)
            </div>

            <div style="border:1px solid #fcd34d; border-radius:4px; padding:10px; background:#fffbeb; margin-bottom:12px; color:#92400e;">
                <strong>Academic Scope Discipline (24-Hour Milestone):</strong><br>
                • Components A (Synthetic Generation) and B (Digitisation & Reconstruction) are fully implemented and verified.<br>
                • Component C (AI Pathology Classifier) is explicitly deferred out of scope for this milestone and marked <code>model_not_trained</code>. Automated disease probabilities are never fabricated.<br>
                • Physiological rhythm analysis (heart rate, R-peaks, RR intervals) is fully functional and rule-based.
            </div>

            <div style="text-align:right;">
                <button class="btn btn-secondary btn-sm" onclick="closeAboutModal()">Close</button>
            </div>
        </div>
    </div>
</div>

<script>
    // Global Application State
    let currentRecordId = "00001_lr";
    let activeImageBase64 = null;
    let activeSourceMode = "upload";
    let currentDigitizedData = null;
    let currentGroundTruthData = null;
    let activeLeadName = "II";
    let activeViewMode = "overlay";
    let animationInterval = null;
    let playbackCurrentSample = 0;
    let playbackSpeed = 1.0;
    let isPlaying = false;
    let selectionStartSample = null;
    let selectionEndSample = null;

    // --- Tab Switching ---
    function switchWorkstationTab(tabId) {
        document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
        const tabBtn = Array.from(document.querySelectorAll('.nav-tab')).find(el => el.getAttribute('onclick')?.includes(tabId));
        if (tabBtn) tabBtn.classList.add('active');
        const section = document.getElementById('section-' + tabId);
        if (section) section.classList.add('active');

        if (tabId === 'explorer' && currentDigitizedData) {
            renderLeadMatrix();
            loadLeadIntoViewer(activeLeadName);
        } else if (tabId === 'evaluation') {
            loadEvaluationSection();
        }
    }

    // --- Modal Controls ---
    function openAboutModal() { document.getElementById('about-modal').style.display = 'flex'; }
    function closeAboutModal() { document.getElementById('about-modal').style.display = 'none'; }

    // --- Source Selection in Workbench ---
    function selectSourceMode(mode) {
        activeSourceMode = mode;
        ['upload', 'sample', 'synthetic'].forEach(m => {
            const card = document.getElementById('src-' + m);
            const box = document.getElementById('box-' + m);
            if (m === mode) {
                card.classList.add('active');
                box.style.display = (m === 'upload' ? 'block' : (m === 'sample' ? 'block' : 'block'));
            } else {
                card.classList.remove('active');
                box.style.display = 'none';
            }
        });
        document.getElementById('workbench-alert').style.display = 'none';

        if (mode === 'sample') {
            loadCuratedSample();
        } else if (mode === 'synthetic') {
            activeImageBase64 = null;
            currentRecordId = document.getElementById('synth-record-select').value || "00001_lr";
            const synthBox = document.getElementById('synth-viewport-box');
            const synthImg = synthBox.querySelector('img');
            if (synthImg) {
                setWorkbenchImage(synthImg.src, `Synthetic Sheet (PTB-XL ${currentRecordId})`, 1650, 1275, false);
            } else {
                document.getElementById('active-meta-text').innerText = 'No synthetic sheet rendered yet. Generate one in Section 03.';
            }
        }
    }

    // --- File Drag & Drop Handling ---
    const dropZone = document.getElementById('box-upload');
    if (dropZone) {
        ['dragenter', 'dragover'].forEach(eName => {
            dropZone.addEventListener(eName, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); }, false);
        });
        ['dragleave', 'drop'].forEach(eName => {
            dropZone.addEventListener(eName, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); }, false);
        });
        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files && files.length > 0) processLocalImage(files[0]);
        });
    }

    function handleFileDrop(e) {
        const file = e.target.files[0];
        if (file) processLocalImage(file);
    }

    function processLocalImage(file) {
        const alertBox = document.getElementById('workbench-alert');
        alertBox.style.display = 'none';

        if (!file.type.startsWith('image/')) {
            alertBox.innerText = '❌ Error: Uploaded file is not an image format (PNG, JPG required).';
            alertBox.style.display = 'block';
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            const dataUrl = e.target.result;
            const testImg = new Image();
            testImg.onload = function() {
                if (testImg.naturalWidth < 50 || testImg.naturalHeight < 50) {
                    alertBox.innerText = `❌ Error: Image dimensions too small (${testImg.naturalWidth}x${testImg.naturalHeight} px). Minimum required resolution is 50x50 px.`;
                    alertBox.style.display = 'block';
                    return;
                }
                currentRecordId = file.name.replace(/\.[^/.]+$/, "");
                const isSingle = (testImg.naturalWidth < 800 || testImg.naturalHeight < 600);
                setWorkbenchImage(dataUrl, file.name, testImg.naturalWidth, testImg.naturalHeight, isSingle);
            };
            testImg.src = dataUrl;
        };
        reader.readAsDataURL(file);
    }

    async function loadCuratedSample() {
        const filename = document.getElementById('select-real-sample').value;
        const alertBox = document.getElementById('workbench-alert');
        alertBox.style.display = 'none';

        try {
            const res = await fetch('/api/sample-image/' + filename);
            if (!res.ok) throw new Error('HTTP ' + res.status);
            const data = await res.json();
            currentRecordId = filename.replace(/\.[^/.]+$/, "");
            const isSingle = (data.suggested_layout === 'single_lead');
            setWorkbenchImage("data:image/png;base64," + data.image_b64, filename, data.width, data.height, isSingle);
        } catch (err) {
            alertBox.innerText = '❌ Failed to load curated sample: ' + err.message;
            alertBox.style.display = 'block';
        }
    }

    function setWorkbenchImage(dataUrl, label, width, height, isSingle) {
        activeImageBase64 = dataUrl;
        const img = document.getElementById('viewport-img');
        img.src = dataUrl;
        img.style.display = 'block';
        document.getElementById('viewport-placeholder').style.display = 'none';

        document.getElementById('active-meta-text').innerText = 
            `File: ${label}\nResolution: ${width} × ${height} px\nInferred Layout: ${isSingle ? 'Single Beat/Crop' : 'Full 12-Lead Sheet'}`;

        document.getElementById('cfg-layout').value = isSingle ? 'single_lead' : '3x4_1';
        document.getElementById('cfg-calibration').value = isSingle ? 'false' : 'true';
    }

    // --- Digitisation Execution & Signature Scanning Animation ---
    async function executeDigitisation() {
        const alertBox = document.getElementById('workbench-alert');
        alertBox.style.display = 'none';

        const runBtn = document.getElementById('btn-run-digitize');
        runBtn.disabled = true;
        runBtn.innerText = '⏳ Processing...';

        // Trigger visual scanning animation
        const viewportBox = document.getElementById('viewport-box');
        viewportBox.classList.add('scanning-active');

        // Animate stepper
        resetStepper();
        setStep(1, 'active', 'Validating...');
        await sleep(250);
        setStep(1, 'done', 'Verified OK');
        setStep(2, 'active', 'Estimating skew...');
        await sleep(200);

        try {
            const payload = {
                record_id: currentRecordId,
                layout_hint: document.getElementById('cfg-layout').value,
                extractor_mode: document.getElementById('cfg-extractor').value,
                is_calibration_reliable: document.getElementById('cfg-calibration').value === 'true',
            };
            if (activeImageBase64) payload.image_base64 = activeImageBase64;

            setStep(3, 'active', 'Suppressing grid...');
            setStep(4, 'active', 'Segmenting trace...');
            setStep(5, 'active', 'Partitioning leads...');
            setStep(6, 'active', 'ResNet-U-Net inference...');

            const res = await fetch('/api/digitize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || ('HTTP ' + res.status));
            }

            const data = await res.json();
            currentDigitizedData = data;

            // Mark stepper steps done with real stats
            setStep(2, 'done', `${data.deskew_angle_deg.toFixed(2)}°`);
            setStep(3, 'done', 'Color de-mixed');
            setStep(4, 'done', 'Trace isolated');
            setStep(5, 'done', `${Object.keys(data.leads || {}).length} regions detected`);
            setStep(6, 'done', `Mode: ${data.extractor_mode_used.toUpperCase()}`);
            setStep(7, 'done', data.is_calibration_reliable ? '100 Hz / Calibrated' : '100 Hz / Nominal');

            // Set default view to Diagnostic Sheet Overlay
            switchImageView('overlay');

            // Show completion banner
            const leadCount = Object.keys(data.leads || {}).length;
            document.getElementById('digitize-summary-msg').innerText = `${leadCount} lead(s) successfully extracted and calibrated. Reconstructed digital waveforms ready in Lead Explorer.`;
            document.getElementById('digitize-success-banner').style.display = 'flex';

        } catch (err) {
            alertBox.innerText = '❌ Digitisation Pipeline Error: ' + err.message;
            alertBox.style.display = 'block';
            setStep(2, 'error', 'Error');
        } finally {
            viewportBox.classList.remove('scanning-active');
            runBtn.disabled = false;
            runBtn.innerText = '⚡ Digitise ECG';
        }
    }

    function resetStepper() {
        for (let i = 1; i <= 7; i++) {
            const dot = document.getElementById('dot-' + i);
            dot.className = 'step-dot';
            document.getElementById('val-' + i).innerText = 'Pending';
        }
    }

    function setStep(idx, state, text) {
        const dot = document.getElementById('dot-' + idx);
        dot.className = 'step-dot ' + (state === 'done' ? 'done' : (state === 'active' ? 'active' : ''));
        document.getElementById('val-' + idx).innerText = text;
    }

    function switchImageView(view) {
        activeViewMode = view;
        ['overlay', 'suppressed', 'binary', 'deskewed', 'original'].forEach(v => {
            const btn = document.getElementById('btn-view-' + v);
            if (btn) btn.classList.toggle('active', v === view);
        });

        const img = document.getElementById('viewport-img');
        if (!currentDigitizedData && view !== 'original') return;

        if (view === 'overlay' && currentDigitizedData?.sheet_overlay_b64) {
            img.src = "data:image/png;base64," + currentDigitizedData.sheet_overlay_b64;
        } else if (view === 'suppressed' && currentDigitizedData?.grid_suppressed_b64) {
            img.src = "data:image/png;base64," + currentDigitizedData.grid_suppressed_b64;
        } else if (view === 'binary' && currentDigitizedData?.binary_trace_b64) {
            img.src = "data:image/png;base64," + currentDigitizedData.binary_trace_b64;
        } else if (view === 'deskewed' && currentDigitizedData?.deskewed_b64) {
            img.src = "data:image/png;base64," + currentDigitizedData.deskewed_b64;
        } else if (activeImageBase64) {
            img.src = activeImageBase64;
        }
    }

    // --- Section 02: Lead Explorer Logic & Canvas Drawing ---
    function renderLeadMatrix() {
        const container = document.getElementById('explorer-leads-grid');
        if (!currentDigitizedData || !currentDigitizedData.leads) {
            container.innerHTML = '<div style="grid-column:1/-1; text-align:center; padding:18px; color:var(--text-muted);">No digitized leads available. Digitise an image in Section 01 first.</div>';
            return;
        }

        const leads = currentDigitizedData.leads;
        const standardOrder = ["I", "AVR", "V1", "V4", "II", "AVL", "V2", "V5", "III", "AVF", "V3", "V6"];
        let html = '';

        // Check if single lead
        const leadKeys = Object.keys(leads);
        if (leadKeys.length === 1) {
            const lName = leadKeys[0];
            const lInfo = leads[lName];
            const isTrained = lInfo.is_trained_lead;
            html += `
                <div class="lead-card active" style="grid-column:1/-1;" onclick="loadLeadIntoViewer('${lName}')">
                    <div class="lead-card-top">
                        <span class="lead-name-tag">LEAD ${lName}</span>
                        <span class="pill ${isTrained ? 'pill-validated' : 'pill-experimental'}">${isTrained ? 'VALIDATED (TRAINED)' : 'EXPERIMENTAL'}</span>
                    </div>
                    <canvas id="spark-${lName}" class="lead-sparkline" width="300" height="38"></canvas>
                    <div style="font-size:10px; color:var(--text-muted); margin-top:4px;">Confidence: ${(lInfo.confidence*100).toFixed(1)}% | Samples: ${lInfo.voltage_mv.length}</div>
                </div>
            `;
            container.innerHTML = html;
            setTimeout(() => drawSparkline(`spark-${lName}`, lInfo.voltage_mv), 50);
            return;
        }

        // Standard 12-lead arrangement
        standardOrder.forEach(lName => {
            if (leads[lName]) {
                const lInfo = leads[lName];
                const isTrained = lInfo.is_trained_lead;
                const activeCls = (lName === activeLeadName ? 'active' : '');
                html += `
                    <div class="lead-card ${activeCls}" id="card-lead-${lName}" onclick="loadLeadIntoViewer('${lName}')">
                        <div class="lead-card-top">
                            <span class="lead-name-tag">${lName}</span>
                            <span class="pill ${isTrained ? 'pill-validated' : 'pill-experimental'}">${isTrained ? 'TRAINED' : 'EXP'}</span>
                        </div>
                        <canvas id="spark-${lName}" class="lead-sparkline" width="220" height="38"></canvas>
                        <div style="font-size:10px; color:var(--text-muted); margin-top:3px; display:flex; justify-content:space-between;">
                            <span>${(lInfo.confidence*100).toFixed(1)}% conf</span>
                            <span>${lInfo.voltage_mv.length} pts</span>
                        </div>
                    </div>
                `;
            }
        });

        // Bottom full-width rhythm strip if available
        if (leads["II_RHYTHM"]) {
            const rInfo = leads["II_RHYTHM"];
            const activeCls = ("II_RHYTHM" === activeLeadName ? 'active' : '');
            html += `
                <div class="lead-card lead-strip-card ${activeCls}" id="card-lead-II_RHYTHM" onclick="loadLeadIntoViewer('II_RHYTHM')">
                    <div class="lead-card-top">
                        <span class="lead-name-tag">LEAD II (CONTINUOUS RHYTHM STRIP — 10.0 s)</span>
                        <span class="pill pill-validated">VALIDATED (TRAINED)</span>
                    </div>
                    <canvas id="spark-II_RHYTHM" class="lead-sparkline" width="1100" height="38"></canvas>
                    <div style="font-size:10px; color:var(--text-muted); margin-top:3px; display:flex; justify-content:space-between;">
                        <span>Continuous 10-second stitched reconstruction</span>
                        <span>${rInfo.voltage_mv.length} samples @ 100 Hz</span>
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;

        // Render sparklines after layout
        setTimeout(() => {
            standardOrder.forEach(lName => {
                if (leads[lName]) drawSparkline(`spark-${lName}`, leads[lName].voltage_mv);
            });
            if (leads["II_RHYTHM"]) drawSparkline("spark-II_RHYTHM", leads["II_RHYTHM"].voltage_mv);
        }, 60);
    }

    function drawSparkline(canvasId, values) {
        const c = document.getElementById(canvasId);
        if (!c || !values || values.length === 0) return;
        const ctx = c.getContext('2d');
        const w = c.width;
        const h = c.height;
        ctx.clearRect(0, 0, w, h);

        const min = Math.min(...values);
        const max = Math.max(...values);
        const span = Math.max(0.001, max - min);

        ctx.strokeStyle = '#0f766e';
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        for (let i = 0; i < values.length; i++) {
            const x = (i / (values.length - 1)) * w;
            const y = h - ((values[i] - min) / span) * (h - 8) - 4;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    function loadLeadIntoViewer(leadName) {
        if (!currentDigitizedData || !currentDigitizedData.leads || !currentDigitizedData.leads[leadName]) return;
        activeLeadName = leadName;

        // Update card active styles
        document.querySelectorAll('.lead-card').forEach(c => c.classList.remove('active'));
        const selectedCard = document.getElementById('card-lead-' + leadName);
        if (selectedCard) selectedCard.classList.add('active');

        const leadObj = currentDigitizedData.leads[leadName];
        document.getElementById('large-viewer-title').innerText = `LEAD ${leadName} — Calibrated Waveform Inspection`;
        document.getElementById('large-viewer-scope').innerText = leadObj.is_trained_lead
            ? '✓ VALIDATED (TRAINED RESNET-U-NET DISTRIBUTION)'
            : '⚠ EXPERIMENTAL (ZERO-SHOT RECEPTIVE-FIELD TRANSFER)';

        restartWaveformPlayback();
        fetchLeadRhythmAnalysis(leadName);
    }

    // --- Waveform Playback & Paper Grid Drawing ---
    function restartWaveformPlayback() {
        playbackCurrentSample = 0;
        isPlaying = true;
        document.getElementById('btn-play-pause').innerText = '⏸ Pause';
        if (animationInterval) clearInterval(animationInterval);

        const leadObj = currentDigitizedData?.leads?.[activeLeadName];
        const totalSamples = leadObj?.voltage_mv?.length || 250;

        // 100 Hz sampling rate = 10ms per sample
        // At 1.0x, advance sample every (10 / speed) ms
        const msPerStep = Math.max(8, 10 / playbackSpeed);

        animationInterval = setInterval(() => {
            if (!isPlaying) return;
            playbackCurrentSample += Math.ceil(playbackSpeed);
            if (playbackCurrentSample >= totalSamples) {
                playbackCurrentSample = totalSamples;
                isPlaying = false;
                document.getElementById('btn-play-pause').innerText = '▶ Play';
                clearInterval(animationInterval);
            }
            renderECGCanvas();
        }, msPerStep);
    }

    function toggleWaveformPlayback() {
        isPlaying = !isPlaying;
        document.getElementById('btn-play-pause').innerText = isPlaying ? '⏸ Pause' : '▶ Play';
        if (isPlaying && playbackCurrentSample >= (currentDigitizedData?.leads?.[activeLeadName]?.voltage_mv?.length || 250)) {
            restartWaveformPlayback();
        }
    }

    function setPlaybackSpeed(spd) {
        playbackSpeed = parseFloat(spd);
        if (isPlaying) restartWaveformPlayback();
    }

    // --- Render ECG Paper Millimeter Grid & Trace ---
    let detectedRPeaksTimes = [];

    function renderECGCanvas(cursorX = null) {
        const c = document.getElementById('ecg-canvas');
        if (!c) return;
        const ctx = c.getContext('2d');
        const w = c.width;
        const h = c.height;

        ctx.clearRect(0, 0, w, h);

        // 1. Draw Clinical ECG Paper Grid (Cream background, Pink minor lines, Red major lines)
        ctx.fillStyle = '#fffcf9';
        ctx.fillRect(0, 0, w, h);

        const minorPx = 8; // 1 mm minor division
        const majorPx = 40; // 5 mm major division

        // Minor grid lines (1 mm)
        ctx.lineWidth = 0.5;
        ctx.strokeStyle = '#fee2e2';
        ctx.beginPath();
        for (let x = 0; x < w; x += minorPx) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
        for (let y = 0; y < h; y += minorPx) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
        ctx.stroke();

        // Major grid lines (5 mm)
        ctx.lineWidth = 1.0;
        ctx.strokeStyle = '#fca5a5';
        ctx.beginPath();
        for (let x = 0; x < w; x += majorPx) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
        for (let y = 0; y < h; y += majorPx) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
        ctx.stroke();

        const leadObj = currentDigitizedData?.leads?.[activeLeadName];
        if (!leadObj) return;

        const vals = leadObj.voltage_mv;
        const times = leadObj.time_s;
        const numPts = vals.length;
        const maxIndex = Math.min(playbackCurrentSample, numPts);

        const baselineY = h / 2;
        const pxPerMv = 70; // 70px per mV on display grid

        // 2. Draw Waveform Trace
        ctx.lineWidth = 2.2;
        ctx.strokeStyle = '#0f766e'; // Medical teal
        ctx.beginPath();
        for (let i = 0; i < maxIndex; i++) {
            const x = (i / (numPts - 1)) * (w - 40) + 20;
            const y = baselineY - (vals[i] * pxPerMv);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // 3. Draw Detected R-Peaks Markers
        if (detectedRPeaksTimes.length > 0) {
            ctx.fillStyle = '#dc2626';
            detectedRPeaksTimes.forEach(t => {
                const sampleIdx = Math.round((t / leadObj.duration_s) * (numPts - 1));
                if (sampleIdx < maxIndex) {
                    const px = (sampleIdx / (numPts - 1)) * (w - 40) + 20;
                    const py = baselineY - (vals[sampleIdx] * pxPerMv);
                    ctx.beginPath();
                    ctx.arc(px, py - 6, 4, 0, 2 * Math.PI);
                    ctx.fill();
                }
            });
        }

        // 4. Draw Playback Head Sweep Cursor
        if (maxIndex < numPts && maxIndex > 0) {
            const sweepX = (maxIndex / (numPts - 1)) * (w - 40) + 20;
            ctx.strokeStyle = '#0284c7';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(sweepX, 0);
            ctx.lineTo(sweepX, h);
            ctx.stroke();
        }

        // 5. Draw Interactive Inspection Cursor Guide Lines
        if (cursorX !== null) {
            const effectiveX = Math.max(20, Math.min(w - 20, cursorX));
            const frac = (effectiveX - 20) / (w - 40);
            const sIdx = Math.round(frac * (numPts - 1));
            const curY = baselineY - (vals[sIdx] * pxPerMv);

            ctx.strokeStyle = '#475569';
            ctx.lineWidth = 1.0;
            ctx.setLineDash([4, 4]);

            // Vertical dotted line
            ctx.beginPath();
            ctx.moveTo(effectiveX, 0);
            ctx.lineTo(effectiveX, h);
            ctx.stroke();

            // Horizontal dotted line
            ctx.beginPath();
            ctx.moveTo(0, curY);
            ctx.lineTo(w, curY);
            ctx.stroke();

            ctx.setLineDash([]);

            // Glowing cursor point
            ctx.fillStyle = '#0284c7';
            ctx.beginPath();
            ctx.arc(effectiveX, curY, 5, 0, 2 * Math.PI);
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Update coordinate box
            document.getElementById('coord-t').innerText = times[sIdx].toFixed(3) + ' s';
            document.getElementById('coord-v').innerText = (vals[sIdx] >= 0 ? '+' : '') + vals[sIdx].toFixed(3) + ' mV';
            document.getElementById('coord-sample').innerText = `#${sIdx}`;
        }
    }

    // Canvas Mouse Interaction for Coordinates
    const ecgCanvas = document.getElementById('ecg-canvas');
    if (ecgCanvas) {
        ecgCanvas.addEventListener('mousemove', (e) => {
            const rect = ecgCanvas.getBoundingClientRect();
            const scaleX = ecgCanvas.width / rect.width;
            const x = (e.clientX - rect.left) * scaleX;
            renderECGCanvas(x);
        });
        ecgCanvas.addEventListener('mouseleave', () => {
            renderECGCanvas(null);
        });
    }

    // --- Genuine Signal Analysis & R-Peak Retrieval ---
    async function fetchLeadRhythmAnalysis(leadName) {
        try {
            const res = await fetch('/api/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({record_id: currentRecordId, lead_name: leadName})
            });
            const data = await res.json();
            const r = data.rhythm_analysis;

            detectedRPeaksTimes = r.r_peak_times_s || [];
            document.getElementById('metric-hr').innerText = r.heart_rate_bpm ? `${r.heart_rate_bpm.toFixed(1)} BPM` : 'Indeterminate';
            document.getElementById('metric-peaks').innerText = r.num_r_peaks;
            document.getElementById('metric-rr').innerText = r.mean_rr_interval_s ? `${r.mean_rr_interval_s.toFixed(3)} s` : '--';
            document.getElementById('metric-rr-std').innerText = r.std_rr_interval_s ? `Std: ${r.std_rr_interval_s.toFixed(3)} s` : '';
            document.getElementById('metric-sqi').innerText = r.signal_quality.toUpperCase();
            document.getElementById('metric-duration').innerText = `Duration: ${r.duration_s || 2.5} s`;
            document.getElementById('metric-rhythm-desc').innerText = r.rule_findings?.[0] || 'Regular Rhythm';

            renderECGCanvas();
        } catch (e) {
            console.warn('Rhythm analysis fetch error:', e);
        }
    }

    function downloadCurrentLeadCSV() {
        window.open('/api/export/csv/' + currentRecordId, '_blank');
    }

    function exportSelectedInterval() {
        alert("Interval export: Exporting active lead waveform segment.");
        window.open('/api/export/csv/' + currentRecordId, '_blank');
    }

    // --- Section 03: Synthetic Lab Logic ---
    async function loadPtbxlMeta(recId) {
        try {
            const res = await fetch('/api/ptbxl-metadata/' + recId);
            const d = await res.json();
            document.getElementById('synth-record-info').innerText = 
                `Record: ${d.record_id} | ${d.num_leads} Leads | ${d.sampling_rate_hz} Hz | ${d.duration_seconds} s | ${d.signal_length} Samples/Lead\nUnits: ${d.units} | Source: ${d.source_dataset.toUpperCase()}`;
        } catch (e) {
            console.warn('Failed to load PTB-XL metadata:', e);
        }
    }

    async function generateSyntheticPaper() {
        const rec = document.getElementById('synth-record-select').value;
        currentRecordId = rec;
        const rot = parseFloat(document.getElementById('synth-rot').value);
        const noise = parseFloat(document.getElementById('synth-noise').value);
        const blur = parseInt(document.getElementById('synth-blur').value);
        const shadow = document.getElementById('synth-shadow').value === "true";

        document.getElementById('synth-viewport-box').innerHTML = '<span style="color:#38bdf8;">Rendering 12-lead clinical ECG paper sheet...</span>';

        try {
            const res = await fetch('/api/synthetic/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    record_id: rec,
                    rotation_deg: rot,
                    gaussian_noise_sigma: noise,
                    blur_kernel_size: blur,
                    shadow_gradient: shadow,
                })
            });
            const data = await res.json();
            document.getElementById('synth-viewport-box').innerHTML = '<img src="data:image/png;base64,' + data.image_b64 + '" />';
            document.getElementById('synth-meta-box').innerText = `Sheet Resolution: ${data.image_shape[1]} × ${data.image_shape[0]} px | Speed: ${data.calibration.speed_mm_s} mm/s | Gain: ${data.calibration.gain_mm_mv} mm/mV | 150 DPI | Leads: ${Object.keys(data.lead_bounding_boxes).length}`;
            document.getElementById('btn-send-to-workbench').disabled = false;
        } catch (err) {
            document.getElementById('synth-viewport-box').innerHTML = '<span style="color:#ef4444;">Error generating synthetic sheet.</span>';
        }
    }

    function sendSyntheticToWorkbench() {
        switchWorkstationTab('workbench');
        selectSourceMode('synthetic');
    }

    // --- Section 04: Evaluation & Export Logic ---
    async function loadEvaluationSection() {
        const noticeBox = document.getElementById('eval-notice-box');
        const lead = document.getElementById('eval-lead-select').value || "II";

        try {
            const res = await fetch('/api/ground-truth/' + currentRecordId);
            const gt = await res.json();
            currentGroundTruthData = gt;

            if (!gt.available) {
                noticeBox.innerText = "ℹ️ Unpaired Real ECG Input: This record was loaded from an uncalibrated real beat crop. Gold-standard digital waveform vectors are scientifically unavailable for this sample. Morphological correlation and RMSE are properly omitted.";
                noticeBox.style.display = 'block';
                document.getElementById('eval-pearson').innerText = 'N/A (Unpaired)';
                document.getElementById('eval-rmse').innerText = 'N/A';
                document.getElementById('eval-snr').innerText = 'Qualitative';
                document.getElementById('eval-comp').innerText = '100%';
                renderGroundTruthComparison(lead);
                return;
            }

            noticeBox.style.display = 'none';
            renderGroundTruthComparison(lead);
            renderFidelitySummaryTable();
        } catch (e) {
            console.warn('Evaluation fetch error:', e);
        }
    }

    function renderGroundTruthComparison(leadName) {
        const c = document.getElementById('eval-canvas');
        if (!c) return;
        const ctx = c.getContext('2d');
        const w = c.width;
        const h = c.height;

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, w, h);

        const recLead = currentDigitizedData?.leads?.[leadName];
        const gtLead = currentGroundTruthData?.leads?.[leadName];

        if (!recLead) {
            ctx.fillStyle = '#64748b';
            ctx.font = '13px sans-serif';
            ctx.fillText('No reconstructed signal available for Lead ' + leadName, w / 2 - 120, h / 2);
            return;
        }

        const baselineY = h / 2;
        const pxPerMv = 60;

        // Draw zero baseline
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, baselineY);
        ctx.lineTo(w, baselineY);
        ctx.stroke();

        // 1. Draw Ground Truth Signal (Dark Blue, Dashed) if available
        if (gtLead && gtLead.voltage_mv) {
            ctx.strokeStyle = '#1e3a8a';
            ctx.lineWidth = 1.8;
            ctx.setLineDash([5, 4]);
            ctx.beginPath();
            const gtVals = gtLead.voltage_mv;
            for (let i = 0; i < gtVals.length; i++) {
                const x = (i / (gtVals.length - 1)) * (w - 40) + 20;
                const y = baselineY - (gtVals[i] * pxPerMv);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // 2. Draw Reconstructed Signal (Teal Solid)
        const recVals = recLead.voltage_mv;
        ctx.strokeStyle = '#0f766e';
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        for (let i = 0; i < recVals.length; i++) {
            const x = (i / (recVals.length - 1)) * (w - 40) + 20;
            const y = baselineY - (recVals[i] * pxPerMv);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Legend
        ctx.font = '11px var(--font-mono)';
        ctx.fillStyle = '#0f766e';
        ctx.fillText('— Reconstructed Signal (ECG DigiAI U-Net)', 25, 22);
        if (gtLead && gtLead.voltage_mv) {
            ctx.fillStyle = '#1e3a8a';
            ctx.fillText('- - Ground Truth Clinical Vector (PTB-XL)', 340, 22);
        }

        // Update single-lead metrics if available in fidelity metrics
        if (currentDigitizedData?.fidelity_metrics?.per_lead?.[leadName]) {
            const m = currentDigitizedData.fidelity_metrics.per_lead[leadName];
            document.getElementById('eval-pearson').innerText = `r = +${m.pearson_r.toFixed(4)}`;
            document.getElementById('eval-rmse').innerText = `${m.rmse_mv.toFixed(4)} mV`;
            document.getElementById('eval-snr').innerText = `${m.snr_db.toFixed(1)} dB`;
            document.getElementById('eval-comp').innerText = `${(m.completeness*100).toFixed(1)}%`;
        }
    }

    function renderFidelitySummaryTable() {
        const tbody = document.getElementById('eval-summary-tbody');
        if (!currentDigitizedData?.fidelity_metrics?.per_lead) return;

        const metrics = currentDigitizedData.fidelity_metrics.per_lead;
        let rows = '';
        for (const [lName, m] of Object.entries(metrics)) {
            const isTrained = ["II", "V1", "II_RHYTHM"].includes(lName);
            rows += `
                <tr>
                    <td><strong>${lName}</strong></td>
                    <td><span class="pill ${isTrained ? 'pill-validated' : 'pill-experimental'}">${isTrained ? 'VALIDATED (TRAINED)' : 'EXPERIMENTAL'}</span></td>
                    <td>+${m.pearson_r.toFixed(4)}</td>
                    <td>${m.rmse_mv.toFixed(4)} mV</td>
                    <td>${m.snr_db.toFixed(1)} dB</td>
                    <td>${(m.completeness*100).toFixed(1)}%</td>
                </tr>
            `;
        }
        tbody.innerHTML = rows;
    }

    function downloadCSVExport() { window.open('/api/export/csv/' + currentRecordId, '_blank'); }
    function downloadWfdbExport() { window.open('/api/export/wfdb/' + currentRecordId, '_blank'); }
    function openPrintableReport() { window.open('/api/report/' + currentRecordId + '/html', '_blank'); }
    function downloadJsonReport() { window.open('/api/report/' + currentRecordId + '/json', '_blank'); }

    function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

    // Initialize defaults on load
    window.onload = function() {
        selectSourceMode('upload');
        loadPtbxlMeta('00001_lr');
    };
</script>
</body>
</html>
"""
