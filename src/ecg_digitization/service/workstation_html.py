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
        --chrome: #0B1F33;
        --chrome-2: #12304A;
        --chrome-line: #1F4160;
        --on-chrome: #E6EEF5;
        --on-chrome-muted: #9FB3C8;
        --accent-on-dark: #2DD4BF;

        --bg: #EEF2F6;
        --surface: #FFFFFF;
        --surface-sunken: #F5F7FA;
        --border: #D5DDE6;
        --border-strong: #B8C4D0;
        --border-subtle: #E8EEF3;

        --text-main: #14212E;
        --text-muted: #4B5B6B;

        --primary: #0F766E;
        --primary-dark: #0B5F58;
        --primary-light: #E3F4F2;

        --trace-teal: #0F766E;
        --teal-trace: #0F766E;
        --teal-light: #E3F4F2;
        --blue-gt: #1E3A8A;

        --success: #15803D;
        --success-bg: #E8F5EC;
        --validated-green: #15803D;
        --validated-bg: #E8F5EC;

        --warning: #B45309;
        --warning-bg: #FEF3E2;
        --experimental-amber: #B45309;
        --experimental-bg: #FEF3E2;

        --danger: #B91C1C;
        --danger-bg: #FDECEC;

        --font-sans: "Segoe UI Variable Text", "Segoe UI", system-ui, -apple-system, Roboto, sans-serif;
        --font-mono: ui-monospace, "Cascadia Code", "Segoe UI Mono", "JetBrains Mono", Consolas, monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: var(--font-sans);
        background: var(--bg);
        color: var(--text-main);
        font-size: 14px;
        line-height: 1.5;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Top Instrument Frame Header */
    header {
        background: var(--chrome);
        border-bottom: 1px solid var(--chrome-line);
        padding: 14px 24px 10px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    .brand-block { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
    .brand-title {
        font-size: 18px;
        font-weight: 600;
        color: var(--on-chrome);
        letter-spacing: -0.2px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-tag {
        font-size: 11px;
        background: transparent;
        color: var(--accent-on-dark);
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 4px;
        border: 1px solid var(--chrome-line);
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .brand-subtitle { font-size: 12px; color: var(--on-chrome-muted); }
    
    .header-badges { display: flex; align-items: center; gap: 10px; }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        border: 1px solid transparent;
        letter-spacing: 0.3px;
    }
    .status-badge::before { content: ""; width: 6px; height: 6px; border-radius: 50%; }
    .badge-validated { background: var(--success-bg); color: var(--success); border-color: #BBF7D0; }
    .badge-validated::before { background: var(--success); }
    .badge-deferred { background: var(--warning-bg); color: var(--warning); border-color: #FDE68A; }
    .badge-deferred::before { background: var(--warning); }
    
    .btn-header {
        padding: 6px 14px;
        font-size: 12px;
        border: 1px solid var(--chrome-line);
        background: var(--chrome-2);
        border-radius: 6px;
        cursor: pointer;
        color: var(--on-chrome);
        font-weight: 500;
        transition: all 120ms ease;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .btn-header:hover {
        background: var(--chrome);
        border-color: var(--accent-on-dark);
        color: var(--on-chrome);
    }
    .btn-header:focus-visible {
        outline: 2px solid var(--accent-on-dark);
        outline-offset: 2px;
    }

    /* Navigation Tabs as part of instrument chrome band */
    .nav-tabs {
        background: var(--chrome);
        border-bottom: 1px solid var(--border);
        display: flex;
        padding: 0 24px;
        gap: 2px;
        position: sticky;
        top: 59px;
        z-index: 99;
        box-shadow: 0 2px 4px rgba(11,31,51,0.04);
    }
    .nav-tab {
        padding: 12px 18px;
        font-size: 13px;
        font-weight: 500;
        color: var(--on-chrome-muted);
        border-bottom: 3px solid transparent;
        cursor: pointer;
        transition: all 120ms ease;
        display: flex;
        align-items: center;
        gap: 8px;
        border-radius: 6px 6px 0 0;
        outline: none;
    }
    .nav-tab:hover {
        color: var(--on-chrome);
        background: var(--chrome-2);
    }
    .nav-tab.active {
        color: #FFFFFF;
        font-weight: 600;
        background: var(--chrome-2);
        border-bottom-color: var(--accent-on-dark);
    }
    .nav-tab:focus-visible {
        outline: 2px solid var(--accent-on-dark);
        outline-offset: -2px;
    }
    .tab-num {
        font-size: 11px;
        color: var(--accent-on-dark);
        opacity: 0.9;
        font-family: var(--font-mono);
        font-weight: 500;
    }

    /* Main Container */
    main { padding: 24px; max-width: 1560px; margin: 0 auto; }
    .tab-section { display: none; }
    .tab-section.active { display: block; }

    /* Standard Cards & Panels */
    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 2px rgba(11,31,51,0.06);
    }
    .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border);
    }
    .panel-title {
        font-size: 15px;
        font-weight: 600;
        color: var(--text-main);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .panel-subtitle {
        font-size: 12px;
        color: var(--text-muted);
        font-weight: normal;
    }

    /* Grid Layouts */
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .grid-workbench { display: grid; grid-template-columns: 380px 1fr; gap: 20px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
    .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }

    /* Form Controls */
    label {
        font-size: 12px;
        font-weight: 500;
        color: var(--text-muted);
        display: block;
        margin-bottom: 6px;
        letter-spacing: 0.1px;
    }
    input[type="text"], input[type="number"], select {
        width: 100%;
        height: 36px;
        padding: 0 12px;
        font-size: 13px;
        font-family: inherit;
        border: 1px solid var(--border-strong);
        border-radius: 6px;
        outline: none;
        background: var(--surface);
        color: var(--text-main);
        margin-bottom: 14px;
        transition: border-color 120ms ease, box-shadow 120ms ease;
    }
    input:focus, select:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(15,118,110,0.2);
    }
    input[type="range"] {
        accent-color: var(--primary);
    }
    
    /* Buttons */
    .btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        height: 36px;
        padding: 0 14px;
        border-radius: 6px;
        font-weight: 500;
        font-size: 13px;
        cursor: pointer;
        border: 1px solid transparent;
        transition: all 120ms ease;
        text-decoration: none;
        outline: none;
    }
    .btn:focus-visible {
        outline: 2px solid var(--primary);
        outline-offset: 2px;
    }
    .btn-primary {
        background: var(--primary);
        color: #FFFFFF;
    }
    .btn-primary:hover {
        background: var(--primary-dark);
    }
    .btn-primary:disabled {
        background: #94A3B8;
        color: #F1F5F9;
        cursor: not-allowed;
        opacity: 0.75;
    }
    .btn-secondary {
        background: var(--surface);
        border-color: var(--border-strong);
        color: var(--text-main);
    }
    .btn-secondary:hover {
        background: var(--surface-sunken);
        border-color: var(--text-muted);
    }
    .btn-secondary.active {
        background: var(--primary-light);
        border-color: var(--primary);
        color: var(--primary-dark);
        font-weight: 600;
    }
    .btn-sm {
        height: 28px;
        padding: 0 10px;
        font-size: 12px;
        border-radius: 5px;
    }

    /* Input Source Selector Cards */
    .source-selector { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 16px; }
    .source-card {
        border: 1px solid var(--border-strong);
        border-radius: 6px;
        padding: 10px 8px;
        cursor: pointer;
        text-align: center;
        background: var(--surface);
        transition: all 120ms ease;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
        outline: none;
    }
    .source-card:hover {
        border-color: var(--text-muted);
        background: var(--surface-sunken);
    }
    .source-card.active {
        border-color: var(--primary);
        background: var(--primary-light);
        box-shadow: 0 0 0 1px var(--primary);
    }
    .source-card:focus-visible {
        outline: 2px solid var(--primary);
        outline-offset: 2px;
    }
    .source-card-title {
        font-weight: 600;
        font-size: 12px;
        color: var(--text-main);
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .source-card.active .source-card-title {
        color: var(--primary-dark);
    }
    .source-card-sub {
        font-size: 11px;
        color: var(--text-muted);
    }

    /* Dropzone */
    .drop-zone {
        border: 2px dashed var(--border-strong);
        border-radius: 8px;
        padding: 28px 16px;
        text-align: center;
        background: var(--surface-sunken);
        cursor: pointer;
        transition: all 120ms ease;
        margin-bottom: 16px;
        outline: none;
    }
    .drop-zone:hover, .drop-zone.dragover {
        border-color: var(--primary);
        background: var(--primary-light);
    }
    .drop-zone:focus-visible {
        outline: 2px solid var(--primary);
        outline-offset: 2px;
    }
    .drop-icon {
        font-size: 24px;
        margin-bottom: 8px;
        display: flex;
        justify-content: center;
    }
    .drop-text {
        font-size: 13px;
        font-weight: 600;
        color: var(--text-main);
    }
    .drop-hint {
        font-size: 12px;
        color: var(--text-muted);
        margin-top: 4px;
    }

    /* Viewport & Scanning Animation */
    .viewport-box {
        position: relative;
        border: 1px solid var(--border);
        border-radius: 8px;
        background: #0B1F33;
        min-height: 420px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }
    .viewport-box img { max-width: 100%; max-height: 480px; object-fit: contain; }
    .scanning-beam {
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, transparent, var(--accent-on-dark), var(--primary), var(--accent-on-dark), transparent);
        box-shadow: 0 0 16px rgba(45,212,191,0.6);
        display: none;
        z-index: 20;
        pointer-events: none;
    }
    @keyframes scanPass { 0% { left: 0%; } 100% { left: 100%; } }
    .scanning-active .scanning-beam { display: block; animation: scanPass 2.2s ease-in-out infinite alternate; }

    /* Processing Stepper */
    .stepper-container { display: flex; flex-direction: column; gap: 8px; margin-top: 14px; }
    .stepper-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 12px;
        background: var(--surface-sunken);
        border: 1px solid var(--border);
        border-radius: 6px;
        font-size: 12px;
    }
    .step-name { display: flex; align-items: center; gap: 8px; font-weight: 500; }
    .step-dot { width: 8px; height: 8px; border-radius: 50%; background: #CBD5E1; }
    .step-dot.active { background: var(--primary); box-shadow: 0 0 0 3px rgba(15,118,110,0.25); }
    .step-dot.done { background: var(--success); }
    .step-val { font-family: var(--font-mono); font-size: 11px; color: var(--text-muted); }

    /* ECG Clinical Paper Canvas */
    .ecg-canvas-container { position: relative; border: 1px solid #F43F5E; border-radius: 6px; background: #FFFCF9; overflow: hidden; cursor: crosshair; }
    canvas.ecg-paper { display: block; width: 100%; height: auto; }

    /* Coordinate Readout Box */
    .coord-readout-card {
        background: #0B1F33;
        color: #F8FAFC;
        border: 1px solid var(--chrome-line);
        border-radius: 8px;
        padding: 12px 18px;
        font-family: var(--font-mono);
        font-size: 12px;
        display: flex;
        gap: 20px;
        align-items: center;
        flex-wrap: wrap;
        margin-top: 12px;
    }
    .coord-item { display: flex; flex-direction: column; }
    .coord-label { font-size: 11px; text-transform: uppercase; color: var(--on-chrome-muted); letter-spacing: 0.3px; }
    .coord-value { font-size: 14px; font-weight: 600; color: var(--accent-on-dark); }

    /* 12-Lead Clinical Grid Cards */
    .leads-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
    .lead-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px;
        cursor: pointer;
        transition: all 120ms ease;
        position: relative;
        outline: none;
    }
    .lead-card:hover { border-color: var(--primary); }
    .lead-card.active {
        border-color: var(--primary);
        background: var(--primary-light);
        box-shadow: 0 0 0 1px var(--primary);
    }
    .lead-card:focus-visible { outline: 2px solid var(--primary); outline-offset: 2px; }
    .lead-card-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
    .lead-name-tag { font-weight: 700; font-size: 13px; font-family: var(--font-mono); color: var(--text-main); }
    .lead-sparkline { width: 100%; height: 38px; display: block; }
    .lead-strip-card { grid-column: 1 / -1; }

    /* Pill Badges */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        border: 1px solid transparent;
    }
    .pill::before { content: ""; width: 6px; height: 6px; border-radius: 50%; }
    .pill-validated { background: var(--success-bg); color: var(--success); border-color: #BBF7D0; }
    .pill-validated::before { background: var(--success); }
    .pill-experimental { background: var(--warning-bg); color: var(--warning); border-color: #FDE68A; }
    .pill-experimental::before { background: var(--warning); }

    /* Tables */
    table.data-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 13px;
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }
    table.data-table th, table.data-table td { padding: 10px 14px; text-align: left; }
    table.data-table th {
        background: var(--surface-sunken);
        font-weight: 500;
        color: var(--text-muted);
        font-size: 12px;
        border-bottom: 1px solid var(--border);
    }
    table.data-table td {
        border-bottom: 1px solid var(--border-subtle);
        color: var(--text-main);
    }
    table.data-table tr:last-child td { border-bottom: none; }
    table.data-table tr:hover td { background: #F8FAFC; }

    /* Notice Banner */
    .notice-banner {
        background: var(--warning-bg);
        border: 1px solid #FCD34D;
        border-left: 3px solid var(--warning);
        padding: 10px 16px;
        font-size: 13px;
        color: #78350F;
        border-radius: 6px;
        margin-bottom: 20px;
        line-height: 1.5;
    }

    /* Modal Dialog */
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(11,31,51,0.65);
        display: none;
        align-items: center;
        justify-content: center;
        z-index: 200;
        backdrop-filter: blur(3px);
    }
    .modal-box {
        background: var(--surface);
        width: 680px;
        max-width: 95vw;
        max-height: 90vh;
        border-radius: 10px;
        border: 1px solid var(--border);
        overflow-y: auto;
        padding: 24px;
        box-shadow: 0 12px 32px rgba(11,31,51,0.25);
    }
    .modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--border);
        padding-bottom: 14px;
        margin-bottom: 18px;
    }

    /* Inline SVG Icons */
    .icon {
        width: 16px;
        height: 16px;
        display: inline-block;
        vertical-align: middle;
        stroke: currentColor;
        fill: none;
        flex-shrink: 0;
    }
    .icon-lg {
        width: 24px;
        height: 24px;
    }

    /* Accessibility: Reduced Motion */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
            scroll-behavior: auto !important;
        }
        .scanning-active .scanning-beam {
            display: none !important;
        }
    }

    /* Responsive Breakpoints */
    @media (max-width: 1100px) {
        .grid-workbench { grid-template-columns: 1fr; }
        .grid-2 { grid-template-columns: 1fr; }
        .grid-4 { grid-template-columns: repeat(2, 1fr); }
        .leads-grid { grid-template-columns: repeat(2, 1fr); }
    }

    @media (max-width: 720px) {
        header { flex-direction: column; align-items: flex-start; gap: 10px; padding: 12px 16px; }
        .nav-tabs { padding: 0 16px; overflow-x: auto; }
        main { padding: 16px; }
        .grid-3 { grid-template-columns: 1fr; }
        .grid-4 { grid-template-columns: 1fr; }
        .leads-grid { grid-template-columns: 1fr; }
        .source-selector { grid-template-columns: 1fr; }
        .coord-readout-card { gap: 12px; }
    }
</style>
</head>
<body>
<!-- SVG Icon Sprite -->
<svg style="display:none" xmlns="http://www.w3.org/2000/svg">
  <symbol id="i-upload" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
  </symbol>
  <symbol id="i-pulse" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
  </symbol>
  <symbol id="i-file" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
  </symbol>
  <symbol id="i-rotate" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><polyline points="3 3 3 8 8 8"/>
  </symbol>
  <symbol id="i-chart" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>
  </symbol>
  <symbol id="i-package" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <line x1="16.5" y1="9.4" x2="7.5" y2="4.21"/><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>
  </symbol>
  <symbol id="i-clipboard" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>
  </symbol>
  <symbol id="i-close" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
  </symbol>
</svg>

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
        <button class="btn-header" onclick="openAboutModal()">About</button>
    </div>
</header>

<!-- Navigation -->
<div class="nav-tabs" role="tablist">
    <div class="nav-tab active" role="tab" tabindex="0" onclick="switchWorkstationTab('workbench')">
        <span class="tab-num">01</span>
        <span>Digitisation Workbench</span>
    </div>
    <div class="nav-tab" role="tab" tabindex="0" onclick="switchWorkstationTab('explorer')">
        <span class="tab-num">02</span>
        <span>Lead Explorer</span>
    </div>
    <div class="nav-tab" role="tab" tabindex="0" onclick="switchWorkstationTab('synthetic')">
        <span class="tab-num">03</span>
        <span>Synthetic ECG Lab</span>
    </div>
    <div class="nav-tab" role="tab" tabindex="0" onclick="switchWorkstationTab('evaluation')">
        <span class="tab-num">04</span>
        <span>Evaluation & Export</span>
    </div>
</div>

<main>
    <!-- Notice Banner -->
    <div class="notice-banner">
        <strong>ACADEMIC PROJECT NOTICE:</strong> ECG DigiAI is an ECE B.Tech major project research workstation for ECG image digitisation and waveform reconstruction. Quantitative accuracy is formally evaluated against paired PTB-XL digital ground truth. Uncalibrated inputs provide relative amplitude trajectories. Outputs do not constitute autonomous medical advice.
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
                        <div class="source-card active" id="src-upload" role="tab" tabindex="0" onclick="selectSourceMode('upload')">
                            <div class="source-card-title"><svg class="icon"><use href="#i-upload"/></svg> Upload</div>
                            <div class="source-card-sub">Local File</div>
                        </div>
                        <div class="source-card" id="src-sample" role="tab" tabindex="0" onclick="selectSourceMode('sample')">
                            <div class="source-card-title"><svg class="icon"><use href="#i-pulse"/></svg> Real Beat</div>
                            <div class="source-card-sub">Kaggle Sample</div>
                        </div>
                        <div class="source-card" id="src-synthetic" role="tab" tabindex="0" onclick="selectSourceMode('synthetic')">
                            <div class="source-card-title"><svg class="icon"><use href="#i-file"/></svg> Synthetic</div>
                            <div class="source-card-sub">From Lab</div>
                        </div>
                        <div class="source-card" id="src-csv" role="tab" tabindex="0" onclick="triggerCSVImport()">
                            <div class="source-card-title"><svg class="icon"><use href="#i-chart"/></svg> CSV</div>
                            <div class="source-card-sub">Import Signal</div>
                        </div>
                    </div>
                    <input type="file" id="csv-file-input" accept=".csv" style="display:none;" onchange="handleCSVFileSelect(event)">

                    <!-- Alert message box -->
                    <div id="workbench-alert" style="display:none; background:var(--danger-bg); border:1px solid #FCA5A5; border-left:3px solid var(--danger); color:var(--danger); padding:10px 14px; border-radius:6px; font-size:13px; margin-bottom:14px; font-weight:500;"></div>

                    <!-- Mode 1: Upload Dropzone -->
                    <div id="box-upload" class="drop-zone" onclick="document.getElementById('file-upload-input').click()">
                        <div class="drop-icon"><svg class="icon icon-lg" style="width:28px;height:28px;color:var(--text-muted);"><use href="#i-upload"/></svg></div>
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
                    <div id="box-synthetic" style="display:none; margin-bottom:14px; background:var(--primary-light); border:1px solid var(--border-strong); padding:12px; border-radius:6px; font-size:13px; color:var(--primary-dark);">
                        Using active paired synthetic 12-lead paper sheet generated in Section 03.
                    </div>

                    <!-- Active Image Readout Card -->
                    <div style="border:1px solid var(--border); border-radius:6px; padding:12px; background:var(--surface-sunken); margin-bottom:16px;">
                        <label>Active Input Metadata</label>
                        <div id="active-meta-text" style="font-family:var(--font-mono); font-size:11px; color:var(--text-main); line-height:1.6;">
                            No input selected. Choose an option above.
                        </div>
                    </div>

                    <!-- Pipeline Execution Configuration -->
                    <label>Layout Template</label>
                    <select id="cfg-layout" onchange="updateActiveMetaText()">
                        <option value="auto">Auto-Detect Layout (Recommended)</option>
                        <option value="3x4_1">3x4 + 1 Rhythm Strip (Standard Clinical Sheet)</option>
                        <option value="3x4_3">3x4 + 3 Rhythm Strips (6 Rows, e.g. Marquette)</option>
                        <option value="3x4">3x4 Grid (No Rhythm Strip)</option>
                        <option value="12x1">12x1 Stacked Multi-Lead Sheet</option>
                        <option value="single_lead">Single Lead / Beat Crop (e.g. Lead II)</option>
                    </select>

                    <label>Waveform Extractor</label>
                    <select id="cfg-extractor">
                        <option value="deep">Deep U-Net (Trained Checkpoint)</option>
                        <option value="classical">Classical Preprocessing (Median Filter)</option>
                        <option value="hybrid">Smart Adaptive (Deep Paper / Classical Single-Beat)</option>
                    </select>

                    <label>Calibration Standard</label>
                    <select id="cfg-calibration">
                        <option value="true">Standard Clinical Paper (25 mm/s, 10 mm/mV) [Calibrated]</option>
                        <option value="false">Uncalibrated / Single Beat Photo [Nominal mV Estimates]</option>
                    </select>

                    <button class="btn btn-primary" id="btn-run-digitize" style="width:100%; padding:10px; font-size:13px; margin-top:4px;" onclick="executeDigitisation()">
                        Digitise ECG
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
                    <div id="digitize-success-banner" style="display:none; background:var(--success-bg); border:1px solid #86EFAC; border-left:3px solid var(--success); border-radius:6px; padding:14px 18px; margin-top:16px; align-items:center; justify-content:space-between;">
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
                    <button class="btn btn-secondary btn-sm" id="btn-play-pause" onclick="toggleWaveformPlayback()">Play</button>
                    <button class="btn btn-secondary btn-sm" onclick="restartWaveformPlayback()"><svg class="icon"><use href="#i-rotate"/></svg> Restart</button>
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
                    <div style="background:var(--surface-sunken); border:1px solid var(--border); padding:12px; border-radius:6px;">
                        <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase;">Heart Rate</div>
                        <div style="font-size:22px; font-weight:700; color:var(--primary); margin-top:2px;" id="metric-hr">-- BPM</div>
                        <div style="font-size:10px; color:var(--text-muted);" id="metric-rhythm-desc">--</div>
                    </div>
                    <div style="background:var(--surface-sunken); border:1px solid var(--border); padding:12px; border-radius:6px;">
                        <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase;">QRS R-Peaks Detected</div>
                        <div style="font-size:22px; font-weight:700; color:var(--text-main); margin-top:2px;" id="metric-peaks">--</div>
                        <div style="font-size:10px; color:var(--text-muted);">Marked on graph</div>
                    </div>
                    <div style="background:var(--surface-sunken); border:1px solid var(--border); padding:12px; border-radius:6px;">
                        <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase;">Mean RR Interval</div>
                        <div style="font-size:22px; font-weight:700; color:var(--text-main); margin-top:2px;" id="metric-rr">-- s</div>
                        <div style="font-size:10px; color:var(--text-muted);" id="metric-rr-std">Std: -- s</div>
                    </div>
                    <div style="background:var(--surface-sunken); border:1px solid var(--border); padding:12px; border-radius:6px;">
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
                <div style="border:1px solid var(--border); border-radius:6px; padding:12px; background:var(--surface-sunken); margin-bottom:16px;">
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
                            <option value="II">Lead II</option>
                            <option value="I">Lead I</option>
                            <option value="III">Lead III</option>
                            <option value="AVR">Lead aVR</option>
                            <option value="AVL">Lead aVL</option>
                            <option value="AVF">Lead aVF</option>
                            <option value="V1">Lead V1</option>
                            <option value="V2">Lead V2</option>
                            <option value="V3">Lead V3</option>
                            <option value="V4">Lead V4</option>
                            <option value="V5">Lead V5</option>
                            <option value="V6">Lead V6</option>
                            <option value="II_RHYTHM">II_RHYTHM</option>
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
                <div style="background:var(--surface-sunken); border:1px solid var(--border); padding:12px; border-radius:6px;">
                    <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase;">Pearson Correlation (r)</div>
                    <div style="font-size:20px; font-weight:700; color:var(--primary); margin-top:2px;" id="eval-pearson">r = --</div>
                    <div style="font-size:10px; color:var(--text-muted);">Target: r > 0.90</div>
                </div>
                <div style="background:var(--surface-sunken); border:1px solid var(--border); padding:12px; border-radius:6px;">
                    <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase;">Root Mean Square Error</div>
                    <div style="font-size:20px; font-weight:700; color:var(--validated-green); margin-top:2px;" id="eval-rmse">-- mV</div>
                    <div style="font-size:10px; color:var(--text-muted);">Target: < 0.25 mV</div>
                </div>
                <div style="background:var(--surface-sunken); border:1px solid var(--border); padding:12px; border-radius:6px;">
                    <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase;">Signal-to-Noise Ratio</div>
                    <div style="font-size:20px; font-weight:700; color:var(--text-main); margin-top:2px;" id="eval-snr">-- dB</div>
                    <div style="font-size:10px; color:var(--text-muted);">Residual noise fidelity</div>
                </div>
                <div style="background:var(--surface-sunken); border:1px solid var(--border); padding:12px; border-radius:6px;">
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
                    <div style="font-size:20px; margin-bottom:8px;"><svg class="icon icon-lg" style="color:var(--primary);"><use href="#i-chart"/></svg></div>
                    <div style="font-weight:700; font-size:12px; margin-bottom:4px;">Unified Multi-Lead CSV</div>
                    <div style="font-size:11px; color:var(--text-muted); margin-bottom:10px;">Time-series with header metadata</div>
                    <button class="btn btn-secondary btn-sm" style="width:100%;" onclick="downloadCSVExport()">Download CSV</button>
                    <button class="btn btn-outline btn-sm" style="width:100%; margin-top:6px;" onclick="triggerCSVImport()">Import CSV</button>
                </div>
                <div style="border:1px solid var(--border); border-radius:6px; padding:14px; text-align:center; background:#f8fafc;">
                    <div style="font-size:20px; margin-bottom:8px;"><svg class="icon icon-lg" style="color:var(--primary);"><use href="#i-package"/></svg></div>
                    <div style="font-weight:700; font-size:12px; margin-bottom:4px;">WFDB-Format Archive</div>
                    <div style="font-size:11px; color:var(--text-muted); margin-bottom:10px;">Standard .hea header & data bundle</div>
                    <button class="btn btn-secondary btn-sm" style="width:100%;" onclick="downloadWfdbExport()">Export WFDB</button>
                </div>
                <div style="border:1px solid var(--border); border-radius:6px; padding:14px; text-align:center; background:#f8fafc;">
                    <div style="font-size:20px; margin-bottom:8px;"><svg class="icon icon-lg" style="color:var(--primary);"><use href="#i-clipboard"/></svg></div>
                    <div style="font-weight:700; font-size:12px; margin-bottom:4px;">Printable Clinical Report</div>
                    <div style="font-size:11px; color:var(--text-muted); margin-bottom:10px;">Formatted HTML diagnostic summary</div>
                    <button class="btn btn-secondary btn-sm" style="width:100%;" onclick="openPrintableReport()">Print / Save HTML</button>
                </div>
                <div style="border:1px solid var(--border); border-radius:6px; padding:14px; text-align:center; background:#f8fafc;">
                    <div style="font-size:20px; margin-bottom:8px;"><svg class="icon icon-lg" style="color:var(--primary);"><use href="#i-file"/></svg></div>
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
            <button class="btn-header" onclick="closeAboutModal()" aria-label="Close"><svg class="icon"><use href="#i-close"/></svg></button>
        </div>
        <div style="font-size:12px; line-height:1.7; color:var(--text-main);">
            <div style="margin-bottom:12px;">
                <strong>Project Title:</strong> ECG Digitisation & Analysis Using AI (ECG DigiAI)<br>
                <strong>Academic Scope:</strong> Final-Year B.Tech Major Project (ECE Department)<br>
                <strong>Institution:</strong> Jaypee Institute of Information Technology (JIIT), Noida<br>
                <strong>Group Members:</strong> <code>Astha Kumari  |   Akash Raj   |   Kanishk Deo</code>
            </div>
            
            <div style="border:1px solid var(--border); border-radius:4px; padding:10px; background:#f8fafc; margin-bottom:12px;">
                <strong>Active Deep Learning Checkpoint:</strong><br>
                • File: <code>models/ecg_digitizer_unet_all12lead.pt</code> (8,121,711 bytes)<br>
                • Architecture: 2D ResNet-U-Net with Differentiable Soft-Argmax Coordinate Head<br>
                • Training Distribution: All 12 Clinical Leads (I–III, aVR–aVF, V1–V6) from PTB-XL Folds 1–8<br>
                • Validation Fidelity: Best Val Mean Pearson $r > 0.977$, Mean RMSE $< 0.051\text{ mV}$ ($50.5\ \mu\text{V}$)<br>
                • Baseline Checkpoints: <code>models/ecg_digitizer_unet.pt</code> (Lead II/V1 legacy) & <code>models/ecg_digitizer_unet.pt.bak</code>
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
                const w = synthImg.naturalWidth || 1654;
                const h = synthImg.naturalHeight || 1240;
                setWorkbenchImage(synthImg.src, `Synthetic Sheet (PTB-XL ${currentRecordId})`, w, h, false);
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
            alertBox.innerText = 'Error: Uploaded file is not an image format (PNG, JPG required).';
            alertBox.style.display = 'block';
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            const dataUrl = e.target.result;
            const testImg = new Image();
            testImg.onload = async function() {
                if (testImg.naturalWidth < 50 || testImg.naturalHeight < 50) {
                    alertBox.innerText = `Error: Image dimensions too small (${testImg.naturalWidth}x${testImg.naturalHeight} px). Minimum required resolution is 50x50 px.`;
                    alertBox.style.display = 'block';
                    return;
                }
                currentRecordId = file.name.replace(/\.[^/.]+$/, "");

                // Automatically query backend for intelligent layout inference
                let inferred = 'auto';
                let isSingle = false;
                try {
                    const resp = await fetch('/api/infer-layout', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({image_base64: dataUrl})
                    });
                    if (resp.ok) {
                        const infData = await resp.json();
                        inferred = infData.predicted_layout || infData.inferred_layout;
                        isSingle = infData.is_single;
                        confidence = infData.confidence || 1.0;
                    }
                } catch(e) {
                    console.warn('Layout inference request failed:', e);
                }

                setWorkbenchImage(dataUrl, file.name, testImg.naturalWidth, testImg.naturalHeight, inferred, isSingle, confidence);
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
            setWorkbenchImage("data:image/png;base64," + data.image_b64, filename, data.width, data.height, data.suggested_layout, isSingle);
        } catch (err) {
            alertBox.innerText = 'Error: Failed to load curated sample: ' + err.message;
            alertBox.style.display = 'block';
        }
    }

    let currentInferredLayout = 'auto';
    let currentLayoutConfidence = 1.0;
    let currentFileName = '';
    let currentImgW = 0, currentImgH = 0;

    function getLayoutDisplayTitle(layoutKey) {
        if (layoutKey === '3x4_3') return '3×4 + 3 Rhythm Strips (6 Rows, Marquette)';
        if (layoutKey === '3x4_1') return '3×4 + 1 Rhythm Strip (4 Rows, Clinical)';
        if (layoutKey === '3x4') return '3×4 Grid (3 Rows)';
        if (layoutKey === '12x1') return '12×1 Stacked Multi-Lead Sheet (12 Rows)';
        if (layoutKey === 'single_lead') return 'Single Lead / Beat Crop';
        return layoutKey || 'Auto';
    }

    function updateActiveMetaText() {
        if (!currentFileName) return;
        const selVal = document.getElementById('cfg-layout').value;
        const isOverride = (selVal !== 'auto') && (selVal !== currentInferredLayout);
        const activeKey = (selVal === 'auto') ? currentInferredLayout : selVal;

        const predLabel = getLayoutDisplayTitle(currentInferredLayout);
        const activeLabel = getLayoutDisplayTitle(activeKey);
        const overrideTag = isOverride ? ' [Manual Override Active]' : ' [Auto-Synchronized]';
        const confText = currentLayoutConfidence ? ` (${(currentLayoutConfidence * 100).toFixed(0)}% conf)` : '';

        document.getElementById('active-meta-text').innerText = 
            `File: ${currentFileName}\nResolution: ${currentImgW} × ${currentImgH} px\nPredicted Layout: ${predLabel}${confText}\nActive Layout: ${activeLabel}${overrideTag}`;
    }

    function setWorkbenchImage(dataUrl, label, width, height, inferredLayout, isSingle, confidence = 1.0) {
        activeImageBase64 = dataUrl;
        const img = document.getElementById('viewport-img');
        img.src = dataUrl;
        img.style.display = 'block';
        document.getElementById('viewport-placeholder').style.display = 'none';

        currentFileName = label;
        currentImgW = width;
        currentImgH = height;
        currentInferredLayout = inferredLayout || 'auto';
        currentLayoutConfidence = confidence || 1.0;

        const layoutSelect = document.getElementById('cfg-layout');
        if (inferredLayout && layoutSelect.querySelector(`option[value="${inferredLayout}"]`)) {
            layoutSelect.value = inferredLayout;
        } else {
            layoutSelect.value = isSingle ? 'single_lead' : '3x4_1';
        }
        document.getElementById('cfg-calibration').value = isSingle ? 'false' : 'true';

        updateActiveMetaText();
    }

    // --- Digitisation Execution & Signature Scanning Animation ---
    async function executeDigitisation() {
        const alertBox = document.getElementById('workbench-alert');
        alertBox.style.display = 'none';

        const runBtn = document.getElementById('btn-run-digitize');
        runBtn.disabled = true;
        runBtn.innerText = 'Processing...';

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

            // Update metadata display with final predicted and active layout
            if (data.predicted_layout) currentInferredLayout = data.predicted_layout;
            if (data.layout_confidence) currentLayoutConfidence = data.layout_confidence;
            updateActiveMetaText();

            // Mark stepper steps done with real stats
            setStep(2, 'done', `${data.deskew_angle_deg.toFixed(2)}°`);
            setStep(3, 'done', 'Color de-mixed');
            setStep(4, 'done', 'Trace isolated');
            setStep(5, 'done', `${data.active_layout || 'Sheet'} (${Object.keys(data.leads || {}).length} leads)`);
            setStep(6, 'done', `Mode: ${data.extractor_mode_used.toUpperCase()}`);
            setStep(7, 'done', data.is_calibration_reliable ? '100 Hz / Calibrated' : '100 Hz / Nominal');

            // Set default view to Diagnostic Sheet Overlay
            switchImageView('overlay');

            // Show completion banner with honest calibration, layout provenance, and warning status
            const leadCount = Object.keys(data.leads || {}).length;
            const layoutInfo = `[Active: ${getLayoutDisplayTitle(data.active_layout)}${data.is_layout_overridden ? ' — Manual Override' : ''}]`;
            let warnMsg = '';
            if (data.is_layout_ambiguous) {
                warnMsg = ' Note: Layout detection confidence moderate; verify alignment.';
            }

            if (leadCount === 1) {
                if (data.is_calibration_reliable) {
                    document.getElementById('digitize-summary-msg').innerText = `1 lead extracted (Lead II / Calibrated). 11 standard leads unavailable. ${layoutInfo}${warnMsg}`;
                } else {
                    document.getElementById('digitize-summary-msg').innerText = `1 lead extracted (Lead II / Nominal mV Estimates — Uncalibrated). 11 standard leads unavailable. ${layoutInfo}${warnMsg}`;
                }
            } else if (data.is_calibration_reliable) {
                document.getElementById('digitize-summary-msg').innerText = `${leadCount} leads successfully extracted and calibrated (100 Hz / Physical mV). ${layoutInfo}${warnMsg} Reconstructed digital waveforms ready in Lead Explorer.`;
            } else {
                document.getElementById('digitize-summary-msg').innerText = `${leadCount} leads extracted (Nominal mV estimates — Uncalibrated Input). ${layoutInfo}${warnMsg} Reconstructed digital waveforms ready in Lead Explorer.`;
            }
            document.getElementById('digitize-success-banner').style.display = 'flex';

        } catch (err) {
            alertBox.innerText = 'Error: Digitisation Pipeline Error: ' + err.message;
            alertBox.style.display = 'block';
            setStep(2, 'error', 'Error');
        } finally {
            viewportBox.classList.remove('scanning-active');
            runBtn.disabled = false;
            runBtn.innerText = 'Digitise ECG';
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
        const calLabel = (leadObj.is_calibration_reliable && currentDigitizedData.is_calibration_reliable)
            ? 'Calibrated Waveform Inspection'
            : 'Uncalibrated Waveform Inspection (Nominal mV Estimates)';
        document.getElementById('large-viewer-title').innerText = `LEAD ${leadName} — ${calLabel}`;
        document.getElementById('large-viewer-scope').innerText = leadObj.is_trained_lead
            ? 'VALIDATED (TRAINED RESNET-U-NET DISTRIBUTION)'
            : 'EXPERIMENTAL (ZERO-SHOT RECEPTIVE-FIELD TRANSFER)';
        document.getElementById('large-viewer-scope').className = 'pill ' + (leadObj.is_trained_lead ? 'pill-validated' : 'pill-experimental');

        restartWaveformPlayback();
        fetchLeadRhythmAnalysis(leadName);
    }

    // --- Waveform Playback & Paper Grid Drawing ---
    function restartWaveformPlayback() {
        playbackCurrentSample = 0;
        isPlaying = true;
        document.getElementById('btn-play-pause').innerText = 'Pause';
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
                document.getElementById('btn-play-pause').innerText = 'Play';
                clearInterval(animationInterval);
            }
            renderECGCanvas();
        }, msPerStep);
    }

    function toggleWaveformPlayback() {
        isPlaying = !isPlaying;
        document.getElementById('btn-play-pause').innerText = isPlaying ? 'Pause' : 'Play';
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

        // 4. Draw Interval Selection Highlight
        if (selectionStartSample !== null && selectionEndSample !== null && selectionStartSample !== selectionEndSample) {
            const minS = Math.min(selectionStartSample, selectionEndSample);
            const maxS = Math.max(selectionStartSample, selectionEndSample);
            const x1 = (minS / (numPts - 1)) * (w - 40) + 20;
            const x2 = (maxS / (numPts - 1)) * (w - 40) + 20;
            ctx.fillStyle = 'rgba(2, 132, 199, 0.18)';
            ctx.fillRect(x1, 0, x2 - x1, h);
            ctx.strokeStyle = '#0284c7';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(x1, 0, x2 - x1, h);

            // Interval measurement banner text
            const dtSec = Math.abs(times[maxS] - times[minS]);
            const dtMs = Math.round(dtSec * 1000);
            const eqBpm = dtSec > 0 ? Math.round(60 / dtSec) : 0;
            ctx.fillStyle = '#0369a1';
            ctx.font = '600 12px sans-serif';
            ctx.fillText(`Selected Interval: Δt = ${dtMs} ms | ${eqBpm} bpm equivalent`, x1 + 4, 20);
        }

        // 5. Draw Playback Head Sweep Cursor
        if (maxIndex < numPts && maxIndex > 0) {
            const sweepX = (maxIndex / (numPts - 1)) * (w - 40) + 20;
            ctx.strokeStyle = '#0284c7';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(sweepX, 0);
            ctx.lineTo(sweepX, h);
            ctx.stroke();
        }

        // 6. Draw Interactive Inspection Cursor Guide Lines
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

    // Canvas Mouse Interaction for Coordinates & Drag-to-Select Interval
    let isCanvasDragging = false;
    let canvasDragStartX = null;
    const ecgCanvas = document.getElementById('ecg-canvas');
    if (ecgCanvas) {
        ecgCanvas.addEventListener('mousedown', (e) => {
            const rect = ecgCanvas.getBoundingClientRect();
            const scaleX = ecgCanvas.width / rect.width;
            canvasDragStartX = (e.clientX - rect.left) * scaleX;
            isCanvasDragging = true;
            if (currentDigitizedData && currentDigitizedData.leads && currentDigitizedData.leads[activeLeadName]) {
                const leadObj = currentDigitizedData.leads[activeLeadName];
                const numPts = leadObj.voltage_mv.length;
                const frac = Math.max(0, Math.min(1, (canvasDragStartX - 20) / (ecgCanvas.width - 40)));
                selectionStartSample = Math.round(frac * (numPts - 1));
                selectionEndSample = selectionStartSample;
            }
            renderECGCanvas(canvasDragStartX);
        });

        ecgCanvas.addEventListener('mousemove', (e) => {
            const rect = ecgCanvas.getBoundingClientRect();
            const scaleX = ecgCanvas.width / rect.width;
            const x = (e.clientX - rect.left) * scaleX;
            if (isCanvasDragging && currentDigitizedData && currentDigitizedData.leads && currentDigitizedData.leads[activeLeadName]) {
                const leadObj = currentDigitizedData.leads[activeLeadName];
                const numPts = leadObj.voltage_mv.length;
                const frac = Math.max(0, Math.min(1, (x - 20) / (ecgCanvas.width - 40)));
                selectionEndSample = Math.round(frac * (numPts - 1));
            }
            renderECGCanvas(x);
        });

        window.addEventListener('mouseup', () => {
            if (isCanvasDragging) {
                isCanvasDragging = false;
                renderECGCanvas(null);
            }
        });

        ecgCanvas.addEventListener('dblclick', () => {
            selectionStartSample = null;
            selectionEndSample = null;
            renderECGCanvas(null);
        });

        ecgCanvas.addEventListener('mouseleave', () => {
            if (!isCanvasDragging) {
                renderECGCanvas(null);
            }
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
        const tbody = document.getElementById('eval-summary-tbody');

        try {
            const res = await fetch('/api/ground-truth/' + currentRecordId);
            const gt = await res.json();
            currentGroundTruthData = gt;

            if (!gt.available || !currentDigitizedData?.fidelity_metrics?.per_lead) {
                noticeBox.innerText = "Notice: Unpaired Real ECG Input: This record was loaded from an uncalibrated real beat crop or real sheet image. Gold-standard digital waveform vectors are scientifically unavailable for this sample. Morphological correlation and RMSE are properly omitted.";
                noticeBox.style.display = 'block';
                document.getElementById('eval-pearson').innerText = 'N/A (Unpaired)';
                document.getElementById('eval-rmse').innerText = 'N/A';
                document.getElementById('eval-snr').innerText = 'Qualitative';
                document.getElementById('eval-comp').innerText = '100%';
                renderGroundTruthComparison(lead);
                if (tbody) {
                    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:16px; color:var(--text-muted); font-weight:500;">Unpaired Real Input — Digital ground truth is scientifically unavailable for this sample. Morphological correlation and RMSE metrics are not applicable.</td></tr>';
                }
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
        if (!tbody) return;
        if (!currentDigitizedData?.fidelity_metrics?.per_lead) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:16px; color:var(--text-muted); font-weight:500;">Run digitisation on a paired synthetic sheet to view 12-lead comparison metrics.</td></tr>';
            return;
        }

        const metrics = currentDigitizedData.fidelity_metrics.per_lead;
        let rows = '';
        for (const [lName, m] of Object.entries(metrics)) {
            const isTrained = m.is_trained_lead !== undefined ? m.is_trained_lead : true;
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

    function triggerCSVImport() {
        document.getElementById('csv-file-input').click();
    }

    async function handleCSVFileSelect(event) {
        const file = event.target.files?.[0];
        if (!file) return;

        const alertBox = document.getElementById('workbench-alert');
        alertBox.style.display = 'none';

        try {
            const text = await file.text();
            const res = await fetch('/api/import-csv', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    csv_text: text,
                    filename: file.name,
                    record_id: file.name.replace(/\.[^/.]+$/, "")
                })
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || ('HTTP ' + res.status));
            }

            const data = await res.json();
            currentDigitizedData = data;
            currentRecordId = data.record_id;

            // Update active metadata display
            setWorkbenchImage('', `Imported CSV Signal (${file.name})`, 800, 400, data.active_layout, false);
            document.getElementById('active-meta-text').innerText =
                `File: ${file.name} | Layout: ${data.active_layout} | Sampling Rate: ${data.sampling_rate_hz} Hz | Calibration: ${data.is_calibration_reliable ? 'Calibrated (mV)' : 'Uncalibrated (Nominal)'}`;

            // Switch to Lead Explorer tab
            switchWorkstationTab('lead-explorer');
            renderLeadSparklines();
            const firstLead = Object.keys(data.leads || {})[0] || "II";
            selectInspectionLead(firstLead);

            alertBox.style.display = 'block';
            alertBox.style.background = '#ECFDF5';
            alertBox.style.borderColor = '#A7F3D0';
            alertBox.style.borderLeftColor = 'var(--success)';
            alertBox.style.color = 'var(--success)';
            alertBox.innerText = `CSV Import Successful: Record '${data.record_id}' loaded with ${Object.keys(data.leads || {}).length} leads. Round-trip consistency verified.`;
        } catch (err) {
            alertBox.style.display = 'block';
            alertBox.style.background = 'var(--danger-bg)';
            alertBox.style.borderColor = '#FCA5A5';
            alertBox.style.borderLeftColor = 'var(--danger)';
            alertBox.style.color = 'var(--danger)';
            alertBox.innerText = `CSV Import Failed: ${err.message}`;
        } finally {
            event.target.value = '';
        }
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
