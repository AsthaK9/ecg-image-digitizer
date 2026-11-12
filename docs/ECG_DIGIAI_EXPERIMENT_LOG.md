# ECG DigiAI — Experiment Log

| Experiment ID | Date / Time | Model / Checkpoint | Dataset & Split | Preprocessing Config | Evaluation Scope | Pearson $r$ | RMSE ($mV$) | Notes & Decisions |
|---|---|---|---|---|---|---|---|---|
| **EXP-001** | 2026-11-10 | `ecg_digitizer_unet.pt` | PTB-XL Folds 1-8 / 9 (Leads II & V1) | Standard synthetic renderer | Validation crops (II & V1) | 0.9971 | 0.0179 | Initial 2-lead proof of concept baseline checkpoint. |
| **EXP-002** | 2026-11-12 | `ecg_digitizer_unet_all12lead.pt` | PTB-XL Folds 1-8 / 9 (All 12 Leads) | Balanced 12-lead sampling | Validation crops (12 leads, 960 crops) | 0.9786 | 0.0505 | 5-epoch training on Apple Silicon MPS. Saved as active checkpoint. |
| **EXP-003** | 2026-11-13 | `ecg_digitizer_unet_all12lead.pt` | Synthetic Sheet 00001_lr (Scenario A) | End-to-end full pipeline | End-to-end 12 leads + rhythm strip (pre-corridor fix) | 0.7162 | 0.1346 | Calibration pulse and lead label text contaminated column 0 limb leads. |
| **EXP-004** | 2026-11-13 | `ecg_digitizer_unet_all12lead.pt` | Synthetic Sheet 00001_lr (Scenario A) | Baseline corridor scanning + pulse separation | End-to-end all 12 leads + II_RHYTHM strip | 0.8134 | 0.1301 | Pulse moved to $\ge 12\text{ px}$ separation; corridor scanning ($y \ge 45\text{ px}$, $base\_y \pm 75\text{ px}$, left 5px margin). Uniform 374px active spans. Lead II jumped to 0.8871, II_RHYTHM to 0.9409, V2 to 0.9316, V6 to 0.9056. Complete and verified. |
