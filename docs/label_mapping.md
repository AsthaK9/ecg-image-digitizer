# PTB-XL Clinical Target Label Mapping & Audit

## 1. Academic Target List vs PTB-XL Ground Truth

Audited against `ptbxl_database.csv` (21,837 records) and `scp_statements.csv` (71 statement codes).

| Target Pathology / Rhythm | PTB-XL Statement Codes | SCP Class / Subclass | Positive Count in PTB-XL | Audit Status | Implementation Strategy |
|---|---|---|---|---|---|
| **Atrial Fibrillation** | `AFIB` | Rhythm / arrhythmia | 1,514 | **Direct Label** | Multi-label classifier target + Rhythm irregularity metric |
| **Bradycardia** | `SBRAD` | Rhythm (Sinus Bradycardia) | 637 | **Direct Label** | Multi-label target + Rule-based verification ($\text{HR} < 60\text{ bpm}$) |
| **Tachycardia** | `STACH`, `SVTAC`, `PSVT` | Rhythm (Sinus & Supraventricular) | 877 | **Direct Label** | Multi-label target + Rule-based verification ($\text{HR} > 100\text{ bpm}$) |
| **ST Elevation** | `STE_` (plus acute injury codes `INJAS`, `INJAL`, `INJIN`, `INJLA`, `INJIL`) | Morphology / Acute Injury | 28 (direct `STE_`), 413 (with acute injury) | **Sparse Direct Label / Proxy** | Marked as low-prevalence direct label; proxy via acute MI injury patterns |
| **ST Depression** | `STD_` | Morphology / Ischemia | 1,009 | **Direct Label** | Multi-label classifier target |
| **Bundle Branch Block** | `CRBBB`, `IRBBB`, `CLBBB`, `ILBBB` | Conduction Disturbance (`CD`) | 2,273 | **Direct Label** | Direct hierarchical/multi-label classification (RBBB / LBBB branches) |
| **Myocardial Infarction** | `IMI`, `ASMI`, `ILMI`, `AMI`, `ALMI`, `LMI`, `IPLMI`, `IPMI`, `PMI` | Diagnostic Class `MI` | 5,486 | **Direct Label** | Super-class `MI` and anatomical sub-types |

---

## 2. Model Decision Discipline & Disclaimers
1. **Rule-Based Support**:
   - Heart rate is computed directly from R-peak detection (when signal quality allows). Bradycardia is defined as $\text{HR} < 60\text{ bpm}$, Tachycardia as $\text{HR} > 100\text{ bpm}$.
   - If the signal contains high noise, low amplitude, or clipping, a `low_signal_quality` warning is attached to the output.
2. **Classifier Readiness**:
   - For pathologies without sufficient data or when a checkpoint is not loaded, the interface returns:
     `model_not_trained` or `Not supported by available labels`.
   - Never fabricate diagnostic predictions, confidence scores, or clinical conclusions. This system operates as a decision-support research pipeline, not an autonomous diagnostic device.
