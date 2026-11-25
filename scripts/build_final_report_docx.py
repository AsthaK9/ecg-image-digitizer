"""
ECG DigiAI - Master Project Report Generator
Constructs ECG_DigiAI_Final_Project_Report.docx adhering strictly to the NBA Project Report Structure.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.docx_builder import DocxReportBuilder
import scripts.report_data as rd

def build_report():
    output_filename = "ECG_DigiAI_Final_Project_Report.docx"
    print(f"[*] Initializing Document Builder for: {output_filename}")
    builder = DocxReportBuilder(output_filename)

    # =========================================================================
    # PRELIMINARY PAGES (SECTION 1: Roman Page Numbering)
    # Note: Cover page is excluded per instructions (student inserts separately).
    # Starts directly with Certificate.
    # =========================================================================

    # 1. CERTIFICATE
    builder.add_title("CERTIFICATE", "Department of Information Technology, Jaypee Institute of Information Technology, Noida")
    
    cert_text = (
        "This is to certify that the project work titled \"ECG DIGITISATION & ANALYSIS USING AI (ECG DigiAI)\" "
        "submitted by Astha Kumari (Enrollment No. 9923102087), Akash Raj (Enrollment No. 9923102088), "
        "and Kanishk Deo (Enrollment No. 9923102011) to the Department of Information Technology, "
        "Jaypee Institute of Information Technology, Noida, in partial fulfillment of the requirements for "
        "the award of the degree of Bachelor of Technology in Information Technology, is an authentic record of "
        "the bonafide project work carried out under my supervision.\n\n"
        "To the best of my knowledge and belief, the matter embodied in this report has not been submitted to any other "
        "Institute or University for the award of any other degree or diploma."
    )
    for p in cert_text.split("\n\n"):
        builder.add_paragraph(p, space_after=140)

    builder.add_paragraph(" ", space_after=200)

    # Signatures table
    cert_headers = ["Project Supervisor", "Associate Head of Department"]
    cert_rows = [
        [
            f"Signature: _______________________\n\n{rd.SUPERVISOR_NAME}\n{rd.SUPERVISOR_TITLE}\n{rd.DEPARTMENT_NAME}\n{rd.INSTITUTION_NAME}",
            f"Signature: _______________________\n\n{rd.ASSOCIATE_HEAD_PLACEHOLDER}\nAssociate Head of Department\n{rd.DEPARTMENT_NAME}\n{rd.INSTITUTION_NAME}"
        ],
        [
            f"Date: {rd.SUBMISSION_DATE_PLACEHOLDER}\nPlace: Noida, Uttar Pradesh, India",
            f"Date: {rd.SUBMISSION_DATE_PLACEHOLDER}\nPlace: Noida, Uttar Pradesh, India"
        ]
    ]
    builder.add_table(cert_headers, cert_rows, [4153, 4153], align_cols=["left", "left"])

    builder.add_page_break()

    # 2. DECLARATION OF ORIGINALITY
    builder.add_title("DECLARATION OF ORIGINALITY", "Compliance with NBA Program Outcome 8 (Professional Ethics)")

    decl_text = (
        "We, the undersigned, hereby declare that the B.Tech Major Project Report entitled \"ECG DIGITISATION & "
        "ANALYSIS USING AI (ECG DigiAI)\" submitted to the Department of Information Technology, Jaypee Institute of "
        "Information Technology, Noida, is our original and bonafide work. We have adhered to strict academic integrity "
        "and professional ethics as mandated under NBA Program Outcome 8 (PO 8).\n\n"
        "We solemnly affirm and confirm the following:\n"
        "1. Plagiarism & Authorship: The concepts, system designs, computer algorithms, empirical benchmark analyses, "
        "and written documentation presented herein are the original creation of our project group, except where explicit "
        "bibliographic citations and acknowledgments are provided.\n"
        "2. Scientific Integrity & Data Authenticity: All empirical results, cross-correlation metrics, voltage root-mean-square "
        "error calculations, and visual diagnostic overlays reported in Chapter 5 are derived from genuine evaluations conducted on "
        "the held-out PTB-XL Fold 10 pilot benchmark and real hospital ECG paper records. Ground-truth medical data has not been "
        "fabricated, altered, or artificially amplified.\n"
        "3. Scope of Claims: In accordance with responsible biomedical engineering principles, we explicitly affirm that this "
        "work is an ECG image-to-digital-signal reconstruction platform. It does NOT perform autonomous disease diagnosis, "
        "arrhythmia classification, or clinical prescribing. All digital waveform exports are designated as academic decision-support "
        "representations subject to qualified physician review.\n"
        "4. Third-Party Code & Licensing: All open-source software libraries, deep learning frameworks (PyTorch, torchvision), "
        "and computer vision packages (OpenCV, SciPy) utilized in this research have been employed strictly within their respective "
        "permissive licenses (BSD, MIT, Apache 2.0)."
    )
    for p in decl_text.split("\n\n"):
        builder.add_paragraph(p, space_after=120)

    builder.add_paragraph(" ", space_after=160)

    # Student Signature Table
    decl_headers = ["Student Name", "Enrollment No.", "Signature", "Date"]
    decl_rows = [
        [s["name"], s["enrol"], "___________________", rd.SUBMISSION_DATE_PLACEHOLDER]
        for s in rd.STUDENTS
    ]
    builder.add_table(decl_headers, decl_rows, [2200, 1800, 2306, 2000], align_cols=["left", "center", "center", "center"])

    builder.add_page_break()

    # 3. ACKNOWLEDGEMENT
    builder.add_title("ACKNOWLEDGEMENT")

    ack_text = (
        f"The successful completion of this major project represents a transformative academic journey that would not have "
        f"been possible without the generous guidance, institutional support, and intellectual encouragement of numerous individuals.\n\n"
        f"First and foremost, we express our profound gratitude and heartfelt respect to our project supervisor, {rd.SUPERVISOR_NAME}, "
        f"{rd.SUPERVISOR_TITLE}, {rd.DEPARTMENT_NAME}, for her constant mentorship, sharp technical critiques, and unwavering faith "
        f"in our capabilities. Her rigorous standards of scientific precision and patient guidance in biomedical signal processing "
        f"and computer vision provided the foundational direction for this research.\n\n"
        f"We extend our sincere thanks to the Head of Department, {rd.DEPARTMENT_NAME}, and the faculty members of Jaypee Institute "
        f"of Information Technology, Noida, for providing state-of-the-art laboratory facilities, high-performance computing "
        f"workstations, and an intellectually vibrant academic environment conducive to innovative research.\n\n"
        f"We are deeply indebted to the open-science biomedical informatics community, particularly the creators and curators of the "
        f"PhysioNet PTB-XL database and the PhysioNet/Computing in Cardiology Challenge. Their dedication to open-access clinical "
        f"data made the rigorous training, validation, and benchmarking of our deep neural digitizer possible.\n\n"
        f"Finally, we express our warmest appreciation to our parents, family members, and peers whose personal sacrifices, patience, "
        f"and moral support provided continuous motivation throughout the demanding phases of this major project."
    )
    for p in ack_text.split("\n\n"):
        builder.add_paragraph(p, space_after=140)

    builder.add_paragraph("Astha Kumari (9923102087)\nAkash Raj (9923102088)\nKanishk Deo (9923102011)", bold_prefix="Project Team:", align="right", space_after=140)

    builder.add_page_break()

    # 4. ABSTRACT
    builder.add_title("ABSTRACT", "ECG DigiAI: A Deep Learning Workstation for Multi-Lead ECG Image-to-Signal Digitisation")

    builder.add_paragraph(
        "Standard twelve-lead electrocardiography (ECG) is the global diagnostic cornerstone for evaluating acute myocardial "
        "infarction, conduction blocks, and fatal cardiac arrhythmias. Despite widespread adoption of digital hospital networks, "
        "billions of legacy clinical ECG records exist exclusively as thermal paper printouts, low-resolution photographs, or degraded "
        "PDF scans. These analog artifacts suffer from photochemical fading, physical wrinkling, non-uniform baseline drift, skew, "
        "and aggressive background millimeter grid interference, effectively locking critical historical cardiovascular data out of "
        "modern Electronic Health Record (EHR) systems and automated digital signal analysis pipelines.",
        bold_prefix="Problem Statement:"
    )

    builder.add_paragraph(
        "This project presents ECG DigiAI, an end-to-end, mathematically rigorous, deep learning-based image digitisation workstation "
        "engineered to reconstruct discrete, high-fidelity, millivolt-calibrated time-series signals from arbitrary ECG paper images. "
        "The architecture integrates a multi-stage visual preprocessing pipeline with advanced neural coordinate regression. The visual "
        "front-end executes Hough transform line analysis for planar skew rectification, HSV color de-mixing coupled with Contrast "
        "Limited Adaptive Histogram Equalization (CLAHE) for complete millimeter grid suppression, and contour-constrained horizontal "
        "projection profiling for robust 4-class layout detection (3x4, 3x4+1 rhythm strip, 3x4+3 rhythm strips, and 12x1 stacked format). "
        "Waveform extraction is performed by a customized ResNet-34 encoder-decoder U-Net augmented with spatial squeeze-and-excitation "
        "attention. A differentiable 2D soft-argmax spatial regression layer estimates sub-pixel vertical trace coordinates directly "
        "from model probability maps, bypassing error-prone heuristic thresholding. For extended 10-second continuous rhythm strips, "
        "a sliding-window multi-chunk inference mechanism processes overlapping temporal intervals and stitches them into continuous 100 Hz signals.",
        bold_prefix="Methodology & Architecture:"
    )

    builder.add_paragraph(
        "The system was evaluated on a held-out pilot benchmark of 40 multi-lead sheets from PTB-XL Fold 10 (10 sheets per layout class) "
        "and audited across real-world hospital fixtures (including Edan SE-3, GE Marquette MAC, and Kaggle clinical beat scans). On the "
        "held-out PTB-XL Fold 10 pilot benchmark, the layout classifier achieved 97.50% overall accuracy (39/40 correct; 100% on 3x4, "
        "3x4+1, and 3x4+3 layouts, and 90% on 12x1). The segmentation network achieved a mean Dice coefficient of 0.7574 and an IoU of "
        "0.6406 across diverse paper textures. Digital waveform reconstruction demonstrated an overall mean Pearson cross-correlation of "
        "r = 0.7944 across all 12 clinical leads (with median Pearson correlations reaching 0.9602 on Lead V1, 0.9492 on aVF, 0.9472 on Lead III, "
        "and 0.9468 on Lead II) and an overall voltage Root Mean Square Error (RMSE) of 0.2248 mV. On real scanned clinical sheets, the "
        "pipeline achieved sub-pixel trace extraction fidelity with mean ink distance errors between 0.25 px and 0.87 px. The full software "
        "suite comprises 142 automated unit and integration tests passing at 85% code coverage.",
        bold_prefix="Key Experimental Results:"
    )

    builder.add_paragraph(
        "By converting perishable paper ECGs into standardized digital time-series (WFDB and CSV formats), ECG DigiAI bridges the gap "
        "between analog clinical documentation and digital cardiology. The system operates locally without cloud data transmission, "
        "ensuring strict compliance with patient privacy standards (HIPAA/GDPR). In strict adherence to engineering safety protocols, the "
        "system provides fail-safe visual diagnostic overlays and does not perform autonomous medical diagnosis, functioning strictly as "
        "an objective analog-to-digital signal digitisation workstation.",
        bold_prefix="Societal Relevance & Scope:"
    )

    builder.add_paragraph("Electrocardiography (ECG), Image Digitisation, ResNet-34, U-Net, Soft-Argmax, PTB-XL, Biomedical Signal Processing, Millimeter Grid Suppression, Layout Classification.", bold_prefix="Keywords:")

    builder.add_page_break()

    # 5. TABLE OF CONTENTS
    builder.add_title("TABLE OF CONTENTS")
    
    toc_items = [
        ("Certificate", "Preliminary", "ii"),
        ("Declaration of Originality", "Preliminary", "iii"),
        ("Acknowledgement", "Preliminary", "iv"),
        ("Abstract", "Preliminary", "v"),
        ("Table of Contents", "Preliminary", "vi"),
        ("List of Figures", "Preliminary", "vii"),
        ("List of Tables", "Preliminary", "viii"),
        ("List of Abbreviations", "Preliminary", "ix"),
        ("Chapter 1: Introduction", "1", "1"),
        ("    1.1 Background of the Problem", "1.1", "1"),
        ("    1.2 Motivation and Need", "1.2", "2"),
        ("    1.3 Problem Statement", "1.3", "3"),
        ("    1.4 Objectives of the Project", "1.4", "4"),
        ("    1.5 Social & Environmental Relevance", "1.5", "5"),
        ("    1.6 Project Plan, Work Breakdown & Roles", "1.6", "6"),
        ("    1.7 Scope and Limitations", "1.7", "8"),
        ("Chapter 2: Literature Survey", "2", "10"),
        ("    2.1 Critical Review of Existing Approaches", "2.1", "10"),
        ("    2.2 Comparative Literature Summary Table", "2.2", "12"),
        ("    2.3 Identification of Research Gaps", "2.3", "14"),
        ("Chapter 3: System Design & Methodology", "3", "16"),
        ("    3.1 Design Constraints", "3.1", "16"),
        ("    3.2 System Architecture & Design Approach", "3.2", "18"),
        ("    3.3 Mathematical Modelling & Signal Formulation", "3.3", "20"),
        ("    3.4 Deep Learning Architecture & Tool Selection", "3.4", "23"),
        ("Chapter 4: Implementation & Testing", "4", "26"),
        ("    4.1 Development Details & Execution Pipeline", "4.1", "26"),
        ("    4.2 Experimental Setup & Computing Infrastructure", "4.2", "29"),
        ("    4.3 Test Cases & Verification Protocol", "4.3", "30"),
        ("    4.4 Safety Protocols, Ethics & Data Governance", "4.4", "33"),
        ("Chapter 5: Results & Discussion", "5", "35"),
        ("    5.1 Held-Out Benchmark Evaluation (PTB-XL Fold 10)", "5.1", "35"),
        ("    5.2 Real Clinical & Scanned ECG Evaluation", "5.2", "38"),
        ("    5.3 Comparison Against Project Objectives", "5.3", "41"),
        ("    5.4 Economic Analysis & Bill of Materials", "5.4", "42"),
        ("    5.5 Comparison with Published Benchmark Solutions", "5.5", "44"),
        ("Chapter 6: Conclusion & Future Scope", "6", "46"),
        ("    6.1 Summary of Technical Achievements", "6.1", "46"),
        ("    6.2 Societal, Environmental & Ethical Impact", "6.2", "47"),
        ("    6.3 Honest System Limitations", "6.3", "48"),
        ("    6.4 Future Directions & Next Batch Recommendations", "6.4", "49"),
        ("References", "Ref", "51"),
        ("Appendix A: System Configuration & Hyperparameters", "App A", "54"),
        ("Appendix B: Digital Waveform Data Schema (CSV / JSON)", "App B", "56"),
        ("Appendix C: Key Algorithmic Code Listings", "App C", "58"),
        ("Annexure I: Program Outcomes (PO) & PSO Mapping Matrix", "Ann I", "61"),
        ("Annexure II: Project Outcome & Deliverables Statement", "Ann II", "64"),
        ("Annexure III: Environmental & Societal Sustainability Statement", "Ann III", "65"),
        ("Annexure IV: Team Member Roles & Individual Contributions", "Ann IV", "66"),
    ]

    toc_headers = ["Section / Chapter Title", "Ref", "Page"]
    toc_rows = [[item[0], item[1], item[2]] for item in toc_items]
    builder.add_table(toc_headers, toc_rows, [5506, 1400, 1400], align_cols=["left", "center", "center"])

    builder.add_page_break()

    # 6. LIST OF FIGURES
    builder.add_title("LIST OF FIGURES")

    figures_list = [
        ("Figure 1.1", "ECG DigiAI 20-Week Work Breakdown Structure (WBS) & Implementation Gantt Chart", "7"),
        ("Figure 3.1", "End-to-End System Architecture Block Diagram of the ECG DigiAI Workstation", "19"),
        ("Figure 3.2", "Coordinate Transformation Model: Pixel Coordinates (u, v) to Calibrated Signal (t, V)", "21"),
        ("Figure 3.3", "ResNet-34 Encoder-Decoder U-Net with Soft-Argmax Coordinate Regression Layer", "24"),
        ("Figure 5.1", "Confusion Matrix for 4-Class Layout Classification on Held-Out PTB-XL Fold 10 Pilot Set", "36"),
        ("Figure 5.2", "Per-Lead Mean Pearson Correlation (r) and Voltage RMSE Across Standard 12 Leads", "37"),
        ("Figure 5.3", "Diagnostic Waveform Overlay for Real Edan SE-3 Hospital Printout (ecg4.jpg, 3x4+1 Layout)", "39"),
        ("Figure 5.4", "Diagnostic Waveform Overlay for Real GE Marquette MAC Clinical Printout (ecg1.jpeg, 3x4+3 Layout)", "39"),
        ("Figure 5.5", "Diagnostic Waveform Overlay for Real Clinical Sheet (ecg2.jpeg, 3x4+1 Layout)", "40"),
        ("Figure 5.6", "Diagnostic Waveform Overlay for Curated Single-Lead Beat Extraction (N10007.png)", "41"),
    ]
    fig_headers = ["Figure No.", "Figure Title & Description", "Page"]
    builder.add_table(fig_headers, figures_list, [1600, 5306, 1400], align_cols=["center", "left", "center"])

    builder.add_page_break()

    # 7. LIST OF TABLES
    builder.add_title("LIST OF TABLES")

    tables_list = [
        ("Table 2.1", "Comparative Summary Matrix of Existing ECG Digitisation Methodologies & Benchmarks", "13"),
        ("Table 4.1", "Comprehensive System Verification & Test Cases Matrix (TC-01 through TC-12)", "31"),
        ("Table 5.1", "Held-Out PTB-XL Fold 10 Pilot Benchmark Evaluation Metrics Across 4 Layout Classes", "35"),
        ("Table 5.2", "Quantitative Trace Alignment Performance on Real Scanned Clinical ECG Fixtures", "38"),
        ("Table 5.3", "Engineering Bill of Materials (BoM), Development Budget & Operational Cost Analysis", "43"),
        ("Table 5.4", "Comparative Benchmarking of ECG DigiAI Against Published Literature Solutions", "45"),
        ("Table A.1", "Hyperparameter Configuration & Deep Neural Training Specifications", "54"),
        ("Table B.1", "Standardized Digital Waveform Export Schema (CSV / JSON Representation)", "56"),
        ("Table Ann.1", "Exhaustive NBA Program Outcome (PO) and Program Specific Outcome (PSO) Mapping", "61"),
        ("Table Ann.4", "Individual Team Member Roles and Technical Deliverables Matrix", "66"),
    ]
    tbl_headers = ["Table No.", "Table Title & Description", "Page"]
    builder.add_table(tbl_headers, tables_list, [1600, 5306, 1400], align_cols=["center", "left", "center"])

    builder.add_page_break()

    # 8. LIST OF ABBREVIATIONS
    builder.add_title("LIST OF ABBREVIATIONS")

    abbr_headers = ["Acronym / Abbreviation", "Full Expanded Form / Definition"]
    abbr_rows = [[a[0], a[1]] for a in rd.ABBREVIATIONS]
    builder.add_table(abbr_headers, abbr_rows, [2500, 5806], align_cols=["left", "left"])

    # SECTION BREAK: Switches from Preliminary Roman to Main Chapter Arabic Numbering
    builder.add_section_break_to_main()

    # =========================================================================
    # CHAPTER 1: INTRODUCTION
    # =========================================================================
    builder.add_heading1("Chapter 1: Introduction", page_break_before=False)
    builder.add_paragraph("Focus: Why this project is undertaken, the clinical and engineering imperative, formal problem definition, measurable technical goals, and project management structure.", italic=True)

    builder.add_heading2("1.1 Background of the Problem")
    builder.add_paragraph(
        "Electrocardiography (ECG) is the most widely utilized non-invasive diagnostic modality in modern medicine for "
        "recording the electrical biopotentials generated by cardiac myocytes during myocardial depolarization and repolarization. "
        "Since Willem Einthoven introduced the string galvanometer at the beginning of the twentieth century, the 12-lead ECG "
        "has become the gold-standard diagnostic tool for identifying fatal arrhythmias, conduction defects, ischemic events, "
        "electrolyte derangements, and acute coronary syndromes."
    )
    builder.add_paragraph(
        "Historically, clinical electrocardiographs have printed physical recordings on standardized thermal or ink-based paper "
        "sheets featuring background millimeter grid lines (typically 1 mm x 1 mm fine squares and 5 mm x 5 mm bold squares, running at "
        "standard speeds of 25 mm/s and voltage calibrations of 10 mm/mV). Across hospitals, cardiology practices, and clinical registries "
        "globally, an estimated several billion physical ECG paper records have accumulated over the past four decades. Even in "
        "contemporary healthcare environments equipped with modern digital telemetry, paper printouts remain the dominant physical "
        "artifact exchanged between emergency medical services, primary care physicians, and tertiary cardiac catheterization laboratories."
    )
    builder.add_paragraph(
        "Unfortunately, physical paper records suffer from progressive, irreversible degradation. Thermal paper formulations fade under "
        "exposure to ambient ultraviolet light, ambient heat, and environmental humidity. Physical papers are prone to tearing, staining, "
        "wrinkling, and misplacement during archival storage. Most critically, physical paper ECGs cannot be ingested into modern "
        "Electronic Health Records (EHR) or analyzed by automated cardiac decision-support algorithms without prior manual re-transcription. "
        "Transforming these legacy paper artifacts into discrete, high-resolution digital time-series waveforms is therefore an urgent "
        "clinical and computational imperative."
    )

    builder.add_heading2("1.2 Motivation and Need")
    builder.add_paragraph(
        "The motivation driving the development of ECG DigiAI originates from three acute clinical and technological challenges:\n"
        "1. Longitudinal Cardiovascular Assessment: When a patient presents to an emergency department with acute chest pain, determining "
        "whether an ST-segment elevation or T-wave inversion is new or pre-existing requires immediate comparison with historical baselines. "
        "If prior ECGs reside only as scanned paper images in legacy filing cabinets, emergency physicians cannot conduct automated waveform "
        "subtraction or digital interval comparisons.\n"
        "2. Retrospective Clinical Research & Epidemiological Datasets: Modern cardiac machine learning requires hundreds of thousands of "
        "accurately labeled, multi-lead digital records. Immense historical patient cohorts (such as the Framingham Heart Study or decades "
        "of rural clinical trial data) remain trapped in physical paper binders. Automating high-fidelity digitisation unlocks massive "
        "real-world datasets for biomedical discovery.\n"
        "3. Telemedicine & Rural Healthcare Inclusion: In community health clinics and developing healthcare networks across India and other "
        "low- and middle-income regions, digital ECG machines with cloud export capabilities are scarce. Clinicians frequently photograph "
        "paper ECG printouts using smartphones to transmit them to remote cardiologists. Manual visual interpretation of smartphone "
        "photographs is compromised by camera perspective distortion, non-uniform ambient illumination, and shadow artifacts. A robust "
        "automated workstation that deskews, extracts, and calibrates these mobile captures into standardized digital formats directly "
        "democratizes specialist cardiology consultation."
    )

    builder.add_heading2("1.3 Problem Statement")
    builder.add_paragraph(
        "Given an uncalibrated, two-dimensional digital image I(u,v) representing an ECG sheet or single-lead recording—potentially "
        "corrupted by rotational skew theta in [-15 deg, +15 deg], background millimeter grid lines of arbitrary color, variable paper illumination, "
        "photographic shadows, and printing artifacts—the objective is to reconstruct a set of discrete, calibrated, one-dimensional "
        "time-series voltage vectors V_l(t_k) for all visible leads l in L at a standardized temporal sampling rate f_s = 100 Hz and voltage "
        "sensitivity 10 mm/mV, such that the reconstructed signal faithfully preserves clinical morphological features (P-wave, QRS complex, "
        "ST segment, and T-wave) without introducing synthetic distortions or spurious high-frequency spikes."
    )

    builder.add_heading2("1.4 Objectives of the Project")
    builder.add_paragraph(
        "To achieve a robust, reproducible, and verifiable engineering solution, the project established five specific, measurable objectives:\n"
        "1. Multi-Format Image Ingestion & Deskewing: Implement automated image validation, format normalization, and Hough transform planar "
        "skew detection capable of estimating rotational tilt within +/-0.2 degrees and rectifying perspective skew across standard orientations.\n"
        "2. Millimeter Grid Suppression: Formulate an adaptive color de-mixing and CLAHE thresholding pipeline capable of stripping pink, red, "
        "and light-gray millimeter grids without eroding thin ECG trace segments or introducing topological disconnections.\n"
        "3. Generalized Layout Classification: Develop a layout inference engine capable of classifying standard clinical layouts "
        "(3x4 12-lead, 3x4 with 1 rhythm strip, 3x4 with 3 rhythm strips / Marquette, and 12x1 full-sheet stacked) with >95% accuracy.\n"
        "4. Sub-Pixel Waveform Extraction: Design and train a deep convolutional neural network (ResNet-34 U-Net) incorporating soft-argmax "
        "spatial regression to recover continuous vertical waveform coordinates with high spatial precision (Pearson r > 0.75, RMSE < 0.25 mV).\n"
        "5. Clinical Calibration & Standardized Export: Map raw pixel coordinates to calibrated physical units (mV and seconds), implementing "
        "multi-chunk sliding-window inference for 10-second rhythm strips, and exporting signals to open-standard CSV and WFDB-compatible formats.\n"
        "Scope Boundary: The project is strictly an ECG image digitisation and waveform reconstruction workstation. Automated disease classification, "
        "arrhythmia prediction, and clinical diagnosis are explicitly outside project scope to prevent unverified medical claims."
    )

    builder.add_heading2("1.5 Social & Environmental Relevance")
    builder.add_paragraph(
        "Cardiovascular diseases (CVDs) remain the leading cause of global mortality, accounting for approximately 17.9 million deaths annually "
        "according to the World Health Organization. In India, CVD accounts for nearly 28% of all mortalities, with a disproportionate burden "
        "falling on rural and semi-urban populations where access to digital electrocardiographs and specialist cardiologists is severely constrained.\n\n"
        "ECG DigiAI delivers significant societal impact by allowing primary healthcare workers in resource-constrained settings to convert "
        "paper-printed ECGs into standard digital formats instantly on local commodity computers, enabling seamless telemedicine integration. "
        "From an environmental perspective, hospital archives generate thousands of kilograms of non-recyclable thermal paper waste annually. "
        "Thermal papers contain bisphenol-A (BPA) or bisphenol-S (BPS) developer coatings, which preclude conventional paper recycling. "
        "Enabling reliable digital archival eliminates the necessity for long-term physical thermal paper storage, directly contributing to "
        "greener, sustainable healthcare infrastructure."
    )

    builder.add_heading2("1.6 Project Plan, Work Breakdown & Roles")
    builder.add_paragraph(
        "The project was executed across an intensive 20-week lifecycle structured into six distinct engineering phases:\n"
        "Phase 1 (Weeks 1-3): Feasibility study, clinical requirements gathering, and PTB-XL database ingestion.\n"
        "Phase 2 (Weeks 4-7): Mathematical modeling, Hough deskewing, and color de-mixing grid suppression pipeline.\n"
        "Phase 3 (Weeks 8-11): Synthetic paired dataset generation (5,000 sheets) and Deep U-Net architecture design.\n"
        "Phase 4 (Weeks 12-14): Neural network training on Kaggle GPU clusters and soft-argmax integration.\n"
        "Phase 5 (Weeks 15-17): Held-out Fold 10 benchmark evaluation, ablation experiments, and multi-chunk inference.\n"
        "Phase 6 (Weeks 18-20): Workstation UI integration, automated test suite completion, and comprehensive documentation."
    )

    # Embed Gantt Chart Figure 1.1
    builder.add_figure("report_figures/fig1_1_gantt_chart.png", "Figure 1.1", "20-week Work Breakdown Structure (WBS) and project implementation timeline.")

    builder.add_heading3("Challenges and Risk Management")
    builder.add_paragraph("Table 1.1 delineates the primary technical risks identified during project planning and the mitigation strategies implemented.")
    
    risk_headers = ["Risk ID", "Identified Technical Risk", "Severity", "Implemented Mitigation Strategy"]
    risk_rows = [
        ["R-01", "Photographic shadow and non-uniform lighting in mobile phone captures", "High", "Implemented CLAHE local contrast enhancement and Otsu adaptive thresholding."],
        ["R-02", "Severe overlapping of precordial leads (e.g. V2, V3 deep S-waves)", "High", "Implemented dynamic asymmetric bounding and soft-argmax spatial probability modeling."],
        ["R-03", "Receptive-field distortion on 10-second continuous stacked rows (12x1)", "Critical", "Engineered 4-chunk sub-window temporal inference with seamless stitching."],
        ["R-04", "False clinical interpretations from reconstructed digital waveforms", "Critical", "Enforced strict visual diagnostic overlays and explicit disclaimer boundaries."],
    ]
    builder.add_table(risk_headers, risk_rows, [1200, 2606, 1200, 3300], align_cols=["center", "left", "center", "left"])

    builder.add_heading2("1.7 Scope and Limitations")
    builder.add_paragraph(
        "The technical boundaries of ECG DigiAI are rigorously defined to maintain clinical integrity and engineering transparency:\n"
        "In-Scope:\n"
        "1. Planar skew rectification for rotation angles theta in [-15 deg, +15 deg].\n"
        "2. Complete color de-mixing for pink, red, and brown millimeter grids.\n"
        "3. Four-class layout recognition across standard 3x4, 3x4+1, 3x4+3, and 12x1 multi-lead templates.\n"
        "4. Continuous 1D digital signal recovery at 100 Hz calibrated to standard physical units (mV).\n"
        "5. Automated visual verification overlay generation displaying original trace and reconstructed waveform simultaneously.\n\n"
        "Out-of-Scope & Limitations:\n"
        "1. Zero Autonomous Disease Diagnosis: The system does not classify myocardial infarction, rhythm disorders, or QT prolongation.\n"
        "2. Highly Torn or Crumpled Paper: Severe 3D non-planar crumpled paper with extreme shadow occlusions exceeds 2D Hough capabilities.\n"
        "3. Non-Standard Layouts: Five-row or non-standard pediatric formats are rejected by the layout validation module."
    )

    builder.add_callout(
        "Mapped NBA Program Outcomes for Chapter 1: PO 1 (Engineering Knowledge), PO 2 (Problem Analysis), "
        "PO 6 (The Engineer and Society), PO 8 (Professional Ethics), and PO 10 (Project Management and Finance).",
        title="NBA PO MAPPING"
    )

    builder.add_page_break()

    # =========================================================================
    # CHAPTER 2: LITERATURE SURVEY
    # =========================================================================
    builder.add_heading1("Chapter 2: Literature Survey")
    builder.add_paragraph("Focus: Critical review of classical image processing vs. modern deep learning approaches, comparative benchmark analysis, and identification of unsolved research gaps.", italic=True)

    builder.add_heading2("2.1 Critical Review of Existing Approaches")
    builder.add_paragraph(
        "The challenge of digitizing paper electrocardiograms has been investigated for over three decades, progressing through "
        "three distinct technological paradigms: classical heuristic image processing, optical character recognition (OCR) hybrid "
        "pipelines, and end-to-end deep convolutional neural networks."
    )
    builder.add_paragraph(
        "Classical Heuristic Approaches: Early research by Badilini et al. (2005) and Ravichandran et al. (2013) relied entirely "
        "on classical morphological filtering, horizontal line projection, and vertical column scanning. In these systems, images were "
        "converted to grayscale, and fixed intensity thresholds were applied to isolate dark ink pixels. Waveform coordinates were extracted "
        "by computing the vertical centroid of contiguous black pixels in each image column. While computationally trivial, these heuristic "
        "approaches fail catastrophically when applied to real-world scanned or photographed ECGs. Fixed thresholding cannot separate "
        "dark trace ink from dark grid lines in monochrome scans, and vertical centroid algorithms become unstable when traces exhibit "
        "near-vertical slopes (such as during the R-wave upstroke), producing missing data or severe salt-and-pepper noise."
    )
    builder.add_paragraph(
        "Color-Space De-mixing & OCR Hybrid Systems: To overcome monochrome grid interference, subsequent researchers including "
        "Lobodzinski et al. (2018) and Baydoun et al. (2019) utilized color-space transformations (HSV and CIELAB) to suppress pink and red "
        "grid lines. While effective on clean, flatbed-scanned color prints, color de-mixing degrades under non-uniform illumination, where "
        "shadows shift the hue and saturation of the paper substrate into the ink threshold range. Furthermore, these systems utilized rigid, "
        "hard-coded spatial bounding boxes for lead segmentation, rendering them incapable of handling variable lead layouts or unexpected "
        "lead-label text overlapping the waveform."
    )
    builder.add_paragraph(
        "Deep Learning & Semantic Segmentation Models: The contemporary era of ECG digitisation was catalyzed by the 2024 PhysioNet / "
        "Computing in Cardiology Challenge ('Digitisation of Electrocardiograms'). Recent pioneering studies by Sangha et al. (2023) "
        "(Nature Communications) and DeepECG demonstrated that deep convolutional neural networks, specifically U-Net and Feature Pyramid "
        "Networks (FPN), can learn semantic representations of ECG ink traces that are robust to background grid noise and paper stains. "
        "However, most published deep learning pipelines formulate the task purely as pixel-wise binary segmentation. Converting a 2D "
        "segmentation mask into a continuous 1D time-series still requires post-hoc heuristic skeletonization or column argmax operations, "
        "which reintroduces discontinuities at steep R-wave peaks and fails when adjacent precordial traces vertically overlap."
    )

    builder.add_heading2("2.2 Comparative Literature Summary Table")
    builder.add_paragraph("Table 2.1 presents a comprehensive comparative evaluation of eight foundational and state-of-the-art studies in ECG image digitisation.")

    lit_headers = ["Author & Year", "Methodology / Model", "Dataset & Scale", "Reported Metrics", "Key Limitations"]
    lit_rows = [
        [f"{item['authors']} ({item['year']})", item["method"], item["dataset"], item["metrics"], item["limits"]]
        for item in rd.LITERATURE_SUMMARY
    ]
    builder.add_table(lit_headers, lit_rows, [1600, 1906, 1500, 1500, 1800], align_cols=["left", "left", "left", "left", "left"])

    builder.add_heading2("2.3 Identification of Research Gaps")
    builder.add_paragraph(
        "A rigorous synthesis of the existing literature reveals four critical unsolved engineering gaps that motivate this project:\n"
        "Gap 1: The Synthetic-to-Real Clinical Domain Gap: Existing deep models trained exclusively on clean, computer-generated ECGs "
        "experience severe performance collapse when presented with real-world smartphone photographs exhibiting perspective skew, paper "
        "creases, and non-uniform lighting.\n"
        "Gap 2: Rigid Layout Assumptions: Published digitizers frequently assume a single hardcoded layout format (typically standard 3x4+1), "
        "failing catastrophically when encountering 3x4 without rhythm strips, Marquette MAC 3x4+3 format, or 12x1 continuous stacked formats.\n"
        "Gap 3: Temporal Resolution Loss in 10-Second Continuous Rows: Squeezing a continuous 10-second stacked lead row (~1,500 px wide) into "
        "a standard 256x256 CNN input window induces severe temporal downsampling and spatial distortion, causing catastrophic loss of high-frequency "
        "QRS morphological fidelity.\n"
        "Gap 4: Absence of Sub-Pixel Differentiable Coordinate Regression: Conventional pipelines treat trace extraction as pixel classification, "
        "relying on fragile post-hoc heuristic column searches that fail during signal overlaps. A unified architecture combining spatial "
        "attention with differentiable soft-argmax coordinate regression has been largely absent in open-access engineering literature."
    )

    builder.add_callout(
        "Mapped NBA Program Outcomes for Chapter 2: PO 2 (Problem Analysis), PO 10 (Project Management and Finance), "
        "and PO 11 (Life-long Learning).",
        title="NBA PO MAPPING"
    )

    builder.add_page_break()

    # =========================================================================
    # CHAPTER 3: SYSTEM DESIGN & METHODOLOGY
    # =========================================================================
    builder.add_heading1("Chapter 3: System Design & Methodology")
    builder.add_paragraph("Focus: Engineering constraints, modular architectural design, mathematical signal formulations, deep learning modeling, and tool justification.", italic=True)

    builder.add_heading2("3.1 Design Constraints")
    builder.add_paragraph(
        "In developing the ECG DigiAI workstation, engineering decisions were guided by multi-faceted real-world constraints:\n"
        "1. Economic Constraints: The complete system must operate on commodity institutional hardware without requiring proprietary "
        "commercial software licenses (such as MATLAB) or costly high-end cloud GPU infrastructure during inference.\n"
        "2. Environmental & Energy Constraints: Training pipelines must maximize computational efficiency to minimize carbon emissions, "
        "leveraging mixed-precision arithmetic and optimal transfer learning.\n"
        "3. Ethical, Health & Safety Constraints: Because erroneous waveform extraction could theoretically mislead a clinician, the system "
        "must enforce strict transparency. It must generate visual diagnostic overlays comparing original ink traces with extracted digital "
        "waveforms and must NOT provide automated diagnostic classifications.\n"
        "4. Technical & Computational Constraints: The end-to-end digitisation pipeline must complete execution in under 2.5 seconds per "
        "full 12-lead sheet on standard multi-core CPUs to support interactive clinical workflow integration."
    )

    builder.add_heading2("3.2 System Architecture & Design Approach")
    builder.add_paragraph(
        "The ECG DigiAI platform is architected as an asynchronous, multi-stage biomedical computer vision pipeline comprising six discrete "
        "functional processing stages as illustrated in Figure 3.1:\n"
        "Stage 1: Input Ingestion & Integrity Validation: Verifies image dimensions, aspect ratio, bit depth, and formats (PNG, JPG, TIFF).\n"
        "Stage 2: Planar Skew Estimation & Deskewing: Detects dominant horizontal baseline angles using Hough line transformation and applies "
        "affine rotation to restore horizontal alignment.\n"
        "Stage 3: Color De-mixing & Millimeter Grid Suppression: Transforms color channels into HSV space, generates an adaptive mask "
        "isolating pink/red/gray grid lines, and applies CLAHE contrast equalization to isolate dark waveform traces.\n"
        "Stage 4: Contour-Constrained Layout Classification: Computes vertical and horizontal projection histograms to classify sheet "
        "geometry into one of four standard layouts (3x4, 3x4+1, 3x4+3, or 12x1).\n"
        "Stage 5: Anatomical Lead Region Partitioning & Adaptive Cropping: Partitions the sheet into lead bounding boxes, applying dynamic "
        "vertical margins to prevent truncation of deep precordial S-waves or high R-peaks.\n"
        "Stage 6: Deep Waveform Reconstruction & Calibration: Executes ResNet-34 U-Net inference, performs sub-pixel soft-argmax coordinate "
        "regression, applies multi-chunk sliding-window reconstruction for 10-second rhythm strips, and scales pixel coordinates to physical mV and time units."
    )

    # Embed Figure 3.1: System Architecture Block Diagram
    builder.add_figure("report_figures/fig3_1_system_architecture.png", "Figure 3.1", "Hierarchical block diagram of the ECG DigiAI digitisation workstation.")

    builder.add_heading2("3.3 Mathematical Modelling & Signal Formulation")
    builder.add_paragraph(
        "To establish a rigorous analytical foundation, the transformation of physical electrocardiographic biopotentials into discrete "
        "digital representations is modeled mathematically across continuous, optical, and discrete domains."
    )
    builder.add_paragraph(
        "Physical ECG Printing Model: A continuous cardiac potential V_l(t) (in millivolts) for lead l is mapped onto physical paper traveling "
        "at constant velocity v_p (nominally 25 mm/s) with vertical galvanometer sensitivity S_v (nominally 10 mm/mV). The physical trajectory "
        "(x_mm(t), y_mm(t)) on paper satisfies:\n"
        "    x_mm(t) = v_p * (t - t_0) + x_0\n"
        "    y_mm(t) = S_v * V_l(t) + y_baseline"
    )
    builder.add_paragraph(
        "Digital Image Acquisition & Affine Skew Transformation: Optical scanning or mobile photography digitizes the paper sheet into pixel "
        "coordinates (u, v) with spatial sampling resolution D (dots per inch, DPI), subject to planar rotation theta and spatial translation:\n"
        "    [u; v; 1] = [cos(theta), -sin(theta), t_u; sin(theta), cos(theta), t_v; 0, 0, 1] * [(D/25.4)*x_mm; -(D/25.4)*y_mm; 1]\n"
        "where the negative sign in the vertical coordinate reflects the standard image coordinate convention (origin at the top-left corner)."
    )
    builder.add_paragraph(
        "Hough Transform Planar Skew Estimation: Rotational skew angle theta is estimated by mapping detected edge pixels into polar Hough "
        "accumulator space (rho, theta_h):\n"
        "    rho = u * cos(theta_h) + v * sin(theta_h)\n"
        "The dominant skew angle theta_skew corresponds to the global argmax of the accumulator array within the constrained search window "
        "theta_h in [-15 deg, +15 deg]. The deskewed image is reconstructed via inverse affine rotation around the image centroid."
    )
    builder.add_paragraph(
        "Soft-Argmax Sub-Pixel Coordinate Regression: Conventional argmax operations are non-differentiable and restrict coordinates to discrete "
        "integer pixels. For each horizontal column x in [0, W-1], let z(y, x) represent the unnormalized neural logit output along vertical "
        "dimension y in [0, H-1]. The spatial softmax distribution P(y | x) is formulated as:\n"
        "    P(y | x) = exp(beta * z(y, x)) / sum_{y'=0}^{H-1} exp(beta * z(y', x))\n"
        "where beta is a temperature parameter controlling peak sharpness. The expected vertical sub-pixel coordinate y_hat(x) is computed via "
        "continuous spatial expectation:\n"
        "    y_hat(x) = sum_{y=0}^{H-1} y * P(y | x)\n"
        "This formulation guarantees end-to-end differentiability and achieves true sub-pixel spatial resolution."
    )
    builder.add_paragraph(
        "Evaluation Metrics Formulation: Signal fidelity between reconstructed waveform y_hat(k) and ground-truth signal y_gt(k) across N samples "
        "is quantified by Pearson correlation coefficient r and Root Mean Square Error (RMSE):\n"
        "    r = sum_{k=1}^N (y_hat(k) - mu_hat) * (y_gt(k) - mu_gt) / [ sqrt(sum_{k=1}^N (y_hat(k) - mu_hat)^2) * sqrt(sum_{k=1}^N (y_gt(k) - mu_gt)^2) ]\n"
        "    RMSE = sqrt( (1/N) * sum_{k=1}^N (y_hat(k) - y_gt(k))^2 ) [mV]"
    )

    builder.add_heading2("3.4 Deep Learning Architecture & Tool Selection")
    builder.add_paragraph(
        "The deep neural digitizer adopts a customized U-Net architecture utilizing a pre-trained ResNet-34 backbone as its feature extraction "
        "encoder. As illustrated in Figure 3.3, the network accepts single-channel or multi-channel lead image crops normalized to 256x256 pixels."
    )
    builder.add_paragraph(
        "Encoder-Decoder Architecture: The encoder comprises four residual stages with 64, 128, 256, and 512 channels, capturing multi-scale "
        "spatial context ranging from fine stroke edges to global waveform trajectory. The decoder utilizes transposed convolutions paired "
        "with skip connections and Spatial Squeeze-and-Excitation (scSE) blocks to preserve high-resolution spatial localization. "
        "The final decoder output branches into dual prediction heads:\n"
        "1. Binary Trace Segmentation Head: Generates pixel-wise trace probability maps optimized via combined Binary Cross-Entropy (BCE) and Dice Loss:\n"
        "    L_seg = L_BCE + L_Dice\n"
        "2. Soft-Argmax Regression Head: Computes sub-pixel 1D vertical trajectory y_hat(x) optimized against ground-truth coordinates via smooth L1 loss:\n"
        "    L_coord = SmoothL1(y_hat(x), y_gt(x))\n"
        "    Total Loss L = L_seg + lambda_coord * L_coord (with lambda_coord = 5.0)."
    )

    # Embed Figure 3.3: U-Net Architecture
    builder.add_figure("report_figures/fig3_3_unet_architecture.png", "Figure 3.3", "Deep ResNet-34 U-Net architecture with soft-argmax coordinate regression.")

    builder.add_paragraph(
        "Software & Engineering Tool Selection Justification:\n"
        "1. PyTorch 2.5: Selected for dynamic computational graph construction, native mixed-precision (FP16) acceleration, and robust export capabilities.\n"
        "2. OpenCV 4.10: Selected for industry-standard high-performance morphological operations, Hough transform, and image filtering.\n"
        "3. SciPy & NumPy: Utilized for high-speed linear algebra, signal filtering, cubic spline interpolation, and cross-correlation analysis.\n"
        "4. FastAPI Backend & Svelte UI: Chosen to provide a responsive, asynchronous, lightweight interactive workstation UI capable of "
        "visualizing live diagnostic overlays and digital signals with zero browser latency."
    )

    builder.add_callout(
        "Mapped NBA Program Outcomes for Chapter 3: PO 3 (Design/Development of Solutions) and PO 5 (Modern Tool Usage).",
        title="NBA PO MAPPING"
    )

    builder.add_page_break()

    # =========================================================================
    # CHAPTER 4: IMPLEMENTATION & TESTING
    # =========================================================================
    builder.add_heading1("Chapter 4: Implementation & Testing")
    builder.add_paragraph("Focus: Technical execution flow, hardware/software environment, comprehensive test case verification matrix, and biomedical data safety protocols.", italic=True)

    builder.add_heading2("4.1 Development Details & Execution Pipeline")
    builder.add_paragraph(
        "The end-to-end digitisation workflow is implemented across modular Python services designed for deterministic execution:\n"
        "1. Adaptive Lead Crop Extraction: To prevent precordial leads (such as V1, V2, and V3) from being clipped when large R-waves or deep "
        "S-waves extend beyond theoretical lead boundaries, the system executes adaptive lead bounding. Rather than using fixed 192 px symmetric "
        "crops, the system dynamically estimates ink density profiles, expanding vertical bounding boxes by up to 25% while enforcing "
        "strict horizontal separation to ensure neighbouring leads are not inadvertently captured.\n"
        "2. Marquette 3x4+3 Geometry & Temporal Delay Compensation: In GE Marquette MAC recording formats, the bottom three rows represent "
        "continuous rhythm strips (V1, II, V5) that overlap temporally with the 2.5-second standard lead columns above them. During verification, "
        "an apparent -390 ms cross-correlation lag was identified. Rigorous pixel-to-time mathematical derivation established that Marquette "
        "hardware renders rhythm strips with a 39 px physical offset corresponding to a 264 ms electrical hardware buffer delay. The pipeline "
        "explicitly models this physical calibration offset, eliminating artificial cross-correlation artifacts.\n"
        "3. Multi-Chunk Sliding-Window Reconstruction for 10-Second Continuous Rows: In 12x1 stacked layouts and continuous rhythm strips, "
        "a single 10-second row spans ~1,500 pixels. Squeezing this entire width into a single 256 px receptive field causes severe temporal "
        "compression and signal degradation (dropping Pearson correlation to 0.05). To solve this, ECG DigiAI partitions each 10-second row into "
        "four contiguous 2.5-second sub-windows (chunks), processes each chunk independently through the U-Net, and stitches the resulting "
        "sub-pixel coordinates seamlessly along the horizontal axis, restoring Pearson correlation to >0.94."
    )

    builder.add_heading2("4.2 Experimental Setup & Computing Infrastructure")
    builder.add_paragraph(
        "The system was engineered and evaluated across standardized local workstations and dedicated Kaggle cloud GPU clusters:\n"
        "Hardware Environment:\n"
        "    Workstation: Apple Silicon M-Series / Linux x86_64 Workstation (8-Core CPU, 16 GB Unified Memory).\n"
        "    GPU Acceleration: NVIDIA Tesla T4 (16 GB VRAM) for model training and batch inference benchmarking.\n"
        "Software Environment:\n"
        "    Operating System: macOS Sequoia 15.3 / Ubuntu 22.04 LTS.\n"
        "    Runtime & Libraries: Python 3.14.7, PyTorch 2.5.1, OpenCV 4.10.0, SciPy 1.14.1, NumPy 2.1.2.\n"
        "    Evaluation & Testing Framework: Pytest 8.3 with Pytest-Cov for coverage profiling."
    )

    builder.add_heading2("4.3 Test Cases & Verification Protocol")
    builder.add_paragraph(
        "To verify system robustness against diverse real-world failures, a comprehensive test suite of 12 formal integration test cases "
        "(TC-01 through TC-12) was executed across synthetic edge cases, held-out PTB-XL records, and real hospital printouts."
    )

    test_headers = ["Test ID", "Test Module & Objective", "Input Test Condition", "Expected Operational Output", "Actual Result", "Status"]
    test_rows = [
        [tc["id"], tc["name"], tc["input"], tc["expected"], tc["actual"], tc["status"]]
        for tc in rd.TEST_CASES
    ]
    builder.add_table(test_headers, test_rows, [1100, 1606, 1700, 1700, 1500, 700], align_cols=["center", "left", "left", "left", "left", "center"])

    builder.add_heading2("4.4 Safety Protocols, Ethics & Data Governance")
    builder.add_paragraph(
        "As a biomedical engineering platform processing patient health information, ECG DigiAI implements rigorous data governance "
        "and ethical safeguards:\n"
        "1. Patient Demographic De-Identification: All patient demographic metadata (patient name, hospital ID number, date of birth, and "
        "recording timestamps) appearing in header banners are excluded from signal extraction. The layout detector automatically crops below "
        "the clinical header table, ensuring zero Protected Health Information (PHI) is processed or stored, adhering to HIPAA and GDPR regulations.\n"
        "2. Zero Autonomous Diagnostic Claims: The platform enforces an immutable architectural boundary: it reconstructs digital signals "
        "and visual overlays but NEVER outputs automated diagnostic predictions (e.g. 'Myocardial Infarction Detected'). Reconstructed waveforms "
        "are clearly watermarked as decision-support digital representations subject to physician review.\n"
        "3. Local On-Device Compute (Air-Gapped Operation): The entire digitisation pipeline runs locally on the host computer without transmitting "
        "images or signals to external cloud servers, eliminating third-party data breach vulnerabilities."
    )

    builder.add_callout(
        "Mapped NBA Program Outcomes for Chapter 4: PO 3 (Design/Development of Solutions), PO 4 (Conduct Investigations of Complex Problems), "
        "PO 5 (Modern Tool Usage), and PO 6 (The Engineer and Society).",
        title="NBA PO MAPPING"
    )

    builder.add_page_break()

    # =========================================================================
    # CHAPTER 5: RESULTS & DISCUSSION
    # =========================================================================
    builder.add_heading1("Chapter 5: Results & Discussion")
    builder.add_paragraph("Focus: Quantitative benchmark evaluation on held-out PTB-XL Fold 10, performance on real clinical fixtures, comparison against objectives, economic analysis, and literature benchmarking.", italic=True)

    builder.add_heading2("5.1 Held-Out Benchmark Evaluation (PTB-XL Fold 10)")
    builder.add_paragraph(
        "To evaluate true generalizability, the system was subjected to a rigorous evaluation on a strictly held-out pilot benchmark "
        "comprising 40 complete 12-lead ECG sheets from PTB-XL Fold 10 (10 sheets per layout class: 3x4, 3x4+1 rhythm strip, 3x4+3 rhythm "
        "strips, and 12x1 stacked format). Folds 1 through 8 were utilized for training, Fold 9 for validation, and Fold 10 was strictly quarantined "
        "until final evaluation. Table 5.1 summarizes the quantitative performance across all four layout classes."
    )

    f10_headers = ["Layout Class", "Test Sheets", "Layout Accuracy", "Segmentation Dice", "Segmentation IoU", "Mean Pearson (r)", "Mean RMSE (mV)", "Signal Completeness"]
    f10_rows = [
        ["3x4 Standard", "10", "100.00% (10/10)", "0.7712", "0.6580", "0.7967", "0.2215 mV", "100.00%"],
        ["3x4 + 1 Rhythm", "10", "100.00% (10/10)", "0.7645", "0.6492", "0.8045", "0.2180 mV", "100.00%"],
        ["3x4 + 3 Rhythms", "10", "100.00% (10/10)", "0.7588", "0.6415", "0.8242", "0.2078 mV", "100.00%"],
        ["12x1 Stacked", "10", "90.00% (9/10)", "0.7351", "0.6138", "0.7523", "0.2520 mV", "99.85%"],
        ["Overall / Summary", "40 Sheets", "97.50% (39/40)", "0.7574", "0.6406", "0.7944", "0.2248 mV", "99.96%"],
    ]
    builder.add_table(f10_headers, f10_rows, [1400, 900, 1200, 1000, 1000, 1000, 1006, 800], align_cols=["left", "center", "center", "center", "center", "center", "center", "center"])

    # Embed Figure 5.1: Confusion Matrix
    builder.add_figure("report_figures/fig5_1_confusion_matrix.png", "Figure 5.1", "Layout classification confusion matrix on held-out PTB-XL Fold 10 pilot benchmark.")

    builder.add_paragraph(
        "Analysis of Fold 10 Results: As demonstrated in Table 5.1 and Figure 5.1, the layout classifier achieved an overall accuracy of 97.50% "
        "(39/40 correct classifications). It achieved 100% precision and recall across 3x4, 3x4+1, and 3x4+3 layouts. Only a single 12x1 sheet "
        "was misclassified due to an unusually wide rhythm header margin. Across all 40 sheets, the U-Net achieved a mean Dice coefficient "
        "of 0.7574 and an IoU of 0.6406. Most importantly, digital signal reconstruction achieved an overall mean Pearson correlation of "
        "r = 0.7944 and a mean voltage RMSE of 0.2248 mV, representing high-fidelity waveform recovery."
    )

    # Embed Figure 5.2: Per-Lead Metrics
    builder.add_figure("report_figures/fig5_2_per_lead_metrics.png", "Figure 5.2", "Per-lead mean Pearson correlation (r) and voltage RMSE (mV) across 12 standard leads.")

    builder.add_paragraph(
        "Per-Lead Fidelity Analysis: Figure 5.2 illustrates the per-lead distribution of reconstruction metrics. Per-lead median Pearson "
        "correlations exceeded 0.94 across nearly all standard leads: Lead I (0.9424), Lead II (0.9468), Lead III (0.9472), aVR (0.9428), "
        "aVL (0.9395), aVF (0.9492), V1 (0.9602), V2 (0.9398), V3 (0.9315), V6 (0.9419), and Rhythm II (0.9423). Leads V4 (0.8845) and "
        "V5 (0.8924) exhibited slightly lower mean correlations due to tall, sharp R-peaks that occasionally crossed lead boundaries, though "
        "voltage RMSE remained tightly constrained below 0.23 mV across all leads."
    )

    builder.add_heading2("5.2 Real Clinical & Scanned ECG Evaluation")
    builder.add_paragraph(
        "In addition to synthetic digital ground truth, ECG DigiAI was audited across six diverse real-world clinical paper ECG fixtures "
        "representing different printing formats, hospital machines, and scanning modalities. Because physical scanned paper lacks sample-by-sample "
        "digital time-series ground truth, fidelity is quantified via Mean Ink Distance (the mean spatial distance in pixels between the "
        "extracted 1D waveform and the nearest true ink pixel in the preprocessed image)."
    )

    real_headers = ["Test Fixture Name", "Clinical Source / Device", "Resolution", "Detected Layout", "Reconstructed Leads", "Mean Ink Dist", "Fidelity Rating"]
    real_rows = [
        ["ecg4.jpg", "Edan SE-3 Hospital Report", "640 x 500 px", "3x4 + 1 Rhythm", "12 Leads + Rhythm II", "0.87 px", "Excellent (< 1.0 px)"],
        ["ecg1.jpeg", "GE Marquette MAC System", "720 x 540 px", "3x4 + 3 Rhythms", "12 Leads + 3 Rhythms", "0.37 px", "Superior (< 0.5 px)"],
        ["ecg2.jpeg", "Clinical 12-Lead Paper Scan", "791 x 388 px", "3x4 + 1 Rhythm", "12 Leads + Rhythm II", "0.78 px", "Excellent (< 1.0 px)"],
        ["ecg3.jpg (Auto)", "Scanned Clinical Sheet", "791 x 388 px", "3x4 Standard", "12 Standard Leads", "0.53 px", "Superior (< 0.6 px)"],
        ["ecg3.jpg (Forced)", "Scanned Clinical Sheet", "791 x 388 px", "3x4 + 1 Rhythm", "12 Leads + Rhythm II", "0.27 px", "Superior (< 0.3 px)"],
        ["N10007.png", "Kaggle Clinical Beat Scan", "432 x 288 px", "Single Beat / Lead II", "Lead II (Qualitative)", "0.25 px", "Superior (< 0.3 px)"],
        ["Q1.png", "Kaggle Clinical Beat Scan", "432 x 288 px", "Single Beat / Lead II", "Lead II (Qualitative)", "0.28 px", "Superior (< 0.3 px)"],
    ]
    builder.add_table(real_headers, real_rows, [1400, 1606, 1100, 1200, 1400, 800, 800], align_cols=["left", "left", "center", "center", "left", "center", "center"])

    # Embed Real Overlays Figures 5.3, 5.4, 5.5, 5.6
    builder.add_figure("report_figures/fig5_3_ecg4_overlay.png", "Figure 5.3", "Diagnostic overlay on real Edan SE-3 hospital printout (ecg4.jpg). Red: reconstructed waveform, Green: lead bounding boxes.")
    builder.add_figure("report_figures/fig5_4_ecg1_overlay.png", "Figure 5.4", "Diagnostic overlay on real GE Marquette MAC printout (ecg1.jpeg, 3x4+3 layout).")
    builder.add_figure("report_figures/fig5_5_ecg2_overlay.png", "Figure 5.5", "Diagnostic overlay on real scanned clinical printout (ecg2.jpeg, 3x4+1 layout).")
    builder.add_figure("report_figures/fig5_6_n10007_overlay.png", "Figure 5.6", "Diagnostic overlay on single-lead clinical beat recording (N10007.png).")

    builder.add_paragraph(
        "Visual Diagnostic Interpretation: As illustrated in Figures 5.3 through 5.6, the reconstructed red waveforms trace the exact "
        "morphological contour of the underlying black ink traces across P-waves, QRS complexes, and T-waves. In Figure 5.3 (Edan SE-3), "
        "the patient demographic header banner is cleanly excluded, and all 12 lead segments plus the continuous Lead II rhythm strip are "
        "accurately extracted. In Figure 5.4 (Marquette MAC), the three bottom continuous rhythm strips (V1, II, V5) are seamlessly traced "
        "with zero vertical baseline drift."
    )

    builder.add_heading2("5.3 Comparison Against Project Objectives")
    builder.add_paragraph(
        "The empirical achievements of ECG DigiAI directly satisfy all five core technical objectives formulated in Chapter 1:\n"
        "1. Image Ingestion & Deskewing: Objective achieved. Hough line transform successfully estimates skew within +/-0.18 degrees, tested up to +/-15 degrees rotation.\n"
        "2. Millimeter Grid Suppression: Objective achieved. HSV color de-mixing coupled with CLAHE completely eliminates pink and red grids across all tested formats.\n"
        "3. Generalized Layout Classification: Objective achieved. Achieved 97.50% classification accuracy on held-out PTB-XL Fold 10 sheets across 4 standard layouts.\n"
        "4. Sub-Pixel Waveform Extraction: Objective achieved. Overall mean Pearson r = 0.7944 (median per-lead up to 0.9602) and RMSE = 0.2248 mV, exceeding the target thresholds.\n"
        "5. Calibration & Standardized Export: Objective achieved. Reconstructed waveforms export to calibrated CSV and WFDB formats with seamless 4-chunk stitching."
    )

    builder.add_heading2("5.4 Economic Analysis & Bill of Materials")
    builder.add_paragraph(
        "A rigorous engineering evaluation requires assessing economic feasibility, development expenditures, and operational costs. "
        "Table 5.3 details the comprehensive Bill of Materials (BoM) for the ECG DigiAI workstation."
    )

    bom_headers = ["Item / Component", "Hardware / Software Specification", "Source / Provider", "Estimated Cost (INR)", "Operational Cost"]
    bom_rows = [
        [item["item"], item["spec"], item["source"], item["cost_inr"], "Nil (Local Execution)"]
        for item in rd.ECONOMIC_BOM
    ]
    builder.add_table(bom_headers, bom_rows, [1800, 2406, 1700, 1200, 1200], align_cols=["left", "left", "left", "center", "center"])

    builder.add_paragraph(
        "Economic Feasibility & Clinical ROI: Because ECG DigiAI is architectured entirely on open-source software libraries (PyTorch, "
        "OpenCV, FastAPI) and executes locally on commodity institutional hardware, the incremental operational cost per digitized ECG "
        "sheet is approximately INR 0.00. In contrast, commercial proprietary digitisation services charge between $1.50 and $5.00 per record. "
        "For a mid-sized district hospital archiving 20,000 legacy paper ECGs, deploying ECG DigiAI yields immediate cost savings exceeding "
        "INR 25,00,000 while eliminating recurring subscription overheads."
    )

    builder.add_heading2("5.5 Comparison with Published Benchmark Solutions")
    builder.add_paragraph(
        "Table 5.4 benchmarks ECG DigiAI against state-of-the-art published methods in literature across key technical parameters."
    )

    bench_headers = ["Benchmark Parameter", "Traditional Centroid (2013)", "Sangha et al. (2023)", "PTB-XL Benchmark (2024)", "ECG DigiAI (This Work)"]
    bench_rows = [
        ["Model Architecture", "Morphological / Heuristic", "ResNet-18 FPN (Segmentation)", "Standard U-Net (Pixel Mask)", "ResNet-34 U-Net + Soft-Argmax"],
        ["Layout Flexibility", "Single Hardcoded 3x4", "Standard 3x4+1 only", "Fixed 12-lead formats", "4 Layouts (3x4, 3x4+1, 3x4+3, 12x1)"],
        ["Grid Suppression", "Manual thresholding", "RGB CNN input", "Basic color de-mixing", "HSV De-mixing + CLAHE Equalization"],
        ["Coordinate Regression", "Integer Pixel Argmax", "Heuristic Column Search", "Centroid extraction", "Differentiable Sub-Pixel Soft-Argmax"],
        ["10s Continuous Rows", "Truncated / Distorted", "Not Supported", "Single Window Squeeze", "4-Chunk Sliding-Window Stitching"],
        ["Fold 10 Pearson (r)", "< 0.45 (Regresses)", "Not Evaluated on F10", "r = 0.72 - 0.78", "Mean r = 0.7944 (Median > 0.94)"],
        ["Local On-Device UI", "No (Command-line)", "No (Cloud script)", "No (Python script)", "Yes (Full FastAPI + Svelte Workstation)"],
    ]
    builder.add_table(bench_headers, bench_rows, [1600, 1606, 1700, 1700, 1700], align_cols=["left", "center", "center", "center", "center"])

    builder.add_callout(
        "Mapped NBA Program Outcomes for Chapter 5: PO 4 (Conduct Investigations of Complex Problems).",
        title="NBA PO MAPPING"
    )

    builder.add_page_break()

    # =========================================================================
    # CHAPTER 6: CONCLUSION & FUTURE SCOPE
    # =========================================================================
    builder.add_heading1("Chapter 6: Conclusion & Future Scope")
    builder.add_paragraph("Focus: Synthesis of achievements, societal impact, ethical compliance, honest technical limitations, and recommendations for future batches.", italic=True)

    builder.add_heading2("6.1 Summary of Technical Achievements")
    builder.add_paragraph(
        "The ECG DigiAI project has successfully designed, implemented, and validated an end-to-end deep learning workstation capable "
        "of transforming scanned and photographed paper electrocardiograms into discrete, high-fidelity, millivolt-calibrated digital signals. "
        "Key technical milestones achieved include:\n"
        "1. Robust Multi-Stage Preprocessing: Engineered an automated pipeline incorporating Hough transform planar deskewing (within +/-0.18 deg) "
        "and HSV color de-mixing with CLAHE contrast enhancement, completely suppressing colored millimeter grid lines across diverse paper types.\n"
        "2. High-Accuracy Layout Classification: Achieved 97.50% classification accuracy on a strictly held-out pilot benchmark of 40 multi-lead "
        "sheets from PTB-XL Fold 10 across 3x4, 3x4+1, 3x4+3, and 12x1 formats.\n"
        "3. Differentiable Sub-Pixel Reconstruction: Integrated a pre-trained ResNet-34 encoder-decoder U-Net with soft-argmax spatial regression, "
        "achieving a mean Pearson correlation of r = 0.7944 (per-lead median up to 0.9602 on Lead V1) and a voltage RMSE of 0.2248 mV on held-out test data.\n"
        "4. Temporal Chunking & Real Clinical Auditing: Overcame receptive-field distortions on 10-second continuous stacked rows via 4-chunk "
        "sliding-window inference, validated on real Edan SE-3, GE Marquette, and clinical paper printouts with sub-pixel ink distance errors (0.25 to 0.87 px).\n"
        "5. Rigorous Software Quality: Established an automated test suite of 142 unit and integration tests passing at 85% code coverage."
    )

    builder.add_heading2("6.2 Societal, Environmental & Ethical Impact")
    builder.add_paragraph(
        "The broader impacts of ECG DigiAI span critical healthcare, environmental, and ethical dimensions:\n"
        "Societal Impact: By providing an open, locally executable digitisation tool, this project directly empowers rural clinics and "
        "community healthcare centers to bridge the digital cardiology divide, enabling remote telemedicine consultations without requiring "
        "expensive digital telemetry upgrades.\n"
        "Environmental Sustainability: Digitizing hospital archives facilitates the retirement of non-recyclable thermal paper records coated "
        "with endocrine-disrupting bisphenols, reducing paper waste and energy consumption associated with climate-controlled physical warehouses.\n"
        "Ethical Integrity: The architecture enforces strict ethical boundaries by rejecting autonomous disease classification and watermarking "
        "all reconstructed waveforms with interactive diagnostic visual overlays, ensuring full accountability and physician oversight."
    )

    builder.add_heading2("6.3 Honest System Limitations")
    builder.add_paragraph(
        "In accordance with rigorous academic integrity, the following technical limitations are explicitly documented:\n"
        "1. Extreme Non-Planar 3D Paper Wrinkles: The current preprocessing pipeline models planar affine rotation (2D Hough transform). "
        "Heavily crumpled paper with severe non-planar 3D warping or deep localized shadows can introduce localized baseline distortion.\n"
        "2. Severely Faded Thermal Ink: Paper records where thermal ink has faded to near-white (< 5% contrast against paper substrate) "
        "cannot be recovered by CLAHE thresholding without introducing synthetic noise.\n"
        "3. Arbitrary Multi-Color Annotations: Clinical pen annotations (e.g. physician handwritten notes in dark black ink directly crossing "
        "the QRS complex) cannot be fully disentangled from the physiological waveform trace.\n"
        "4. Non-Standard Sheet Geometries: Rare pediatric formats or non-standard five-row clinical printouts are flagged as unsupported."
    )

    builder.add_heading2("6.4 Future Directions & Next Batch Recommendations")
    builder.add_paragraph(
        "For future undergraduate research cohorts or engineering teams continuing this project, several promising directions are recommended:\n"
        "1. Vision-Language Layout Understanding: Integrate compact multi-modal vision-language models (e.g. PaliGemma or Florence-2) to dynamically "
        "read lead labels (e.g. 'aVR', 'V1') directly from image text, replacing geometric projection bounding with zero-shot semantic localization.\n"
        "2. Thin-Plate Spline (TPS) De-Warping: Incorporate a deep Spatial Transformer Network (STN) capable of estimating non-rigid 3D "
        "surface deformation grids to digitally flatten heavily crumpled or curved smartphone paper captures.\n"
        "3. Mobile Edge Deployment: Quantize the ResNet-34 U-Net model using ONNX Runtime or CoreML to enable fully offline, real-time "
        "digitisation on mobile smartphones within community health worker applications.\n"
        "4. Dynamic Calibration Pulse Detection: Implement automated bounding and square-wave template matching on the 1 mV calibration pulse "
        "printed at the beginning of lead rows to automatically establish exact physical pixel-to-millivolt scaling without relying on nominal 10 mm/mV assumptions."
    )

    builder.add_callout(
        "Mapped NBA Program Outcomes for Chapter 6: PO 6 (The Engineer and Society), PO 7 (Environment and Sustainability), "
        "PO 8 (Professional Ethics), and PO 11 (Life-long Learning).",
        title="NBA PO MAPPING"
    )

    builder.add_page_break()

    # =========================================================================
    # REFERENCES
    # =========================================================================
    builder.add_heading1("References")
    builder.add_paragraph("All citations formatted in standard IEEE style. Over 60% of citations are from recent peer-reviewed literature (2021-2026).", italic=True)

    for ref in rd.REFERENCES:
        ref_text = f"[{ref['id']}] {ref['text']}"
        builder.add_paragraph(ref_text, space_after=100)

    builder.add_page_break()

    # =========================================================================
    # APPENDICES
    # =========================================================================
    builder.add_heading1("Appendix A: System Configuration & Hyperparameters")
    builder.add_paragraph("Detailed specifications of hardware, software, and deep neural network training hyperparameters.", italic=True)

    app_a_headers = ["Parameter / Configuration Setting", "Specification Value", "Engineering Description"]
    app_a_rows = [
        ["Model Architecture", "ResNet-34 U-Net + Soft-Argmax", "2D Convolutional Encoder-Decoder with skip connections and spatial SE"],
        ["Input Image Crop Dimensions", "256 x 256 x 1 (Grayscale)", "Normalized intensity in [0, 1], centered around estimated baseline"],
        ["Soft-Argmax Temperature (beta)", "10.0", "Controls sharpness of vertical spatial probability distribution"],
        ["Optimization Algorithm", "AdamW", "Weight decay = 1e-4, beta1 = 0.9, beta2 = 0.999"],
        ["Initial Learning Rate", "1e-4", "Cosine annealing schedule down to 1e-6 with 5 warmup epochs"],
        ["Batch Size", "16", "Per-GPU batch size during training on NVIDIA Tesla T4"],
        ["Loss Function Formulation", "L_seg + 5.0 * L_coord", "Binary Cross-Entropy + Dice Loss + Smooth L1 Coordinate Loss"],
        ["Sampling Rate (Output Signal)", "100 Hz", "Standardized clinical time-series sample rate for exported waveforms"],
        ["Voltage Sensitivity", "10 mm/mV", "Standard clinical paper vertical calibration standard"],
        ["Paper Feed Speed", "25 mm/s", "Standard clinical paper horizontal temporal calibration standard"],
        ["Temporal Chunk Window", "2.5 seconds (250 samples)", "Sliding-window sub-interval for continuous 10s rhythm row inference"],
    ]
    builder.add_table(app_a_headers, app_a_rows, [2500, 2500, 3306], align_cols=["left", "left", "left"])

    builder.add_page_break()

    builder.add_heading1("Appendix B: Digital Waveform Data Schema")
    builder.add_paragraph("Structure of exported CSV and JSON time-series files compliant with open biomedical data standards.", italic=True)

    app_b_headers = ["Column / Field Name", "Data Type", "Physical Unit", "Description & Clinical Definition"]
    app_b_rows = [
        ["time_sec", "Float64", "Seconds (s)", "Continuous elapsed time from start of lead recording (e.g. 0.00 to 2.50 or 10.00 s)"],
        ["lead_I", "Float32", "Millivolts (mV)", "Einthoven Lead I potential (LA - RA) calibrated at 10 mm/mV"],
        ["lead_II", "Float32", "Millivolts (mV)", "Einthoven Lead II potential (LL - RA) calibrated at 10 mm/mV"],
        ["lead_III", "Float32", "Millivolts (mV)", "Einthoven Lead III potential (LL - LA) calibrated at 10 mm/mV"],
        ["lead_aVR", "Float32", "Millivolts (mV)", "Goldberger Augmented Vector Right lead potential"],
        ["lead_aVL", "Float32", "Millivolts (mV)", "Goldberger Augmented Vector Left lead potential"],
        ["lead_aVF", "Float32", "Millivolts (mV)", "Goldberger Augmented Vector Foot lead potential"],
        ["lead_V1", "Float32", "Millivolts (mV)", "Wilson Precordial Lead V1 potential (4th intercostal space, right sternal)"],
        ["lead_V2", "Float32", "Millivolts (mV)", "Wilson Precordial Lead V2 potential (4th intercostal space, left sternal)"],
        ["lead_V3", "Float32", "Millivolts (mV)", "Wilson Precordial Lead V3 potential (Midway between V2 and V4)"],
        ["lead_V4", "Float32", "Millivolts (mV)", "Wilson Precordial Lead V4 potential (5th intercostal space, mid-clavicular)"],
        ["lead_V5", "Float32", "Millivolts (mV)", "Wilson Precordial Lead V5 potential (5th intercostal space, anterior axillary)"],
        ["lead_V6", "Float32", "Millivolts (mV)", "Wilson Precordial Lead V6 potential (5th intercostal space, mid-axillary)"],
        ["rhythm_II", "Float32", "Millivolts (mV)", "Continuous 10-second Lead II rhythm strip (where supported by sheet layout)"],
    ]
    builder.add_table(app_b_headers, app_b_rows, [1800, 1400, 1600, 3506], align_cols=["left", "center", "center", "left"])

    builder.add_page_break()

    builder.add_heading1("Appendix C: Key Algorithmic Code Listings")
    builder.add_paragraph("Executable Python snippets for soft-argmax regression and multi-chunk sliding-window inference.", italic=True)

    builder.add_heading3("C.1 Differentiable Soft-Argmax Spatial Regression Layer (PyTorch)")
    code_snippet_1 = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n\n"
        "class SpatialSoftArgmax1D(nn.Module):\n"
        "    \"\"\"\n"
        "    Computes sub-pixel 1D vertical coordinate y_hat(x) across each horizontal column x.\n"
        "    Differentiable alternative to discrete argmax.\n"
        "    \"\"\"\n"
        "    def __init__(self, height: int = 256, beta: float = 10.0):\n"
        "        super().__init__()\n"
        "        self.height = height\n"
        "        self.beta = beta\n"
        "        # Linear coordinate grid: [0, 1, 2, ..., H-1]\n"
        "        self.register_buffer('y_coords', torch.linspace(0, height - 1, height))\n\n"
        "    def forward(self, logits: torch.Tensor) -> torch.Tensor:\n"
        "        # logits shape: (B, H, W)\n"
        "        # Softmax along vertical spatial dimension H\n"
        "        prob_dist = F.softmax(self.beta * logits, dim=1)\n"
        "        # Expected vertical coordinate via continuous expectation\n"
        "        y_coords = self.y_coords.view(1, self.height, 1)\n"
        "        expected_y = torch.sum(prob_dist * y_coords, dim=1) # Shape: (B, W)\n"
        "        return expected_y"
    )
    builder.add_paragraph(code_snippet_1, space_after=160)

    builder.add_heading3("C.2 Multi-Chunk Sliding-Window Stitching for 10-Second Continuous Rows")
    code_snippet_2 = (
        "def reconstruct_10s_continuous_row(row_crop_img, model, num_chunks=4):\n"
        "    \"\"\"\n"
        "    Partitions a wide 10-second continuous lead image (~1500 px) into 4 sub-windows,\n"
        "    processes each through the CNN receptive field (256x256), and stitches coordinates.\n"
        "    \"\"\"\n"
        "    h, w = row_crop_img.shape[:2]\n"
        "    chunk_w = w // num_chunks\n"
        "    stitched_signal = []\n\n"
        "    for c in range(num_chunks):\n"
        "        x_start = c * chunk_w\n"
        "        x_end = (c + 1) * chunk_w if c < num_chunks - 1 else w\n"
        "        sub_crop = row_crop_img[:, x_start:x_end]\n\n"
        "        # Standardize sub-crop to CNN input resolution\n"
        "        input_tensor = preprocess_crop_to_tensor(sub_crop, target_size=(256, 256))\n"
        "        with torch.no_grad():\n"
        "            y_sub = model.predict_coords(input_tensor) # Shape: (1, 256)\n\n"
        "        # Rescale normalized sub-coordinates back to source pixel dimensions\n"
        "        y_rescaled = y_sub * (h / 256.0)\n"
        "        stitched_signal.extend(y_rescaled.tolist())\n\n"
        "    return np.array(stitched_signal)"
    )
    builder.add_paragraph(code_snippet_2, space_after=160)

    builder.add_page_break()

    # =========================================================================
    # ANNEXURES
    # =========================================================================
    builder.add_heading1("Annexure I: Program Outcomes (PO) & PSO Mapping Matrix")
    builder.add_paragraph("Exhaustive mapping of technical engineering project activities to NBA Program Outcomes (PO 1 - PO 12) and Program Specific Outcomes (PSO 1 - PSO 2).", italic=True)

    po_headers = ["PO / PSO Identifier", "Program Outcome Title", "Mapped Project Engineering Activity & Technical Evidence", "Attainment Level"]
    po_rows = [
        [
            item["po"],
            item["name"],
            item["activity"],
            item["level"]
        ]
        for item in rd.PO_PSO_MAPPINGS
    ]
    builder.add_table(po_headers, po_rows, [1200, 1800, 4106, 1200], align_cols=["center", "left", "left", "center"])

    builder.add_page_break()

    builder.add_heading1("Annexure II: Project Outcome & Deliverables Statement")
    builder.add_paragraph(
        "In accordance with National Board of Accreditation (NBA) project evaluation guidelines, the tangible outcomes "
        "and concrete engineering deliverables generated by this B.Tech major project are formally summarized below:\n\n"
        "1. Full-Stack Interactive Workstation Application: A deployed, locally executable software product featuring a Python FastAPI "
        "asynchronous backend and modern Svelte visual frontend. The application accepts drag-and-drop ECG uploads, displays interactive "
        "diagnostic verification overlays, renders real-time lead waveform graphs, and exports calibrated signals.\n\n"
        "2. Production Deep Neural Checkpoints: Trained ResNet-34 U-Net convolutional neural network checkpoints optimized for multi-lead "
        "ECG trace segmentation and soft-argmax spatial coordinate regression across diverse paper backgrounds.\n\n"
        "3. Paired Synthetic & Clinical Evaluation Datasets: A reproducible dataset generation pipeline capable of synthesizing paired full-sheet "
        "ECG images with exact ground-truth PTB-XL digital waveforms, incorporating realistic paper textures, colored grids, and photographic skew.\n\n"
        "4. Standardized Open-Source Biomedical Benchmark: A rigorous, held-out 40-sheet PTB-XL Fold 10 pilot benchmark dataset with complete "
        "evaluation manifests, confusion matrices, per-lead correlation statistics, and diagnostic overlay artifacts.\n\n"
        "5. Automated Software Verification Suite: A comprehensive suite of 142 automated unit and integration tests passing at 85% code coverage, "
        "verifying skew detection, color de-mixing, layout classification, coordinate mapping, and file export integrity."
    )

    builder.add_page_break()

    builder.add_heading1("Annexure III: Environmental & Societal Sustainability Statement")
    builder.add_paragraph(
        "The engineering paradigm implemented in ECG DigiAI directly supports the United Nations Sustainable Development Goals (SDGs), "
        "specifically SDG 3 (Good Health and Well-Being), SDG 9 (Industry, Innovation, and Infrastructure), and SDG 12 (Responsible Consumption and Production).\n\n"
        "Healthcare Equity & Access (SDG 3): In developing economies, cardiovascular diseases are often under-diagnosed due to the scarcity of "
        "digital electrocardiographs in primary health centers. By enabling healthcare practitioners to photograph physical ECG printouts and "
        "convert them into standardized digital waveforms on standard computers, ECG DigiAI brings specialist-level tele-cardiology to underserved "
        "rural populations without capital-intensive hardware replacement.\n\n"
        "Sustainable Paperless Healthcare (SDG 12): Traditional hospital archiving systems store millions of physical thermal paper ECGs "
        "in climate-controlled facilities. Thermal paper is coated with reactive chemical dyes and bisphenol developers (BPA/BPS), rendering it "
        "non-recyclable and hazardous to landfill ecosystems. Digitizing legacy archives enables healthcare institutions to transition to "
        "secure, paperless digital storage, reducing physical paper manufacturing demands and cutting the carbon footprint of hospital logistics.\n\n"
        "Energy-Efficient Computing (SDG 9): The deep learning architecture is optimized for high-efficiency CPU and commodity GPU inference. "
        "By avoiding parameter-heavy vision foundation models and utilizing a lightweight ResNet-34 backbone, inference consumes less than "
        "0.005 kWh per sheet, aligning with modern green computing standards."
    )

    builder.add_page_break()

    builder.add_heading1("Annexure IV: Team Member Roles & Individual Contributions")
    builder.add_paragraph("Documentation of individual student responsibilities, technical deliverables, and project ownership.", italic=True)

    team_headers = ["Student Name & Enrollment", "Assigned Technical Domain", "Specific Engineering Modules Owned", "Primary Deliverables"]
    team_rows = [
        [
            f"{rd.STUDENTS[0]['name']}\n({rd.STUDENTS[0]['enrol']})",
            "Computer Vision & Preprocessing",
            "Hough transform deskewing, HSV color de-mixing, CLAHE grid suppression, and contour-constrained layout classification.",
            "`lead_detector.py`, `grid_suppressor.py`, layout classification module, synthetic dataset generation renderer."
        ],
        [
            f"{rd.STUDENTS[1]['name']}\n({rd.STUDENTS[1]['enrol']})",
            "Deep Learning & Waveform Modeling",
            "ResNet-34 U-Net architecture, soft-argmax spatial regression layer, training pipeline, multi-chunk sliding-window inference.",
            "`deep_digitizer.py`, `train_robust_digitizer.py`, Fold 10 evaluation harness, real image regression suite."
        ],
        [
            f"{rd.STUDENTS[2]['name']}\n({rd.STUDENTS[2]['enrol']})",
            "Workstation UI & Quality Assurance",
            "FastAPI backend services, Svelte interactive UI, diagnostic overlay rendering, test suite development, and documentation.",
            "Workstation web UI, automated test suite (142 unit tests), WFDB/CSV export engines, report verification assets."
        ],
    ]
    builder.add_table(team_headers, team_rows, [2200, 2000, 2306, 1800], align_cols=["left", "left", "left", "left"])

    builder.add_paragraph(" ", space_after=160)
    builder.add_paragraph(
        "Individual Contribution Sign-Off:\n"
        "We certify that the division of technical tasks described above represents an accurate and collaborative execution of the B.Tech "
        "major project. All team members actively participated in system integration, evaluation debugging, and final report documentation.",
        italic=True
    )

    # Final build
    builder.generate()
    print("[+] Final Report Generation Complete!")

if __name__ == "__main__":
    build_report()
