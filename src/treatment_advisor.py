"""
Phase 5: Medical & Clinical Treatment Advisory System
Provides evidence-based clinical recommendations, follow-up schedules,
and intervention plans based on detected Diabetic Retinopathy severity grade.
"""

TREATMENT_PROTOCOLS = {
    0: {
        "severity_title": "Grade 0: No Diabetic Retinopathy",
        "risk_level": "Low / Routine Monitoring",
        "color_badge": "#2ec4b6",
        "description": "No microaneurysms, hemorrhages, or retinal lesions detected on fundus examination.",
        "follow_up": "Annual screening (every 12 months) by an optometrist or ophthalmologist.",
        "glycemic_target": "Maintain strict HbA1c < 7.0% and target blood pressure < 130/80 mmHg.",
        "clinical_actions": [
            "Continue annual routine dilated eye examinations.",
            "Maintain optimal blood glucose control (HbA1c monitoring every 3 months).",
            "Control blood pressure and lipid profile (serum cholesterol).",
            "Educate patient on maintaining healthy lifestyle and regular physical activity."
        ],
        "medication_laser": "No immediate ocular medication or laser treatment required."
    },
    1: {
        "severity_title": "Grade 1: Mild Non-Proliferative DR (NPDR)",
        "risk_level": "Mild Risk / Close Monitoring",
        "color_badge": "#ffbf00",
        "description": "Presence of isolated microaneurysms only. Small vessel dilation observed.",
        "follow_up": "Follow-up screening within 6 to 12 months.",
        "glycemic_target": "Strict HbA1c target < 7.0%, aggressive blood pressure control (< 130/80 mmHg).",
        "clinical_actions": [
            "Schedule follow-up dilated fundus evaluation within 6–12 months.",
            "Review diabetic management plan with endocrinologist or primary care physician.",
            "Monitor lipid levels (statins as indicated by PCP).",
            "Patient counseling regarding early signs of visual changes (floaters, blurriness)."
        ],
        "medication_laser": "No intraocular injections or laser therapy indicated at this stage."
    },
    2: {
        "severity_title": "Grade 2: Moderate Non-Proliferative DR (NPDR)",
        "risk_level": "Moderate Risk / Specialist Referral",
        "color_badge": "#ff9f1c",
        "description": "Microaneurysms, hard exudates, venous bleeding, and intraretinal hemorrhages detected in 1–3 quadrants.",
        "follow_up": "Comprehensive ophthalmologist evaluation within 3 to 6 months.",
        "glycemic_target": "Tight glycemic control under physician supervision.",
        "clinical_actions": [
            "Refer to a Retina Specialist / Ophthalmologist within 3–6 months.",
            "Perform Optical Coherence Tomography (OCT) scan to evaluate Macular Edema (DME).",
            "Assess for signs of Clinically Significant Macular Edema (CSME).",
            "Optimize systemic hypertension and renal function monitoring."
        ],
        "medication_laser": "Anti-VEGF therapy (e.g., Ranibizumab/Aflibercept) or focal laser photocoagulation if Macular Edema is present."
    },
    3: {
        "severity_title": "Grade 3: Severe Non-Proliferative DR (NPDR)",
        "risk_level": "High Risk / Urgent Specialist Intervention",
        "color_badge": "#e63946",
        "description": ">20 intraretinal hemorrhages in 4 quadrants, venous beading in 2+ quadrants, or IRMA in 1+ quadrant (4-2-1 rule).",
        "follow_up": "Urgent retina specialist consultation within 2 to 4 weeks.",
        "glycemic_target": "Strict metabolic control; avoid rapid drops in blood sugar without specialist oversight.",
        "clinical_actions": [
            "URGENT referral to Retina Specialist within 2–4 weeks.",
            "Perform OCT and Fundus Fluorescein Angiography (FFA) to map capillary non-perfusion.",
            "Prepare for prophylactic laser treatment or anti-VEGF therapy before progression to proliferative phase.",
            "Close patient surveillance for neovascularization signs."
        ],
        "medication_laser": "Panretinal Photocoagulation (PRP) evaluation or intraocular Anti-VEGF injections indicated to reduce ischemia."
    },
    4: {
        "severity_title": "Grade 4: Proliferative Diabetic Retinopathy (PDR)",
        "risk_level": "Critical Risk / Emergency Ocular Intervention",
        "color_badge": "#9d0208",
        "description": "Active neovascularization (new abnormal blood vessels), vitreous hemorrhage, preretinal hemorrhage, or tractional retinal detachment.",
        "follow_up": "IMMEDIATE Emergency evaluation by a Retina Specialist (within 24–48 hours).",
        "glycemic_target": "Multidisciplinary hospital care (Endocrinology + Retina Specialist team).",
        "clinical_actions": [
            "IMMEDIATE EMERGENCY evaluation by a Vitreoretinal Specialist.",
            "Assess risk of vision loss, tractional retinal detachment, or neovascular glaucoma.",
            "Immediate Panretinal Photocoagulation (PRP) laser therapy.",
            "Prompt intravitreal Anti-VEGF anti-angiogenic injections.",
            "Vitreoretinal surgery (Pars Plana Vitrectomy - PPV) if persistent vitreous hemorrhage or tractional detachment occurs."
        ],
        "medication_laser": "Urgent Panretinal Photocoagulation (PRP) + Intravitreal Anti-VEGF injections (Aflibercept / Faricimab) + Pars Plana Vitrectomy if surgical."
    }
}

def get_treatment_advisory(severity_class, csme_risk=False):
    """
    Returns clinical treatment dictionary for specified severity class (0-4)
    with dynamic CSME Macular Edema risk assessment.
    """
    cls = int(severity_class)
    if cls not in TREATMENT_PROTOCOLS:
        cls = 0

    import copy
    protocol = copy.deepcopy(TREATMENT_PROTOCOLS[cls])

    if csme_risk:
        protocol['csme_alert'] = "⚠️ HIGH RISK: Clinically Significant Macular Edema (CSME) detected! Lesion within 1.0 Disc Diameter of Fovea."
        protocol['clinical_actions'].insert(0, "URGENT OCT (Optical Coherence Tomography) Scan to measure Macular Center Thickness.")
        protocol['clinical_actions'].insert(1, "Evaluate for Focal Laser Photocoagulation or Anti-VEGF Intravitreal Injection to prevent central vision loss.")
        if cls <= 1:
            protocol['risk_level'] = "Elevated Risk / Macular Edema Alert"
    else:
        protocol['csme_alert'] = "Normal Macular Foveal Clearance (No CSME lesion within 1.0 Disc Diameter of Fovea)."

    return protocol
