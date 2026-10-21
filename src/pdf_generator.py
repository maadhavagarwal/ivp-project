import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from src import config
from src.treatment_advisor import get_treatment_advisory

def generate_html_report(patient_info, result_dict, output_html_path):
    """
    Generate an HTML medical diagnostic report including quantitative IVP biomarkers.
    """
    feats = result_dict.get('features', {})
    csme_risk = feats.get('csme_risk', False)
    treatment = get_treatment_advisory(result_dict['class_id'], csme_risk=csme_risk)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format Biomarkers
    vessel_dens = f"{feats.get('vessel_density', 0.0) * 100:.2f}%"
    frangi_score = f"{feats.get('frangi_score', 0.0) * 100:.2f}%"
    gabor_score = f"{feats.get('gabor_score', 0.0):.1f}"
    exudate_ratio = f"{feats.get('exudate_ratio', 0.0) * 100:.2f}% ({feats.get('exudate_count', 0)} spots)"
    micro_cnt = f"{feats.get('lesion_count', 0)} spots"
    fovea_dist = f"{feats.get('fovea_lesion_distance_dd', 99.0):.2f} DD"
    csme_badge = "<span style='color: #d90429; font-weight: bold;'>⚠️ HIGH CSME RISK (Immediate OCT Required)</span>" if csme_risk else "<span style='color: #2ec4b6; font-weight: bold;'>✅ Normal Clearance</span>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Diabetic Retinopathy Diagnostic Report</title>
        <style>
            body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 30px; color: #2b2d42; }}
            .header {{ text-align: center; border-bottom: 3px solid #3a86ef; padding-bottom: 10px; margin-bottom: 20px; }}
            .header h1 {{ color: #1d3557; margin-bottom: 5px; }}
            .header p {{ color: #6c757d; font-size: 14px; margin: 0; }}
            .patient-box {{ background: #f8f9fa; border-left: 5px solid #3a86ef; padding: 15px; margin-bottom: 20px; border-radius: 4px; }}
            .grid {{ display: flex; gap: 20px; margin-bottom: 20px; }}
            .col {{ flex: 1; }}
            .badge {{ display: inline-block; padding: 6px 12px; color: #fff; font-weight: bold; border-radius: 20px; font-size: 14px; background: {treatment['color_badge']}; }}
            .section-title {{ border-bottom: 2px solid #e9ecef; padding-bottom: 5px; color: #1d3557; margin-top: 25px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #dee2e6; padding: 10px 14px; text-align: left; font-size: 14px; }}
            th {{ background-color: #f1f3f5; color: #1d3557; }}
            ul {{ line-height: 1.6; }}
            .csme-alert {{ background-color: #ffe3e3; border-left: 5px solid #e63946; padding: 12px; margin: 15px 0; color: #900; font-weight: bold; }}
            .footer {{ margin-top: 40px; text-align: center; font-size: 12px; color: #adb5bd; border-top: 1px solid #dee2e6; padding-top: 15px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Ophthalmic Diagnostic Report</h1>
            <p>Automated Retinal Scanning & Diabetic Retinopathy Assessment System</p>
        </div>

        <div class="patient-box">
            <div class="grid">
                <div class="col">
                    <strong>Patient ID / Name:</strong> {patient_info.get('name', 'N/A')}<br>
                    <strong>Age / Sex:</strong> {patient_info.get('age', 'N/A')} / {patient_info.get('sex', 'N/A')}<br>
                </div>
                <div class="col">
                    <strong>Scan Date:</strong> {timestamp}<br>
                    <strong>Eye Examined:</strong> {patient_info.get('eye', 'Right Eye (OD)')}<br>
                </div>
            </div>
        </div>

        <h3 class="section-title">Diagnostic Classification</h3>
        <p>
            <strong>Severity Stage:</strong> <span class="badge">{treatment['severity_title']}</span><br><br>
            <strong>Confidence Score:</strong> {result_dict['confidence'] * 100:.1f}%<br>
            <strong>Risk Assessment:</strong> {treatment['risk_level']}<br>
            <strong>Clinical Summary:</strong> {treatment['description']}
        </p>

        {f'<div class="csme-alert">{treatment.get("csme_alert", "")}</div>' if csme_risk else ''}

        <h3 class="section-title">Quantitative IVP Biomarker Analysis</h3>
        <table>
            <thead>
                <tr>
                    <th>Biomarker / Physical Feature</th>
                    <th>Measured Quantitative Value</th>
                    <th>Clinical Interpretation</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Vessel Area Density</strong></td>
                    <td>{vessel_dens}</td>
                    <td>Retinal vascular coverage index</td>
                </tr>
                <tr>
                    <td><strong>Frangi Vesselness Score</strong></td>
                    <td>{frangi_score}</td>
                    <td>Hessian multiscale tubular structure contrast</td>
                </tr>
                <tr>
                    <td><strong>Gabor Wavelet Contrast</strong></td>
                    <td>{gabor_score}</td>
                    <td>12-Orientation anisotropic vessel sharpness</td>
                </tr>
                <tr>
                    <td><strong>Hard Exudates (Lipid Leaks)</strong></td>
                    <td>{exudate_ratio}</td>
                    <td>Bright yellow lipid exudation density</td>
                </tr>
                <tr>
                    <td><strong>Microaneurysms (Red Lesions)</strong></td>
                    <td>{micro_cnt}</td>
                    <td>Isolated bleeding capillaries / microvascular lesions</td>
                </tr>
                <tr>
                    <td><strong>CSME Macular Clearance</strong></td>
                    <td>{fovea_dist} ({csme_badge})</td>
                    <td>Proximity to Fovea (threshold &le; 1.0 Disc Diameter)</td>
                </tr>
            </tbody>
        </table>

        <h3 class="section-title">Evidence-Based Treatment Protocol & Clinical Plan</h3>
        <p><strong>Follow-Up Schedule:</strong> {treatment['follow_up']}</p>
        <p><strong>Glycemic & Systemic Target:</strong> {treatment['glycemic_target']}</p>
        
        <h4>Recommended Clinical Actions:</h4>
        <ul>
            {"".join([f"<li>{action}</li>" for action in treatment['clinical_actions']])}
        </ul>

        <p><strong>Medication & Laser Therapy Guidance:</strong><br>{treatment['medication_laser']}</p>

        <div class="footer">
            <p>This automated report was generated using Computer Vision & PyTorch Deep Learning. Clinical decisions should be confirmed by a board-certified ophthalmologist.</p>
        </div>
    </body>
    </html>
    """

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_html_path

def generate_pdf_summary_image(patient_info, result_dict, raw_bgr, heatmap_rgb, output_pdf_path):
    """
    Generates a comprehensive PDF summary document by rendering a structured high-res canvas image
    including the quantitative IVP biomarkers table and clinical treatment plan.
    """
    feats = result_dict.get('features', {})
    csme_risk = feats.get('csme_risk', False)
    treatment = get_treatment_advisory(result_dict['class_id'], csme_risk=csme_risk)
    
    # Create blank canvas (High-res A4 proportions: 900 x 1350)
    w, h = 900, 1350
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Draw Header
    draw.rectangle([(0, 0), (w, 95)], fill=(29, 53, 87))
    draw.text((35, 20), "DIABETIC RETINOPATHY DIAGNOSTIC REPORT", fill=(255, 255, 255))
    draw.text((35, 55), "Automated IVP & Deep Learning Clinical Advisory System", fill=(200, 225, 255))

    # Patient Details Box
    draw.rectangle([(35, 115), (w-35, 185)], fill=(245, 247, 250), outline=(200, 210, 225))
    draw.text((55, 130), f"Patient Name: {patient_info.get('name', 'Anonymous')}", fill=(0, 0, 0))
    draw.text((55, 155), f"Age / Sex: {patient_info.get('age', 'N/A')} / {patient_info.get('sex', 'N/A')}", fill=(0, 0, 0))
    draw.text((520, 130), f"Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fill=(0, 0, 0))
    draw.text((520, 155), f"Examined Eye: {patient_info.get('eye', 'Right Eye')}", fill=(0, 0, 0))

    # Severity Banner
    draw.rectangle([(35, 200), (w-35, 260)], fill=(230, 240, 255), outline=(50, 120, 220))
    draw.text((55, 220), f"DIAGNOSIS: {treatment['severity_title']}", fill=(20, 40, 100))
    draw.text((580, 220), f"Confidence: {result_dict['confidence']*100:.1f}%", fill=(0, 100, 50))

    # Paste Images side by side (370 x 300 each)
    img_y = 280
    img_size = (380, 280)
    if raw_bgr is not None:
        raw_pil = Image.fromarray(cv2.cvtColor(cv2.resize(raw_bgr, img_size), cv2.COLOR_BGR2RGB))
        canvas.paste(raw_pil, (55, img_y))
        draw.text((55, img_y + 285), "Raw Retinal Fundus Scan", fill=(0, 0, 0))

    if heatmap_rgb is not None:
        hmap_pil = Image.fromarray(cv2.resize(heatmap_rgb, img_size))
        canvas.paste(hmap_pil, (465, img_y))
        draw.text((465, img_y + 285), "Grad-CAM Lesion Heatmap Overlay", fill=(0, 0, 0))

    # IVP Biomarker Table Box
    table_y = img_y + 325
    draw.text((35, table_y), "QUANTITATIVE IVP BIOMARKER ANALYSIS", fill=(29, 53, 87))
    draw.line([(35, table_y + 22), (w-35, table_y + 22)], fill=(200, 200, 200), width=2)
    
    # Biomarker measurements grid
    draw.rectangle([(35, table_y + 30), (w-35, table_y + 165)], fill=(250, 252, 255), outline=(190, 205, 230))
    
    v_dens = f"{feats.get('vessel_density', 0.0) * 100:.2f}%"
    frangi = f"{feats.get('frangi_score', 0.0) * 100:.2f}%"
    gabor = f"{feats.get('gabor_score', 0.0):.1f}"
    exudates = f"{feats.get('exudate_ratio', 0.0) * 100:.2f}% ({feats.get('exudate_count', 0)} spots)"
    lesions = f"{feats.get('lesion_count', 0)} spots"
    csme_text = f"HIGH CSME RISK ({feats.get('fovea_lesion_distance_dd', 99.0):.2f} DD)" if csme_risk else f"Normal Clearance ({feats.get('fovea_lesion_distance_dd', 99.0):.2f} DD)"
    
    draw.text((55, table_y + 45), f"• Vessel Density: {v_dens}", fill=(20, 20, 20))
    draw.text((55, table_y + 75), f"• Frangi Vessel Score: {frangi}", fill=(20, 20, 20))
    draw.text((55, table_y + 105), f"• Gabor Orientation Score: {gabor}", fill=(20, 20, 20))
    
    draw.text((480, table_y + 45), f"• Hard Exudate Area: {exudates}", fill=(20, 20, 20))
    draw.text((480, table_y + 75), f"• Microaneurysms: {lesions}", fill=(20, 20, 20))
    csme_color = (190, 20, 20) if csme_risk else (0, 120, 60)
    draw.text((480, table_y + 105), f"• Macula / CSME: {csme_text}", fill=csme_color)

    if csme_risk:
        draw.rectangle([(35, table_y + 175), (w-35, table_y + 205)], fill=(255, 230, 230), outline=(230, 50, 50))
        draw.text((50, table_y + 182), "⚠️ URGENT: Clinically Significant Macular Edema (CSME) detected! OCT Scan required.", fill=(180, 0, 0))
        plan_y = table_y + 225
    else:
        plan_y = table_y + 185

    # Clinical Protocols
    draw.text((35, plan_y), "CLINICAL RECOMMENDATIONS & TREATMENT PLAN", fill=(29, 53, 87))
    draw.line([(35, plan_y + 22), (w-35, plan_y + 22)], fill=(200, 200, 200), width=2)

    y_pos = plan_y + 35
    draw.text((35, y_pos), f"• Follow-up Schedule: {treatment['follow_up']}", fill=(0, 0, 0))
    y_pos += 26
    draw.text((35, y_pos), f"• Glycemic & Systemic Target: {treatment['glycemic_target']}", fill=(0, 0, 0))
    y_pos += 32

    draw.text((35, y_pos), "Key Clinical Actions:", fill=(0, 0, 0))
    y_pos += 22
    for act in treatment['clinical_actions'][:4]:
        draw.text((55, y_pos), f"- {act}", fill=(50, 50, 50))
        y_pos += 22

    y_pos += 10
    draw.text((35, y_pos), f"Medication / Laser Guidance: {treatment['medication_laser']}", fill=(180, 40, 40))

    # Footer
    draw.text((35, h - 35), "Automated IVP + AI Report. For clinical confirmation by licensed ophthalmologist.", fill=(150, 150, 150))

    # Save to PDF
    canvas.save(output_pdf_path, "PDF", resolution=100.0)
    return output_pdf_path

if __name__ == "__main__":
    print("Testing PDF Report Generator with Biomarkers...")
    res = {
        'class_id': 2,
        'confidence': 0.89,
        'features': {
            'vessel_density': 0.124,
            'frangi_score': 0.082,
            'gabor_score': 34.5,
            'exudate_ratio': 0.008,
            'exudate_count': 6,
            'lesion_count': 14,
            'fovea_lesion_distance_dd': 0.85,
            'csme_risk': True
        }
    }
    p_info = {'name': 'John Doe', 'age': '54', 'sex': 'Male', 'eye': 'Right Eye (OD)'}
    html_p = os.path.join(config.OUTPUT_DIR, "test_report.html")
    pdf_p = os.path.join(config.OUTPUT_DIR, "test_report.pdf")
    generate_html_report(p_info, res, html_p)
    print("Generated HTML report with biomarkers successfully:", html_p)
