import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from src import config
from src.treatment_advisor import get_treatment_advisory

def generate_html_report(patient_info, result_dict, output_html_path):
    """
    Generate an HTML medical diagnostic report.
    """
    treatment = get_treatment_advisory(result_dict['class_id'])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
            ul {{ line-height: 1.6; }}
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
    Generates a PDF summary document by rendering a structured high-res canvas image.
    """
    treatment = get_treatment_advisory(result_dict['class_id'])
    
    # Create blank canvas
    w, h = 800, 1100
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Draw Header
    draw.rectangle([(0, 0), (w, 100)], fill=(29, 53, 87))
    draw.text((30, 25), "DIABETIC RETINOPATHY DIAGNOSTIC REPORT", fill=(255, 255, 255))
    draw.text((30, 60), "Automated IVP & Deep Learning Clinical Advisory System", fill=(200, 225, 255))

    # Patient Details Box
    draw.rectangle([(30, 120), (w-30, 200)], fill=(245, 247, 250), outline=(200, 210, 225))
    draw.text((50, 135), f"Patient Name: {patient_info.get('name', 'Anonymous')}", fill=(0, 0, 0))
    draw.text((50, 165), f"Age / Sex: {patient_info.get('age', 'N/A')} / {patient_info.get('sex', 'N/A')}", fill=(0, 0, 0))
    draw.text((450, 135), f"Scan Date: {datetime.now().strftime('%Y-%m-%d')}", fill=(0, 0, 0))
    draw.text((450, 165), f"Examined Eye: {patient_info.get('eye', 'Right Eye')}", fill=(0, 0, 0))

    # Severity Banner
    draw.rectangle([(30, 220), (w-30, 280)], fill=(230, 240, 255), outline=(50, 120, 220))
    draw.text((50, 240), f"DIAGNOSIS: {treatment['severity_title']}", fill=(20, 40, 100))
    draw.text((500, 240), f"Confidence: {result_dict['confidence']*100:.1f}%", fill=(0, 100, 50))

    # Paste Images side by side
    if raw_bgr is not None:
        raw_pil = Image.fromarray(cv2.cvtColor(cv2.resize(raw_bgr, (320, 320)), cv2.COLOR_BGR2RGB))
        canvas.paste(raw_pil, (50, 310))
        draw.text((50, 640), "Raw Retinal Fundus Scan", fill=(0, 0, 0))

    if heatmap_rgb is not None:
        hmap_pil = Image.fromarray(cv2.resize(heatmap_rgb, (320, 320)))
        canvas.paste(hmap_pil, (430, 310))
        draw.text((430, 640), "Grad-CAM Lesion Heatmap", fill=(0, 0, 0))

    # Clinical Protocols
    draw.text((30, 680), "CLINICAL RECOMMENDATIONS & TREATMENT PLAN", fill=(29, 53, 87))
    draw.line([(30, 705), (w-30, 705)], fill=(200, 200, 200), width=2)

    y_pos = 720
    draw.text((30, y_pos), f"• Follow-up Schedule: {treatment['follow_up']}", fill=(0, 0, 0))
    y_pos += 30
    draw.text((30, y_pos), f"• Glycemic & Systemic Target: {treatment['glycemic_target']}", fill=(0, 0, 0))
    y_pos += 40

    draw.text((30, y_pos), "Key Clinical Actions:", fill=(0, 0, 0))
    y_pos += 25
    for act in treatment['clinical_actions']:
        draw.text((50, y_pos), f"- {act}", fill=(50, 50, 50))
        y_pos += 25

    y_pos += 15
    draw.text((30, y_pos), f"Medication / Laser Guidance: {treatment['medication_laser']}", fill=(180, 40, 40))

    # Save to PDF
    canvas.save(output_pdf_path, "PDF", resolution=100.0)
    return output_pdf_path

if __name__ == "__main__":
    print("Testing PDF Report Generator...")
    res = {'class_id': 2, 'confidence': 0.89}
    p_info = {'name': 'John Doe', 'age': '54', 'sex': 'Male', 'eye': 'Right Eye (OD)'}
    html_p = os.path.join(config.OUTPUT_DIR, "test_report.html")
    pdf_p = os.path.join(config.OUTPUT_DIR, "test_report.pdf")
    generate_html_report(p_info, res, html_p)
    print("Generated HTML report:", html_p)
