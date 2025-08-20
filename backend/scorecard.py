from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt
import io
import os

def _make_chart(metrics_dict):
    names = list(metrics_dict.keys())
    vals = [float(v) for v in metrics_dict.values()]
    plt.figure(figsize=(6,3))
    plt.bar(range(len(vals)), vals, color="#0677ce")
    plt.xticks(range(len(vals)), names, rotation=45, ha='right')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def generate_scorecard(metrics_dict, path="scorecard.pdf"):
    # Chart image
    img_buf = _make_chart(metrics_dict)
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "EthixAI - Ethical AI Scorecard")
    c.setFont("Helvetica", 10)
    y = height - 80
    for metric, value in metrics_dict.items():
        c.drawString(60, y, f"{metric}: {round(value, 4)}")
        y -= 14
    # Chart
    img = ImageReader(img_buf)
    c.drawImage(img, 50, y - 160, width=500, height=150, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()
