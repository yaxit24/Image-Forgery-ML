# src/report_exporter.py
"""
Export forensic reports as PDF or Markdown files.

PDF:  Uses fpdf2 (lightweight, no system deps like wkhtmltopdf)
MD:   Plain text with optional base64-embedded images
"""

import io
import base64
import numpy as np
from PIL import Image


def _pil_to_bytes(pil_img, fmt="PNG"):
    """Convert PIL image to bytes."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


def _numpy_to_pil(arr):
    """Convert numpy array (HxW or HxWx3, float or uint8) to PIL Image."""
    if arr.dtype != np.uint8:
        # Normalize float images to 0-255
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")


def export_markdown(report_text, images=None):
    """
    Export report as a Markdown file with optional base64-embedded images.

    Args:
        report_text: str, the markdown report
        images:      dict of {name: numpy_array or PIL.Image}, optional

    Returns:
        bytes: UTF-8 encoded markdown content
    """
    md = report_text + "\n\n"

    if images:
        md += "---\n\n### Analysis Images\n\n"
        for name, img in images.items():
            if isinstance(img, np.ndarray):
                img = _numpy_to_pil(img)
            img_bytes = _pil_to_bytes(img)
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            md += f"**{name}**\n\n"
            md += f"![{name}](data:image/png;base64,{b64})\n\n"

    return md.encode("utf-8")


def export_pdf(report_text, images=None, title="Forensic Analysis Report"):
    """
    Export report as a PDF using fpdf2.

    Args:
        report_text: str, the report content (markdown stripped to plain text)
        images:      dict of {name: PIL.Image or numpy_array}, optional
        title:       str, PDF title

    Returns:
        bytes: PDF file content
    """
    from fpdf import FPDF
    import tempfile, os

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ---- Title page ----
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 20, title, ln=True, align="C")
    pdf.ln(10)

    # ---- Body ----
    pdf.set_font("Helvetica", "", 10)

    # Simple markdown → PDF rendering
    for line in report_text.split("\n"):
        stripped = line.strip()

        if stripped.startswith("## "):
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 8, stripped[3:], ln=True)
            pdf.set_font("Helvetica", "", 10)
        elif stripped.startswith("### "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 7, stripped[4:], ln=True)
            pdf.set_font("Helvetica", "", 10)
        elif stripped.startswith("- "):
            pdf.cell(5)   # indent
            pdf.cell(0, 6, "  • " + stripped[2:], ln=True)
        elif stripped.startswith("|"):
            # Simple table row — render as fixed-width text
            pdf.set_font("Courier", "", 9)
            pdf.cell(0, 5, stripped, ln=True)
            pdf.set_font("Helvetica", "", 10)
        elif stripped.startswith("*") and stripped.endswith("*"):
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 6, stripped.strip("*"), ln=True)
            pdf.set_font("Helvetica", "", 10)
        elif stripped == "---":
            pdf.ln(3)
            pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 170, pdf.get_y())
            pdf.ln(3)
        elif stripped:
            # Remove markdown bold markers for PDF
            clean = stripped.replace("**", "")
            pdf.multi_cell(0, 6, clean)
        else:
            pdf.ln(3)

    # ---- Images ----
    if images:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Analysis Images", ln=True)
        pdf.ln(5)

        for name, img in images.items():
            if isinstance(img, np.ndarray):
                img = _numpy_to_pil(img)

            # Save to temp file (fpdf2 needs a file path)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp.name)
            tmp.close()

            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, name, ln=True)

            # Fit image to page width
            page_w = pdf.w - 2 * pdf.l_margin
            img_w, img_h = img.size
            ratio = min(page_w / img_w, 80 / img_h)  # max 80mm height
            pdf.image(tmp.name, w=img_w * ratio, h=img_h * ratio)
            pdf.ln(5)

            os.unlink(tmp.name)

    # Output
    return pdf.output()
