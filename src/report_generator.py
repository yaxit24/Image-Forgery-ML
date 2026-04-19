# src/report_generator.py
"""
Structured forensic report generation.

Primary:  Groq API (Llama 3.1 8B) with grounded prompts
Fallback: Template-based report (no LLM needed)

The Groq prompt is grounded — it only uses provided analysis data,
reducing hallucination risk. If the API is unavailable, the template
fallback produces an equivalent structured report.
"""

import os
import json
from datetime import datetime


# ---- Groq API Report ----

SYSTEM_PROMPT = """You are a digital forensics expert writing a concise analysis report.
Rules:
- Use ONLY the provided analysis data. Do not invent findings.
- Be precise with numbers. Cite model names and confidence values.
- If models disagree, state the disagreement clearly.
- Keep the abstract to 2-3 sentences.
- Use the exact section headings provided."""

REPORT_TEMPLATE_PROMPT = """Given the following image forensic analysis data, write a structured report.

**Analysis Data:**
```json
{analysis_json}
```

**Output the report in this exact format:**

## Forensic Analysis of {filename}

### Abstract
(2-3 sentence summary of findings)

### Key Findings
(Bullet points of evidence with metrics)

### Suggested Tampering Type
(splicing / copy-move / compression artifacts / unknown — with brief justification)

### Technical Details
(Which models were used, their individual scores, localization stats)

### Conclusion
(Final verdict: Authentic or Forged, with confidence level and caveats)"""


def generate_report_groq(analysis_data, filename="uploaded_image"):
    """
    Generate a forensic report using Groq API.

    Args:
        analysis_data: dict with ensemble results, detector stats, localization stats
        filename:      original image filename

    Returns:
        str: markdown-formatted report, or None if API fails
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        analysis_json = json.dumps(analysis_data, indent=2, default=str)
        user_prompt = REPORT_TEMPLATE_PROMPT.format(
            analysis_json=analysis_json,
            filename=filename,
        )

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,    # low temp = more factual
            max_tokens=1024,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Groq API error: {e}")
        return None


# ---- Template Fallback ----

def generate_report_fallback(analysis_data, filename="uploaded_image"):
    """
    Generate a structured report using pure templates (no LLM).
    Always works — used when Groq API is unavailable.
    """
    ensemble = analysis_data.get("ensemble", {})
    ela      = analysis_data.get("ela_stats", {})
    res      = analysis_data.get("residual_stats", {})
    block    = analysis_data.get("block_stats", {})
    copy     = analysis_data.get("copy_stats", {})
    loc      = analysis_data.get("localization", {})

    verdict    = ensemble.get("verdict", "unknown").upper()
    confidence = ensemble.get("confidence", 0)
    agreement  = ensemble.get("agreement", "unknown")
    details    = ensemble.get("details", "")
    cnn_conf   = ensemble.get("cnn_conf", 0)
    rf_conf    = ensemble.get("rf_conf", 0)

    tampered_pct   = loc.get("tampered_area_pct", 0)
    num_regions    = loc.get("num_regions", 0)
    largest_pct    = loc.get("largest_region_pct", 0)

    n_matches  = copy.get("n_good_matches", 0)
    n_clusters = copy.get("n_clusters", 0)

    # Determine tampering type
    if n_clusters > 0 and n_matches > 10:
        tampering_type = "Copy-Move"
        type_reason = f"Detected {n_clusters} cluster(s) of {n_matches} matched keypoints."
    elif tampered_pct > 5:
        tampering_type = "Splicing (likely)"
        type_reason = f"Localized tampered region covers {tampered_pct}% of the image."
    else:
        tampering_type = "Unknown / Subtle"
        type_reason = "No strong structural indicators of a specific manipulation type."

    # Build findings
    findings = []
    if ela.get("ela_mean", 0) > 15:
        findings.append(f"Elevated ELA mean ({ela['ela_mean']:.1f}) suggests compression inconsistencies.")
    if ela.get("ela_max", 0) > 100:
        findings.append(f"High ELA peak ({ela['ela_max']:.1f}) in localized regions.")
    if res.get("res_std", 0) > 0.01:
        findings.append(f"Residual noise variance ({res['res_std']:.4f}) indicates sensor pattern anomalies.")
    if n_clusters > 0:
        findings.append(f"Copy-move detector found {n_clusters} duplicated region cluster(s).")
    if tampered_pct > 0:
        findings.append(f"Localization mask covers {tampered_pct}% of image ({num_regions} region(s)).")
    if not findings:
        findings.append("No strong indicators of tampering detected by any analysis method.")

    findings_text = "\n".join(f"- {f}" for f in findings)

    # Caveats
    caveats = []
    if agreement == "partial":
        caveats.append("Models produced conflicting results — interpret with caution.")
    if confidence < 0.7:
        caveats.append(f"Confidence is moderate ({confidence:.0%}). Consider manual review.")
    caveat_text = " ".join(caveats) if caveats else "Analysis methods are in agreement."

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"""## Forensic Analysis of {filename}
*Generated: {now}*

### Abstract
This image has been classified as **{verdict}** with {confidence:.0%} ensemble confidence. \
{"The analysis detected suspicious regions covering " + f"{tampered_pct}% of the image. " if tampered_pct > 2 else ""}\
{details}

### Key Findings
{findings_text}

### Suggested Tampering Type
**{tampering_type}** — {type_reason}

### Technical Details
| Model | Result | Confidence |
|-------|--------|------------|
| MobileNetV2 CNN | {ensemble.get('cnn_label', 'N/A')} | {cnn_conf:.0%} |
| Random Forest (M1) | {ensemble.get('rf_label', 'N/A')} | {rf_conf:.0%} |
| **Ensemble** | **{verdict}** | **{confidence:.0%}** |

Localization: {tampered_pct}% tampered area, {num_regions} region(s), largest = {largest_pct}%.
Model agreement: {agreement}.

### Conclusion
**Verdict: {verdict}** ({confidence:.0%} confidence). {caveat_text}
"""
    return report.strip()


def generate_report(analysis_data, filename="uploaded_image"):
    """
    Generate a report — tries Groq API first, falls back to template.

    Returns:
        (report_text, source)
        source: "groq" or "template"
    """
    report = generate_report_groq(analysis_data, filename)
    if report:
        return report, "groq"

    return generate_report_fallback(analysis_data, filename), "template"
