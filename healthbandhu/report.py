"""Phase 12: automated clinical report generator (PDF via ReportLab).

Uses only ReportLab's built-in Helvetica font, so no font files are needed.
All user-provided text is XML-escaped before being placed in Paragraphs.
"""

from __future__ import annotations

import io
import uuid
from datetime import datetime
from typing import Dict, Optional
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import config

INK = colors.HexColor("#1F2544")
INDIGO = colors.HexColor("#2F3E8F")
MUTED = colors.HexColor("#5B6178")
RULE = colors.HexColor("#D9DDEA")
TINT = colors.HexColor("#EEF0F7")

TRIAGE_COLORS = {
    "CRITICAL": colors.HexColor("#B42318"),
    "HIGH RISK": colors.HexColor("#C2410C"),
    "MODERATE RISK": colors.HexColor("#A16207"),
    "LOW RISK": colors.HexColor("#15803D"),
}


def _pct(p: Optional[float]) -> str:
    return "-" if p is None else f"{p * 100:.1f}%"


def _points(delta: float) -> str:
    value = delta * 100
    return "0.0 pts" if abs(value) < 0.05 else f"{value:+.1f} pts"


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle("hb_body", parent=base["Normal"], fontName="Helvetica", fontSize=9.5,
                          leading=13.5, textColor=INK, alignment=TA_LEFT)
    return {
        "body": body,
        "small": ParagraphStyle("hb_small", parent=body, fontSize=8, leading=11, textColor=MUTED),
        "h2": ParagraphStyle("hb_h2", parent=body, fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                             spaceBefore=10, spaceAfter=5, textColor=INDIGO),
        "cell": ParagraphStyle("hb_cell", parent=body, fontSize=8.8, leading=11.5),
        "cell_bold": ParagraphStyle("hb_cell_bold", parent=body, fontName="Helvetica-Bold", fontSize=8.8,
                                    leading=11.5),
        "triage_level": ParagraphStyle("hb_triage_level", parent=body, fontName="Helvetica-Bold", fontSize=15,
                                       leading=19, textColor=colors.white),
        "triage_text": ParagraphStyle("hb_triage_text", parent=body, fontSize=9.5, leading=13,
                                      textColor=colors.white),
        "bullet": ParagraphStyle("hb_bullet", parent=body, leftIndent=10, bulletIndent=0, spaceAfter=2),
    }


def _table(rows, col_widths, header=True, zebra=True):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT", repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), TINT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INDIGO)]
    t.setStyle(TableStyle(style))
    return t


def _decorate_page(canvas, doc):
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(INDIGO)
    canvas.rect(0, height - 16 * mm, width, 16 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawString(18 * mm, height - 10.3 * mm, config.APP_TITLE)
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(width - 18 * mm, height - 10.3 * mm, "Clinical decision support report")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(18 * mm, 10.5 * mm, "Screening support only. Not a medical diagnosis. "
                                          "Confirm all findings with a qualified clinician.")
    canvas.drawRightString(width - 18 * mm, 10.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf_report(result, patient: Optional[Dict] = None) -> bytes:
    """Render a DiagnosisResult (healthbandhu.models) as PDF bytes."""
    patient = patient or {}
    s = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=24 * mm, bottomMargin=22 * mm,
        title=f"{config.APP_TITLE} report", author=config.APP_TITLE,
    )
    content_width = A4[0] - 36 * mm
    story = []

    try:
        generated = datetime.fromisoformat(result.generated_at).strftime("%d %b %Y, %H:%M %Z").strip()
    except ValueError:
        generated = result.generated_at

    # -- report details ------------------------------------------------------
    def val(key):
        v = str(patient.get(key) or "").strip()
        return escape(v) if v else "Not provided"

    meta = [
        [Paragraph("<b>Report ID</b>", s["cell"]), Paragraph(uuid.uuid4().hex[:10].upper(), s["cell"]),
         Paragraph("<b>Generated</b>", s["cell"]), Paragraph(escape(generated), s["cell"])],
        [Paragraph("<b>Patient</b>", s["cell"]), Paragraph(val("name"), s["cell"]),
         Paragraph("<b>Age / sex</b>", s["cell"]),
         Paragraph(f"{val('age')} / {val('sex')}", s["cell"])],
    ]
    story.append(_table(meta, [24 * mm, 55 * mm, 24 * mm, content_width - 103 * mm], header=False))
    story.append(Spacer(1, 8))

    # -- triage ---------------------------------------------------------------
    em = result.emergency
    triggers = ", ".join(f"{escape(sym)} (+{w})" for sym, w in em.triggers) or "none reported"
    triage_rows = [
        [Paragraph(f"Triage: {escape(em.level.title())}", s["triage_level"])],
        [Paragraph(escape(em.action), s["triage_text"])],
        [Paragraph(f"Emergency score {em.score}. Red-flag symptoms: {triggers}.", s["triage_text"])],
    ]
    triage = Table(triage_rows, colWidths=[content_width], hAlign="LEFT")
    triage.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TRIAGE_COLORS.get(em.level, INDIGO)),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
        ("TOPPADDING", (0, 1), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -2), 2),
    ]))
    story.append(triage)
    if em.is_urgent:
        story.append(Spacer(1, 3))
        story.append(Paragraph(escape(config.EMERGENCY_CONTACT_NOTE), s["small"]))

    # -- symptoms ---------------------------------------------------------------
    story.append(Paragraph("Reported symptoms", s["h2"]))
    story.append(Paragraph(escape(", ".join(result.symptoms)), s["body"]))

    # -- differential ------------------------------------------------------------
    story.append(Paragraph("Possible conditions", s["h2"]))
    conf = result.confidence
    story.append(Paragraph(
        f"Confidence: <b>{escape(conf.level.title())}</b>. The most likely condition has a probability of "
        f"{_pct(conf.probability)}, {conf.margin * 100:.1f} points ahead of the next."
        + (" The top results are close, so treat the ranking with caution." if conf.ambiguous else ""),
        s["body"]))
    story.append(Spacer(1, 5))
    header = ["#", "Condition", "Ensemble", "DNN", "Naive Bayes", "Rule score"]
    rows = [[Paragraph(f"<b>{h}</b>", s["cell"]) for h in header]]
    for c in result.candidates:
        rows.append([
            Paragraph(str(c.rank), s["cell"]),
            Paragraph(escape(c.name.capitalize()), s["cell_bold"] if c.rank == 1 else s["cell"]),
            Paragraph(_pct(c.probability), s["cell_bold"] if c.rank == 1 else s["cell"]),
            Paragraph(_pct(c.dnn_probability), s["cell"]),
            Paragraph(_pct(c.nb_probability), s["cell"]),
            Paragraph("-" if c.rule_score is None else f"{c.rule_score:.2f}", s["cell"]),
        ])
    story.append(_table(rows, [8 * mm, content_width - 92 * mm, 22 * mm, 20 * mm, 22 * mm, 20 * mm]))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        f"Ensemble = {result.dnn_weight:.2f} x deep neural network + {1 - result.dnn_weight:.2f} x Bernoulli "
        "Naive Bayes. Rule score is an independent symptom-profile match from 0 to 1.", s["small"]))

    # -- explanation ---------------------------------------------------------------
    top_name = escape(result.top.name)
    block = [Paragraph(f"Why {top_name} ranked first", s["h2"]),
             Paragraph("Each reported symptom was removed in turn. Influence is the symptom's share of the "
                       "evidence for this condition (negative values argue against it). Probability change "
                       "is how much the probability falls when the symptom is removed.", s["small"]),
             Spacer(1, 4)]
    rows = [[Paragraph("<b>Symptom</b>", s["cell"]), Paragraph("<b>Influence</b>", s["cell"]),
             Paragraph("<b>Probability change</b>", s["cell"])]]
    for contrib in result.contributions[:12]:
        rows.append([Paragraph(escape(contrib.symptom.capitalize()), s["cell"]),
                     Paragraph("0.0%" if abs(contrib.influence) < 0.0005 else f"{contrib.influence * 100:+.1f}%",
                               s["cell"]),
                     Paragraph(_points(contrib.impact), s["cell"])])
    block.append(_table(rows, [content_width - 80 * mm, 35 * mm, 45 * mm]))
    story.append(KeepTogether(block))

    if result.unreported_typical:
        story.append(Spacer(1, 6))
        items = ", ".join(f"{escape(sym)} ({p * 100:.0f}% of cases)" for sym, p in result.unreported_typical)
        story.append(Paragraph(f"<b>Often seen with {top_name} but not reported:</b> {items}.", s["body"]))

    # -- rule cross-check -----------------------------------------------------------
    if result.rule_matches:
        block = [Paragraph("Rule-based cross-check", s["h2"])]
        rows = [[Paragraph("<b>Condition</b>", s["cell"]), Paragraph("<b>Score</b>", s["cell"]),
                 Paragraph("<b>Key symptoms matched</b>", s["cell"])]]
        for m in result.rule_matches:
            matched = ", ".join(m.matched) if m.matched else "none"
            rows.append([Paragraph(escape(m.disease.capitalize()), s["cell"]),
                         Paragraph(f"{m.score:.2f}", s["cell"]),
                         Paragraph(f"{len(m.matched)} of {m.key_count}: {escape(matched)}", s["cell"])])
        block.append(_table(rows, [55 * mm, 16 * mm, content_width - 71 * mm]))
        story.append(KeepTogether(block))

    # -- recommendations ------------------------------------------------------------
    block = [Paragraph("Recommendations", s["h2"])]
    for rec in result.recommendations:
        block.append(Paragraph(escape(rec), s["bullet"], bulletText="\u2022"))
    story.append(KeepTogether(block))

    story.append(Spacer(1, 10))
    note = Table([[Paragraph(escape(config.DISCLAIMER), s["small"])]], colWidths=[content_width], hAlign="LEFT")
    note.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, RULE), ("BACKGROUND", (0, 0), (-1, -1), TINT),
                              ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                              ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(note)
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Models used: {escape(', '.join(result.models_used))}.", s["small"]))

    doc.build(story, onFirstPage=_decorate_page, onLaterPages=_decorate_page)
    return buf.getvalue()
