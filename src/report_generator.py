"""ReportGenerator — produces downloadable PDF soil audit reports.

What it generates:
  A multi-page PDF where each page covers one district:
  - SBRS score, risk category, and dominant risk factor
  - Component scores breakdown (which parameter is hurting the most)
  - Projected SBRS values table (year 0 → year N)
  - Actionable recommendations for the farmer

The PDF is returned as bytes so Streamlit can offer it as a download button.

Usage:
  generator = ReportGenerator()
  pdf_bytes = generator.generate_pdf(
      district_data=df,
      sbrs_results=sbrs_df,
      projections=projections_df,
      recommendations={"Latur": ["Apply compost...", ...]},
  )
  # Pass pdf_bytes to st.download_button(data=pdf_bytes, ...)

Helper:
  sanitize_filename(name) — strips path traversal characters from filenames
"""

from __future__ import annotations

import re
from io import BytesIO

import pandas as pd
from fpdf import FPDF


def sanitize_filename(name: str) -> str:
    """Remove path traversal and dangerous characters from a filename string.

    Removes: ``../``, ``./``, ``/``, ``\\``, null bytes, and any remaining
    characters that are not alphanumeric, spaces, hyphens, underscores, or dots.

    Examples:
        >>> sanitize_filename("../etc/passwd")
        'etcpasswd'
        >>> sanitize_filename("district/name")
        'districtname'
        >>> sanitize_filename("safe_name-01")
        'safe_name-01'
    """
    # Remove null bytes
    name = name.replace("\x00", "")
    # Remove path traversal sequences (order matters: ../ before ./ before /)
    name = name.replace("../", "").replace("./", "")
    # Remove remaining slashes and backslashes
    name = name.replace("/", "").replace("\\", "")
    # Strip any remaining non-safe characters (keep alphanumeric, space, -, _, .)
    name = re.sub(r"[^\w\s\-.]", "", name)
    return name


class ReportGenerator:
    """Generate PDF soil audit reports for one or more districts."""

    # Column label mapping for component scores
    _COMPONENT_LABELS: dict[str, str] = {
        "organic_carbon_norm": "Organic Carbon",
        "irrigation_stress_norm": "Irrigation Stress",
        "fertilizer_load_norm": "Fertilizer Load",
        "crop_diversity_index_norm": "Crop Diversity Index",
        "rainfall_variability_norm": "Rainfall Variability",
    }

    def generate_pdf(
        self,
        district_data: pd.DataFrame,
        sbrs_results: pd.DataFrame,
        projections: pd.DataFrame,
        recommendations: dict[str, list[str]],
    ) -> bytes:
        """Render a PDF soil audit report and return it as bytes.

        For each district in *sbrs_results* the report includes:
        - District name, SBRS score, risk category, and dominant factor
        - Component scores breakdown
        - Bankruptcy year (if any)
        - Recommendations list
        - Projected SBRS values table (year 0 to max year)

        Args:
            district_data: Raw district parameter DataFrame (used for context).
            sbrs_results: DataFrame with columns ``district``, ``sbrs``,
                ``risk_category``, ``dominant_factor``, ``component_scores``,
                and optionally ``bankruptcy_year``.
            projections: Long-format projection DataFrame with columns
                ``district``, ``year``, and ``sbrs``.
            recommendations: Mapping of district name → list of recommendation strings.

        Returns:
            PDF document as a ``bytes`` object.

        Raises:
            RuntimeError: If PDF generation fails due to a rendering or library error.
        """
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_margins(left=15, top=15, right=15)

            for _, row in sbrs_results.iterrows():
                district = str(row["district"])
                safe_district = sanitize_filename(district)  # noqa: F841 — used for filename context
                sbrs = float(row["sbrs"])
                risk_category = str(row["risk_category"])
                dominant_factor = str(row["dominant_factor"])
                component_scores: dict = row.get("component_scores", {}) or {}
                bankruptcy_year = row.get("bankruptcy_year", None)

                pdf.add_page()

                # ── Title ──────────────────────────────────────────────────
                pdf.set_font("Helvetica", style="B", size=16)
                pdf.cell(0, 10, "Soil Audit Report", new_x="LMARGIN", new_y="NEXT", align="C")
                pdf.ln(2)

                # ── District header ────────────────────────────────────────
                pdf.set_font("Helvetica", style="B", size=13)
                pdf.cell(0, 8, f"District: {district}", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1)

                # ── SBRS summary ───────────────────────────────────────────
                pdf.set_font("Helvetica", size=11)
                pdf.cell(0, 7, f"SBRS Score: {sbrs:.1f} / 100", new_x="LMARGIN", new_y="NEXT")
                pdf.cell(0, 7, f"Risk Category: {risk_category}", new_x="LMARGIN", new_y="NEXT")

                dominant_label = self._COMPONENT_LABELS.get(dominant_factor, dominant_factor)
                pdf.cell(0, 7, f"Dominant Risk Factor: {dominant_label}", new_x="LMARGIN", new_y="NEXT")

                if bankruptcy_year is not None:
                    pdf.set_font("Helvetica", style="B", size=11)
                    pdf.set_text_color(200, 0, 0)
                    pdf.cell(0, 7, f"Bankruptcy Year: {bankruptcy_year}", new_x="LMARGIN", new_y="NEXT")
                    pdf.set_text_color(0, 0, 0)
                    pdf.set_font("Helvetica", size=11)
                else:
                    pdf.cell(0, 7, "Bankruptcy Year: Not projected within horizon", new_x="LMARGIN", new_y="NEXT")

                pdf.ln(3)

                # ── Component scores ───────────────────────────────────────
                pdf.set_font("Helvetica", style="B", size=12)
                pdf.cell(0, 8, "Component Scores Breakdown", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", size=10)

                if component_scores:
                    col_w = [100, 40]
                    pdf.set_fill_color(230, 230, 230)
                    pdf.cell(col_w[0], 7, "Parameter", border=1, fill=True)
                    pdf.cell(col_w[1], 7, "Weighted Score", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
                    pdf.set_fill_color(255, 255, 255)
                    for param, score in component_scores.items():
                        label = self._COMPONENT_LABELS.get(param, param)
                        pdf.cell(col_w[0], 7, label, border=1)
                        pdf.cell(col_w[1], 7, f"{float(score):.4f}", border=1, new_x="LMARGIN", new_y="NEXT")
                else:
                    pdf.cell(0, 7, "No component score data available.", new_x="LMARGIN", new_y="NEXT")

                pdf.ln(3)

                # ── Recommendations ────────────────────────────────────────
                pdf.set_font("Helvetica", style="B", size=12)
                pdf.cell(0, 8, "Recommendations", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", size=10)

                district_recs = recommendations.get(district, [])
                if district_recs:
                    for i, rec in enumerate(district_recs, start=1):
                        pdf.multi_cell(0, 6, f"{i}. {rec}", new_x="LMARGIN", new_y="NEXT")
                else:
                    pdf.cell(0, 7, "No recommendations available for this district.", new_x="LMARGIN", new_y="NEXT")

                pdf.ln(3)

                # ── Projection table ───────────────────────────────────────
                district_proj = projections[projections["district"] == district].copy()
                if not district_proj.empty:
                    district_proj = district_proj.sort_values("year")
                    pdf.set_font("Helvetica", style="B", size=12)
                    pdf.cell(0, 8, "Projected SBRS Values", new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Helvetica", size=10)

                    col_w2 = [60, 60]
                    pdf.set_fill_color(230, 230, 230)
                    pdf.cell(col_w2[0], 7, "Year", border=1, fill=True)
                    pdf.cell(col_w2[1], 7, "SBRS", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
                    pdf.set_fill_color(255, 255, 255)

                    for _, proj_row in district_proj.iterrows():
                        pdf.cell(col_w2[0], 7, str(int(proj_row["year"])), border=1)
                        pdf.cell(col_w2[1], 7, f"{float(proj_row['sbrs']):.1f}", border=1, new_x="LMARGIN", new_y="NEXT")

            # ── Output ─────────────────────────────────────────────────────
            buf = BytesIO()
            pdf.output(buf)
            return buf.getvalue()

        except Exception as exc:
            raise RuntimeError(f"PDF generation failed: {exc}") from exc
