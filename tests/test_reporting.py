"""Unit tests for HTML reporting."""

from reporting.html_report import generate_html_report
from common.types import CurveData


def test_html_report_generation(tmp_path):
    report_file = tmp_path / "test_report.html"
    c1 = CurveData(
        name="Test Curve",
        x_name="Voltage",
        x_unit="V",
        y_name="Current",
        y_unit="A",
        x_data=[0.0, 1.0, 2.0],
        y_data=[0.0, 0.5, 1.0]
    )
    generate_html_report(
        device_name="TestDevice",
        output_path=report_file,
        curves=[c1],
        metrics={"static_rmse_pct": 0.5}
    )
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "PieceSpice Validation Report: TestDevice" in content
    assert "plotly-2.35.2.min.js" in content
