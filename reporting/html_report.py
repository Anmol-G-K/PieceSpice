"""Interactive HTML report generation with Jinja2 and Plotly.

The output is a single HTML file; Plotly itself is loaded from the CDN
referenced in the template (no local server, but a browser needs network
access to render the charts).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import jinja2
import plotly.graph_objects as go
from common.types import CurveData


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PieceSpice Report: {{ device_name }}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 24px; background: #f8fafc; }
        .container { max-width: 1100px; margin: 0 auto; background: #fff; padding: 32px; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
        h1 { color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }
        h2 { color: #1e293b; margin-top: 28px; }
        .card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 20px 0; }
        .card { background: #f1f5f9; padding: 16px; border-radius: 8px; border-left: 4px solid #3b82f6; }
        .card-title { font-size: 0.85em; color: #64748b; text-transform: uppercase; font-weight: bold; }
        .card-value { font-size: 1.6em; font-weight: bold; color: #0f172a; margin-top: 4px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }
        th { background: #f8fafc; color: #475569; font-size: 0.9em; text-transform: uppercase; }
        .chart-box { margin: 24px 0; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; background: #ffffff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>PieceSpice Validation Report: {{ device_name }}</h1>
        <p><strong>Generated At:</strong> {{ timestamp }} | <strong>Model Tier:</strong> Tier {{ tier }}</p>
        
        {% if metric_cards %}
        <h2>Executive Performance Summary</h2>
        <div class="card-grid">
            {% for card in metric_cards %}
            <div class="card">
                <div class="card-title">{{ card.label }}</div>
                <div class="card-value">{{ card.value }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <h2>Datasheet Curve Comparison</h2>
        {% for chart in charts %}
        <div class="chart-box">
            <div id="{{ chart.id }}"></div>
        </div>
        {% endfor %}
    </div>

    <script>
        {% for chart in charts %}
        Plotly.newPlot('{{ chart.id }}', {{ chart.data | safe }}, {{ chart.layout | safe }});
        {% endfor %}
    </script>
</body>
</html>
"""


_METRIC_CARDS = {
    "static_rmse_pct": ("Static RMSE", "%"),
    "loss_error_pct": ("Switching Loss Error", "%"),
    "speedup_factor": ("Transient Speedup", "\u00d7"),
}


def _metric_cards(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    """Format the metrics the caller actually supplied.

    No defaults and no placeholder values: a metric that was not measured is
    simply absent from the report.
    """
    cards: List[Dict[str, str]] = []
    for key, value in metrics.items():
        label, suffix = _METRIC_CARDS.get(
            key, (key.replace("_", " ").title(), ""))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            cards.append({"label": label, "value": str(value)})
        elif suffix:
            cards.append({"label": label, "value": f"{value:.2f}{suffix}"})
        else:
            cards.append({"label": label, "value": f"{value:.4g}"})
    return cards


def generate_html_report(
    device_name: str,
    output_path: Path,
    curves: List[CurveData],
    metrics: Dict[str, Any],
    tier: int = 2,
):
    """Generate a single-file interactive HTML validation report.

    ``metrics`` is rendered verbatim; no value is invented when a metric is
    absent (the summary section is omitted when the dict is empty).
    """
    charts = []
    for idx, curve in enumerate(curves):
        chart_id = f"chart_{idx}"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve.x_data,
            y=curve.y_data,
            mode='lines+markers',
            name=f"{curve.name}"
        ))
        fig.update_layout(
            title=f"{curve.name} - {curve.x_name} vs {curve.y_name}",
            xaxis_title=f"{curve.x_name} [{curve.x_unit}]",
            yaxis_title=f"{curve.y_name} [{curve.y_unit}]",
            template="plotly_white",
            # STRICT REQUIREMENT: Legend at EAST outside plotting area
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02
            )
        )
        plotly_json = fig.to_plotly_json()
        charts.append({
            "id": chart_id,
            "data": json.dumps(plotly_json["data"]),
            "layout": json.dumps(plotly_json["layout"])
        })

    template = jinja2.Template(HTML_TEMPLATE)
    rendered = template.render(
        device_name=device_name,
        timestamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        tier=tier,
        metric_cards=_metric_cards(metrics),
        charts=charts
    )
    output_path.write_text(rendered, encoding="utf-8")
