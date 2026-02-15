"""Risk monitoring dashboard with Dash + Plotly."""

import json
import logging
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/processed")


def create_dashboard(
    portfolio_returns: pd.Series,
    weights: pd.Series,
    risk_summary: dict,
    rolling_sharpe: pd.Series,
    turnover: pd.Series | None = None,
    metrics: dict | None = None,
) -> dash.Dash:
    """Create and return a Dash app for risk monitoring."""
    app = dash.Dash(__name__)

    cumulative = (1 + portfolio_returns).cumprod()
    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()

    # Build KPI cards
    kpi_cards = [
        _kpi_card("Sharpe", f"{risk_summary.get('sharpe', 0):.2f}"),
        _kpi_card("Max DD", f"{risk_summary.get('max_drawdown', 0):.1%}"),
        _kpi_card("Ann. Return", f"{risk_summary.get('annualized_return', 0):.1%}"),
        _kpi_card("VaR (95%)", f"{risk_summary.get('var_95_historical', 0):.4f}"),
        _kpi_card("CVaR (95%)", f"{risk_summary.get('cvar_95', 0):.4f}"),
    ]
    if metrics:
        if "avg_turnover" in metrics:
            kpi_cards.append(_kpi_card("Avg Turnover", f"{metrics['avg_turnover']:.1%}"))
        if "optimizer_method" in metrics:
            kpi_cards.append(
                _kpi_card("Optimizer", metrics["optimizer_method"].replace("_", " ").title())
            )
        if "total_cost_bps" in metrics:
            kpi_cards.append(_kpi_card("Cost (bps)", f"{metrics['total_cost_bps']:.0f}"))

    # Build chart rows
    row1 = [
        dcc.Graph(figure=_cumulative_returns_chart(cumulative), style={"flex": "1"}),
        dcc.Graph(figure=_weights_chart(weights), style={"flex": "1"}),
    ]
    row2 = [
        dcc.Graph(figure=_drawdown_chart(drawdown), style={"flex": "1"}),
        dcc.Graph(figure=_rolling_sharpe_chart(rolling_sharpe), style={"flex": "1"}),
    ]
    rows = [
        html.Div(row1, style={"display": "flex", "gap": "20px"}),
        html.Div(row2, style={"display": "flex", "gap": "20px"}),
    ]

    if turnover is not None:
        row3 = [
            dcc.Graph(figure=_turnover_chart(turnover), style={"flex": "1"}),
            dcc.Graph(figure=_hhi_chart(weights), style={"flex": "1"}),
        ]
        rows.append(html.Div(row3, style={"display": "flex", "gap": "20px"}))

    app.layout = html.Div(
        [
            html.H1("Signum — Risk Dashboard"),
            html.Div(
                kpi_cards,
                style={
                    "display": "flex", "gap": "15px", "marginBottom": "30px",
                    "flexWrap": "wrap",
                },
            ),
            *rows,
        ],
        style={"padding": "20px", "fontFamily": "system-ui"},
    )

    return app


def _kpi_card(label: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontSize": "13px", "color": "#666"}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": "bold"}),
        ],
        style={
            "padding": "12px 20px",
            "border": "1px solid #e0e0e0",
            "borderRadius": "8px",
            "minWidth": "120px",
        },
    )


def _cumulative_returns_chart(cumulative: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative.index, y=cumulative.values, mode="lines", name="Portfolio",
    ))
    fig.update_layout(
        title="Cumulative Returns", xaxis_title="Date", yaxis_title="Growth of $1",
    )
    return fig


def _weights_chart(weights: pd.Series) -> go.Figure:
    # Sort by weight descending for readability
    weights = weights.sort_values(ascending=False)
    fig = go.Figure(data=[go.Bar(x=weights.index, y=weights.values)])
    fig.update_layout(title="Portfolio Weights", xaxis_title="Ticker", yaxis_title="Weight")
    return fig


def _drawdown_chart(drawdown: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown.values,
            fill="tozeroy", mode="lines", name="Drawdown",
            line={"color": "red"},
        )
    )
    fig.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown %")
    return fig


def _rolling_sharpe_chart(rolling_sharpe: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index, y=rolling_sharpe.values, mode="lines", name="Sharpe",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Rolling Sharpe Ratio (60d)", xaxis_title="Date", yaxis_title="Sharpe",
    )
    return fig


def _turnover_chart(turnover: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=turnover.index, y=turnover.values, name="Turnover"))
    avg = turnover.mean()
    fig.add_hline(y=avg, line_dash="dash", line_color="orange",
                  annotation_text=f"avg={avg:.1%}")
    fig.update_layout(
        title="Portfolio Turnover per Rebalance",
        xaxis_title="Date", yaxis_title="Turnover (fraction)",
    )
    return fig


def _hhi_chart(weights: pd.Series) -> go.Figure:
    """Show position concentration metrics."""
    hhi = (weights ** 2).sum()
    eff_n = 1.0 / hhi if hhi > 0 else 0
    equal_weight = 1.0 / len(weights) if len(weights) > 0 else 0

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=eff_n,
        title={"text": "Effective # of Bets"},
        gauge={
            "axis": {"range": [0, len(weights)]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, len(weights) * 0.3], "color": "red"},
                {"range": [len(weights) * 0.3, len(weights) * 0.7], "color": "yellow"},
                {"range": [len(weights) * 0.7, len(weights)], "color": "green"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": 1.0 / equal_weight if equal_weight > 0 else 0,
            },
        },
    ))
    fig.update_layout(height=300)
    return fig


def _load_backtest_results():
    """Load persisted backtest results from data/processed/."""
    returns_path = RESULTS_DIR / "backtest_returns.parquet"

    if not returns_path.exists():
        logger.warning("No backtest results found. Run 'make backtest' first.")
        return None

    returns_df = pd.read_parquet(returns_path)
    portfolio_returns = returns_df["return"]

    weights = pd.read_json(RESULTS_DIR / "backtest_weights.json", typ="series")

    from python.portfolio.risk import RiskEngine

    returns_matrix = pd.DataFrame({"portfolio": portfolio_returns})
    engine = RiskEngine(returns_matrix, pd.Series({"portfolio": 1.0}))
    risk_summary = engine.summary()
    rolling_sharpe = engine.rolling_sharpe(window=60)

    # Load turnover if available
    turnover = None
    turnover_path = RESULTS_DIR / "backtest_turnover.parquet"
    if turnover_path.exists():
        turnover = pd.read_parquet(turnover_path)["turnover"]

    # Load metrics if available
    metrics = None
    metrics_path = RESULTS_DIR / "backtest_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return portfolio_returns, weights, risk_summary, rolling_sharpe, turnover, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    result = _load_backtest_results()
    if result is not None:
        returns, weights, risk, rolling, turnover, metrics = result
        app = create_dashboard(returns, weights, risk, rolling, turnover, metrics)
    else:
        # Fallback to synthetic demo
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
        weights = pd.Series(
            {"AAPL": 0.3, "MSFT": 0.25, "GOOG": 0.2, "AMZN": 0.15, "META": 0.1}
        )
        risk = {
            "sharpe": 1.2, "max_drawdown": -0.15, "annualized_return": 0.12,
            "var_95_historical": -0.018, "cvar_95": -0.025,
        }
        rolling = returns.rolling(60).mean() / returns.rolling(60).std() * np.sqrt(252)
        app = create_dashboard(returns, weights, risk, rolling)

    app.run(debug=True, port=8050)
