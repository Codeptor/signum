"""Risk monitoring dashboard with Dash + Plotly."""

import logging

import dash
from dash import dcc, html, Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def create_dashboard(
    portfolio_returns: pd.Series,
    weights: pd.Series,
    risk_summary: dict,
    rolling_sharpe: pd.Series,
) -> dash.Dash:
    """Create and return a Dash app for risk monitoring."""
    app = dash.Dash(__name__)

    cumulative = (1 + portfolio_returns).cumprod()
    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()

    app.layout = html.Div(
        [
            html.H1("Quant Platform — Risk Dashboard"),
            # KPI cards
            html.Div(
                [
                    _kpi_card("Sharpe Ratio", f"{risk_summary.get('sharpe', 0):.2f}"),
                    _kpi_card("Max Drawdown", f"{risk_summary.get('max_drawdown', 0):.2%}"),
                    _kpi_card("Ann. Return", f"{risk_summary.get('annualized_return', 0):.2%}"),
                    _kpi_card("VaR (95%)", f"{risk_summary.get('var_95_historical', 0):.4f}"),
                    _kpi_card("CVaR (95%)", f"{risk_summary.get('cvar_95', 0):.4f}"),
                ],
                style={"display": "flex", "gap": "20px", "marginBottom": "30px"},
            ),
            # Charts row 1
            html.Div(
                [
                    dcc.Graph(
                        figure=_cumulative_returns_chart(cumulative),
                        style={"flex": "1"},
                    ),
                    dcc.Graph(
                        figure=_weights_chart(weights),
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "20px"},
            ),
            # Charts row 2
            html.Div(
                [
                    dcc.Graph(
                        figure=_drawdown_chart(drawdown),
                        style={"flex": "1"},
                    ),
                    dcc.Graph(
                        figure=_rolling_sharpe_chart(rolling_sharpe),
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "20px"},
            ),
        ],
        style={"padding": "20px", "fontFamily": "system-ui"},
    )

    return app


def _kpi_card(label: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontSize": "14px", "color": "#666"}),
            html.Div(value, style={"fontSize": "24px", "fontWeight": "bold"}),
        ],
        style={
            "padding": "15px 25px",
            "border": "1px solid #e0e0e0",
            "borderRadius": "8px",
            "minWidth": "140px",
        },
    )


def _cumulative_returns_chart(cumulative: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative.values, mode="lines", name="Portfolio"))
    fig.update_layout(title="Cumulative Returns", xaxis_title="Date", yaxis_title="Growth of $1")
    return fig


def _weights_chart(weights: pd.Series) -> go.Figure:
    fig = go.Figure(
        data=[go.Treemap(labels=weights.index, parents=[""] * len(weights), values=weights.values)]
    )
    fig.update_layout(title="Portfolio Weights")
    return fig


def _drawdown_chart(drawdown: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill="tozeroy",
            mode="lines",
            name="Drawdown",
            line={"color": "red"},
        )
    )
    fig.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown %")
    return fig


def _rolling_sharpe_chart(rolling_sharpe: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, mode="lines", name="Sharpe"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="Rolling Sharpe Ratio (60d)", xaxis_title="Date", yaxis_title="Sharpe")
    return fig


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
    weights = pd.Series({"AAPL": 0.3, "MSFT": 0.25, "GOOG": 0.2, "AMZN": 0.15, "META": 0.1})
    risk = {
        "sharpe": 1.2,
        "max_drawdown": -0.15,
        "annualized_return": 0.12,
        "var_95_historical": -0.018,
        "cvar_95": -0.025,
    }
    rolling = returns.rolling(60).mean() / returns.rolling(60).std() * np.sqrt(252)

    app = create_dashboard(returns, weights, risk, rolling)
    app.run(debug=True, port=8050)
