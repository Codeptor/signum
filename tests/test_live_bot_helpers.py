"""Tests for live_bot helpers: _verify_order_fill timeout path, _seconds_until.

These are critical untested helpers in the live trading path.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

from python.brokers.base import BrokerOrder


# ===========================================================================
# _verify_order_fill — timeout path (the only untested branch)
# ===========================================================================


class TestVerifyOrderFillTimeout:
    """The existing tests cover immediate-fill, delayed-fill, and rejected.
    This class covers the TIMEOUT path that was missing.
    """

    @patch("time.sleep", return_value=None)
    def test_timeout_when_order_stays_open(self, _sleep):
        """Order that never reaches terminal state returns 'timeout'."""
        from examples.live_bot import _verify_order_fill, ORDER_POLL_TIMEOUT_SECS

        # Create a mock broker where get_order always returns 'new'
        mock_broker = MagicMock()
        mock_order = MagicMock()
        mock_order.status = "new"
        mock_broker.get_order.return_value = mock_order

        result = _verify_order_fill(mock_broker, "order-stuck", "AAPL", 10.0)

        assert result["status"] == "timeout"
        assert result["filled_qty"] == 0
        assert result["filled_avg_price"] is None
        assert result["symbol"] == "AAPL"
        assert result["order_id"] == "order-stuck"

    @patch("time.sleep", return_value=None)
    def test_timeout_when_get_order_returns_none(self, _sleep):
        """If get_order keeps returning None, should still timeout gracefully."""
        from examples.live_bot import _verify_order_fill

        mock_broker = MagicMock()
        mock_broker.get_order.return_value = None

        result = _verify_order_fill(mock_broker, "order-ghost", "MSFT", 5.0)

        assert result["status"] == "timeout"
        assert result["filled_qty"] == 0

    @patch("time.sleep", return_value=None)
    def test_timeout_fills_just_before_deadline(self, _sleep):
        """Order that fills on the last poll before timeout should return 'filled'."""
        from examples.live_bot import (
            _verify_order_fill,
            ORDER_POLL_INTERVAL_SECS,
            ORDER_POLL_TIMEOUT_SECS,
        )

        # Calculate number of polls: timeout / interval
        n_polls = int(ORDER_POLL_TIMEOUT_SECS / ORDER_POLL_INTERVAL_SECS)

        mock_broker = MagicMock()
        call_count = 0

        def get_order_side_effect(oid):
            nonlocal call_count
            call_count += 1
            order = MagicMock()
            # Fill on the last poll
            if call_count >= n_polls:
                order.status = "filled"
                order.qty = 10.0
                order.filled_avg_price = 155.0
            else:
                order.status = "new"
            return order

        mock_broker.get_order.side_effect = get_order_side_effect

        result = _verify_order_fill(mock_broker, "order-slow", "AAPL", 10.0)
        assert result["status"] == "filled"
        assert result["filled_avg_price"] == 155.0

    @patch("time.sleep", return_value=None)
    def test_cancelled_is_terminal(self, _sleep):
        """Both 'canceled' and 'cancelled' spellings are handled."""
        from examples.live_bot import _verify_order_fill

        for spelling in ("canceled", "cancelled"):
            mock_broker = MagicMock()
            mock_order = MagicMock()
            mock_order.status = spelling
            mock_order.qty = 10.0
            mock_order.filled_avg_price = None
            mock_broker.get_order.return_value = mock_order

            result = _verify_order_fill(mock_broker, f"order-{spelling}", "AAPL", 10.0)
            assert result["status"] == spelling

    @patch("time.sleep", return_value=None)
    def test_expired_is_terminal(self, _sleep):
        from examples.live_bot import _verify_order_fill

        mock_broker = MagicMock()
        mock_order = MagicMock()
        mock_order.status = "expired"
        mock_order.qty = 10.0
        mock_order.filled_avg_price = None
        mock_broker.get_order.return_value = mock_order

        result = _verify_order_fill(mock_broker, "order-expired", "AAPL", 10.0)
        assert result["status"] == "expired"


# ===========================================================================
# _seconds_until
# ===========================================================================


class TestSecondsUntil:
    """Tests for the _seconds_until helper that handles timezone-aware/naive datetimes."""

    def test_tz_aware_datetime_in_future(self):
        from examples.live_bot import _seconds_until

        future = datetime.now(tz=timezone.utc) + timedelta(hours=2)
        secs = _seconds_until(future)
        # Should be close to 7200 seconds
        assert 7100 < secs < 7300

    def test_tz_naive_datetime_in_future(self):
        from examples.live_bot import _seconds_until

        # Naive datetime — should be treated as UTC
        future = datetime.utcnow() + timedelta(hours=1)
        secs = _seconds_until(future)
        assert 3500 < secs < 3700

    def test_past_datetime_returns_minimum(self):
        """Past timestamps should return at least 60 seconds (safety minimum)."""
        from examples.live_bot import _seconds_until

        past = datetime.now(tz=timezone.utc) - timedelta(hours=1)
        secs = _seconds_until(past)
        assert secs == 60.0  # Floor

    def test_iso_string_in_future(self):
        from examples.live_bot import _seconds_until

        future = datetime.now(tz=timezone.utc) + timedelta(hours=3)
        iso = future.isoformat()
        secs = _seconds_until(iso)
        assert 10700 < secs < 10900

    def test_iso_string_with_z_suffix(self):
        from examples.live_bot import _seconds_until

        future = datetime.now(tz=timezone.utc) + timedelta(hours=1)
        # Replace +00:00 with Z (common in API responses)
        iso = future.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        secs = _seconds_until(iso)
        assert 3500 < secs < 3700

    def test_invalid_string_returns_fallback(self):
        from examples.live_bot import _seconds_until

        secs = _seconds_until("not-a-date")
        assert secs == 3600.0  # 1 hour fallback

    def test_unknown_type_returns_fallback(self):
        from examples.live_bot import _seconds_until

        secs = _seconds_until(12345)
        assert secs == 3600.0  # 1 hour fallback

    def test_none_returns_fallback(self):
        from examples.live_bot import _seconds_until

        secs = _seconds_until(None)
        assert secs == 3600.0

    def test_minimum_60_seconds(self):
        """Even very near-future times should return at least 60 seconds."""
        from examples.live_bot import _seconds_until

        # 10 seconds from now → should be clamped to 60
        near = datetime.now(tz=timezone.utc) + timedelta(seconds=10)
        secs = _seconds_until(near)
        assert secs == 60.0


# ===========================================================================
# _send_alert — fire-and-forget
# ===========================================================================


class TestSendAlert:
    def test_no_webhook_url_does_nothing(self):
        """When ALERT_WEBHOOK_URL is not set, _send_alert is a no-op."""
        with patch("examples.live_bot.ALERT_WEBHOOK_URL", None):
            from examples.live_bot import _send_alert

            # Should not raise
            _send_alert("test alert")

    def test_webhook_failure_swallowed(self):
        """Network errors in _send_alert must never propagate."""
        with patch("examples.live_bot.ALERT_WEBHOOK_URL", "https://hooks.example.com/test"):
            with patch("urllib.request.urlopen", side_effect=Exception("network down")):
                from examples.live_bot import _send_alert

                # Should not raise
                _send_alert("test alert")
