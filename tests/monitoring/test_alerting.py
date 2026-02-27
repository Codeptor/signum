"""Tests for python.monitoring.alerting — multi-channel alert delivery.

Covers:
  - Email sending (SMTP) with mocked smtplib
  - Webhook delivery with mocked urllib
  - Rate limiting behavior
  - Heartbeat cooldown
  - send_trade_summary formatting
  - get_alerting_status introspection
  - Severity routing
  - Fire-and-forget guarantee (errors swallowed)
"""

from unittest.mock import MagicMock, patch  # noqa: I001

import pytest


# ---------------------------------------------------------------------------
# Fixtures — isolate module-level state between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_alerting_state():
    """Reset module-level rate limit and heartbeat state between tests."""
    import python.monitoring.alerting as mod

    mod._alert_timestamps.clear()
    mod._last_heartbeat_ts = float("-inf")
    yield


@pytest.fixture()
def _email_configured(monkeypatch):
    """Configure email env vars for tests."""
    monkeypatch.setattr("python.monitoring.alerting.SMTP_HOST", "smtp.test.com")
    monkeypatch.setattr("python.monitoring.alerting.SMTP_PORT", 587)
    monkeypatch.setattr("python.monitoring.alerting.SMTP_USER", "bot@test.com")
    monkeypatch.setattr("python.monitoring.alerting.SMTP_PASSWORD", "secret")
    monkeypatch.setattr("python.monitoring.alerting.SMTP_FROM", "bot@test.com")
    monkeypatch.setattr("python.monitoring.alerting.ALERT_EMAIL_TO", "user@test.com")
    monkeypatch.setattr("python.monitoring.alerting.SMTP_USE_TLS", True)


@pytest.fixture()
def _webhook_configured(monkeypatch):
    """Configure webhook URL for tests."""
    monkeypatch.setattr(
        "python.monitoring.alerting.ALERT_WEBHOOK_URL",
        "https://hooks.example.com/test",
    )


@pytest.fixture()
def _nothing_configured(monkeypatch):
    """Ensure no alerting channels are configured."""
    monkeypatch.setattr("python.monitoring.alerting.SMTP_HOST", "")
    monkeypatch.setattr("python.monitoring.alerting.SMTP_USER", "")
    monkeypatch.setattr("python.monitoring.alerting.SMTP_PASSWORD", "")
    monkeypatch.setattr("python.monitoring.alerting.ALERT_EMAIL_TO", "")
    monkeypatch.setattr("python.monitoring.alerting.ALERT_WEBHOOK_URL", "")


# ===========================================================================
# send_alert — core dispatch
# ===========================================================================


class TestSendAlert:
    def test_no_channels_configured_is_noop(self, _nothing_configured):
        """send_alert does nothing when no channels are configured."""
        from python.monitoring.alerting import AlertSeverity, send_alert

        # Should not raise
        send_alert("test message", AlertSeverity.CRITICAL)

    def test_webhook_called_when_configured(self, _webhook_configured):
        """send_alert POSTs to webhook URL."""
        from python.monitoring.alerting import send_alert

        with patch("python.monitoring.alerting.urllib.request.urlopen") as mock_urlopen:
            send_alert("hello from test")
            mock_urlopen.assert_called_once()

    def test_webhook_failure_swallowed(self, _webhook_configured):
        """Webhook network errors must never propagate."""
        from python.monitoring.alerting import send_alert

        with patch(
            "python.monitoring.alerting.urllib.request.urlopen",
            side_effect=Exception("network down"),
        ):
            # Must not raise
            send_alert("test alert")

    def test_email_sent_in_background(self, _email_configured):
        """Email is sent via background thread when SMTP is configured."""
        from python.monitoring.alerting import AlertSeverity, send_alert

        with patch("python.monitoring.alerting.smtplib.SMTP") as mock_smtp_cls:
            mock_server = MagicMock()
            mock_smtp_cls.return_value = mock_server

            send_alert("test email alert", AlertSeverity.INFO)

            # Wait for background thread to complete
            import threading

            for t in threading.enumerate():
                if t.daemon and t.name != "MainThread":
                    t.join(timeout=5)

            mock_smtp_cls.assert_called_once_with("smtp.test.com", 587, timeout=15)
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("bot@test.com", "secret")
            mock_server.sendmail.assert_called_once()
            # Verify recipient
            call_args = mock_server.sendmail.call_args
            assert call_args[0][0] == "bot@test.com"  # from
            assert call_args[0][1] == ["user@test.com"]  # to

    def test_email_failure_swallowed(self, _email_configured):
        """SMTP errors must never propagate."""
        from python.monitoring.alerting import send_alert

        with patch(
            "python.monitoring.alerting.smtplib.SMTP",
            side_effect=Exception("SMTP refused"),
        ):
            send_alert("test alert")
            # Wait for thread
            import threading

            for t in threading.enumerate():
                if t.daemon and t.name != "MainThread":
                    t.join(timeout=5)
            # No exception = pass

    def test_severity_prefix_in_subject(self, _email_configured):
        """Email subject includes severity prefix."""
        from python.monitoring.alerting import AlertSeverity, send_alert

        with patch("python.monitoring.alerting.smtplib.SMTP") as mock_smtp_cls:
            mock_server = MagicMock()
            mock_smtp_cls.return_value = mock_server

            send_alert("Bad thing happened", AlertSeverity.CRITICAL)

            import threading

            for t in threading.enumerate():
                if t.daemon and t.name != "MainThread":
                    t.join(timeout=5)

            # Check the email message contains CRITICAL prefix
            call_args = mock_server.sendmail.call_args
            msg_str = call_args[0][2]  # third arg is the message string
            assert "[Signum CRITICAL]" in msg_str

    def test_custom_subject(self, _email_configured):
        """Custom subject is used when provided."""
        from python.monitoring.alerting import AlertSeverity, send_alert

        with patch("python.monitoring.alerting.smtplib.SMTP") as mock_smtp_cls:
            mock_server = MagicMock()
            mock_smtp_cls.return_value = mock_server

            send_alert("details here", AlertSeverity.INFO, subject="Custom Title")

            import threading

            for t in threading.enumerate():
                if t.daemon and t.name != "MainThread":
                    t.join(timeout=5)

            call_args = mock_server.sendmail.call_args
            msg_str = call_args[0][2]
            assert "Custom Title" in msg_str

    def test_multiple_recipients(self, monkeypatch):
        """Comma-separated ALERT_EMAIL_TO sends to all recipients."""
        monkeypatch.setattr("python.monitoring.alerting.SMTP_HOST", "smtp.test.com")
        monkeypatch.setattr("python.monitoring.alerting.SMTP_PORT", 587)
        monkeypatch.setattr("python.monitoring.alerting.SMTP_USER", "bot@test.com")
        monkeypatch.setattr("python.monitoring.alerting.SMTP_PASSWORD", "secret")
        monkeypatch.setattr("python.monitoring.alerting.SMTP_FROM", "")
        monkeypatch.setattr(
            "python.monitoring.alerting.ALERT_EMAIL_TO",
            "a@test.com, b@test.com, c@test.com",
        )

        from python.monitoring.alerting import send_alert

        with patch("python.monitoring.alerting.smtplib.SMTP") as mock_smtp_cls:
            mock_server = MagicMock()
            mock_smtp_cls.return_value = mock_server

            send_alert("multi-recipient test")

            import threading

            for t in threading.enumerate():
                if t.daemon and t.name != "MainThread":
                    t.join(timeout=5)

            call_args = mock_server.sendmail.call_args
            recipients = call_args[0][1]
            assert recipients == ["a@test.com", "b@test.com", "c@test.com"]


# ===========================================================================
# Rate limiting
# ===========================================================================


class TestRateLimiting:
    def test_rate_limit_blocks_after_threshold(self, _webhook_configured):
        """Alerts are dropped after exceeding rate limit."""
        import python.monitoring.alerting as mod
        from python.monitoring.alerting import AlertSeverity, send_alert

        with patch("python.monitoring.alerting.urllib.request.urlopen") as mock_urlopen:
            # Send exactly at the limit
            for i in range(mod._RATE_LIMIT_MAX_ALERTS):
                send_alert(f"msg {i}", AlertSeverity.WARNING)

            assert mock_urlopen.call_count == mod._RATE_LIMIT_MAX_ALERTS

            # Next one should be rate-limited
            send_alert("one more", AlertSeverity.WARNING)
            assert mock_urlopen.call_count == mod._RATE_LIMIT_MAX_ALERTS  # no increment

    def test_critical_bypasses_rate_limit(self, _webhook_configured):
        """CRITICAL severity always bypasses rate limiting."""
        import python.monitoring.alerting as mod
        from python.monitoring.alerting import AlertSeverity, send_alert

        with patch("python.monitoring.alerting.urllib.request.urlopen") as mock_urlopen:
            # Fill up the rate limit
            for i in range(mod._RATE_LIMIT_MAX_ALERTS):
                send_alert(f"msg {i}", AlertSeverity.WARNING)

            # CRITICAL should still get through
            send_alert("CRITICAL bypasses", AlertSeverity.CRITICAL)
            assert mock_urlopen.call_count == mod._RATE_LIMIT_MAX_ALERTS + 1


# ===========================================================================
# Heartbeat
# ===========================================================================


class TestHeartbeat:
    def test_heartbeat_sends_first_time(self, _webhook_configured):
        """First heartbeat always sends."""
        from python.monitoring.alerting import send_heartbeat

        with patch("python.monitoring.alerting._send_webhook") as mock_wh:
            send_heartbeat("all is well")
            assert mock_wh.call_count == 1

    def test_heartbeat_cooldown(self, _webhook_configured):
        """Second heartbeat within cooldown is suppressed."""
        from python.monitoring.alerting import send_heartbeat

        with patch("python.monitoring.alerting._send_webhook") as mock_wh:
            send_heartbeat("first")
            send_heartbeat("second — should be suppressed")
            assert mock_wh.call_count == 1

    def test_heartbeat_force_bypasses_cooldown(self, _webhook_configured):
        """force=True bypasses heartbeat cooldown."""
        from python.monitoring.alerting import send_heartbeat

        with patch("python.monitoring.alerting._send_webhook") as mock_wh:
            send_heartbeat("first")
            send_heartbeat("forced", force=True)
            assert mock_wh.call_count == 2


# ===========================================================================
# Trade summary
# ===========================================================================


class TestTradeSummary:
    def test_trade_summary_format(self, _webhook_configured):
        """send_trade_summary formats correctly and sends."""
        from python.monitoring.alerting import send_trade_summary

        with patch("python.monitoring.alerting.urllib.request.urlopen") as mock_urlopen:
            send_trade_summary(
                filled=8,
                partial=1,
                failed=1,
                total=10,
                positions={"AAPL": 0.15, "MSFT": 0.12, "GOOG": 0.10},
                equity=100_000.0,
            )
            assert mock_urlopen.call_count == 1
            # Verify the payload contains the summary
            call_args = mock_urlopen.call_args
            request_obj = call_args[0][0]
            import json

            payload = json.loads(request_obj.data.decode("utf-8"))
            text = payload["text"]
            assert "8 filled" in text
            assert "$100,000.00" in text
            assert "AAPL" in text

    def test_trade_summary_warning_on_failures(self, _webhook_configured):
        """Trade summary uses WARNING severity when there are failures."""
        from python.monitoring.alerting import send_trade_summary

        with patch("python.monitoring.alerting._send_webhook") as mock_wh:
            with patch("python.monitoring.alerting._send_email"):
                send_trade_summary(filled=5, partial=0, failed=2, total=7)
                # Should have been called (WARNING severity)
                mock_wh.assert_called_once()


# ===========================================================================
# get_alerting_status
# ===========================================================================


class TestAlertingStatus:
    def test_status_when_configured(self, _email_configured, _webhook_configured):
        """Status reports both channels as configured."""
        from python.monitoring.alerting import get_alerting_status

        status = get_alerting_status()
        assert status["email_configured"] is True
        assert status["email_transport"] == "smtp"
        assert status["webhook_configured"] is True
        assert status["smtp_host"] == "smtp.test.com"
        assert status["recipients"] == ["user@test.com"]

    def test_status_when_unconfigured(self, _nothing_configured):
        """Status reports both channels as unconfigured."""
        from python.monitoring.alerting import get_alerting_status

        status = get_alerting_status()
        assert status["email_configured"] is False
        assert status["email_transport"] == "none"
        assert status["webhook_configured"] is False
        assert status["recipients"] == []

    def test_status_sendgrid_transport(self, monkeypatch, _webhook_configured):
        """Status reports SendGrid when configured."""
        monkeypatch.setattr("python.monitoring.alerting.SENDGRID_API_KEY", "SG.test")
        monkeypatch.setattr("python.monitoring.alerting.SENDGRID_FROM_EMAIL", "bot@test.com")
        monkeypatch.setattr("python.monitoring.alerting.ALERT_EMAIL_TO", "user@test.com")
        from python.monitoring.alerting import get_alerting_status

        status = get_alerting_status()
        assert status["email_configured"] is True
        assert status["email_transport"] == "sendgrid"
        assert status["sendgrid_configured"] is True


# ===========================================================================
# _send_alert wrapper in live_bot.py
# ===========================================================================


class TestLiveBotSendAlert:
    """Verify the live_bot._send_alert wrapper delegates correctly."""

    def test_send_alert_delegates_to_alerting_module(self):
        """_send_alert in live_bot.py calls the centralized send_alert."""
        with patch("examples.live_bot.send_alert") as mock_send:
            from examples.live_bot import _send_alert

            _send_alert("test message")
            mock_send.assert_called_once()

    def test_send_alert_passes_severity(self):
        """_send_alert forwards severity to the alerting module."""
        with patch("examples.live_bot.send_alert") as mock_send:
            from examples.live_bot import _send_alert
            from python.monitoring.alerting import AlertSeverity

            _send_alert("critical thing", AlertSeverity.CRITICAL)
            mock_send.assert_called_once_with("critical thing", severity=AlertSeverity.CRITICAL)


# ===========================================================================
# HTML email formatting
# ===========================================================================


class TestEmailFormatting:
    def test_html_contains_severity_and_message(self):
        """HTML email body includes severity label and message text."""
        from python.monitoring.alerting import AlertSeverity, _format_email_html

        html = _format_email_html("Test alert message", AlertSeverity.WARNING)
        assert "warning" in html.lower()
        assert "Test alert message" in html

    def test_html_critical_uses_red_color(self):
        """CRITICAL alerts use red color in HTML."""
        from python.monitoring.alerting import AlertSeverity, _format_email_html

        html = _format_email_html("Oh no", AlertSeverity.CRITICAL)
        assert "#cc3333" in html


# ===========================================================================
# SendGrid transport
# ===========================================================================


class TestSendGridTransport:
    def test_sendgrid_called_when_configured(self, monkeypatch):
        """SendGrid HTTP API is used when SENDGRID_API_KEY is set."""
        monkeypatch.setattr("python.monitoring.alerting.SENDGRID_API_KEY", "SG.testkey")
        monkeypatch.setattr("python.monitoring.alerting.SENDGRID_FROM_EMAIL", "bot@test.com")
        monkeypatch.setattr("python.monitoring.alerting.ALERT_EMAIL_TO", "user@test.com")
        from python.monitoring.alerting import AlertSeverity, send_alert

        with patch("python.monitoring.alerting.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock(status=202)
            send_alert("test via sendgrid", AlertSeverity.INFO)

            import threading

            for t in threading.enumerate():
                if t.daemon and t.name != "MainThread":
                    t.join(timeout=5)

            # Should have been called (for SendGrid API)
            assert mock_urlopen.call_count >= 1
            # Verify it hit the SendGrid endpoint
            call_args = mock_urlopen.call_args_list
            sendgrid_calls = [
                c
                for c in call_args
                if hasattr(c[0][0], "full_url") and "sendgrid" in c[0][0].full_url
            ]
            assert len(sendgrid_calls) >= 1

    def test_sendgrid_preferred_over_smtp(self, monkeypatch):
        """When both SendGrid and SMTP are configured, SendGrid is used."""
        monkeypatch.setattr("python.monitoring.alerting.SENDGRID_API_KEY", "SG.testkey")
        monkeypatch.setattr("python.monitoring.alerting.SENDGRID_FROM_EMAIL", "bot@sg.com")
        monkeypatch.setattr("python.monitoring.alerting.SMTP_HOST", "smtp.test.com")
        monkeypatch.setattr("python.monitoring.alerting.SMTP_USER", "user@test.com")
        monkeypatch.setattr("python.monitoring.alerting.SMTP_PASSWORD", "secret")
        monkeypatch.setattr("python.monitoring.alerting.ALERT_EMAIL_TO", "user@test.com")
        from python.monitoring.alerting import _send_email

        with patch("python.monitoring.alerting._send_email_sendgrid") as mock_sg:
            with patch("python.monitoring.alerting._send_email_smtp") as mock_smtp:
                _send_email("test", "body")
                mock_sg.assert_called_once()
                mock_smtp.assert_not_called()

    def test_sendgrid_failure_swallowed(self, monkeypatch):
        """SendGrid API errors must never propagate."""
        monkeypatch.setattr("python.monitoring.alerting.SENDGRID_API_KEY", "SG.testkey")
        monkeypatch.setattr("python.monitoring.alerting.SENDGRID_FROM_EMAIL", "bot@sg.com")
        monkeypatch.setattr("python.monitoring.alerting.ALERT_EMAIL_TO", "user@test.com")
        from python.monitoring.alerting import _send_email_sendgrid

        with patch(
            "python.monitoring.alerting.urllib.request.urlopen",
            side_effect=Exception("SendGrid down"),
        ):
            # Must not raise
            _send_email_sendgrid("test", "body")
