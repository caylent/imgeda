"""Tests for Lambda handler."""

from __future__ import annotations

from imgeda.lambda_handler.handler import handler


class TestLambdaHandler:
    def test_stub_returns_501(self) -> None:
        result = handler({"bucket": "test", "keys": []}, None)
        assert result["statusCode"] == 501

    def test_stub_returns_body(self) -> None:
        result = handler({}, None)
        assert "body" in result
        assert "not yet implemented" in result["body"].lower()
