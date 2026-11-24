"""CLI workbench smoke tests (no interactive terminal required)."""

from typer.testing import CliRunner

from cli.main import app

_runner = CliRunner()


def test_models_catalog_lists_configured_vendors():
    result = _runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "Model catalog" in result.output
    assert "cpm2-1200" in result.output
    assert "SpiceModels files" in result.output


def test_doctor_checks_environment():
    result = _runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Environment" in result.output
    assert "configured vendors" in result.output


def test_ui_is_safe_without_console():
    result = _runner.invoke(app, ["ui"])
    assert result.exit_code == 0
    assert ("interactive input" in result.output.lower()
            or "needs a terminal" in result.output.lower()
            or "interactive mode" in result.output.lower())


def test_root_without_command_does_not_crash():
    result = _runner.invoke(app, [])
    assert result.exit_code == 0
