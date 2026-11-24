"""Tests for config/yaml_io.py — input loading, output writing, key conventions."""


import pytest
import yaml

from config.yaml_io import (
    KEYS,
    OUTPUT_DIR,
    load_input,
    load_output,
    save_output,
)


def test_input_config_loads_for_irfs7730pbf():
    cfg = load_input("irfs7730pbf")
    assert cfg["device"] == "irfs7730pbf"
    assert cfg["subcircuit"] == "irfb_s_sl7730pbf"
    cond = cfg["conditions"]
    # numeric types (YAML 1.1 would parse 1.0e6 as a string; must be float)
    assert isinstance(cond["F_meas"], float)
    assert cond["F_meas"] == pytest.approx(1e6)
    assert cond["Id_th_target"] == pytest.approx(250e-6)
    assert cond["Id_load"] == 100.0
    assert cond["Vbus"] == 38.0
    assert cond["temperature"] == 25.0


def test_load_input_missing_vendor_raises_helpfully():
    with pytest.raises(FileNotFoundError, match="No input config"):
        load_input("does_not_exist")


def test_save_and_roundtrip_output(tmp_path, monkeypatch):
    monkeypatch.setattr("config.yaml_io.OUTPUT_DIR", tmp_path / "outputs")
    p = save_output(
        "irfs7730pbf", "test_tb", "pass",
        conditions_used={"Vbus": 38.0},
        extracted={"vth_v": 4.3, "curve_id": [1.0, 2.0]},
        notes="unit test",
    )
    assert p.exists()
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert data["device"] == "irfs7730pbf"
    assert data["testbench"] == "test_tb"
    assert data["status"] == "pass"
    assert data["conditions_used"] == {"Vbus": 38.0}
    assert data["extracted"]["vth_v"] == 4.3
    assert data["notes"] == "unit test"
    # roundtrip via helper
    back = load_output("irfs7730pbf", "test_tb")
    assert back == data


def test_numpy_types_serialized(tmp_path, monkeypatch):
    import numpy as np
    monkeypatch.setattr("config.yaml_io.OUTPUT_DIR", tmp_path / "outputs")
    p = save_output(
        "irfs7730pbf", "np_tb", "pass",
        conditions_used={},
        extracted={"arr": np.array([1.5, 2.5]), "scalar": np.float64(3.5)},
    )
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert data["extracted"]["arr"] == [1.5, 2.5]
    assert data["extracted"]["scalar"] == 3.5


def test_invalid_status_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("config.yaml_io.OUTPUT_DIR", tmp_path / "outputs")
    with pytest.raises(ValueError, match="status must be"):
        save_output("irfs7730pbf", "x", "failed", {}, {})


def test_canonical_keys_present_for_all_written_outputs():
    """Every real output file in the tree must use canonical extracted keys."""
    known = set(KEYS) | {"notes_key"} | {
        f"curve_id_vgs_{v}_v" for v in range(0, 100)
    }
    # %g-formatted Vgs family keys (e.g. curve_id_vgs_4.5_v) are canonical:
    import re
    vgs_family_re = re.compile(r"^curve_id_vgs_[\d.]+_v$")
    for path in OUTPUT_DIR.glob("*/*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["status"] in ("pass", "partial", "blocked")
        for key in data.get("extracted", {}):
            if vgs_family_re.match(key):
                continue
            assert key in known, f"{path.name}: non-canonical key '{key}'"


def test_existing_outputs_cover_all_passing_curves():
    outputs = {p.stem for p in OUTPUT_DIR.glob("irfs7730pbf/*.yaml")}
    expected = {
        "curve1_output_characteristics",
        "curve2_transfer_characteristic",
        "curve3_rds_on",
        "curve4_capacitances",
        "curve5_gate_charge",
        "curve6_body_diode",
    }
    assert expected <= outputs
