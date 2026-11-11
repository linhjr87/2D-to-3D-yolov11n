import shutil

import pytest
import yaml

from conftest import CONFIGS
from fpyolo.schema import load_schema


def test_v0_counts_and_indices():
    s = load_schema(CONFIGS / "classes_v0.yaml")
    assert [c.semantic_id for c in s.for_model("D")] == list(range(3, 33))
    assert [c.semantic_id for c in s.for_model("W")] == [1, 2, 33]  # wall, curtain wall, railing
    assert s.yolo_index("D")[3] == 0 and s.yolo_index("D")[32] == 29
    assert s.by_schema_id(19).name == "airconditioner" and s.by_schema_id(19).group == "appliance-hvac"
    assert s.by_schema_id(31).semantic_id == 1
    assert s.yolo_names("D")[0] == "single_door"
    assert s.yolo_names("W") == {0: "wall", 1: "curtain_wall", 2: "railing"}
    assert len(set(s.yolo_names("D").values())) == 30  # names confirmed in reports/class_names.md


def _write(path, doc):
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return path


def _base():
    return yaml.safe_load((CONFIGS / "classes_v0.yaml").read_text(encoding="utf-8"))


def test_rejects_bad_group_and_duplicates(spaced_tmp):
    bad_group = _base()
    bad_group["classes"][0]["group"] = "bogus"
    with pytest.raises(ValueError):
        load_schema(_write(spaced_tmp / "bad_group.yaml", bad_group))

    dup_sid = _base()
    dup_sid["classes"][1]["schema_id"] = dup_sid["classes"][0]["schema_id"]
    with pytest.raises(ValueError):
        load_schema(_write(spaced_tmp / "dup_sid.yaml", dup_sid))

    dup_sem = _base()
    dup_sem["classes"][1]["semantic_id"] = dup_sem["classes"][0]["semantic_id"]
    with pytest.raises(ValueError):
        load_schema(_write(spaced_tmp / "dup_sem.yaml", dup_sem))


def test_disabled_classes_excluded(spaced_tmp):
    doc = _base()
    for c in doc["classes"]:
        if c["name"] == "curtain_wall":
            c["enabled"] = False
    s = load_schema(_write(spaced_tmp / "disabled.yaml", doc))
    assert [c.semantic_id for c in s.for_model("W")] == [1, 33]
    assert s.yolo_index("W") == {1: 0, 33: 1}  # railing moves up; wall keeps index 0


def test_load_from_path_with_spaces(spaced_tmp):
    dst = spaced_tmp / "classes_v0.yaml"
    shutil.copy(CONFIGS / "classes_v0.yaml", dst)
    assert len(load_schema(dst).for_model("D")) == 30
