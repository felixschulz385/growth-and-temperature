import os

from src.data.common import statusfile


def test_read_missing_returns_none(tmp_path):
    assert statusfile.read(str(tmp_path / "nope.json")) is None


def test_write_then_read_roundtrips_and_stamps_updated_at(tmp_path):
    path = str(tmp_path / "_status" / "unit.json")
    statusfile.write(path, {"status": "retrying", "attempts": 2})
    data = statusfile.read(path)
    assert data["status"] == "retrying"
    assert data["attempts"] == 2
    assert "updated_at" in data


def test_write_is_atomic_no_tmp_file_left_behind(tmp_path):
    path = str(tmp_path / "unit.json")
    statusfile.write(path, {"status": "ok"})
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")


def test_read_corrupt_file_returns_none_not_raise(tmp_path):
    path = str(tmp_path / "corrupt.json")
    os.makedirs(tmp_path, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("{not valid json")
    assert statusfile.read(path) is None


def test_remove_missing_file_is_a_noop(tmp_path):
    statusfile.remove(str(tmp_path / "nope.json"))  # must not raise


def test_status_path_sanitizes_unit_id():
    path = statusfile.status_path("/data/raw", "2020/h12v09")
    assert path == os.path.join("/data/raw", "_status", "2020_h12v09.json")


def test_list_status_filenames_empty_when_subdir_missing(tmp_path):
    assert statusfile.list_status_filenames(str(tmp_path)) == set()


def test_list_status_filenames_lists_what_was_written(tmp_path):
    statusfile.write(statusfile.status_path(str(tmp_path), "a"), {"status": "retrying"})
    statusfile.write(statusfile.status_path(str(tmp_path), "b/c"), {"status": "retrying"})
    assert statusfile.list_status_filenames(str(tmp_path)) == {"a.json", "b_c.json"}
