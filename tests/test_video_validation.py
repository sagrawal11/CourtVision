"""Tests for the upload container check in cv/analysis_job.py (audit FILE_UPLOADS #14).

assert_is_video must accept MP4/MOV containers and reject anything else before it
reaches the CV pipeline. Run with: pytest tests/test_video_validation.py
"""

import importlib.util
from pathlib import Path

import pytest

# Load cv/analysis_job.py as a standalone module rather than `from cv.analysis_job`
# — importing it as a package submodule would run cv/__init__.py, which eagerly
# imports the torch/ultralytics detectors (not needed here and absent in CI). Its
# top-level imports are stdlib only, so a direct file load is torch-free.
_spec = importlib.util.spec_from_file_location(
    "analysis_job_under_test",
    Path(__file__).resolve().parent.parent / "cv" / "analysis_job.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
assert_is_video = _mod.assert_is_video


def _write(tmp_path, data: bytes):
    p = tmp_path / "upload.bin"
    p.write_bytes(data)
    return str(p)


@pytest.mark.parametrize(
    "header",
    [
        b"\x00\x00\x00\x18ftypmp42",   # standard MP4
        b"\x00\x00\x00\x14ftypqt  ",   # QuickTime .mov
        b"\x00\x00\x00\x08moov____",   # movie box first
        b"\x00\x00\x00\x08mdat____",   # media data box first
    ],
)
def test_accepts_mp4_mov_containers(tmp_path, header):
    assert_is_video(_write(tmp_path, header + b"\x00" * 32))  # no raise


@pytest.mark.parametrize(
    "data",
    [
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0d",              # PNG image
        b"PK\x03\x04\x14\x00\x00\x00\x08\x00\x00\x00",     # ZIP archive
        b"<!DOCTYPE html><html><body>hi</body></html>",    # HTML
        b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00",        # ELF binary
    ],
)
def test_rejects_non_video(tmp_path, data):
    with pytest.raises(ValueError):
        assert_is_video(_write(tmp_path, data))


def test_rejects_truncated_file(tmp_path):
    with pytest.raises(ValueError):
        assert_is_video(_write(tmp_path, b"abc"))
