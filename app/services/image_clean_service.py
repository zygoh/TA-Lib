"""
AI 生图元数据清洗（clean-image standard 模式同源：ffmpeg 双遍重编码 + exiftool 双遍清元数据）。
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_REQUIRED_BINARIES = ("ffmpeg", "exiftool")
_TEMP_PASS1 = "pass1.jpg"
_DEFAULT_QUALITY = 92
_MIN_QUALITY = 1
_MAX_QUALITY = 100
_ASPECT_TOLERANCE = 0.02
_TARGET_W = 1536
_TARGET_H = 1024


class ImageCleanError(Exception):
    """图片清洗失败。"""


def check_image_clean_deps() -> list[str]:
    missing: list[str] = []
    for name in _REQUIRED_BINARIES:
        if shutil.which(name) is None:
            missing.append(name)
    return missing


def _ffmpeg_quality(quality: int) -> int:
    return round((100 - quality) * 31 / 100 + 1)


def _run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise ImageCleanError(stderr or f"command failed: {' '.join(cmd)}") from exc


def _reencode(input_path: Path, output_path: Path, *, quality: int) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-q:v",
            str(_ffmpeg_quality(quality)),
            "-map_metadata",
            "-1",
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
            str(output_path),
        ]
    )


def _strip_metadata(path: Path) -> None:
    _run(["exiftool", "-all=", "-overwrite_original", "-quiet", str(path)])


def _metadata_field_count(path: Path) -> int:
    try:
        result = subprocess.run(
            ["exiftool", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return -1
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    return len(lines)


def _probe_dimensions(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except Exception as exc:
        raise ImageCleanError(f"cannot read output dimensions: {exc}") from exc


def _within_aspect_tolerance(width: int, height: int) -> bool:
    if width <= 0 or height <= 0:
        return False
    ratio = width / height
    target = _TARGET_W / _TARGET_H
    return abs(ratio - target) / target <= _ASPECT_TOLERANCE


def clean_image_file(
    input_path: Path,
    output_path: Path,
    *,
    mode: str = "standard",
    quality: int = _DEFAULT_QUALITY,
) -> dict:
    """
    清洗本地图片文件。

    mode:
      - standard: 4-pass（ffmpeg → exiftool → ffmpeg → exiftool），输出 JPEG
      - strip-only: 仅 exiftool，保留原格式
    """
    missing = check_image_clean_deps()
    if missing:
        raise ImageCleanError(f"missing dependencies: {', '.join(missing)}")

    mode = (mode or "standard").strip().lower()
    if mode not in {"standard", "strip-only"}:
        raise ImageCleanError(f"unsupported mode: {mode}")

    quality = max(_MIN_QUALITY, min(_MAX_QUALITY, int(quality)))
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.is_file():
        raise ImageCleanError(f"input not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_metadata_fields = _metadata_field_count(input_path)
    original_size = input_path.stat().st_size

    if mode == "strip-only":
        shutil.copy2(input_path, output_path)
        _strip_metadata(output_path)
    else:
        with tempfile.TemporaryDirectory(prefix="ta-lib-image-clean-") as tmp:
            tmp_dir = Path(tmp)
            pass1 = tmp_dir / _TEMP_PASS1
            _reencode(input_path, pass1, quality=quality)
            _strip_metadata(pass1)
            _reencode(pass1, output_path, quality=quality)
            _strip_metadata(output_path)

    if not output_path.is_file():
        raise ImageCleanError("clean produced no output file")

    width, height = _probe_dimensions(output_path)
    if not _within_aspect_tolerance(width, height):
        raise ImageCleanError(
            f"output aspect not 4:3 landscape: {width}x{height} (expected ~{_TARGET_W}x{_TARGET_H})"
        )

    cleaned_metadata_fields = _metadata_field_count(output_path)
    cleaned_size = output_path.stat().st_size
    content_type = "image/jpeg" if mode == "standard" else _guess_mime(output_path)

    report = {
        "mode": mode,
        "quality": quality if mode == "standard" else None,
        "width": width,
        "height": height,
        "content_type": content_type,
        "original_size": original_size,
        "cleaned_size": cleaned_size,
        "original_metadata_fields": original_metadata_fields,
        "cleaned_metadata_fields": cleaned_metadata_fields,
        "metadata_removed": max(0, original_metadata_fields - cleaned_metadata_fields)
        if original_metadata_fields >= 0 and cleaned_metadata_fields >= 0
        else None,
    }
    logger.info(
        "image clean ok mode=%s %sx%s meta %s→%s size %s→%s",
        mode,
        width,
        height,
        original_metadata_fields,
        cleaned_metadata_fields,
        original_size,
        cleaned_size,
    )
    return report


def clean_image_bytes(
    data: bytes,
    filename: str,
    *,
    mode: str = "standard",
    quality: int = _DEFAULT_QUALITY,
) -> tuple[bytes, str, dict]:
    """清洗内存中的图片，返回 (body, content_type, report)。"""
    if not data:
        raise ImageCleanError("empty image payload")

    suffix = Path(filename or "image.png").suffix or ".png"
    with tempfile.TemporaryDirectory(prefix="ta-lib-image-clean-") as tmp:
        tmp_dir = Path(tmp)
        inp = tmp_dir / f"input{suffix}"
        if mode == "standard":
            out = tmp_dir / "output.jpg"
        else:
            out = tmp_dir / f"output{suffix}"
        inp.write_bytes(data)
        report = clean_image_file(inp, out, mode=mode, quality=quality)
        return out.read_bytes(), report["content_type"], report


def _guess_mime(path: Path) -> str:
    import mimetypes

    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("image/"):
        return mime
    return "application/octet-stream"
