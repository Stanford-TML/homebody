"""Check the assets against their manifests, read-only."""
from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def safe_path(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or ".." in parts.parts or "\\" in relative:
        raise ValueError("unsafe_path")
    path = root
    for part in parts.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("symlink")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("escaped_path")
    return path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root: Path) -> dict:
    issues = []
    checked = set()
    hashed_models = set()
    manifests = [("assets/manifest.json", root, "path"),
                 ("assets/models/manifest.json", root / "assets/models", "file")]
    for name, base, key in manifests:
        try:
            manifest = json.loads(safe_path(root, name).read_text())
            if manifest["schema"] != 1 or not isinstance(manifest["files"], list):
                raise ValueError("unsupported_manifest")
            seen = set()
            for index, record in enumerate(manifest["files"]):
                label = f"{name}:record:{index}"
                try:
                    path = safe_path(base, record[key])
                    if path in seen:
                        raise ValueError("duplicate_manifest_entry")
                    seen.add(path)
                    expected_size = record["bytes"]
                    if type(expected_size) is not int or expected_size < 0:
                        raise ValueError("invalid_size")
                    if path.stat().st_size != expected_size:
                        raise ValueError("size_mismatch")
                    if name == "assets/models/manifest.json" and "sha256" not in record:
                        raise ValueError("model_sha256_missing")
                    if "sha256" in record:
                        expected_hash = record["sha256"]
                        if (not isinstance(expected_hash, str) or len(expected_hash) != 64 or
                                any(char not in "0123456789abcdef" for char in expected_hash)):
                            raise ValueError("invalid_sha256")
                        if digest(path) != expected_hash:
                            raise ValueError("hash_mismatch")
                        if name == "assets/models/manifest.json":
                            hashed_models.add(path.relative_to(root).as_posix())
                    checked.add(path.relative_to(root).as_posix())
                except (OSError, KeyError, TypeError, ValueError) as error:
                    issues.append({"location": label, "reason": type(error).__name__})
        except (OSError, KeyError, TypeError, ValueError) as error:
            issues.append({"location": name, "reason": type(error).__name__})
    for xml in (root / "assets").rglob("*.xml"):
        try:
            tree = ET.parse(safe_path(root, xml.relative_to(root).as_posix()))
            compiler = tree.find("compiler")
            for node in tree.iter():
                if "file" not in node.attrib:
                    continue
                prefix = ""
                if compiler is not None:
                    prefix = compiler.get({"mesh": "meshdir", "texture": "texturedir"}.get(
                        node.tag, ""), "")
                reference = xml.parent / prefix / node.attrib["file"]
                safe_path(root, reference.relative_to(root).as_posix()).stat()
        except (OSError, ValueError, ET.ParseError) as error:
            issues.append({"location": xml.relative_to(root).as_posix(),
                           "reason": f"xml_reference_{type(error).__name__}"})
    for index, path in enumerate((root / "assets").rglob("*")):
        if path.is_symlink():
            issues.append({"location": "assets", "reason": "symlink"})
        elif path.is_file() and path.name not in ("manifest.json", ".DS_Store"):
            relative = path.relative_to(root).as_posix()
            if relative not in checked:
                issues.append({"location": "assets", "reason": "unlisted_asset",
                               "file_id": f"asset:{index}"})
            if relative.startswith("assets/models/") and relative not in hashed_models:
                issues.append({"location": "assets/models", "reason": "model_sha256_not_verified",
                               "file_id": f"asset:{index}"})
    return {"ok": not issues, "listed_file_count": len(checked), "issues": issues,
            "note": "Integrity checks do not establish redistribution rights or physical accuracy."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = verify(args.root.resolve())
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
