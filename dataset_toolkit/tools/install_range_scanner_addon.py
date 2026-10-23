import argparse
import importlib
import subprocess
import sys
from pathlib import Path

import bpy


TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADDON_ZIP = TOOLKIT_ROOT / "extensions" / "blainder-range-scanner" / "range_scanner.zip"
DEFAULT_MODULE = "range_scanner"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install and enable the bundled range_scanner add-on.")
    parser.add_argument("--addon-zip", type=Path, default=DEFAULT_ADDON_ZIP,
                        help="Bundled range-scanner add-on ZIP file.")
    parser.add_argument("--module", default=DEFAULT_MODULE,
                        help="Blender add-on module name inside the ZIP.")
    parser.add_argument("--install-pyyaml", action="store_true",
                        help="Install the PyYAML dependency into Blender's bundled Python.")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(argv)


def ensure_pyyaml(*, install: bool) -> None:
    try:
        importlib.import_module("yaml")
        return
    except ModuleNotFoundError:
        if not install:
            raise RuntimeError(
                "PyYAML is required by range_scanner. Re-run with --install-pyyaml "
                "(or install PyYAML into Blender's Python yourself)."
            ) from None

    subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "PyYAML==6.0.2",
    ], check=True)
    importlib.import_module("yaml")


def install_addon(addon_zip: Path, module: str) -> None:
    if not addon_zip.is_file():
        raise FileNotFoundError(addon_zip)
    bpy.ops.preferences.addon_install(filepath=str(addon_zip), overwrite=True)
    bpy.ops.preferences.addon_enable(module=module)

    import addon_utils
    loaded, enabled = addon_utils.check(module)
    if not (loaded and enabled):
        raise RuntimeError(f"Blender did not enable add-on {module!r}")
    importlib.import_module(module)
    bpy.ops.wm.save_userpref()


def main() -> None:
    args = parse_args()
    ensure_pyyaml(install=args.install_pyyaml)
    install_addon(args.addon_zip.resolve(), args.module)
    print(f"Installed and enabled {args.module} from {args.addon_zip.resolve()}")


if __name__ == "__main__":
    main()
