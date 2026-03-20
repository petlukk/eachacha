"""CLI entry point: eachacha-grep."""

import sys


def main():
    """Entry point for eachacha-grep console script.
    Delegates to the standalone eachacha_grep.py in the project root.
    For pip-installed usage, import and use eachacha.search directly.
    """
    # Import the standalone grep script's main
    import os
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    grep_path = os.path.join(script_dir, "eachacha_grep.py")
    if os.path.exists(grep_path):
        # Running from source tree
        import importlib.util
        spec = importlib.util.spec_from_file_location("eachacha_grep", grep_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
    else:
        print("eachacha-grep: use 'python3 eachacha_grep.py' from the project directory,",
              file=sys.stderr)
        print("or use the Python API: from eachacha import search",
              file=sys.stderr)
        sys.exit(1)
