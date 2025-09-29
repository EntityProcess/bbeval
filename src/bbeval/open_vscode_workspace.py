#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Opens VS Code with the configured workspace.

Resolves the workspace path from:
1) --workspace-path command-line argument (highest precedence)
2) EVAL_CARGOWISE_WORKSPACE_PATH environment variable
3) .env file in the same folder as this script

Then launches `code <workspace>` with fallbacks for `code.cmd` and an optional
`CODE_CLI_PATH` environment variable. Emits clear errors if the CLI or workspace are missing.

On Windows, the --focus flag attempts to bring the newly opened VS Code window to the foreground.
Emits a final machine-parseable status line: OPEN_VSCODE_RESULT=<json>
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Check if we're on Windows
_IS_WINDOWS = sys.platform == "win32"

# Try to import Windows-specific modules. They are only needed for the --focus feature.
_HAS_WIN32_MODULES = False
if _IS_WINDOWS:
    try:
        import win32con
        import win32gui
        _HAS_WIN32_MODULES = True
    except ImportError:
        pass  # Modules not available, focus feature will be disabled


def parse_dotenv(file_path: Path) -> dict[str, str]:
    """Parses a simple .env file and returns a dictionary."""
    if not file_path.is_file():
        return {}
    env_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                env_map[key] = value
    return env_map


def resolve_workspace_path(override_path: str | None) -> Path:
    """Resolves the .code-workspace path with a specific precedence."""
    if override_path:
        return Path(override_path).resolve()

    if "EVAL_CARGOWISE_WORKSPACE_PATH" in os.environ:
        return Path(os.environ["EVAL_CARGOWISE_WORKSPACE_PATH"]).resolve()

    script_dir = Path(__file__).parent
    dotenv_path = script_dir / '.env'
    env_map = parse_dotenv(dotenv_path)
    if "EVAL_CARGOWISE_WORKSPACE_PATH" in env_map:
        return Path(env_map["EVAL_CARGOWISE_WORKSPACE_PATH"]).resolve()

    raise FileNotFoundError(
        "EVAL_CARGOWISE_WORKSPACE_PATH not set. Provide --workspace-path or set env variable or add it to .env."
    )


def get_code_cli() -> str:
    """Finds the VS Code CLI executable."""
    if "CODE_CLI_PATH" in os.environ:
        return os.environ["CODE_CLI_PATH"]
    
    # shutil.which checks the system's PATH
    if cli_path := shutil.which('code'):
        return cli_path
    if _IS_WINDOWS and (cli_path := shutil.which('code.cmd')):
        return cli_path

    raise FileNotFoundError("VS Code CLI not found. Ensure 'code' is on PATH or set CODE_CLI_PATH.")


def focus_vscode_window(workspace_path: Path) -> bool:
    """
    On Windows, polls for and attempts to bring the VS Code window to the foreground.
    This is a best-effort operation.
    """
    if not _HAS_WIN32_MODULES:
        return False
        
    title_key = workspace_path.stem  # Filename without extension
    deadline = time.monotonic() + 10  # 10-second timeout
    
    hwnd = [None] # Use a list to allow modification inside the callback
    
    def enum_callback(handle, _):
        if win32gui.IsWindowVisible(handle) and win32gui.GetWindowText(handle):
            if title_key in win32gui.GetWindowText(handle):
                # Check process name to be more specific
                proc_name = "Code.exe"
                try:
                    # VS Code stable and insiders process names
                    pids = win32gui.GetWindowThreadProcessId(handle)
                    # We are only interested in process ID which is on second index
                    # This API is inconsistent so we are handling both tuple and int
                    pid = pids
                    if type(pids) == tuple and len(pids) > 1:
                        pid = pids[1]
                    import psutil # lazy import to avoid dependency if not on windows with focus
                    proc_name = psutil.Process(pid).name()
                except (ImportError, Exception):
                    pass # psutil not installed or process exited, fallback to title check

                if proc_name.lower() in ('code.exe', 'code - insiders.exe'):
                     hwnd[0] = handle

    while time.monotonic() < deadline:
        win32gui.EnumWindows(enum_callback, None)
        if hwnd[0]:
            try:
                # Restore if minimized and bring to front
                win32gui.ShowWindow(hwnd[0], win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd[0])
                print(f"Brought VS Code window to foreground: *{title_key}*", file=sys.stderr)
                return True
            except Exception:
                # Fallback via WScript.Shell if available
                try:
                    import win32com.client
                    shell = win32com.client.Dispatch("WScript.Shell")
                    shell.AppActivate(win32gui.GetWindowText(hwnd[0]))
                    print(f"Brought VS Code window to foreground (fallback): *{title_key}*", file=sys.stderr)
                    return True
                except Exception:
                    pass # Both methods failed
        time.sleep(0.2)
        
    print(f"Could not locate VS Code window matching title: *{title_key}* within timeout", file=sys.stderr)
    return False


def open_and_focus_workspace(workspace_path: str, focus: bool = False, verbose: bool = True) -> bool:
    """
    Opens a VS Code workspace and optionally focuses it.
    
    Args:
        workspace_path: Path to the .code-workspace file
        focus: Whether to attempt focusing the window
        verbose: Whether to print status messages
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ws_path = Path(workspace_path).resolve()
        if not ws_path.is_file():
            raise FileNotFoundError(f"Workspace not found: {ws_path}")

        code_cli = get_code_cli()

        # Use subprocess.Popen for a non-blocking call, similar to Process::Start
        subprocess.Popen([code_cli, str(ws_path)], shell=_IS_WINDOWS)

        focused = False
        if focus:
            if _IS_WINDOWS:
                if _HAS_WIN32_MODULES:
                    # Give VS Code a moment to start before trying to focus
                    time.sleep(1.0)
                    focused = focus_vscode_window(ws_path)
                else:
                    if verbose:
                        print("  Focus requested but win32 modules not available; skipping focus.", file=sys.stderr)
            else:
                if verbose:
                    print("  Focus requested but OS is not Windows; skipping focus.", file=sys.stderr)
        
        return True

    except (FileNotFoundError, Exception) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return False


def main():
    """Main script execution."""
    parser = argparse.ArgumentParser(description="Opens a VS Code workspace.")
    parser.add_argument(
        "-w", "--workspace-path",
        help="Explicit path to a .code-workspace file."
    )
    parser.add_argument(
        "-f", "--focus",
        action="store_true",
        help="On Windows, attempt to bring the VS Code window to the foreground."
    )
    args = parser.parse_args()

    try:
        ws_path = resolve_workspace_path(args.workspace_path)
        if not ws_path.is_file():
            raise FileNotFoundError(f"Workspace not found: {ws_path}")

        code_cli = get_code_cli()

        print(f"\033[96mOpening VS Code workspace:\033[0m")
        print(f"\033[96m  {ws_path}\033[0m")
        print(f"\033[36mUsing CLI: {code_cli}\033[0m")

        # Use subprocess.Popen for a non-blocking call, similar to Process::Start
        subprocess.Popen([code_cli, str(ws_path)], shell=_IS_WINDOWS)

        focused = False
        if args.focus:
            if _IS_WINDOWS:
                if _HAS_WIN32_MODULES:
                    focused = focus_vscode_window(ws_path)
                else:
                    print("  Focus requested but win32 modules not available; skipping focus.", file=sys.stderr)
            else:
                print("  Focus requested but OS is not Windows; skipping focus.", file=sys.stderr)
        
        # Emit machine-parseable summary
        result = {
            "launched": True,
            "focusRequested": args.focus,
            "focused": focused,
            "workspace": str(ws_path),
            "cli": code_cli,
        }
        json_output = json.dumps(result, separators=(',', ':'))
        print(f"OPEN_VSCODE_RESULT={json_output}")

    except (FileNotFoundError, Exception) as e:
        print(f"\033[91mERROR: {e}\033[0m", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()