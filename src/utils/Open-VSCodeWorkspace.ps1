#!/usr/bin/env pwsh
<#
.SYNOPSIS
Opens VS Code with the configured workspace.

.DESCRIPTION
Resolves the workspace path from:
1) -WorkspacePath parameter (highest precedence)
2) EVAL_CARGOWISE_WORKSPACE_PATH environment variable
3) .env file in the same folder as this script

Then launches `code <workspace>` with fallbacks for `code.cmd` and an optional
`CODE_CLI_PATH` environment variable. Emits clear errors if the CLI or workspace are missing.

.NOTES
- Use `-Focus` on Windows to attempt to bring the newly opened VS Code window to the foreground.
 - Emits a final machine-parseable status line: `OPEN_VSCODE_RESULT=<json>` where JSON contains
     `{ "launched": bool, "focusRequested": bool, "focused": bool, "workspace": string, "cli": string }`.

.PARAMETER WorkspacePath
Explicit path to a `.code-workspace` file.

.PARAMETER Focus
On Windows, attempt to bring the newly opened VS Code window to the foreground.

.EXAMPLE
./Open-VSCodeWorkspace.ps1

.EXAMPLE
./Open-VSCodeWorkspace.ps1 -WorkspacePath "C:\\path\\to\\CargoWise.code-workspace"

.EXAMPLE
./Open-VSCodeWorkspace.ps1 -WorkspacePath "C:\\path\\to\\CargoWise.code-workspace" -Focus
#>
[CmdletBinding()] param(
    [Parameter()][string]$WorkspacePath,
    [switch]$Focus
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-EnvFromDotEnv {
    param([string]$FilePath)
    if (-not (Test-Path -LiteralPath $FilePath)) { return @{} }
    $map = @{}
    foreach ($line in Get-Content -LiteralPath $FilePath -Encoding UTF8) {
        if ($line -match '^[ \t]*#') { continue }
        if ($line -match '^[ \t]*$') { continue }
        if ($line -match '^(?<k>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?<v>.+)$') {
            $k = $Matches.k
            $v = $Matches.v.Trim()
            # Remove optional quotes
            if ($v.StartsWith('"') -and $v.EndsWith('"')) { $v = $v.Trim('"') }
            if ($v.StartsWith("'") -and $v.EndsWith("'")) { $v = $v.Trim("'") }
            $map[$k] = $v
        }
    }
    return $map
}

function Resolve-WorkspacePath {
    param([string]$Override)
    if ($Override) { return (Resolve-Path -LiteralPath $Override).ProviderPath }

    if ($env:EVAL_CARGOWISE_WORKSPACE_PATH) {
        return (Resolve-Path -LiteralPath $env:EVAL_CARGOWISE_WORKSPACE_PATH).ProviderPath
    }

    $scriptDir = Split-Path -LiteralPath $PSCommandPath -Parent
    $dotEnvPath = Join-Path $scriptDir '.env'
    $envMap = Get-EnvFromDotEnv -FilePath $dotEnvPath
    if ($envMap.ContainsKey('EVAL_CARGOWISE_WORKSPACE_PATH')) {
        return (Resolve-Path -LiteralPath $envMap['EVAL_CARGOWISE_WORKSPACE_PATH']).ProviderPath
    }

    throw "EVAL_CARGOWISE_WORKSPACE_PATH not set. Provide -WorkspacePath or set env variable or add it to .env."
}

function Get-CodeCli {
    if ($env:CODE_CLI_PATH) { return $env:CODE_CLI_PATH }
    if (Get-Command code -ErrorAction SilentlyContinue) { return 'code' }
    if (Get-Command code.cmd -ErrorAction SilentlyContinue) { return 'code.cmd' }
    throw "VS Code CLI not found. Ensure 'code' is on PATH or set CODE_CLI_PATH to the full path."
}

function Invoke-FocusVSCodeWindowIfRequested {
    param(
        [Parameter(Mandatory)][string]$WorkspacePath,
        [switch]$Focus
    )
    if (-not $Focus) { return $false }
    if (-not $IsWindows) {
        Write-Verbose 'Focus requested but OS is not Windows; skipping focus.'
        return $false
    }

    # Title key is typically the workspace name (without extension) in the VS Code window title.
    $titleKey = [IO.Path]::GetFileNameWithoutExtension($WorkspacePath)
    if (-not $titleKey) { return }

    try {
        Add-Type -ErrorAction SilentlyContinue @"
using System;
using System.Runtime.InteropServices;
public static class Win32 {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow);
}
"@
    } catch { }

    $deadline = [DateTime]::UtcNow.AddSeconds(10)
    $pattern = [regex]::Escape($titleKey)
    $broughtToFront = $false
    do {
        # VS Code stable and insiders process names
        $procs = @()
        foreach ($n in @('Code','Code - Insiders')) {
            try { $procs += Get-Process -Name $n -ErrorAction SilentlyContinue } catch { }
        }
        foreach ($p in $procs) {
            if (-not $p.MainWindowHandle -or $p.MainWindowHandle -eq 0) { continue }
            $title = $p.MainWindowTitle
            if (-not $title) { continue }
            if ($title -match $pattern) {
                try {
                    # Restore and bring to foreground
                    [void][Win32]::ShowWindowAsync($p.MainWindowHandle, 9) # SW_RESTORE
                    [void][Win32]::SetForegroundWindow($p.MainWindowHandle)
                    $broughtToFront = $true
                    break
                } catch {
                    # Fallback via WScript.Shell if available
                    try {
                        $wshell = New-Object -ComObject WScript.Shell
                        if ($wshell) { [void]$wshell.AppActivate($title) }
                        $broughtToFront = $true
                        break
                    } catch { }
                }
            }
        }
        if (-not $broughtToFront) { Start-Sleep -Milliseconds 200 }
    } while (-not $broughtToFront -and [DateTime]::UtcNow -lt $deadline)

    if ($broughtToFront) {
        Write-Verbose "Brought VS Code window to foreground: *$titleKey*"
    } else {
        Write-Verbose "Could not locate VS Code window matching title: *$titleKey* within timeout"
    }
    return $broughtToFront
}

function Main {
    $ws = Resolve-WorkspacePath -Override $WorkspacePath
    if (-not (Test-Path -LiteralPath $ws)) { throw "Workspace not found: $ws" }

    $code = Get-CodeCli
    Write-Host "Opening VS Code workspace:" -ForegroundColor Cyan
    Write-Host "  $ws" -ForegroundColor Cyan
    Write-Host "Using CLI: $code" -ForegroundColor DarkCyan

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $code
    $psi.ArgumentList.Add($ws) | Out-Null
    $psi.UseShellExecute = $true
    [void][System.Diagnostics.Process]::Start($psi)

    # Optionally bring the new window to foreground (best-effort on Windows)
    $focused = Invoke-FocusVSCodeWindowIfRequested -WorkspacePath $ws -Focus:$Focus

    # Emit machine-parseable summary for automation consumers
    $result = [ordered]@{
        launched       = $true
        focusRequested = [bool]$Focus
        focused        = [bool]$focused
        workspace      = $ws
        cli            = $code
    }
    $json = ($result | ConvertTo-Json -Compress)
    Write-Output "OPEN_VSCODE_RESULT=$json"
}

try {
    Main
} catch {
    Write-Error $_
    exit 1
}
