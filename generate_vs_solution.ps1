param(
    [string]$BuildDir = "build-vs",
    [string]$Generator = "Visual Studio 17 2022",
    [string]$Platform = "x64",
    [switch]$Open
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildPath = Join-Path $RepoRoot $BuildDir

Write-Host "Repository root: $RepoRoot"
Write-Host "Build directory: $BuildPath"
Write-Host "Generator: $Generator"
Write-Host "Platform: $Platform"

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "cmake is not available in PATH."
}

New-Item -ItemType Directory -Force -Path $BuildPath | Out-Null

cmake -S $RepoRoot -B $BuildPath -G $Generator -A $Platform

$Solution = Get-ChildItem -Path $BuildPath -Filter *.sln | Select-Object -First 1
if ($null -eq $Solution) {
    throw "CMake completed, but no .sln file was generated."
}

Write-Host ""
Write-Host "Generated solution:" $Solution.FullName
Write-Host "Open it with Visual Studio, or run:"
Write-Host "  devenv `"$($Solution.FullName)`""

if ($Open) {
    $Devenv = Get-Command devenv -ErrorAction SilentlyContinue
    if ($null -eq $Devenv) {
        throw "Visual Studio devenv was not found in PATH."
    }

    Write-Host ""
    Write-Host "Opening solution in Visual Studio..."
    & $Devenv.Source $Solution.FullName
}
