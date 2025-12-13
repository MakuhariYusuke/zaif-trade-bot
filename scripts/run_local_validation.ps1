param(
    [string]$PythonVersions = '3.10,3.11,3.13',
    [string]$ImageTagPrefix = 'ztb-ci'
)

Write-Host "Running local validation using Docker. This requires Docker Desktop or daemon to be available."

$versions = $PythonVersions -split ',' | ForEach-Object { $_.Trim() }
foreach ($ver in $versions) {
    $tag = "$ImageTagPrefix:$ver"
    Write-Host "Building Docker image for Python $ver -> $tag"
    docker build --progress=plain --build-arg PYTHON_VERSION=$ver -t $tag -f docker/ci.Dockerfile .

    Write-Host "Running validation in container ($tag)"
    docker run --rm -v ${PWD}:/workspace -w /workspace $tag
}

Write-Host "Local validation runs finished. Check output above for any failures."
