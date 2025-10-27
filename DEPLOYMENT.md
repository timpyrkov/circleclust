# Deployment Guide for CircleClust

This guide will help you deploy CircleClust to PyPI and Read the Docs.

## Prerequisites

1. Create accounts on:
   - [PyPI](https://pypi.org/account/register/)
   - [TestPyPI](https://test.pypi.org/account/register/) (for testing)
   - [Read the Docs](https://readthedocs.org/accounts/signup/)

2. Install required tools:
   ```bash
   pip install build twine
   ```

## Step 1: Prepare for PyPI

### 1.1 Test the build locally

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build the distribution
python -m build

# Verify the distribution
ls -la dist/
```

### 1.2 Test on TestPyPI first

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Install from TestPyPI to test
pip install --index-url https://test.pypi.org/simple/ circleclust
```

### 1.3 Upload to PyPI

```bash
# Upload to production PyPI
python -m twine upload dist/*
```

You'll need to enter your PyPI credentials (username and password or API token).

## Step 2: Set up Read the Docs

### 2.1 Import the project

1. Go to https://readthedocs.org/dashboard/
2. Click "Import a Project"
3. Select "Connect Repository"
4. Choose GitHub and authorize
5. Select your `circleclust` repository
6. Leave default settings and click "Next"

### 2.2 Configure the project

Recommended settings:
- **Name**: circleclust
- **Default branch**: master (or main)
- **Python configuration file**: `docs/requirements.txt`
- **Install Project**: Yes
- **Requirements file**: `docs/requirements.txt`
- **Documentation type**: Sphinx lead

### 2.3 Add .readthedocs.yml (already created)

The `.readthedocs.yml` file at the root of your repository configures the build:

```yaml
version: 2

build:
  image: latest

python:
  version: "3.11"
  install:
    - requirements: docs/requirements.txt

sphinx:
  configuration: docs/conf.py
```

### 2.4 Trigger the build

1. Go to your project on Read the Docs
2. Click "Build version" for your latest version
3. Wait for the build to complete

Your documentation will be available at: `https://circleclust.readthedocs.io/`

## Step 3: Verify the deployment

### Check PyPI
Visit: https://pypi.org/project/circleclust/

### Check Read the Docs
Visit: https://circleclust.readthedocs.io/

### Test installation from PyPI
```bash
pip install circleclust
```

## Version updates

When updating the package:

1. Update `version` in `setup.py`
2. Update `__version__` in `circleclust/__init__.py` (if using dynamic version)
3. Commit and push to GitHub
4. Build and upload to PyPI:
   ```bash
   rm -rf build/ dist/ *.egg-info
   python -m build
   python -m twine upload dist/*
   ```
5. Read the Docs will automatically rebuild from your GitHub repo

## Troubleshooting

### Common issues

1. **PyPI upload fails**: Make sure you increment the version number
2. **Read the Docs build fails**: Check the build logs for missing dependencies
3. **Documentation not updating**: Clear the Read the Docs cache or rebuild manually

## Useful commands

```bash
# Test build locally
python -m build

# Check what files will fit in the package
python -m build --sdist
tar -tzf dist/circleclust-0.0.1.tar.gz

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to production PyPI
twine upload dist/*
```

