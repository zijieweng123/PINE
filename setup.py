from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
REQUIREMENTS = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()


setup(
    name="pine-iqa",
    version=VERSION,
    description="PINE for no-reference image quality assessment",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[line for line in REQUIREMENTS if line and not line.startswith("#")],
    python_requires=">=3.8",
)
