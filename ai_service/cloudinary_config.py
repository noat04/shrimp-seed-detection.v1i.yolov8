"""Load optional Cloudinary credentials from environment variables."""

import os
from pathlib import Path

import cloudinary
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

_CLOUDINARY_ENV = {
    "cloud_name": os.getenv("CLOUDINARY_CLOUD_NAME"),
    "api_key": os.getenv("CLOUDINARY_API_KEY"),
    "api_secret": os.getenv("CLOUDINARY_API_SECRET"),
}
_provided = {key for key, value in _CLOUDINARY_ENV.items() if value}
_missing = set(_CLOUDINARY_ENV) - _provided

if _provided and _missing:
    missing_env_names = {
        "cloud_name": "CLOUDINARY_CLOUD_NAME",
        "api_key": "CLOUDINARY_API_KEY",
        "api_secret": "CLOUDINARY_API_SECRET",
    }
    missing = ", ".join(missing_env_names[key] for key in sorted(_missing))
    raise RuntimeError(f"Incomplete Cloudinary configuration. Missing: {missing}")

CLOUDINARY_CONFIGURED = len(_provided) == len(_CLOUDINARY_ENV)

if CLOUDINARY_CONFIGURED:
    cloudinary.config(
        cloud_name=_CLOUDINARY_ENV["cloud_name"],
        api_key=_CLOUDINARY_ENV["api_key"],
        api_secret=_CLOUDINARY_ENV["api_secret"],
        secure=True,
    )
