"""py2app build configuration for Decipher.app"""
from setuptools import setup

APP = ["src/main.py"]
DATA_FILES = [
    ("resources/dictionaries", ["resources/dictionaries/english_common.txt"]),
]
OPTIONS = {
    "py2app": {
        "iconfile": "resources/icons/decipher.icns",
        "packages": ["PySide6", "anthropic", "keyring", "PIL"],
        "plist": {
            "CFBundleName": "Decipher",
            "CFBundleDisplayName": "Decipher",
            "CFBundleIdentifier": "com.decipher.app",
            "CFBundleVersion": "0.1.0",
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
        },
    }
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options=OPTIONS,
    setup_requires=["py2app"],
)
