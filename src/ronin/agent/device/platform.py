from enum import Enum


class Platform(str, Enum):
    """Supported target platform types for mobile automation."""
    ANDROID = "android"
    IOS = "ios"
