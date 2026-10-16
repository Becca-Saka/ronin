from enum import Enum


class Platform(str, Enum):
    """Supported target platform types for mobile automation."""
    ANDROID = "android"
    IOS = "ios"

    @classmethod
    def named(cls, value) -> "Platform":
        """The platform a name stands for.

        A name Ronin does not know stops the run. Treating anything that is
        not "android" as iOS turned a typo into a run against the wrong
        platform, which fails far away from the word that caused it.

        Raises:
            UsageError: If the name is not a platform Ronin supports.
        """
        from ronin.errors import UsageError

        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            known = ", ".join(platform.value for platform in cls)
            raise UsageError(f"Unknown platform '{value}'. Ronin supports: {known}") from None
