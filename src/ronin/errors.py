"""The failures Ronin recognises, as distinct from the ones it is surprised by.

A device that cannot be driven, a screen that cannot be read, an action the
device refused and an artifact that could not be written each lead somewhere
different, and `except Exception` collapses them into one.

An adapter over a driver, a subprocess or the filesystem may still catch
broadly, since an SDK can raise almost anything, but it translates what it
caught into one of these. Above that line handlers name what they expect.
"""


class RoninError(Exception):
    """Base for a failure Ronin recognised and can describe."""


class DeviceUnavailable(RoninError):
    """The device, emulator or its tooling could not carry out an instruction."""


class ObservationUnavailable(RoninError):
    """The screen could not be read, so there is nothing to decide from."""


class ActionRejected(RoninError):
    """The device did not do what the action asked, and nothing may assume it did."""


class ArtifactWriteFailed(RoninError):
    """A report, log or screenshot could not be written."""


class UsageError(RoninError):
    """Ronin was asked for something it cannot find or use.

    The message is for the person who typed the command, and says what to fix.
    """
