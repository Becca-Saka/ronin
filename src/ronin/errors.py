"""The failures Ronin recognises, and the ones that surprise it.

Each of these leads somewhere different, and `except Exception` collapses them
into one:

- a device Ronin cannot drive
- a screen Ronin cannot read
- an action the device refused
- an artifact Ronin could not write
- a command that asked for something absent

An adapter over a driver, a subprocess or the filesystem may still catch
broadly, because an SDK can raise almost anything. It then translates what it
caught into one of these types. Above that line, each handler names what it
expects.
"""


class RoninError(Exception):
    """Base for a failure Ronin recognises and can describe."""


class DeviceUnavailable(RoninError):
    """Ronin could not drive the device, the emulator or its tooling."""


class ObservationUnavailable(RoninError):
    """Ronin could not read the screen, so it has nothing to decide from."""


class ActionRejected(RoninError):
    """The device did not do what the action asked. Nothing may assume it did."""


class ArtifactWriteFailed(RoninError):
    """Ronin could not write a report, a log or a screenshot."""


class UsageError(RoninError):
    """The command asked for something Ronin cannot find or use.

    The message goes to the person who typed the command. It says what to fix.
    """
