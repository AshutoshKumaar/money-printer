class ShortsAutomationError(Exception):
    """Base exception for expected automation failures."""


class ConfigurationError(ShortsAutomationError):
    """Raised when runtime configuration is invalid."""


class GenerationError(ShortsAutomationError):
    """Raised when content generation fails."""


class UploadError(ShortsAutomationError):
    """Raised when YouTube upload fails."""
