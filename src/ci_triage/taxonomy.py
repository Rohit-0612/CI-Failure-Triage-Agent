"""Controlled vocabulary for CI failures.

Two separate axes, because they answer different questions:
- FailedStage: *what* part of the pipeline failed (tests, lint, install...).
- FailureCategory: *why* it failed (the root error class).

An ImportError during pytest collection is stage=TEST but category=IMPORT_ERROR.
When several categories match one failure, PRECEDENCE picks the most specific one.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


class FailureCategory(StrEnum):
    TEST_FAILURE = "TEST_FAILURE"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    TYPE_ERROR = "TYPE_ERROR"  # static type checking (mypy/pyright), not runtime TypeError
    IMPORT_ERROR = "IMPORT_ERROR"
    DEPENDENCY_FAILURE = "DEPENDENCY_FAILURE"
    BUILD_FAILURE = "BUILD_FAILURE"
    LINT_FAILURE = "LINT_FAILURE"
    FORMAT_FAILURE = "FORMAT_FAILURE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
    DOCKER_FAILURE = "DOCKER_FAILURE"
    CI_CONFIGURATION_FAILURE = "CI_CONFIGURATION_FAILURE"
    NETWORK_FAILURE = "NETWORK_FAILURE"
    TIMEOUT = "TIMEOUT"
    FLAKY = "FLAKY"  # only assignable with hindsight (same code later passed)
    UNKNOWN = "UNKNOWN"


class FailedStage(StrEnum):
    TEST = "test"
    LINT = "lint"
    FORMAT = "format"
    TYPECHECK = "typecheck"
    BUILD = "build"
    INSTALL = "install"
    SETUP = "setup"
    OTHER = "other"


# Most specific root cause first. Generic stage-like categories (TEST_FAILURE) come
# last because they are symptoms that more specific errors usually explain.
PRECEDENCE: tuple[FailureCategory, ...] = (
    FailureCategory.TIMEOUT,
    FailureCategory.CI_CONFIGURATION_FAILURE,
    FailureCategory.DOCKER_FAILURE,
    FailureCategory.DEPENDENCY_FAILURE,
    FailureCategory.SYNTAX_ERROR,
    FailureCategory.IMPORT_ERROR,
    FailureCategory.NETWORK_FAILURE,
    FailureCategory.ENVIRONMENT_FAILURE,
    FailureCategory.CONFIGURATION_ERROR,
    FailureCategory.FORMAT_FAILURE,
    FailureCategory.LINT_FAILURE,
    FailureCategory.TYPE_ERROR,
    FailureCategory.BUILD_FAILURE,
    FailureCategory.TEST_FAILURE,
    FailureCategory.FLAKY,
    FailureCategory.UNKNOWN,
)


def pick_primary(categories: Iterable[FailureCategory]) -> FailureCategory:
    found = set(categories)
    for category in PRECEDENCE:
        if category in found:
            return category
    return FailureCategory.UNKNOWN
