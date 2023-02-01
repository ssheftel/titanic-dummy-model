"""
placeholder.py

Placeholder for module containing one job step(s).
"""

from edscommon.job_api.context import Context
from edscommon.job_api.resources.base import DefaultResource


def some_step(context: Context[DefaultResource]) -> None:
    """
    Doc string describing some step.
    """
    context.log.debug("Starting some_step")
