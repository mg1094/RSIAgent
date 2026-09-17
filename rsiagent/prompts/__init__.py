"""Prompt templates for each agent role (paper Appendix B).

Prompts are kept separate from agents so that role *boundaries* can be audited
by reading one file.  Every passage the paper quotes is marked as such in the
module that holds it.
"""

from . import actor, curriculum, learning, verifier

__all__ = ["actor", "curriculum", "learning", "verifier"]
