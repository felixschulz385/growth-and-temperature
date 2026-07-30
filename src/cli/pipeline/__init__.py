"""The ``pipeline`` CLI domain.

docs/design/09-integrated-pipeline.md §8: replaces ``download`` and
``preprocess run``/``preprocess transfer`` with one domain covering a
source's whole fetch/prepare/grid lifecycle. ``assemble`` and ``analysis``
stay separate (different cardinality -- see the design doc).
"""

from .commands import register

__all__ = ["register"]
