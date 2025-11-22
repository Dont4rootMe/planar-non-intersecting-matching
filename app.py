"""Convenience launcher for the planar matching application.

This redirects ``python app.py`` to the new UI implemented in
``planar_matching_app/app.py``.
"""

from __future__ import annotations

import sys

from planar_matching_app.app import main


if __name__ == "__main__":
    sys.exit(main())

