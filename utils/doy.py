"""Acquisition-date parsing helpers."""

from datetime import datetime

import numpy as np


def parse_yyyymmdd(value):
    text = str(int(value))
    if len(text) != 8:
        raise ValueError("Invalid YYYYMMDD acquisition date: {!r}".format(value))
    return datetime.strptime(text, "%Y%m%d")


def dates_to_doy(dates):
    """Convert an iterable of YYYYMMDD values to 1-based day-of-year."""
    return np.asarray(
        [parse_yyyymmdd(value).timetuple().tm_yday for value in dates],
        dtype=np.int16,
    )
