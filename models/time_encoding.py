"""Fixed sinusoidal encoding of real calendar day-of-year values."""

import math

import torch
from torch import nn


def calendar_day_of_year(dates):
    """Convert integer YYYYMMDD tensors to real calendar DOY; padding 0 stays 0."""
    dates = dates.to(dtype=torch.long)
    valid = dates > 0
    safe = torch.where(valid, dates, torch.full_like(dates, 19000101))
    year = safe // 10000
    month = (safe // 100) % 100
    day = safe % 100
    if torch.any((month < 1) | (month > 12) | (day < 1) | (day > 31)):
        raise ValueError("Dates must use valid-looking YYYYMMDD integers")
    cumulative = torch.tensor(
        [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
        dtype=torch.long,
        device=dates.device,
    )
    doy = cumulative[month] + day
    leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
    doy = doy + (leap & (month > 2)).to(dtype=torch.long)
    return torch.where(valid, doy, torch.zeros_like(doy))


class SinusoidalDOYEncoding(nn.Module):
    """Parameter-free sinusoidal encoding driven by calendar DOY, not index."""

    def __init__(self, hidden_dim=128):
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0 or hidden_dim % 2:
            raise ValueError("hidden_dim must be a positive even integer")
        self.hidden_dim = hidden_dim
        frequencies = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )
        self.register_buffer("frequencies", frequencies, persistent=True)

    def forward(self, dates):
        doy = calendar_day_of_year(dates)
        angles = doy.to(dtype=self.frequencies.dtype).unsqueeze(-1) * self.frequencies
        output = torch.zeros(
            (*dates.shape, self.hidden_dim),
            dtype=self.frequencies.dtype,
            device=dates.device,
        )
        output[..., 0::2] = torch.sin(angles)
        output[..., 1::2] = torch.cos(angles)
        return output * (dates > 0).unsqueeze(-1).to(dtype=output.dtype)

