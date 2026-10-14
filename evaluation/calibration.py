"""Validation-only scalar temperature fitting, matching the paper implementation."""

import math

import numpy as np
import torch


def softmax_temperature(logits, temperature):
    if float(temperature) <= 0:
        raise ValueError("Temperature must be positive")
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    values -= values.max(axis=1, keepdims=True)
    result = np.exp(values)
    return result / result.sum(axis=1, keepdims=True)


def fit_temperature(logits, labels):
    """Fit one positive scalar to Validation NLL; never call on Test."""
    logits_tensor = torch.as_tensor(logits, dtype=torch.float64)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=200, tolerance_grad=1e-10,
        tolerance_change=1e-12, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = torch.nn.functional.cross_entropy(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_temperature.detach()).clamp(0.05, 20.0))
    if not math.isfinite(temperature) or temperature <= 0:
        raise FloatingPointError("Invalid fitted temperature")
    return temperature
