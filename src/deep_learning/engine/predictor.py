from collections.abc import Callable

import numpy
import torch
from torch import nn


def transform(tensor: numpy.ndarray) -> numpy.ndarray:
    tensor = tensor.transpose(2, 0, 1)
    tensor = numpy.ascontiguousarray(tensor)
    tensor = tensor.astype(numpy.float32)
    tensor = tensor / 255
                   
    return tensor


def predict_single_image(
    model: nn.Module,
    classes: list[str],
    device: torch.device,
    tensor: numpy.ndarray,
    transform: Callable[[numpy.ndarray], numpy.ndarray] = transform,
):
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor. Received {tensor.ndim}D")

    tensor = transform(tensor)
    tensor = numpy.expand_dims(tensor, axis=0)
    tensor = torch.from_numpy(tensor).to(device)

    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        higher_prob, higher_prob_index = torch.max(probabilities, dim=1)

    probs_dict = {
        cls_name: round(prob.item(), 4)
        for cls_name, prob in zip(classes, probabilities[0])
    }

    return {
        "predicted_class": classes[higher_prob_index.item()],
        "certainty": round(higher_prob.item(), 4),
        "probabilities": probs_dict,
    }
