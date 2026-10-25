from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from deep_learning.config import (
    DEVICE,
    LEARNING_RATE_BACKBONE,
    LEARNING_RATE_HEAD,
    MODEL_PATH,
    MODELS_DIR,
    NUM_EPOCHS_PHASE_1,
    NUM_EPOCHS_PHASE_2,
)
from deep_learning.data.dataset import (
    test_data_loader,
    train_data_loader,
    validation_data_loader,
)
from deep_learning.engine.trainer import evaluate, train_one_epoch
from deep_learning.models.base_model import BaseModel
from deep_learning.models.resnet_custom import ResNetCustom


def train_model(
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Adam,
    epochs: int,
    device: torch.device,
    save_path: str | Path = MODEL_PATH,
):
    best_loss: float | None = None

    for epoch in range(epochs):
        epoch_loss, epoch_acc = train_one_epoch(
            model,
            data_loader=train_data_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        eval_loss, eval_acc = evaluate(
            model,
            data_loader=validation_data_loader,
            criterion=criterion,
            device=device,
        )

        if best_loss is None or eval_loss < best_loss:
            best_loss = eval_loss

            torch.save(model.state_dict(), save_path)

        print(f" Epoch {epoch + 1} ".center(60, "="))
        print(f"Training Loss / Accuracy: {epoch_loss:.4f} / {epoch_acc:.2%}")
        print(f"Validation Loss / Accuracy: {eval_loss:.4f} / {eval_acc:.2%}")


def test_model(
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
):
    test_loss, test_acc = evaluate(
        model, data_loader=test_data_loader, criterion=criterion, device=device
    )

    print(" Final Validation ".center(60, "="))
    print(f"Training Loss / Accuracy: {test_loss:.4f} / {test_acc:.2%}")


def train_model_transfer_learning(model: BaseModel, save_path: str | Path = MODEL_PATH):
    criterion = nn.CrossEntropyLoss()

    model.freeze_gradients()
    model.switch_head()
    model.to(DEVICE)

    optimizer_phase_1 = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_HEAD)

    train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer_phase_1,
        epochs=NUM_EPOCHS_PHASE_1,
        device=DEVICE,
        save_path=save_path,
    )

    model.unfreeze_last_backbone_layer()

    optimizer_phase_2 = torch.optim.Adam(
        [
            {
                "params": model.get_last_backbone_layer_parameters(),
                "lr": LEARNING_RATE_BACKBONE,
            },
            {"params": model.get_head_parameters(), "lr": LEARNING_RATE_HEAD},
        ]
    )

    train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer_phase_2,
        epochs=NUM_EPOCHS_PHASE_2,
        device=DEVICE,
        save_path=save_path,
    )

    test_model(model=model, criterion=criterion, device=DEVICE)


if __name__ == "__main__":
    model = ResNetCustom(num_classes=3)

    train_model_transfer_learning(model, save_path=MODELS_DIR / "resnet.pth")

