from PIL import ImageOps
from deep_learning.engine.grad_cam import GradCAM
import numpy
from PIL import Image
from deep_learning.config import MODEL_PATH
import torch
from deep_learning.models.resnet_custom import ResNetCustom
from deep_learning.config import DEVICE
from torchvision import models

model = ResNetCustom(num_classes=3)
model.switch_head()
model.load_state_dict(torch.load(MODEL_PATH))
model.to(DEVICE)

model.eval()

input_tensor = (
    torch.from_numpy(
        numpy.ascontiguousarray(
            numpy.array(
                ImageOps.fit(
                    Image.open(
                        "/home/jos/Desktop/Machine Learning/Deep Learning/TorchVision/data/Test/Rust/82add70df6ab2854.jpg"
                    ).convert("RGB"),
                    (224, 224),
                )
            )
            .astype(numpy.float32)
            .transpose(2, 0, 1)
            / 255
        )
    )
    .unsqueeze(dim=0)
    .to(DEVICE)
)

print(input_tensor.shape)

with GradCAM(model=model, target_layer=model.model.layer4) as grad_cam:
    heatmap_tensor = grad_cam.generate_heatmap(input_tensor=input_tensor)

    print(heatmap_tensor)
