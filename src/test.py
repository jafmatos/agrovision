import numpy
import torch
from PIL import Image, ImageOps

from deep_learning.config import DEVICE, MODEL_PATH
from deep_learning.engine.grad_cam import GradCAM
from deep_learning.models.resnet_custom import ResNetCustom

model = ResNetCustom(num_classes=3)
model.switch_head()
model.load_state_dict(torch.load(MODEL_PATH))
model.to(DEVICE)

model.eval()

image = ImageOps.fit(
    Image.open(
        "/home/jos/Desktop/Machine Learning/Deep Learning/TorchVision/data/Test/Rust/82add70df6ab2854.jpg"
    ).convert("RGB"),
    (224, 224),
)

input_tensor = (
    torch.from_numpy(
        numpy.ascontiguousarray(
            numpy.array(image).astype(numpy.float32).transpose(2, 0, 1) / 255
        )
    )
    .unsqueeze(dim=0)
    .to(DEVICE)
)

with GradCAM(model=model, target_layer=model.model.layer4) as grad_cam:
    heatmap_tensor = grad_cam.generate_heatmap(input_tensor=input_tensor)
    heatmap_image = Image.fromarray(heatmap_tensor)

    image_with_heatmap = Image.blend(image, heatmap_image, alpha=0.4)

    image_with_heatmap.save("image_with_heatmap_rust.png")