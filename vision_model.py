import torch
import torchvision

class VisionModel(torch.nn.Module):
    def __init__(self, model_selection: str, num_classes: int = 10, *args, **kwargs) -> None:
        super(VisionModel, self).__init__()
        self.model_selection = self.name = model_selection
        self.in_channels = 3  # Gunakan 3 untuk gambar RGB

        if self.model_selection == "densenet":
            self.model = torchvision.models.densenet121(pretrained=True)
            self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, num_classes)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, data, *args, **kwargs) -> torch.Tensor:
        x = self.model(data)
        return self.softmax(x)
