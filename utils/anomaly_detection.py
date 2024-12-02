import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from alibi_detect.cd import ClassifierUncertaintyDrift


def initialize_uncertainty_detector(ref_data, device):
    encoding_dim = 32
    clf = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    clf.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    clf.maxpool = nn.Identity()
    num_features = clf.fc.in_features
    clf.fc = nn.Linear(num_features, encoding_dim)
    clf = clf.to(device).eval()
    cd = ClassifierUncertaintyDrift(ref_data, model=clf, backend="pytorch", p_val=0.02, preds_type="logits")
    return cd


def detect_anomaly(dataset, model):
    anomaly_detected = None

    # if len(dataset.poisoned_classes) == 0:
    #     return anomaly_detected

    labels = ["No!", "Yes!"]
    
    if not model.buffer.is_empty():
        all_data = model.buffer.get_all_data()
        ref_images = all_data[0]
        if ref_images.size(0) > 0:
            anomaly_detector = initialize_uncertainty_detector(ref_images, model.device)
            incoming_images = []
            for batch in dataset.train_loader:
                img_batch = batch[0]
                incoming_images.append(img_batch)
            incoming_images = torch.cat(incoming_images, dim=0)
            preds = anomaly_detector.predict(incoming_images)
            anomaly_detected = (preds['data']['is_drift'], 
                                float(', '.join([f'{p_val:.5f}' for p_val in preds['data']['p_val']])))
            print(f"Anomaly detected? {labels[preds['data']['is_drift']]}")
            print(f"Feature-wise p-values: {', '.join([f'{p_val:.3f}' for p_val in preds['data']['p_val']])}")

    return anomaly_detected