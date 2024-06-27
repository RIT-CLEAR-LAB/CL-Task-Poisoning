import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from functools import partial
from alibi_detect.cd.pytorch import preprocess_drift
from alibi_detect.cd import MMDDrift, ClassifierUncertaintyDrift

def initialize_detector(ref_data, device):
    encoding_dim = 32
    encoder_net = nn.Sequential(
        nn.Conv2d(3, 64, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(64, 128, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(128, 512, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2048, encoding_dim)
    ).to(device).eval()

    preprocess_fn = partial(preprocess_drift, model=encoder_net, device=device, batch_size=512)
    cd = MMDDrift(ref_data, backend='pytorch', p_val=.05, preprocess_fn=preprocess_fn, n_permutations=100)

    return cd

def detect_drift(drifting_classes, train_loader, model):
    if len(drifting_classes) == 0:
        return

    labels = ['No!', 'Yes!']

    for cls in drifting_classes:
        filtered_images = []
        for img_batch, target_batch, _ in train_loader:
            mask = target_batch == cls
            selected_images = img_batch[mask]
            filtered_images.append(selected_images)

        new_images =  torch.cat(filtered_images, dim=0)

        if new_images.size(0) > 0:
            ref_samples = model.buffer.get_class_data(cls)
            drift_detector = initialize_detector(ref_samples, model.device)
            preds = drift_detector.predict(new_images)
            print('')
            print(f"Drift in class {cls}? {labels[preds['data']['is_drift']]}")
            print(f'p-value: {preds["data"]["p_val"]:.3f}')
            print(f'MMD-Distance: {preds["data"]["distance"]:.3f}')

def initialize_uncertainty_detector(ref_data, device):
    encoding_dim = 32
    clf = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    clf.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    clf.maxpool = nn.Identity()
    num_features = clf.fc.in_features
    clf.fc = nn.Linear(num_features, encoding_dim)
    clf = clf.to(device).eval()
    cd = ClassifierUncertaintyDrift(ref_data, model=clf, backend='pytorch', p_val=0.05, preds_type='logits')

    return cd

def detect_uncertainty_drift(drifting_classes, train_loader, model):
    if len(drifting_classes) == 0:
        return

    labels = ['No!', 'Yes!']

    for cls in drifting_classes:
        filtered_images = []
        for img_batch, target_batch, _ in train_loader:
            mask = target_batch == cls
            selected_images = img_batch[mask]
            filtered_images.append(selected_images)

        new_images =  torch.cat(filtered_images, dim=0)

        if new_images.size(0) > 0:
            ref_samples = model.buffer.get_class_data(cls)
            drift_detector = initialize_uncertainty_detector(ref_samples, model.device)
            preds = drift_detector.predict(new_images)
            print(f"Drift in class {cls}? {labels[preds['data']['is_drift']]}")
            print(f"Feature-wise p-values: {', '.join([f'{p_val:.3f}' for p_val in preds['data']['p_val']])}")

            if preds['data']['is_drift']:       # removing drifted samples from buffer
                model.buffer.flush_class(cls)
