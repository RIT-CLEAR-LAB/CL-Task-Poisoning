import numpy as np
import torch
import torch.nn as nn
from functools import partial
from alibi_detect.cd.pytorch import preprocess_drift
from alibi_detect.cd import MMDDrift
from timeit import default_timer as timer

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

    # define preprocessing function
    preprocess_fn = partial(preprocess_drift, model=encoder_net, device=device, batch_size=512)

    # initialise drift detector
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
            t = timer()
            preds = drift_detector.predict(new_images)
            dt = timer() - t
            print('')
            print(f"Drift in class {cls}? {labels[preds['data']['is_drift']]}")
            print(f'p-value: {preds["data"]["p_val"]:.3f}')
            print(f'MMD-Distance: {preds["data"]["distance"]:.3f}')
            print(f'Time (s) {dt:.3f}')
