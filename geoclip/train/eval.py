import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from geopy.distance import geodesic as GD

def distance_accuracy(targets, preds, dis=2500, gps_gallery=None):
    total = len(targets)
    correct = 0
    gd_avg = 0

    for i in range(total):
        gd = GD(gps_gallery[preds[i]], targets[i]).km
        gd_avg += gd
        if gd <= dis:
            correct += 1

    gd_avg /= total
    return correct / total, gd_avg

def eval_images(val_dataloader, model, device="cuda"):
    model.eval()
    preds = []
    targets = []

    gps_gallery = model.gps_gallery.to(device)  # Ensure GPS gallery is on the correct device

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Evaluating")):
            try:
                # Unpack the batch safely
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    imgs, labels = batch[:2]
                else:
                    print(f"Unexpected batch structure in batch {batch_idx}: {type(batch)}")
                    continue  # Skip this batch

                # Convert labels to tensor if they're tuples
                if isinstance(labels, tuple):
                    labels = torch.tensor(labels, dtype=torch.float32)
                elif isinstance(labels, np.ndarray):
                    labels = torch.from_numpy(labels).float()

                if not isinstance(labels, torch.Tensor):
                    print(f"Unexpected label type in batch {batch_idx}: {type(labels)}")
                    continue  # Skip this batch

                if labels.dim() != 2 or labels.shape[1] != 2:
                    print(f"Unexpected label shape in batch {batch_idx}: {labels.shape}")
                    continue  # Skip this batch

                labels = labels.cpu().numpy()
                imgs = imgs.to(device)

                # Get predictions (probabilities for each location based on similarity)
                logits_per_image = model(imgs, gps_gallery)
                
                # Ensure logits_per_image has the expected shape
                if logits_per_image.dim() != 2 or logits_per_image.shape[0] != imgs.shape[0]:
                    print(f"Unexpected logits shape in batch {batch_idx}: {logits_per_image.shape}")
                    continue

                probs = logits_per_image.softmax(dim=-1)
                
                # Predict gps location with the highest probability (index)
                outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()
                
                preds.append(outs)
                targets.append(labels)

                if batch_idx % 100 == 0:
                    print(f"Processed batch {batch_idx}. Current preds length: {len(preds)}, targets length: {len(targets)}")

            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                print(f"Batch content: {batch}")
                continue

    if len(preds) == 0:
        print("No predictions were made. Check the data and model.")
        return {}

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    print(f"Final shapes - preds: {preds.shape}, targets: {targets.shape}")

    model.train()

    distance_thresholds = [2500, 750, 200, 25, 1] # km
    accuracy_results = {}
    for dis in distance_thresholds:
        acc, avg_distance_error = distance_accuracy(targets, preds, dis, gps_gallery.cpu().numpy())
        print(f"Accuracy at {dis} km: {acc}, Average Distance Error: {avg_distance_error}")
        accuracy_results[f'acc_{dis}_km'] = acc

    return accuracy_results