import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

def train(train_dataloader, model, optimizer, epoch, batch_size, device, scheduler=None, criterion=nn.CrossEntropyLoss()):
    print(f"Starting Epoch {epoch}")
    model.train()
    total_loss = 0
    num_batches = 0

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    for i, batch in bar:
        try:
            # Unpack the batch
            imgs, gps = zip(*batch)
            
            # Stack the images and GPS coordinates
            imgs = torch.stack(imgs).to(device)
            gps = torch.stack(gps).to(device)

            # Create targets based on the actual batch size
            targets_img_gps = torch.arange(imgs.size(0), device=device)

            # Ensure gps is 2D: [batch_size, 2]
            if gps.dim() == 1:
                gps = gps.view(-1, 2)
            
            gps_queue = model.module.get_gps_queue() if isinstance(model, nn.DataParallel) else model.get_gps_queue()

            optimizer.zero_grad()

            # Append GPS Queue & Queue Update
            gps_all = torch.cat([gps, gps_queue], dim=0)
            if isinstance(model, nn.DataParallel):
                model.module.dequeue_and_enqueue(gps)
            else:
                model.dequeue_and_enqueue(gps)

            # Forward pass
            logits_img_gps = model(imgs, gps_all)

            # Compute the loss
            img_gps_loss = criterion(logits_img_gps, targets_img_gps)
            loss = img_gps_loss

            # Backpropagate
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            bar.set_description(f"Epoch {epoch} loss: {loss.item():.5f}")

            # Print debugging information every 10 batches
            # if i % 10 == 0:
            #     print(f"\nBatch {i}:")
            #     print(f"  Images shape: {imgs.shape}")
            #     print(f"  GPS shape: {gps.shape}")
            #     print(f"  GPS queue shape: {gps_queue.shape}")
            #     print(f"  Loss: {loss.item():.5f}")

        except RuntimeError as e:
            print(f"\nError in batch {i}: {str(e)}")
            print(f"  Images shape: {imgs.shape if 'imgs' in locals() else 'N/A'}")
            print(f"  GPS shape: {gps.shape if 'gps' in locals() else 'N/A'}")
            print(f"  GPS queue shape: {gps_queue.shape if 'gps_queue' in locals() else 'N/A'}")
            continue

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / num_batches
    print(f"\nEpoch {epoch} average loss: {avg_loss:.5f}")
    return avg_loss