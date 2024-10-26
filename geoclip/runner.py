import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from geoclip.model.GeoCLIP import GeoCLIP
from geoclip.train.dataloader import GeoDataLoader, img_train_transform, img_val_transform
from geoclip.train.train import train
from geoclip.train.eval import eval_images
import os

def collate_fn(batch):
    return [item for item in batch if item is not None]

def main():
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-6
    device = "cuda"

    csv_file = "/data/geolocation/panorama_log_new.csv"
    img_dir = "/data/geolocation/panos_only_mid"
    test_file_path = "./geoclip_models/test_write.txt"
    try:
        os.makedirs("./geoclip_models", exist_ok=True)
        with open(test_file_path, "w") as f:
            f.write("Test write")
        os.remove(test_file_path)
        print("Successfully tested write access to ./geoclip_models")
    except Exception as e:
        print(f"Error: Cannot write to ./geoclip_models. {str(e)}")
        print("Please ensure the directory exists and has write permissions.")
        return

    model = GeoCLIP(from_pretrained=True)
    model.to(device)

    latest_model_path = './geoclip_models/model_1e6_9.pth' # Last model used in training
    if os.path.exists(latest_model_path):
        print(f"Loading model from {latest_model_path}")
        model.load_state_dict(torch.load(latest_model_path))
    else:
        print(f"Warning: {latest_model_path} not found. Starting from scratch.")


    full_dataset = GeoDataLoader(csv_file, img_dir, transform=img_train_transform())

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = img_val_transform()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=30, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=30, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train(train_dataloader, model, optimizer, epoch, batch_size, device)
        print("Saving model")
        torch.save(model.state_dict(), f"./geoclip_models/model_1e6_{epoch+10}.pth")

    torch.save(model.state_dict(), "/home/billy/projects/models/FINAL_model.pth")

if __name__ == "__main__":
    main()
