<div align="center">    
 
# 🌎 GeoViT, based on GeoCLIP, for worldwide image geolocalization.

## 📎 Getting Started: Training

This repo contains a PyTorch implementation of GeoViT, a model for geolocalizing images based on GeoCLIP.
All data locations are found in runner.py.

```
python runner.py
```

## 📎 Getting Started: Predictions

```python
from geoclip import GeoCLIP
from custom_model import modify_geoclip_architecture, CustomGeoCLIPModel

# Initialize base GeoCLIP model
base_model = GeoCLIP()

# Modify the architecture and create custom model
custom_model = modify_geoclip_architecture(base_model)

# Load custom trained weights
model_path = "path/to/custom_model_weights.pth"
state_dict = torch.load(model_path, map_location=torch.device('cuda'))
custom_model.load_state_dict(state_dict)

# Use the custom model for predictions
image_path = "image.png"
top_pred_gps, top_pred_prob = custom_model.predict(image_path, top_k=5)

print("Top 5 GPS Predictions")
print("=====================")
for i in range(5):
    lat, lon = top_pred_gps[i]
    print(f"Prediction {i+1}: ({lat:.6f}, {lon:.6f})")
    print(f"Probability: {top_pred_prob[i]:.6f}")
    print("")
```
## Citation

```
@inproceedings{geoclip,
  title={GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization},
  author={Vivanco, Vicente and Nayak, Gaurav Kumar and Shah, Mubarak},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```