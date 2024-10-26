<div align="center">    
 
# 🌎 GeoViT, based on GeoCLIP, for worldwide image geolocalization.

## 📎 Getting Started: API

You can install GeoCLIP's module using pip:

```
pip install geoclip
```

or directly from source:

```
git clone https://github.com/VicenteVivan/geo-clip
cd geo-clip
python setup.py install
```

## 🗺️📍 Worldwide Image Geolocalization

![ALT TEXT](/figures/inference.png)

### Usage: GeoCLIP Inference

```python
import torch
from geoclip import GeoCLIP

model = GeoCLIP()

image_path = "image.png"

top_pred_gps, top_pred_prob = model.predict(image_path, top_k=5)

print("Top 5 GPS Predictions")
print("=====================")
for i in range(5):
    lat, lon = top_pred_gps[i]
    print(f"Prediction {i+1}: ({lat:.6f}, {lon:.6f})")
    print(f"Probability: {top_pred_prob[i]:.6f}")
    print("")
```

## 🌐 Worldwide GPS Embeddings

In our paper, we show that once trained, our location encoder can assist other geo-aware neural architectures. Specifically, we explore our location encoder's ability to improve multi-class classification accuracy. We achieved state-of-the-art results on the Geo-Tagged NUS-Wide Dataset by concatenating GPS features from our pre-trained location encoder with an image's visual features. Additionally, we found that the GPS features learned by our location encoder, even without extra information, are effective for geo-aware image classification, achieving state-of-the-art performance in the GPS-only multi-class classification task on the same dataset.

![ALT TEXT](/figures/downstream-task.png)

### Usage: Pre-Trained Location Encoder

```python
import torch
from geoclip import LocationEncoder

gps_encoder = LocationEncoder()

gps_data = torch.Tensor([[40.7128, -74.0060], [34.0522, -118.2437]])  # NYC and LA in lat, lon
gps_embeddings = gps_encoder(gps_data)
print(gps_embeddings.shape) # (2, 512)
```

### Usage: Custom GeoCLIP-based Model

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

## Acknowledgments

This project incorporates code from Joshua M. Long's Random Fourier Features Pytorch. For the original source, visit [here](https://github.com/jmclong/random-fourier-features-pytorch).

## Citation

```
@inproceedings{geoclip,
  title={GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization},
  author={Vivanco, Vicente and Nayak, Gaurav Kumar and Shah, Mubarak},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
