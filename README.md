# A multi-view contrastive learning framework for spatial embeddings in risk modeling

In this repository, we provide the code and the trained models as described in our paper: 

> Holvoet, F., Blier-Wong, C., & Antonio, K. (2025). A multi-view contrastive learning framework for spatial embeddings in risk modeling. *arXiv preprint arXiv:XXXX.XXXXX*.

Paper is available as pre-print via Arxiv: **[arXiv preprint arXiv:XXXX.XXXXX](https://arxiv.com)**

## Contents 

The repository contains the following files: 

- The multiview spatial embedding model:
    - Folder "config" contains the parameters of the model and the locations to all data for training
    - Folder "datamodules" contains the code to read in the data when training the model
    - Folder "positional_encoding" contains the code for the location encoder. Multiple encoders are given, the one used in the paper is the  spherical_harmonics_ylm.py. For explanations on the other, see the [GitHub Repository of the  SatCLIP model](https://github.com/microsoft/satclip). 

- Environment specifications for training the spatial embedding model:
    - requirements.txt
    - environment.yml

- Example on how to use the trained models to extract embeddings
    - Add_embeddings_to_data.ipynb: A Jupyter notebook showing how to add embeddings to a data set containing a latitude and a longitude columns. 

## Using the pretrained models

The trained models, as described in Section 3.4 of the paper, can be downloaded via HuggingFace.

There are five different models available in our HuggingFace repository:

- `EU16_GS32_OSM16.ckpt`
- `EU16_OSM16.ckpt`
- `EU32_GS96_OSM32.ckpt`
- `EU64_GS64.ckpt`
- `EU8_GS32_OSM32.ckpt`

Example on how to download the models and calculate embeddings for a list of latitude, longitude coordinates:

```python
from huggingface_hub import hf_hub_download
from load_lightweight import get_mvloc_encoder
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example coordinates (latitude, longitude) of various European cities
c = torch.tensor([
    (50.8503, 4.3517),   # Brussels
    (48.8566, 2.3522),   # Paris
    (51.5074, -0.1278),  # London
    (52.5200, 13.4050),  # Berlin
    (41.9028, 12.4964),  # Rome
    (40.4168, -3.7038),  # Madrid
    (59.3293, 18.0686),  # Stockholm
    (60.1699, 24.9384),  # Helsinki
    (47.4979, 19.0402),  # Budapest
    (48.2082, 16.3738),  # Vienna
], dtype=torch.float32) 

model = get_mvloc_encoder(
    hf_hub_download("FreekH/multiview_spatial_embedding", "MODEL_NAME.ckpt"),
    device=device
)
model.to(device)

with torch.no_grad():
    emb = model(c.to(device).double()).detach().cpu().numpy()
```

Replace `MODEL_NAME.ckpt` with the desired model filename from the list above. The Jupyter Notebook Add_embeddings_to_data.ipynb contains a function to systematically add embeddings to a data set containing a latitude and a longitude feature. 

## Citation

```bibtex
@article{holvoet2025multiview,
  title={A multi-view contrastive learning framework for spatial embeddings in risk modeling},
  author={Holvoet, Freek and Blier-Wong, Christopher and Antonio, Katrien},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```
