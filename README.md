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

To download any of these models, use the following code:

```python
from huggingface_hub import hf_hub_download

# Download the trained model
model_path = hf_hub_download(
    repo_id="FreekH/multiview_spatial_embedding",
    filename="MODEL_NAME.ckpt",
    cache_dir="./models"
)
```

Replace `MODEL_NAME.ckpt` with the desired model filename from the list above.

Or using the HuggingFace CLI:

```bash
huggingface-cli download FreekH/multiview_spatial_embedding MODEL_NAME.ckpt --local-dir ./models
```

## Citation

```bibtex
@article{holvoet2025multiview,
  title={A multi-view contrastive learning framework for spatial embeddings in risk modeling},
  author={Holvoet, Freek and Blier-Wong, Christopher and Antonio, Katrien},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```
