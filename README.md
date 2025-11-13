# A multi-view contrastive learning framework for spatial embeddings in risk modeling

In this repository, we provide the code and the trained models as described in our paper: 

**A multi-view contrastive learning framework for spatial embeddings in risk modeling** \
Freek Holvoet, Christopher Blier-Wong and Katrien Antonio

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

- Already trained models, as described in the paper in Section 3.4
    - Trained_Model/EU16_OSM16.ckpt
    - The other models mentioned in Section 3.4 of the paper are to large in size to upload to GitHub. We are looking for a solution for this. If needed, please contact me at freek.holvoet@kuleuven.be

- Example on how to use the trained models to extract embeddings
    - Trained_Model/Add_embeddings_to_data.ipynb: A Jupyter notebook showing how to add embeddings to a data set containing a latitude and a longitude columns. 

## Citation

If you use this work in your research, please cite:

> Holvoet, F., Blier-Wong, C., & Antonio, K. (2025). A multi-view contrastive learning framework for spatial embeddings in risk modeling. *arXiv preprint arXiv:XXXX.XXXXX*.

Bibtex version:
```bibtex
@article{holvoet2024multiview,
  title={A multi-view contrastive learning framework for spatial embeddings in risk modeling},
  author={Holvoet, Freek and Blier-Wong, Christopher and Antonio, Katrien},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}