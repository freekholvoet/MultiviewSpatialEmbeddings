from main import *

def get_satclip(ckpt_path, device, return_all=False):
    ckpt = torch.load(ckpt_path,map_location=device)
    ckpt['hyper_parameters'].pop('eval_downstream')
    ckpt['hyper_parameters'].pop('air_temp_data_path')
    ckpt['hyper_parameters'].pop('election_data_path')
    lightning_model = SatCLIPLightningModule(**ckpt['hyper_parameters']).to(device)

    lightning_model.load_state_dict(ckpt['state_dict'])
    lightning_model.eval()

    geo_model = lightning_model.model

    if return_all:
        return geo_model
    else:
        return geo_model.location
    

def get_MCS(ckpt_path, device, return_all=False):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    # Remove unwanted keys from hyperparameters
    unwanted_keys = ['eval_downstream', 'air_temp_data_path', 'election_data_path', '_instantiator']
    for key in unwanted_keys:
        ckpt['hyper_parameters'].pop(key, None)
    
    # Filter out any other unexpected keys
    valid_keys = SatCLIPLightningModule.__init__.__code__.co_varnames
    filtered_hyper_parameters = {k: v for k, v in ckpt['hyper_parameters'].items() if k in valid_keys and k != 'checkpoint_path'}
    
    lightning_model = SatCLIPLightningModule(**filtered_hyper_parameters).to(device)
    lightning_model.load_state_dict(ckpt['state_dict'])
    lightning_model.eval()

    geo_model = lightning_model.model

    if return_all:
        return geo_model
    else:
        return geo_model.location
