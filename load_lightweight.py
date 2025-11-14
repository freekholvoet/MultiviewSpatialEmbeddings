import torch
from location_encoder import get_neural_network, get_positional_encoding, LocationEncoder

def get_mvloc_encoder(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    hp = ckpt['hyper_parameters']

    posenc = get_positional_encoding(
        hp['pos_encoder'],
        hp['legendre_polys'],
        hp['harmonics_calculation'],
        hp['min_radius'],
        hp['max_radius'],
        hp['frequency_num']
    )
    
    nnet = get_neural_network(
        hp['loc_encoder'],
        posenc.embedding_dim,
        hp['Combined_dim'],
        hp['loc_capacity'],
        hp['loc_layers'],
    )

    # only load nnet params from state dict
    state_dict = ckpt['state_dict']
    state_dict = {k[k.index('nnet'):]: state_dict[k] 
                  for k in state_dict.keys() if 'nnet' in k}
    
    loc_encoder = LocationEncoder(posenc, nnet).double()
    loc_encoder.load_state_dict(state_dict)
    loc_encoder.eval()

    # Clear unnecessary data from checkpoint to save memory
    del ckpt
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return loc_encoder        
