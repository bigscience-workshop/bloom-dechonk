import argparse
import os
from copy import deepcopy

import torch

def count_parameters(model):
    """
    Copied from: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def select_layers_from_strategy(strategy, n_layers, downsampling_rate):
    """
        Given the downsampling strategy, the number of layers and the downsampling rate
        Return a dictionary that maps the layers we want to select from the source model 
        to the layers we want to assign on the shriked model, in the format
        {source_model_key:target_model_key, ... }
        Note that this is applied only on the attention block layers (named as `h` in `BloomModel`)

        Example:
            downsampling rate = 0.5 (ie consider half of the layers)
            Source Model: 
                h.0 | h.1 | ... | h.30
            Target Model:
                if strategy = 'first':
                    returns {h.0:h.0, h.1:h.1, ... h.14:h.14} (consider the first half of the layers)
                if strategy = 'last':
                    returns {h.15:h.0, h.16:h.0, ... , h.30:h.14} (consider the second half of the layers)
                    So h.15 of the source model will correspond to h.0 in the target model, etc.
                if strategy = 'step':
                    returns {h.0:h.0, h.2:h.1, h.4:h.0, ... , h.30:h.14}
                    We consider the layers of the source model with a step of 1/downsample_rate
        Args:
            strategy (`str`, *required*):
                layer selection strategy, must be `first`, `last` or `step` (see comments above)
            n_layers (`int`, *required*):
                total number of layers in the source model
            downsampling_rate (`float`, *required*):
                depth downsampling rate
    """
    selecting_rate = int(1/(downsampling_rate))

    if strategy == "first":
        array_layers = ["h."+str(i) for i in range(n_layers//selecting_rate)]
    elif strategy == "last":
        array_layers = ["h."+str(i) for i in range(int(n_layers//selecting_rate), n_layers)]
    elif strategy == "step":
        array_layers = ["h."+str(i) for i in range(n_layers) if i % int(selecting_rate) == 0]
    else:
        raise NotImplementedError("Unknown strategy: {}".format(strategy))
    return {layer:"h."+str(i) for i, layer in enumerate(array_layers)}

def convert_config_to_downsized_config(config, hidden_size_downsampling_rate, depth_downsampling_rate, aggregation_strategy, layer_selection_strategy):
    """
        Defines the new hyperparameters of the shrinked model, given the downsampling rates.
        Creates a new config file and returns it.
        Args:
            config (`transformers.BloomConfig`, *required*):
                input old model's config
            hidden_downsampling_rate (`float`, *required*):
                hidden states downsampling rate
            depth_downsampling_rate (`float`, *required*):
                depth downsampling rate
            aggregation_strategy (`str`, *required*):
                weights matrices aggregation strategy
            layer_selection_strategy (`str`, *required*):
                layer selection strategy - see the function `select_layers_from_strategy` for more details
    """
    old_hidden_size, old_num_layer, old_num_heads = config.hidden_size, config.n_layer, config.n_head
    
    downsized_config = deepcopy(config)
    downsized_config.width_downsampling_rate = hidden_size_downsampling_rate
    downsized_config.depth_downsampling_rate = depth_downsampling_rate
    
    downsized_config.hidden_size = int(old_hidden_size * hidden_size_downsampling_rate)
    downsized_config.n_layer = int(old_num_layer * depth_downsampling_rate)
    downsized_config.n_head = int(old_num_heads * hidden_size_downsampling_rate)

    downsized_config.weights_aggregation_strategy = aggregation_strategy
    downsized_config.layer_selection_strategy = layer_selection_strategy
    return downsized_config

def select_keys_from_state_dict(state_dict, dict_layers):
    """
        Given the old model's state dict and a dictionnary that maps 
        which layer to keep for the shrinked model, returns the preprocessed 
        state dict
        Example:
            Source state_dict.keys() contain:
                word_embeddings.weight, h.0.self_attention.qkv.weight, h.1.self_attention.qkv.weight, ... , 
                h.30.self_attention.qkv.weight
            dict_layers:
                {h.0:h.0, h.2:h.1, ... , h.30:h.14}
            Returned state_dict.keys() will contain:
                word_embeddings.weight, h.0.self_attention.qkv.weight, h.1.self_attention.qkv.weight, ... , 
                h.14.self_attention.qkv.weight - with h.1.self_attention.qkv.weight corresponding to h.2.self_attention.qkv.weight,
                and h.14.self_attention.qkv.weight corresponding to h.30.self_attention.qkv.weight etc.
            
        Args:
            state_dict (`torch.state_dict`, *required*):
                input old model's state_dict
            dict_layers (`Dict[`str`, `str`], *required*):
                dictionary mapping the keys we want to keep for the layers
    """
    processed_state_dict = {}
    for key in state_dict.keys():
        if key.startswith("h."):
            prefix = ".".join(key.split('.')[:2])
            if prefix in dict_layers.keys():
                new_key = key.replace(prefix, dict_layers[prefix])
                processed_state_dict[new_key] = state_dict[key]
        else:
            processed_state_dict[key] = state_dict[key]
    return processed_state_dict

def map_key_to_downsized_model(hidden_size, key):
    """
        This function maps the key of the old model to the corresponding new dimensions of the new model

        Args:
            hidden_size (`int`, *required*):
                Targed downsampled hidden_size.
            key (`str`, *required*):
                model state dict key that needs to be processed
        Returns:
            splitting_dimensions (`Tuple[`int`, `int`]`):
                Desired target weight matrix dimension for the corresponding input key.
    """
    if key.endswith("dense.weight"):
        splitting_dimensions = (hidden_size, hidden_size)
    elif key.endswith("dense_h_to_4h.weight"):
        splitting_dimensions = (4*hidden_size, hidden_size)
    elif key.endswith("dense_h_to_4h.bias"):
        splitting_dimensions = (4*hidden_size,)
    elif key.endswith("dense_4h_to_h.weight"):
        splitting_dimensions = (hidden_size, 4*hidden_size)
    elif key.endswith("self_attention.query_key_value.bias"):
        splitting_dimensions = (3*hidden_size,)
    elif key.endswith("self_attention.query_key_value.weight"):
        splitting_dimensions = (3*hidden_size, hidden_size)
    else:
        splitting_dimensions = (hidden_size,)
    return splitting_dimensions


def process_weights(weights, aggregation_strategy, splitting_dimensions):
    """
        This function takes the weights of the old model and downsamples them to the new model's dimensions
    """
    if len(splitting_dimensions) == 1:
        splitting_size = splitting_dimensions[0]
        splitted_tensor = torch.stack(weights.split(splitting_size, dim=-1))
    elif len(splitting_dimensions) == 2:
        splitting_size_x, splitting_size_y = splitting_dimensions
        for i, splitting_size in enumerate(splitting_dimensions):
            if i == 0:
                splitted_tensor = torch.stack(weights.split(splitting_size, dim=0))
            else:
                splitted_tensor = torch.stack(splitted_tensor.split(splitting_size, dim=-1))
        n_folds = splitted_tensor.shape[0] * splitted_tensor.shape[1]
        splitted_tensor = splitted_tensor.contiguous().view(n_folds, splitting_size_x, splitting_size_y)

    if aggregation_strategy == "mean":
        return torch.mean(splitted_tensor, dim=0)
    elif aggregation_strategy == "first":
        return splitted_tensor[0]
    elif aggregation_strategy == "last":
        return splitted_tensor[-1]
    else:
        raise ValueError("Unknown aggregation strategy: {}".format(aggregation_strategy))

def downsize_state_dict(state_dict, downsized_model_config, aggregation_strategy="mean"):
    """
        Take the old model's state dict and downsize it. The state dict should be already preprocessed
        by the function `select_keys_from_state_dict` that will select only the relevant keys (relevant layers)
        Args:
            state_dict (`torch.state_dict`, *required*):
                input state dict of the old model
            downsized_model_config (`transformers.BloomConfig`, *required*):
                config class of the shrinked model
            aggregation_strategy (`str`, *required*):
                aggregation strategy for the weights matrices. We first divide the weights matrix
                into (1/(down_sampling_rate**2)) chunks for linear weights and (1/downsampling_rate)
                for bias terms
                - "first": Select only the first chunk
                - "last": Select only the last chunk
                - "mean": Average all the chunks 
    """
    downsized_state_dict = {}
    for key in state_dict.keys():
        splitting_dimensions = map_key_to_downsized_model(downsized_model_config.hidden_size, key)
        weight_tensor = state_dict[key]
        processed_weights = process_weights(weight_tensor, aggregation_strategy, splitting_dimensions)
        downsized_state_dict[key] = processed_weights
    return downsized_state_dict