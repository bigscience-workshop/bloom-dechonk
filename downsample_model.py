# Import libraries
import argparse
import os
from copy import deepcopy

import torch
from transformers import BloomConfig, BloomModel

model_name = "bigscience/bloom-6b3"
output_model_name = "bloom-6b3-shrinked"
downsampling_rate = 0.25
aggregation_strategy = "mean"
layer_selection_strategy = "step"
push_to_hub = True


def count_parameters(model):
    """
    Copied from: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def select_layers_from_strategy(strategy, n_layers, downsampling_rate):
    selecting_rate = int(1/(2*downsampling_rate))

    if strategy == "first":
        array_layers = ["h."+str(i) for i in range(selecting_rate)]
    elif strategy == "last":
        array_layers = ["h."+str(i) for i in range(int(selecting_rate), n_layers)]
    elif strategy == "step":
        array_layers = ["h."+str(i) for i in range(n_layers) if i % int(selecting_rate) == 0]
    else:
        raise NotImplementedError("Unknown strategy: {}".format(strategy))
    return {layer:"h."+str(i) for i, layer in enumerate(array_layers)}

def convert_config_to_downsized_config(config, downsampling_rate, aggregation_strategy, layer_selection_strategy):
    """
        Step1: Define the new hyperparameters given the source model and the downsampling rate. This is done in the config file
    """
    old_hidden_size, old_num_layer = config.hidden_size, config.n_layer
    
    downsized_config = deepcopy(config)
    downsized_config.downsampling_rate = downsampling_rate
    downsized_config.hidden_size = int(old_hidden_size * downsampling_rate * 2)
    downsized_config.n_layer = int(old_num_layer * downsampling_rate * 2)
    downsized_config.weights_aggregation_strategy = aggregation_strategy
    downsized_config.layer_selection_strategy = layer_selection_strategy

    return downsized_config

def select_keys_from_state_dict(state_dict, dict_layers):
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


def process_weights(weights, aggregation_strategy, downsampling_rate, splitting_dimensions=1):
    """
        This function takes the weights of the old model and downsamples them to the new model's dimensions
    """
    if len(splitting_dimensions) == 1:
        splitting_size = splitting_dimensions[0]
        splitted_tensor = torch.stack(weights.split(splitting_size, dim=-1))
    elif len(splitting_dimensions) == 2:
        splitting_size_x, splitting_size_y = splitting_dimensions
        n_folds = int(1/downsampling_rate)
        for i, splitting_size in enumerate(splitting_dimensions):
            if i == 0:
                splitted_tensor = torch.stack(weights.split(splitting_size, dim=0))
            else:
                splitted_tensor = torch.stack(splitted_tensor.split(splitting_size, dim=-1))
        splitted_tensor = splitted_tensor.contiguous().view(n_folds, splitting_size_x, splitting_size_y)

    if aggregation_strategy == "mean":
        return torch.mean(splitted_tensor, dim=0)
    elif aggregation_strategy == "first":
        return splitted_tensor[0]
    elif aggregation_strategy == "last":
        return splitted_tensor[-1]
    else:
        raise ValueError("Unknown aggregation strategy: {}".format(aggregation_strategy))

def downsize_state_dict(state_dict, downsized_model_config, downsampling_rate, aggregation_strategy="mean"):
    """
        The state dict needs to be pre-processed by popping the keys that are not needed (aka the layers that we want to discard)
        Step5: Load the source model's weights into the new model and save the new model. Let the user define the aggregation strategy
    """
    downsized_state_dict = {}
    for key in state_dict.keys():
        splitting_dimensions = map_key_to_downsized_model(downsized_model_config.hidden_size, key)
        weight_tensor = state_dict[key]
        processed_weights = process_weights(weight_tensor, aggregation_strategy, downsampling_rate, splitting_dimensions)
        downsized_state_dict[key] = processed_weights
    return downsized_state_dict

def main():
    config_old_model = BloomConfig.from_pretrained(model_name, use_auth_token=True)
    old_model = BloomModel.from_pretrained(model_name, use_auth_token=True, torch_dtype="auto")

    downsized_config = convert_config_to_downsized_config(config_old_model, downsampling_rate, aggregation_strategy, layer_selection_strategy)
    downsized_model = BloomModel(downsized_config)

    old_model_state_dict = old_model.state_dict()

    mapping_new_keys = select_layers_from_strategy(layer_selection_strategy, config_old_model.n_layer, downsampling_rate)
    old_model_state_dict = select_keys_from_state_dict(old_model_state_dict, mapping_new_keys)

    downsized_state_dict = downsize_state_dict(old_model_state_dict, downsized_config, downsampling_rate, aggregation_strategy)
    downsized_model.load_state_dict(downsized_state_dict)

    if push_to_hub:
        downsized_config.push_to_hub(output_model_name, revision="{}_{}_{}".format(downsampling_rate, aggregation_strategy, layer_selection_strategy), use_auth_token=True, organization="bigscience")
        downsized_model.push_to_hub(output_model_name, revision="{}_{}_{}".format(downsampling_rate, aggregation_strategy, layer_selection_strategy), use_auth_token=True, organization="bigscience")
    else:
        torch.save(downsized_model.state_dict(), os.path.join(output_model_name, 'pytorch_model.bin'))


if __name__ == "__main__":
    main()