import argparse
import os

import torch
from transformers import BloomConfig, BloomModel

from src.downsampling import convert_config_to_downsized_config, select_layers_from_strategy, select_keys_from_state_dict, downsize_state_dict

def main(args):
    config_old_model = BloomConfig.from_pretrained(args.model_name, use_auth_token=True)
    old_model = BloomModel.from_pretrained(args.model_name, use_auth_token=True, torch_dtype="auto")

    downsized_config = convert_config_to_downsized_config(config_old_model, args.hidden_downsampling_rate, args.layer_downsampling_rate, args.aggregation_strategy, args.layer_selection_strategy)
    downsized_model = BloomModel(downsized_config)

    old_model_state_dict = old_model.state_dict()

    mapping_new_keys = select_layers_from_strategy(args.layer_selection_strategy, config_old_model.n_layer, args.layer_downsampling_rate)
    old_model_state_dict = select_keys_from_state_dict(old_model_state_dict, mapping_new_keys)

    downsized_state_dict = downsize_state_dict(old_model_state_dict, downsized_config, args.aggregation_strategy)
    downsized_model.load_state_dict(downsized_state_dict)

    if args.push_to_hub:
        downsized_config.push_to_hub(args.output_model_name, use_auth_token=True, organization="bigscience")
        downsized_model.push_to_hub(args.output_model_name, use_auth_token=True, organization="bigscience")
    else:
        if not os.path.exists(args.output_model_name):
            os.makedirs(args.output_model_name)
        torch.save(downsized_model.state_dict(), os.path.join(args.output_model_name, 'pytorch_model.bin'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downsize a Bloom model to a smaller model')
    parser.add_argument('--model_name', type=str, default="bigscience/bigscience-small-testing", help='Name of the model to downsize - must be on the Hub')
    parser.add_argument('--output_model_name', type=str, default="bigscience/bigscience-small-testing-shrinked", help='Name of the output model')
    parser.add_argument('--hidden_downsampling_rate', type=float, default=0.5, help='Downsampling rate for the hidden layers')
    parser.add_argument('--layer_downsampling_rate', type=float, default=0.5, help='Downsampling rate for the layers')
    parser.add_argument('--aggregation_strategy', type=str, default="mean", help='Aggregation strategy for the weights matrices', choices=["mean", "first", "last"])
    parser.add_argument('--layer_selection_strategy', type=str, default="step", help='Layer selection strategy', choices=["first", "last", "step"])
    parser.add_argument('--push_to_hub', action='store_true', help='Push the model to the Hub')
    args = parser.parse_args()
    main(args)