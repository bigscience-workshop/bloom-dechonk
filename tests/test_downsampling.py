import unittest
import torch
from transformers import BloomConfig, BloomModel

from src.downsampling import select_layers_from_strategy, convert_config_to_downsized_config, select_keys_from_state_dict, downsize_state_dict, map_key_to_downsized_model

class BloomShrinkingTest(unittest.TestCase):
    """
        Testing suite for the model shrinking functions.
    """

    def setUp(self):
        super().setUp()
        self.path_bigscience_debug_model = "bigscience/bigscience-small-testing"
        self.n_layers = 6
        self.depth_downsampling_rate = 0.5
        self.width_downsampling_rate = 0.5
    
    def test_select_layers_from_strategy(self):
        """
            Tests the select_layers_from_strategy function.
        """
        expected_first = {"h.0":"h.0", "h.1":"h.1", "h.2":"h.2"}
        expected_last = {"h.3":"h.0", "h.4":"h.1", "h.5":"h.2"}
        expected_step = {"h.0":"h.0", "h.2":"h.1", "h.4":"h.2"}
        strategies_to_test = {
            "first":expected_first,
            "last":expected_last,
            "step":expected_step,
        }
        for strategy in strategies_to_test.keys():
            self.assertDictEqual(select_layers_from_strategy(strategy, self.n_layers, self.depth_downsampling_rate), strategies_to_test[strategy])
    
    def test_convert_config_to_downsized_config(self):
        """
            Loads the config file from bigscience/bigscience-small-testing and verifies that the new config file is correct.
        """
        config = BloomConfig.from_pretrained(self.path_bigscience_debug_model)
        downsized_config = convert_config_to_downsized_config(config, self.width_downsampling_rate, self.depth_downsampling_rate, "mean", "first")

        self.assertEqual(downsized_config.n_layer, config.n_layer * self.depth_downsampling_rate)
        self.assertEqual(downsized_config.hidden_size, config.hidden_size * self.width_downsampling_rate)
        self.assertEqual(downsized_config.n_head, config.n_head * self.width_downsampling_rate)

        self.assertEqual(downsized_config.weights_aggregation_strategy, "mean")
        self.assertEqual(downsized_config.layer_selection_strategy, "first")

        self.assertEqual(downsized_config.depth_downsampling_rate, self.depth_downsampling_rate)
        self.assertEqual(downsized_config.width_downsampling_rate, self.width_downsampling_rate)

    def test_map_key_to_downsized_model(self):
        """
            Tests the map_key_to_downsized_model function.
            This function has to return a tuple of size 2 if the input is 
            a weight matrix, and a tuple of size 1 if the input is a bias vector.
        """
        keys_to_test = [
            "dense.weight",
            "dense.bias",
            "dense_h_to_4h.weight",
            "dense_h_to_4h.bias",
            "dense_4h_to_h.weight",
            "dense_4h_to_h.bias",
            "self_attention.query_key_value.bias",
            "self_attention.query_key_value.weight",
        ]
        dummy_hidden_size = 16
        for key in keys_to_test:
            if key.endswith("weight"):
                self.assertEqual(len(map_key_to_downsized_model(dummy_hidden_size, key)), 2)
            elif key.endswith("bias"):
                self.assertEqual(len(map_key_to_downsized_model(dummy_hidden_size, key)), 1)

    def test_logits(self):
        """
            Downsizes the debug model and verifies that the logits are correct.
        """
        expected_logits = torch.tensor([ 2.421438694000244e-08, -1.862645149230957e-08, -3.725290298461914e-09])
        input_ids = torch.LongTensor([1, 2, 3])

        config_old_model = BloomConfig.from_pretrained(self.path_bigscience_debug_model)
        old_model = BloomModel.from_pretrained(self.path_bigscience_debug_model, torch_dtype="auto")

        downsized_config = convert_config_to_downsized_config(config_old_model, self.width_downsampling_rate, self.depth_downsampling_rate, "mean", "step")
        downsized_model = BloomModel(downsized_config)

        old_model_state_dict = old_model.state_dict()

        mapping_new_keys = select_layers_from_strategy("step", config_old_model.n_layer, self.depth_downsampling_rate)
        old_model_state_dict = select_keys_from_state_dict(old_model_state_dict, mapping_new_keys)

        downsized_state_dict = downsize_state_dict(old_model_state_dict, downsized_config, "mean")
        downsized_model.load_state_dict(downsized_state_dict)

        downsized_model = downsized_model.eval()
        predicted_logits = downsized_model(input_ids).last_hidden_state.mean(dim=-1)
        
        try:
            torch.testing.assert_close(predicted_logits, expected_logits, atol=0.0, rtol=0.0) # raises an exception if the two tensors are not close
        except:
            self.assertTrue(False)