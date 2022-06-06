# Import libraries

# Import bloom models

# Step1: Define the new hyperparameters given the source model and the downsampling rate. This can be done in the config file
# Step2: Define the new model given this config file
# Step3: Load the source model
# Step4: Load the new model
# Step5: Load the source model's weights into the new model and save the new model. Let the user define the aggregation strategy
# Step6: Save the new model's weights and push them on the Hub

def convert_config_to_downsized_config(config, downsampling_rate):
    # Step1: Define the new hyperparameters given the source model and the downsampling rate. This can be done in the config file
    pass

def create_downsized_model(config):
    # Step2: Define the new model given this config file
    pass

def load_source_model(model_name):
    # Step3: Load the source model
    pass

def load_downsized_model(config, model_name):
    # Step4: Load the new model
    pass

def load_downsized_model_weights(config, source_model, downsized_model, model_name, aggregation_strategy):
    # Step5: Load the source model's weights into the new model and save the new model. Let the user define the aggregation strategy
    pass

def save_downsized_model_weights(config, model_name):
    # Step6: Save the new model's weights and push them on the Hub
    pass