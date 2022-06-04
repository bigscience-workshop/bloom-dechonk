import logging
from typing import Callable, Optional

import transformers

logger = logging.getLogger(__file__)

load_raw_shuffled_datasets: Optional[Callable] = None
Trainer: Optional[transformers.Trainer] = None


try:
     import _training_setup_overrides  # custom code for using your cluster's internal database
     load_raw_shuffled_datasets = getattr(_training_setup_overrides, 'load_raw_shuffled_datasets', None)
     Trainer = getattr(_training_setup_overrides, 'Trainer', None)
     logger.warning(f"Using _training_setup_overrides ({_training_setup_overrides})")
     logger.warning(f"Overriding load_raw_shuffled_datasets: {load_raw_shuffled_datasets is not None}")
     logger.warning(f"Overriding Trainer: {Trainer is not None}")

except ImportError as e:
     logger.warning("Not using _training_setup_overrides")
