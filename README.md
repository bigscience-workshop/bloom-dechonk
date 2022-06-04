# bloom-dechonk
A repo for running model shrinking experiments.

The training script is a slightly modified version of [run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)
from transformers.

### References:
* A PR that adds bloom model to HF Transformers: https://github.com/huggingface/transformers/pull/17474
* Base model checkpoint: ([bloom-6b3](https://huggingface.co/bigscience/bloom-6b3/tree/e1f323d102aee6128c6e5045b99bb8e5015f828f))
* Training logs & config for base model ([tr11f-6B3-logs](https://huggingface.co/bigscience/tr11f-6B3-logs/tensorboard))
* [Todo: link to full train&val dataset on bigscience or -- copy from bigscience-catalogue-lm-data]
* [Todo: a link to public tensorboard logs]
* [Todo: a list of model downsizing scripts]
* [Todo: link do discussion threads suggested during last call]

### Known issues:
* warmup steps and total steps in the training script (below) were chosen by a random guess, they may be suboptimal,  
* the training / validation splits are *not* the same as in the main bloom training,
* batch skipping is not properly validated; if you restart training, you may (or may not) train on some batches twice,
* would be better to make env.sh into a dockerfile, using ubuntu as parent layer


### Training code

The code requires recent datasets and a development version of Transformers that implements the Bloom model:
```
pip install https://github.com/younesbelkada/transformers/archive/ba1d9fc05fda160bda968cc77c4c5dbb21049aa9.zip
pip install datasets==2.2.2 accelerate==0.9.0 fairscale==0.4.6
```

The full installation script can be found in [env.sh](./env.sh). It assumes clean ubuntu/debian installation and runs.
__Please do not run this script before you look inside.__

Once you have set up all the dependencies, you can begin training with the following command 
```bash
export INPUT_PATH=. SNAPSHOT_PATH=./snapshots LOGS_PATH=./logs OMP_NUM_THREADS=32
export DATASET_NAME_OR_PATH=TODO DATASET_CONFIG_NAME=TODO RUN_NAME=EXP_NAME_HERE
export INITIAL_MODEL_PATH=SAVED_MODEL_AND_TOKENIZER

# training hyperparameters were adapted from https://huggingface.co/bigscience/tr11f-6B3-logs/tensorboard?scroll=1#text
# except learning rate and warmup steps, which were chosen based on model's learning rate during initial checkpoint
# this code assumes 8 gpus. For a different setup, change gradient_accumulation_steps or  per_device_train_batch_size
# to get the global batch size of 2^20 (~1M tokens) 

torchrun --nproc_per_node=gpu ./run_clm.py --do_train --do_eval \
    --model_name $INITIAL_MODEL_PATH --tokenizer_name $INITIAL_MODEL_PATH \
    --dataset_name $DATASET_NAME_OR_PATH --dataset_config_name $DATASET_CONFIG_NAME --run_name $RUN_NAME \
    --block_size 2048 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 16 \
    --learning_rate 0.00008 --max_grad_norm 1.0 --lr_scheduler_type cosine --max_steps 31250 --warmup_steps 1000 \
    --adam_epsilon 1e-8 --weight_decay 0.1 --adam_beta1 0.9 --adam_beta2 0.95 --fp16=True \
    --ddp_find_unused_parameters 0 --cache_dir $INPUT_PATH/data/cache --output_dir $SNAPSHOT_PATH --overwrite_output_dir=True \
    --logging_dir $LOGS_PATH --report_to tensorboard --logging_first_step --logging_steps 100 \
    --evaluation_strategy steps --eval_steps 1000 --prediction_loss_only --eval_subset_size 512 \
    --save_steps 1000 --save_total_limit 2 --seed 42 --dataloader_num_workers 4
```

# Model shrinking code

[TODO]