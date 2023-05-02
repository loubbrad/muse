## Creating a dataset for fine-tuning

This repo (functionally) contains a old development version of my python library pyroll. To generate fine-tuning dataset from a directory of midi files you can do the following

```
from muse import datasets
from muse.models import tokenizer

# Generates a piano-roll dataset recursively from midi files in'./mydir'
dataset = datasets.PianoRollDataset.build(
    "./mydir",
    recur=True,
    div = 4, 
)

# Saves this to a json file
dataset.to_json("./my_dataset.json")

# The tokenizer contains methods for processing piano-roll into src, tgt torch.Tensor
tokenizer = tokenizer.MakedLMPretrainTokenizer()

# Load datatset into a torch.utils.data.Dataset child class
train_dataset = datasets.TrainDataset.from_pianoroll_dataset(
    dataset,
    tokenizer,
    split = 'train',
)

# Or alternatively from a json
train_dataset = datasets.TrainDataset.from_json(
    './my_dataset.json',
    tokenizer,
    key="train",
)
```

## Fine-tuning

You can fine-tune MUSE on this dataset my executing the command:

```
python muse/train.py --mode maskedlm-pretrain \
    -c models/params.ckpt \
    -d my_dataset.json \
    --workers 1 \
    --gpus 1 \
    --epochs 50
```

Note we use mode=maskedlm-pretrain because this uses the standard tokenizer for masked language modelling. 

## Sampling

generate.py contains code for Gibbs sampling. The generate_fugue function demonstrates how I use Gibbs sampling to generate fugues from an initial prompt. If you want to try this yourself you can do so by downloading the fugue weights / prompts using:

```
make fugue_params
make fugue_data
python muse/generate.py
```
