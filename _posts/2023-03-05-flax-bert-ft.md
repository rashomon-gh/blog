---
title: Fine tuning a pretrained model from Hugging Face Transformers with flax
date: 2023-03-05
permalink: /posts/2023-03-05-flax-bert-ft
tags:
    - nlp
    - flax
    - deeplearning
    - finetuning
---

Pre-trained models are great. They're trained on a lot of data us normies probably won't be able to compile by ourselves and they also require a lot of compute to train from scratch. Ever since BERT was released, the NLP community has been using pre-trained models to fine-tune on their own datasets. This is a great way to leverage the power of these models without having to train them from scratch.

(The last two sentences were suggested by Copilot. I don't disagree but don't blame me for plagiarism.)

So to pay homage to the model that brought the ImageNet moment to NLP, I will show you how you can take a pre-trained BERT model from Huggingface and train it on a dataset for movie reviews.

The dataset can be found here: [Pang & Lee, 2004](http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz)

Also, if you're interested in the paper behind the dataset:

```bibtex
@inproceedings{pang-lee-2004-sentimental,
    title = "A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts",
    author = "Pang, Bo  and
      Lee, Lillian",
    booktitle = "Proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics ({ACL}-04)",
    month = jul,
    year = "2004",
    address = "Barcelona, Spain",
    url = "https://aclanthology.org/P04-1035",
    doi = "10.3115/1218955.1218990",
    pages = "271--278",
}
```

To keep the main focus on the fine-tuning process, I will abstract the data preprocessing in a separate python script which can be found [here](https://github.com/ShawonAshraf/annotated-jax/blob/main/utils/pre_polarity.py).


### Dependencies

Before you begin,  use the [toml file](https://github.com/ShawonAshraf/annotated-jax/blob/main/pyproject.toml) to create an env with `uv`. 


### Dataset and Dataloaders


```python
from utils.pre_polarity import prepare_dataset
from IPython.display import clear_output

main_dataset = prepare_dataset()
clear_output()
```


```python
from loguru import logger

logger.info(f"Total dataset size: {len(main_dataset)}")
logger.info("Creating Train and Test Splits.")
train_test_dict = main_dataset.train_test_split(test_size=0.2)

train_dataset = train_test_dict["train"]
test_dataset = train_test_dict["test"]

logger.info(f"Train dataset size: {len(train_dataset)}")
logger.info(f"Test dataset size: {len(test_dataset)}")

logger.info("Creating Train Dev Split from Train Dataset.")
train_dev_dict = train_dataset.train_test_split(test_size=0.2)


train_dataset = train_dev_dict["train"]
dev_dataset = train_dev_dict["test"]

logger.info(f"Train dataset size: {len(train_dataset)}")
logger.info(f"Dev dataset size: {len(dev_dataset)}")
```

```bash
2024-07-07 02:58:23.659 | INFO     | __main__:<module>:3 - Total dataset size: 2000
2024-07-07 02:58:23.660 | INFO     | __main__:<module>:4 - Creating Train and Test Splits.
2024-07-07 02:58:23.664 | INFO     | __main__:<module>:10 - Train dataset size: 1600
2024-07-07 02:58:23.664 | INFO     | __main__:<module>:11 - Test dataset size: 400
2024-07-07 02:58:23.665 | INFO     | __main__:<module>:13 - Creating Train Dev Split from Train Dataset.
2024-07-07 02:58:23.667 | INFO     | __main__:<module>:20 - Train dataset size: 1280
2024-07-07 02:58:23.668 | INFO     | __main__:<module>:21 - Dev dataset size: 320
```


```python
import numpy as np
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer


class PolarityReviewDataset(Dataset):

    def __init__(self, dataset_split: HFDataset, 
                 tokenizer_model_name: str = "google-bert/bert-base-uncased",
                 max_len: int = 512):
        self.ds = dataset_split
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.MAX_LEN = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        review = self.ds[idx]["text"]
        label = self.ds[idx]["label"]

        # encode review text
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.MAX_LEN,
            truncation=True,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="np" # return numpy arrays
        )
        
        return encoding["input_ids"], encoding["attention_mask"], np.array([label])
```


```python
trainset = PolarityReviewDataset(train_dataset)
devset = PolarityReviewDataset(dev_dataset)
testset = PolarityReviewDataset(test_dataset)
```


```python
import jax_dataloader as jdl

BATCH_SIZE = 24 # Max I could load on an RTX 3090
train_loader = jdl.DataLoader(
    trainset, "pytorch", batch_size=BATCH_SIZE, shuffle=True)
val_loader = jdl.DataLoader(
    devset, "pytorch", batch_size=BATCH_SIZE, shuffle=False)
test_loader = jdl.DataLoader(
    testset, "pytorch", batch_size=BATCH_SIZE, shuffle=False)
```

### Model Definition

I would urge you to pay special attention to this part if you're coming from pytorch. Jax works differently. So does Flax. Although BERT is available as a Flax module on the HF hub, the loading process is different than that of the pytorch version. 

First of all, Flax models are immutable pytrees. Pytorch models are a container of tensors which can be mutated. So you can update or assign new params to a Pytorch model on the fly. The same is not possible with Flax models. 

Second, you can't take a Flax model with pretrained params and just assign it to a flax model with the same architecture. You have to unfreeze the new model params, then overwrite them with the pretrained params and then freeze them again. It's like opening a pack of chips and sealing it back again so that nobody knows that you ate some.

Let's check some code first then I will explain.




```python
from transformers import FlaxAutoModel


def load_model(model_name: str = "google-bert/bert-base-uncased") -> tuple:
    model = FlaxAutoModel.from_pretrained(model_name)
    clear_output()
    
    # extract the module and the params
    module = model.module
    pretrained_params = {"params": model.params}
    
    return module, pretrained_params
```

As you can see, I extracted the flax module and the params from the model. Now I will define a new model and assign these params to that one. 


```python
import flax.linen as nn
from flax.core.frozen_dict import unfreeze, freeze
import jax.numpy as jnp



class SentimentCLF(nn.Module):
    backbone: nn.Module # the pretrained model

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray, attention_mask: jnp.ndarray) -> jnp.ndarray:
        # forward pass
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # pooler_output
        out = out.pooler_output
        
        # pass through a dense layer that projects to 2 labels types
        out = nn.Dense(2)(out)
        return out
```


```python
bert_module, pretrained_params = load_model()
```


```python
import jax

rng = jax.random.key(42)
model = SentimentCLF(bert_module)

sample_data = trainset[0]
input_ids, attention_mask, label = sample_data

params = model.init(rng, input_ids, attention_mask)
```

Unfreeze and Freeze, basically.

![freeze.png](./assets/freeze.png)


```python
# unfreeze the new model
params_unfrozen = unfreeze(params)
params_unfrozen["params"]["backbone"] = pretrained_params["params"]
```


```python
# freeze back
params = freeze(params_unfrozen)
```

That's it. Now you can train this model as any other flax model.

### Training


```python
import optax

@jax.jit
def calculate_loss(params, input_ids, attention_mask, label):
    logits = model.apply(params, input_ids, attention_mask)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, label)
    # typical numpy array thing
    # should be a scalar
    return loss[0]
```


```python
@jax.jit
def batched_loss(params, input_ids, attention_masks, labels):
    batch_loss = jax.vmap(calculate_loss, in_axes=(None, 0, 0, 0))(
        params, input_ids, attention_masks, labels)
    return batch_loss.mean(axis=-1)
```


```python
from flax.training import train_state

clipper = optax.clip_by_global_norm(1.0)

tx = optax.chain(optax.adam(learning_rate=2e-5),
                 optax.clip_by_global_norm(1.0))

initial_state = train_state.TrainState.create(
    apply_fn=model.apply,
    tx=tx,
    params=params,
)
criterion = jax.value_and_grad(batched_loss)
```


```python
from sklearn.metrics import f1_score
from tqdm.auto import tqdm, trange


@jax.jit
def test_step(state, batch):
    input_ids, attention_masks, _ = batch

    def infer(params, input_ids, attention_mask):
        logits = model.apply(params, input_ids, attention_mask)
        return jax.nn.softmax(logits, axis=-1)

    probas = jax.vmap(jax.jit(infer), in_axes=(None, 0, 0))(
        state.params, input_ids, attention_masks)

    return probas


def evaluate(state, test_loader):
    scores = list()
    for batch in tqdm(test_loader):
        labels = batch[2]
        probas = test_step(state, batch)
        preds = np.argmax(probas, axis=-1)
        # f1 score, never trust simple accuracy!
        f1s = f1_score(labels, preds)

        scores.append(f1s)

    return np.array(scores).mean(axis=-1)
```


```python
@jax.jit
def train_step(state, batch):
    input_ids, attention_masks, labels = batch
    loss_value, grads = criterion(
        state.params, input_ids, attention_masks, labels)
    updated_state = state.apply_gradients(grads=grads)
    return loss_value, updated_state


@jax.jit
def validation_step(state, batch):
    input_ids, attention_masks, labels = batch
    loss_value, _ = criterion(state.params, input_ids, attention_masks, labels)
    return loss_value


```


```python
def train(state, epochs, train_loader, val_loader):
    steps = 0
    train_losses = []
    mean_val_losses = []

    # =============
    for e in trange(epochs):
        for batch in tqdm(train_loader, desc="train_step"):
            train_loss, state = train_step(state, batch)
            steps += 1

            # log every 200 steps
            if steps % 40 == 0:
                train_losses.append(train_loss)

                # run validation
                validation_losses = []
                for batch in tqdm(val_loader, desc="validation_step"):
                    val_loss = validation_step(state, batch)
                    validation_losses.append(val_loss)

                mean_val_loss = np.array(validation_losses).mean(axis=-1)
                mean_val_losses.append(mean_val_loss)

                logger.info(
                    f"Epoch : {e + 1} :: Step : {steps} :: Loss/Train : {train_loss} :: Loss/Validation : {mean_val_loss}")

    # ============
    return state, train_losses, mean_val_losses


# ============
trained_state, train_losses, mean_val_losses = train(
    initial_state, 2, train_loader, val_loader)
```

```bash
 0%|          | 0/2 [00:00<?, ?it/s]
train_step:   0%|          | 0/54 [00:00<?, ?it/s]
validation_step:   0%|          | 0/14 [00:00<?, ?it/s]
2024-07-07 02:59:13.291 | INFO     | __main__:train:25 - Epoch : 1 :: Step : 40 :: Loss/Train : 0.5054243206977844 :: Loss/Validation : 0.3673432469367981
train_step:   0%|          | 0/54 [00:00<?, ?it/s]
validation_step:   0%|          | 0/14 [00:00<?, ?it/s]
2024-07-07 02:59:40.769 | INFO     | __main__:train:25 - Epoch : 2 :: Step : 80 :: Loss/Train : 0.23009130358695984 :: Loss/Validation : 0.3605358302593231
```

Kinda sus, looks like slight ovefitting but let's do an eval first and then I will explain.

### Evaluation

Always evaluate your models. You don't leave home without brushing teeth in the morning, do you?


```python
score = evaluate(trained_state, test_loader)
logger.info(f"Test F1 Score: {score}")
```

```bash
  0%|          | 0/17 [00:00<?, ?it/s]
2024-07-07 02:59:58.041 | INFO     | __main__:<module>:2 - Test F1 Score: 0.8734399378232043
```

The main problem with this dataset is that the inputs are longer than what BERT can handle (512). In the tokeniser I added truncation. This leads to information loss, so the model is basically reading halfway through the texts and is forced to make a hasty decision about the label. 

Either way, our goal was to fine tune a model, we did that. In a real life scenario, always check your data and what you want from your model before you burn electricity.

