---
title: Model Checkpointing using Orbax
date: 2023-03-01
permalink: /posts/2023-03-01-orbax-ckpt
tags:
    - nlp
    - flax
    - deeplearning
---

So say you've trained a model using flax, it trained fine, has a nice learning curve (train vs validation) and now you want to save it. Or, you want to save checkpoints of the model during specific stages of the training process and later, use the best checkpoints for inference. Technically, all flax modules are dataclasses and params (part of model state in flax) are what store the model, so what we need to do for checkpointing is to persist the params. 

However, that poses on small problem. You see, params from flax modules are not regular python data types. They're tree maps. In other libraries, e.g. pytorch, you can save a state dict as a regular python dictionary. Such isn't compatible with flax. Instead, we use a package called [orbax](https://github.com/google/orbax) for checkpointing. 

Let's train a small cnn on cifar 10  and then add checkpointing to it. You can skip directly to the [Checkpointing](#checkpointing) section if you want as the ones before it are just setup code. 

### Dependencies

Before you begin,  use the [toml file](https://github.com/ShawonAshraf/annotated-jax/blob/main/pyproject.toml) to create an env with `uv`. 


```python
import torch
import jax_dataloader as jdl
import torchvision
import torchvision.transforms as transforms


class ToNumpy:
    def __call__(self, x: torch.Tensor):
        return x.numpy()
    
    
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ToNumpy()])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)


batch_size = 128

trainloader = jdl.DataLoader(trainset, backend="pytorch", batch_size=batch_size,
                             shuffle=True)
testloader = jdl.DataLoader(testset, backend="pytorch", batch_size=batch_size,
                            shuffle=False)

# classes in cifar10
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```


```python
import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn


from einops import rearrange


class ConvNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        # convs
        out = nn.Conv(features=6, kernel_size=(5, 5))(x)
        out = nn.max_pool(out, window_shape=(2, 2))
        out = nn.Conv(features=16, kernel_size=(5, 5))(out)
        out = nn.max_pool(out, window_shape=(2, 2))

        # flatten into a vector
        # skip the batch dim
        if len(x.shape) > 3:
            out = rearrange(x, "batch c h w -> batch (c h w)")
        else:
            out = out.flatten()

        # dense
        out = nn.Dense(features=120)(out)
        out = nn.Dense(features=84)(out)
        out = nn.Dense(features=10)(out)

        return out
    

# ======================
model = ConvNet()
rng = jax.random.key(0)
params = model.init(rng, jnp.empty((3, 32, 32)))

# run a sample forward pass
logits = model.apply(params, jnp.empty((3, 32, 32)))
print(logits.shape)
```


```bash
(10,)
```



```python
import optax
from tqdm.notebook import trange, tqdm
from flax.training import train_state


@jax.jit
def calculate_loss(params, x, y):
    logits = model.apply(params, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return loss


@jax.jit
def batched_loss(params, xs, ys):
    batch_loss = jax.vmap(calculate_loss, in_axes=(None, 0, 0))(params, xs, ys)
    return batch_loss.mean(axis=-1)


optimiser = optax.adam(learning_rate=0.001)
criterion = jax.value_and_grad(batched_loss)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimiser
)


@jax.jit
def train_step(state, batch):
    loss_value, grads = criterion(state.params, *batch)
    updated_state = state.apply_gradients(grads=grads)
    return loss_value, updated_state
```


```python
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


@jax.jit
def test_step(state, xs):
    def infer(params, x):
        logits = model.apply(params, x)
        return jax.nn.softmax(logits, axis=-1)

    preds = jax.vmap(jax.jit(infer), in_axes=(None, 0))(state.params, xs)
    return preds


def evaluate(state, test_loader):
    scores = list()
    for batch in tqdm(test_loader):
        xs, ys = batch
        preds = test_step(state, xs)
        preds = jnp.argmax(preds, axis=-1)
        f1 = f1_score(preds, ys, average="micro")
        scores.append(f1)

    return np.array(scores).mean(axis=-1)


def custom_classification_report(state, test_loader):
    preds = list()
    actual = list()
    for batch in tqdm(test_loader):
        xs, ys = batch
        pred = test_step(state, xs)
        pred = jnp.argmax(pred, axis=-1).tolist()
        preds.extend(pred)
        actual.extend(ys.tolist())

    clf = classification_report(preds, actual, target_names=classes)
    print(clf)
```

## Checkpointing

Let's first prepare orbax and then go through how checkpointing via it works. 


```python
!rm -rf /tmp/cnn_cifar10_checkpointing_example
# delete the existing checkpoints
```


```python
import orbax
from flax.training import orbax_utils

# since everything in jax is a pytree
# the checkpoints are basically the pytree versions of the params
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
# checkpoint manager for managing how many checkpoints to keep
# keep a max of 5 checkpoints
options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
# structure for the checkpoint
# will be used by orbax later to restore the saved model
# you can also add other information regarding the model
# but make sure to keep the structure consistent
ckpt = {
    "state": state,
}
# tell orbax to use the structure
save_args = orbax_utils.save_args_from_target(ckpt)
# add the save path
save_path = "/tmp/cnn_cifar10_checkpointing_example"
# ckpt manager for versioning
checkpoint_manager = orbax.checkpoint.CheckpointManager(save_path, orbax_checkpointer, options)
```

### Breaking down the mess

Since we're training using train_state from flax, we update the params with the `train_state`. So it would be wise to checkpoint the entire state instead of the params only. Now, there are two stages of checkpointing using Orbax. First, we have `PyTreeCheckpointer`, which can save a single checkpoint. It doesn't keep track of updates over training iterations so no matter how many iterations your training runs for, it'll only save a checkpoint on the first call. To track different checkpoints throughout training, we need `CheckpointManager`.

Now, orbax will save the state as a pytree object. But if we want to restore a full state from it, the checkpoint manager provides no such direct API. So we have to improvise a bit (just follow through, comes right after the training function). Furthermore, orbax has no clue about the data structure of your checkpoints. All it cares is that it gets a pytree, as long as you supply the structure for it. The `ckpt` dict here provides the structure to orbax. It basically acts as a schema for the checkpoints.

Sounds complicated? Kinda is. You see in Pytorch, you can just use the model class and map a saved state_dict to it. Could flax have made it simpler? May be! You can read more [here](https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html).

*N.B: Checkpoint manager creates the checkpoint directory during init and maintains the directory and checkpoint metadata (paths etc.) as state. So if you want to rerun the training, delete the save_path directory first and re init the manager.*

Now we can add the checkpointing code inside the `train` function.


```python
def train(state, epochs, train_loader, test_loader, ckpt_manager=checkpoint_manager, save_args=save_args):
    steps = 0
    losses = []
    f1_scores = []
    
    lowest_loss = np.inf

    # =============
    for e in trange(epochs):
        for batch in tqdm(train_loader):
            loss, state = train_step(state, batch)
            steps += 1

            # log every 200 steps
            if steps % 200 == 0:
                losses.append(loss)

                # run evaluation
                print("Evaluating ... ")
                score = evaluate(state, test_loader)

                f1_scores.append(score)

                print(f"Epoch : {e + 1} :: Step : {steps} :: Loss : {loss} :: F1 : {score}")
                
                # checkpoint only if train loss decreases
                # ideally one would checkpoint on validation loss
                # but we don't have a validation split on the dataset
                # take this as an example
            
                if loss < lowest_loss:
                    print("Checkpointing")
                    lowest_loss = loss
                    # save model ckpt
                    ckpt = {
                        "state": state
                    }
                    ckpt_manager.save(steps, ckpt, save_kwargs={"save_args": save_args})

    # ============
    return state, losses, f1_scores
```


```python
_, _, _ = train(state, 5, trainloader, testloader)
# don't really need these as we'll be restoring checkpoints
```


```bash
  0%|          | 0/5 [00:00<?, ?it/s]

  0%|          | 0/391 [00:00<?, ?it/s]

Evaluating ... 

  0%|          | 0/79 [00:00<?, ?it/s]

Epoch : 1 :: Step : 200 :: Loss : 1.6986947059631348 :: F1 : 0.4495648734177215
Checkpointing

  0%|          | 0/391 [00:00<?, ?it/s]

Evaluating ... 

  0%|          | 0/79 [00:00<?, ?it/s]

Epoch : 2 :: Step : 400 :: Loss : 1.507976770401001 :: F1 : 0.4664754746835443
Checkpointing
Evaluating ... 

  0%|          | 0/79 [00:00<?, ?it/s]

Epoch : 2 :: Step : 600 :: Loss : 1.4895985126495361 :: F1 : 0.4799248417721519
Checkpointing

  0%|          | 0/391 [00:00<?, ?it/s]

Evaluating ... 

  0%|          | 0/79 [00:00<?, ?it/s]

Epoch : 3 :: Step : 800 :: Loss : 1.3953628540039062 :: F1 : 0.48783623417721517
Checkpointing
Evaluating ... 

  0%|          | 0/79 [00:00<?, ?it/s]

Epoch : 3 :: Step : 1000 :: Loss : 1.5259835720062256 :: F1 : 0.49831882911392406

  0%|          | 0/391 [00:00<?, ?it/s]

Evaluating ... 

  0%|          | 0/79 [00:00<?, ?it/s]

Epoch : 4 :: Step : 1200 :: Loss : 1.2466219663619995 :: F1 : 0.5143393987341772
Checkpointing
Evaluating ... 

  0%|          | 0/79 [00:00<?, ?it/s]

Epoch : 4 :: Step : 1400 :: Loss : 1.433868408203125 :: F1 : 0.4990110759493671

  0%|          | 0/391 [00:00<?, ?it/s]

Evaluating ... 

  0%|          | 0/79 [00:00<?, ?it/s]

Epoch : 5 :: Step : 1600 :: Loss : 1.349440574645996 :: F1 : 0.5150316455696202
Evaluating ... 

  0%|          | 0/79 [00:00<?, ?it/s]

Epoch : 5 :: Step : 1800 :: Loss : 1.4629029035568237 :: F1 : 0.5240308544303798
```


Now let's check the saved checkpoints.


```python
import os

print(os.listdir(save_path))
```

```bash
['200', '400', '600', '800', '1200']
```


Question is though, how to figure out which one is the best? If you check back on the checkpointing condition, we only saved the params when there was a new min train loss. So, based on the steps count, the last one is our best choice.


```python
latest_step = checkpoint_manager.latest_step()
print(latest_step)
```

```bash
1200
```


Time to restore the model and run some inference on it. I'm just going to iterate through the test loader again here.


```python
def restore_state_from_step(step, ckpt_manager=checkpoint_manager):
    restored_ckpt = ckpt_manager.restore(step)
    restored_params = restored_ckpt["state"]["params"]
    
    # create a new train state object from the params
    restored_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=restored_params,
        tx=optimiser
    )

    return restored_state


restored_state = restore_state_from_step(latest_step)
```

So what we have to do to restore the model is to load the latest checkpoint, take the params from it, create a new state and return it.


```python
evaluate(restored_state, testloader)
```

```bash
0%|          | 0/79 [00:00<?, ?it/s]

0.5143393987341772
```


```python
custom_classification_report(restored_state, testloader)
```


```bash
  0%|          | 0/79 [00:00<?, ?it/s]

              precision    recall  f1-score   support

       plane       0.59      0.53      0.56      1106
         car       0.57      0.67      0.62       859
        bird       0.38      0.34      0.36      1101
         cat       0.38      0.36      0.37      1052
        deer       0.46      0.45      0.45      1013
         dog       0.36      0.50      0.42       721
        frog       0.64      0.51      0.57      1251
       horse       0.55      0.60      0.57       918
        ship       0.67      0.63      0.65      1070
       truck       0.55      0.60      0.57       909

    accuracy                           0.51     10000
   macro avg       0.51      0.52      0.51     10000
weighted avg       0.52      0.51      0.51     10000
```    

