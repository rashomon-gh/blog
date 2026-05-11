---
title: "Paper Summary : Feng et al. (2018) : Pathologies of Neural Models Make Interpretations Difficult"
date: 2021-08-07
permalink: /posts/2021-08-07-feng-patho-nm
tags:
    - nlp
    - review
---

## Paper Information

[Feng et al. (2018) : Pathologies of Neural Models Make Interpretations Difficult. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. ](https://www.aclweb.org/anthology/D18-1407/)


## Motivation

In NLP, neural models learn features from words. However, neural models tend to provide confident outputs even with inputs which are nonsensical to humans, for example, they may be grammatically incorrect or have no actual meaning in terms of human judgement. The existing interpretability methods, based on importance of words as features, fail to explain this behaviour. The authors of this paper wanted to focus on the limitations of the existing interpretation methods by pointing out this behaviour, which they term as *the pathological behaviour of neural models*. They propose *input reduction* to iteratively eliminate the least important words in the input and then use these reduced inputs to fine-tune and interpret the behaviour of the models. 

## Data

The authors made use of three datasets -  [SQUAD](https://aclanthology.org/D16-1264/), [SNLI](https://nlp.stanford.edu/projects/snli/) and [VQA](https://visualqa.org/).

## Method

### Input Reduction
The authors wanted to reduce inputs to the length which will not affect the model confidence and maintain the pathological behaviour as stated above. To achieve this, they needed to first find the least important words and then eliminate them.

### Finding  least important words
Neural models focus on features which are important. The authors however, wanted to focus on what is *unimportant* or the least important. To measure the importance of a word, they have proposed *efficient gradient based approximation*. By this method, the importance of a word is the dot product of the word embedding and the gradient of the model output with respect to the word embedding.

### Eliminating unimportant words

Based on the importance computed from the gradient based approximation, the words with the lowest importance values are eliminated iteratively from an input. The condition for the iterative process was that the model confidence should remain unchanged. 


### Augmenting reduced inputs with Beam Search
Iterative reduction can reduce an input to a single word, reducing model output confidence. As such, the authors used beam search to find the shortest possible reduced input. Here,  they have used a threshold value \\( k \\) for the beam search so that the input does not get reduced further and still manages to maintain model confidence. 

### Human judgement

To judge how humans accuracy on the reduced inputs, the authors collect 200 correctly classified examples from each dataset, reduce them and then ask two groups of crowd workers to judge these reduced inputs. One group was asked to judge the original examples and the other was asked to check the reduced ones. The authors also wanted to know how random a reduced example would appear to a human. As such, they randomly eliminated words from examples and then asked the crowd workers to compare the original examples and their reduced counterparts.


### Mitigating model pathologies

The authors propose to use entropy based regularization to mitigate the model pathologies. The goal here is to fine tune an existing model to increase maximum likelihood for original examples and also to reduce the entropy on the reduced examples.


## Result

Human accuracy drops significantly when they are asked to judge reduced inputs but the neural models retain their performance. The authors state that the neural models are overfitting and as a result are overconfident of their predictions. This behaviour makes confidence an unreliable metric to judge a neural models predictions. Also, the models tend to retain their prediction performance until the input representation sees a shift in its heatmap and even a slight shift in the heatmap can significantly affect the model output. This explains how models remain confident on their prediction on reduced inputs. Post regularization, human accuracy increases in reduced examples and alongside that the crowd workers find newly generated random examples more preferable than the reduced ones they saw earlier.

## Conclusion and Critical Reflection 

In this paper, the authors have proposed a method which is akin to [adversarial methods](https://arxiv.org/abs/1412.6572) but use entropy based regularization to set on a different objective. Their work finds pathological behaviour in neural models and also proposes a method to mitigate the issue and also, reducing inputs to counter model overconfidence. Alongside these the authors show that confidence can not be a good measure of model prediction. 

However, they do not provide any concrete information on the baseline model. Also, while discussing the results of their proposed regularization approach they fail to provide a clarification on why their regularization improved human accuracy or the necessity of the process. Since they had used randomly generated examples in that stage, the improved accuracy could have been a result of the randomness as well. On top of this, they provide no figures on how many randomly generated examples were there or how many were used in the *they vs. Random experiment*. 


