---
title: "Paper Summary : Yao, Zhang et al. (2020): Non-deterministic and emotional chatting machine: learning emotional conversation generation using condition"
date: 2021-07-19
permalink: /posts/2021-07-19-yao-chat-machine
tags:
    - nlp
    - review
---


## Paper information
[Yao, Zhang et al. (2020): Non-deterministic and emotional chatting machine: learning emotional conversation generation using conditional variational autoencoders, Neural Computing and Applications.](https://link.springer.com/article/10.1007/s00521-020-05338-z)

## Motivation
Conversational systems like chat bots generate responses based on the text provided as input. While they can generate responses based on a specific emotion with related context, these responses are often generic and repetitive. Previous works on this topic or only based on understanding context and associated emotion and ignore the generic nature of the generated response. In this paper, the authors propose a system which tackles this problem by combining the previous works of understanding emotion and context and combining it with a non-deterministic response generator.

## Data 
Emotional Conversation Generation Challenge Dataset, which consists of postings and follow-up comments from the Chinese social media platform Weibo. This dataset contains more than 1 million utterance-response pairs. The datset also contains 6 emotion categories - Anger, Disgust, Happiness, Like, Sadness and Other.

## Method
The authors present two models for their proposed system, NECM  or Non-deterministic and emotional chatting machine. The first model, NECM-Z&E, is a concatenation of outputs from two encoders - Post encoder and Response encoder. The Post encoder approximates a representation given input text, emotion and a suitable response. The Response encoder approximates the representation from input text alone, learning the semantic features.

The second model NECM-GMM approxiates representations with Gaussian Mixture Model for all emotion categories. and generates responses using an attention based GRU or gated recurrent unit using the outputs from either NECM-Z&E or NECM-GMM as input. Also to predict the desired emotion category of the representation from NECM-GMM the authors used a Multi Layer Perceptron.

## Main Result
The authors achieved lower perplexity and higher accuracy compared to existing sequence based models on automatic evaluation of generated reponses. For manual evaluation, the human evaluators had moderate agreement. Also, NECM-GMM performs better than NECM-Z&E in terms of syntactic and affective diversity, which indicates that the desired emotion for a response can be classified from input representation without supervision.  


## Critical Reflection, Limitations

The authors based their work on two cases of conditional probabilities. One is the probability of a representation vector given input text, a suitable response and a desired emotion category. The another is that the probability given just the input text. While this works as expected for the dataset the authors used, which is a considerable large corpus, this method may not produce expected results for smaller sized corpora or low resource languages. Also the authors do not report the metrics for MLP which classifies emotion from input.
