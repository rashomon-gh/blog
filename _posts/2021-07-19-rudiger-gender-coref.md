---
title: "Paper Summary : Rudinger et al. (2018) : Gender Bias in Coreference Resolution"
date: 2021-07-19
permalink: /posts/2021-07-19-rudiger-gender-coref
tags:
    - nlp
    - review
---

## Paper Information

[Rudinger et al. (2018) : Gender Bias in Coreference Resolution, Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers). ](https://www.aclweb.org/anthology/N18-2002)

## Motivation

Data driven systems potentially acquire and exhibit human bias in their outputs, such as bias towards a specific gender, race etc. In natural language processing these systems can vary from sentiment analysis, machine translation and so on. In this paper, the authors empirically evaluate and confirm the  presence of systematic gender bias in three existing coreference resolution systems. The authors show how specific gender pronouns are more associated with specific types of occupations. Alongside this, they propose _Winogender schemas_ for generating evaluation data and correlate gender to occupation corefernce with real life employment data.


## Data
The authors make use of [B&L](https://www.aclweb.org/anthology/P06-1005)which is a large list of English nouns and noun phrases with gender and number counts, collected from web news. They also used employment data of the period 2015-16 from Bureau of Labor Statistics to correlate between gender statistics in B&L and the aforementioned employment data. 

## Method

### Winogender schemas
The authors wanted to discover for which cases or inputs a gender pronoun (male/female/neutral) gets coreferred to an occupation. Using the _Winogender schemas_, they created an evaluation set  of sentence templates containing *three expressions of interest :  occupation*, *participant* and *pronoun*. They used a list of 60 occupations and for each of them created two sentence templates, 120 in total. In one sentence they coreferred *pronoun* to
*occupation* and *pronoun* to *participant* in the other.

### Validation
Each of the *Winogender schemas* based sentence templates have one correct answer which is either the *occupation* or the *participant* from the contained expression. The authors wanted to evaluate which of these two get associated with the *pronoun*.


## Result
The authors evaluated three systems, terming them as [RULE](https://www.aclweb.org/anthology/W11-1902), [STAT](https://www.aclweb.org/anthology/D13-1203) and [NEURAL](https://arxiv.org/abs/1609.08667). The evaluation indicated that the masculine or male pronouns are coreferred more to *occupation* than the female pronouns while the neutral pronouns are randomly coreferred to either *occupation* or *participant* due to ambiguity. The results also showed that the gender preference of these systems for occupations correlated to real life employment statistics and are more leaned or skewed towards males. The authors also go on to say that these systems *overgeneralize* on gender terms.

## Conclusion and Critical Reflection

This paper shows the presence of gender bias in existing coreference systems and how it correlates to real life  gender bias in the employment sector and that the males get preference for certain occupations. However, correlation does not always imply causation and this preferential behavior in some cases can be due to factors other than just gender. There are historical, religious, social and political biases which give more preference to males over females. Also the authors disregard the complexity of the sentences in the evaluation set, which, based on the sense of pronouns and context can affect the results. Another point to consider here is that  *Winogender schemas* can detect the presence of bias but can not rule out its absence. This may produce unexpected results if the same evaluation is applied on other systems.
