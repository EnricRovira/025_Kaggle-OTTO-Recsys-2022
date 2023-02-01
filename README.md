# Kaggle-OTTO-Recsys-2022

![Header](/images/header.png)

https://www.kaggle.com/competitions/otto-recommender-system/overview/

---

This repo conteins my training script for candidate generation items in order to feed them to the ranker model.

A simple bert base model in which items are feed in a sequence way, ordered by time. A transformer apply attentions weights and i tye the embeddings as the tranposed matrix of the embeddings is the same shape as output shich results in a huge gaining of time performance. 
