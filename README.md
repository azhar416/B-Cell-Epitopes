# B Cell Epitopes
---

The TF pipeline lets each run select one classifier head with
`CFG.CLASSIFIER_TYPE`: `pooled_mlp` (default), `cnn`, `attention_pooling`, or
`bigru`. Set `USE_SMOTE=False` to use these heads; the separate embedding-SMOTE
workflow uses its own MLP and rejects other classifier selections. The selected
head is stored in the checkpoint metadata.