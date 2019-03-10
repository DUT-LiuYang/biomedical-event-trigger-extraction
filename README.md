# biomedical-event-trigger-extraction
> Some models for biomedical event trigger extraction by Liu Yang (DUT NLP Lab).

## 项目介绍
本项目是DUTNLP生物事件抽取相关代码重构与创新，完成后将在MLEE以及BioNLP的语料上进行实验。

## 观点
当前生物医学事件论文的一些常见问题：

- 多个单词组成的触发词，只识别第一个单词或者将其看作多个触发词。

- 文本中的信息利用的不够，大多数论文只是简单地做句子分类或者是序列标注。

## 模型介绍
- play_model.py:    The baseline model. baseline 模型，基于双向GRU的序列标注。
- RNN_base_self_attention_model.py: The proposed model in our bibm 2017 paper.

- self_attention_model.py:  The proposed model in our bibm 2017 paper without attention labels.
- self_attention_model_2.py:  The model to test the location of self-attention.

## 关于
biomedical event trigger extraction by 未来数据研究所 LiuYang.

```
那日少年薄春衫，明月照银簪。
```
