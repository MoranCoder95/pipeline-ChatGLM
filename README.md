# 流水线系统(pipeline)构建本地知识库的ChatGLM问答系统实现

# 介绍
🤖️ 利用 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) + [pipelines](https://github.com/PaddlePaddle) 实现的基于本地知识的 ChatGLM 应用。

-高效的本地查询：充分利用百度NLP能力并不断集成其他NLP能力；

-可定制性：基于流水线系统的本地知识库可以根据具体需求进行高度定制；

-方便展示：内置UI界面方便展示；

-代码清晰：相对其他实现本项目代码逻辑清晰明了。


## 硬件需求

- ChatGLM-6B 模型硬件需求

    注：如未将模型下载至本地，请执行前检查`$HOME/.cache/huggingface/`文件夹剩余空间，模型文件下载至本地需要 15 GB 存储空间。

    模型下载方法可参考ChatGLM官网（本项目源码部署运行自动下载ChatGLM-6B相应模型） 。
  
    | **量化等级**   | **最低 GPU 显存**（推理） | **最低 GPU 显存**（高效参数微调） |
    | -------------- | ------------------------- | --------------------------------- |
    | FP16（无量化） | 13 GB                     | 14 GB                             |
    | INT8           | 8 GB                     | 9 GB                             |
    | INT4           | 6 GB                      | 7 GB                              |


## 开发部署

### 软件需求

本项目已在Windows Python 3.8 - 3.10，CUDA 11.7 环境下完成测试。

### 从本地加载模型

项目启动，默认从huggingface自动下载模型及相应文件，存储位于`$HOME/.cache/huggingface/`文件夹

## 开发计划

- [ ] pipeline-ChatGLM
  - [x] 项目基础构建
  - [ ] pdf论文文本识别能力增强
  - [ ] ChatGPT API 接入
- [ ] Demo
  - [ ] 本地pdf知识库(>1G)下ChatGLM问答
  - [ ] 模型微调
- [ ] 增加更多 LLM 模型支持
  - [x] [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)
  - [x] [THUDM/chatglm-6b-int8](https://huggingface.co/THUDM/chatglm-6b-int8)
  - [x] [THUDM/chatglm-6b-int4](https://huggingface.co/THUDM/chatglm-6b-int4)
  - [x] [THUDM/chatglm-6b-int4-qe](https://huggingface.co/THUDM/chatglm-6b-int4-qe)
  - [ ] [ClueAI/ChatYuan-large-v2](https://huggingface.co/ClueAI/ChatYuan-large-v2)
  - [ ] [fnlp/moss-moon-003-sft](https://huggingface.co/fnlp/moss-moon-003-sft)
- [ ] 增加更多 Embedding 模型支持
  - [ ] [shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese)
  - [ ] [GanymedeNil/text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese)
- [ ] 增加 API 支持
  - [ ] 利用 fastapi 实现 API 部署方式
- [ ] 低代码支持

# 模型栈

## 召回排序模型

基于百度飞桨NLP能力，`Pipelines`的主要API的模型如下

### DensePassageRetriever

除了`DensePassageRetriever`的默认模型外，还可以选择下面的模型试试效果：

| 模型                                    | 语言    | 模型详细信息                                                 |
| --------------------------------------- | ------- | ------------------------------------------------------------ |
| rocketqa-zh-base-query-encoder          | Chinese | 12-layer, 768-hidden, 12-heads, 118M parameters. Trained on DuReader retrieval text. |
| rocketqa-zh-medium-query-encoder        | Chinese | 6-layer, 768-hidden, 12-heads, 75M parameters. Trained on DuReader retrieval text. |
| rocketqa-zh-mini-query-encoder          | Chinese | 6-layer, 384-hidden, 12-heads, 27M parameters. Trained on DuReader retrieval text. |
| rocketqa-zh-micro-query-encoder         | Chinese | 4-layer, 384-hidden, 12-heads, 23M parameters. Trained on DuReader retrieval text. |
| rocketqa-zh-nano-query-encoder          | Chinese | 4-layer, 312-hidden, 12-heads, 18M parameters. Trained on DuReader retrieval text. |
| rocketqav2-en-marco-query-encoder       | English | 12-layer, 768-hidden, 12-heads, 118M parameters. Trained on MSMARCO. |
| ernie-search-base-dual-encoder-marco-en | English | 12-layer, 768-hidden, 12-heads, 118M parameters. Trained on MSMARCO. |

### ErnieRanker

类似地`ErnieRanker`可以选择下面的模型试试效果：

| 模型                                      | 语言    | 模型详细信息                                                 |
| ----------------------------------------- | ------- | ------------------------------------------------------------ |
| rocketqa-base-cross-encoder               | Chinese | 12-layer, 768-hidden, 12-heads, 118M parameters. Trained on DuReader retrieval text. |
| rocketqa-medium-cross-encoder             | Chinese | 6-layer, 768-hidden, 12-heads, 75M parameters. Trained on DuReader retrieval text. |
| rocketqa-mini-cross-encoder               | Chinese | 6-layer, 384-hidden, 12-heads, 27M parameters. Trained on DuReader retrieval text. |
| rocketqa-micro-cross-encoder              | Chinese | 4-layer, 384-hidden, 12-heads, 23M parameters. Trained on DuReader retrieval text. |
| rocketqa-nano-cross-encoder               | Chinese | 4-layer, 312-hidden, 12-heads, 18M parameters. Trained on DuReader retrieval text. |
| rocketqav2-en-marco-cross-encoder         | English | 12-layer, 768-hidden, 12-heads, 118M parameters. Trained on Trained on MSMARCO. |
| ernie-search-large-cross-encoder-marco-en | English | 24-layer, 768-hidden, 12-heads, 118M parameters. Trained on Trained on MSMARCO. |

### ErnieReader

`ErnieReader`目前提供了一个预置的模型：

| 模型                                    | 语言    | 模型详细信息                                                 |
| --------------------------------------- | ------- | ------------------------------------------------------------ |
| ernie-gram-zh-finetuned-dureader-robust | Chinese | 12-layer, 768-hidden, 12-heads, 118M parameters. Trained on DuReader Robust Text. |











## 常见问题

🎉 pipeline-ChatGLM 项目还在不断完善，如果你也对本项目感兴趣，欢迎讨论交流。 

交流&问答群：835323155



