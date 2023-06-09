# Demo说明

❤️❤️❤️❤️❤️❤️"交流&问答群：835323155"

各个Demo功能由简单到复杂进行，方便进行基于本地知识库的大模型知识问答，如有问题和建议欢迎和我们同学交流！



## `semantic_search_example.py`

难度指数：⭐

![Demo](.\assets\img\semantic_search_example.png)

🍬介绍

- 实现基于dureader数据集（运行自动下载）的语义搜索；
- 利用了PaddlePaddle的Dense Passage Retrieval (DPR)和ErnieRanker模型来实现高效的检索和排序；
- 支持FAISS和Milvus搜索引擎来存储和查询段落；
- gradio库用于创建用户友好的界面，与语义搜索系统进行交互；

🦎注意

- 运行前请删除生成的`faiss_document_store.db`文件；



## `answer_question_chatGLM_example.py`

难度指数：⭐

![image-20230609095420881](.\assets\img\answer_question_chatGLM_example.png)

🍬介绍

- 实现基于ChatGLM-6B（运行自动下载）的简单问答系统
- gradio库用于创建用户友好的界面，与语义搜索系统进行交互；

🦎注意

- 



## `chat_txt_chatGLM_example.py`

难度指数：⭐⭐

![Demo](.\assets\img\chat_txt_chatGLM_example.png)

🍬介绍

- 实现基于ChatGLM-6B（运行自动下载）下的本地dureader数据集（运行自动下载）知识问答；
- 利用了PaddlePaddle的Dense Passage Retrieval (DPR)和ErnieRanker模型来实现高效的检索和排序，返回相关Query段落构建提示信息 `PromptTemplate("请根据以下背景资料回答问题：\n 背景资料：{documents} \n问题：{query}")`
- 支持FAISS和Milvus搜索引擎来存储和查询段落；
- gradio库用于创建用户友好的界面，`Search Results`显示最终回答，`Documents`显示本地检索结果；

🦎注意

- 运行前请删除生成的`faiss_document_store.db`文件；
- 如显存有限（RTX 3060 12G）建议最小ChatGLM-6B版本；



## `chat_pdf_chatGLM_example.py`

难度指数：⭐⭐

![Demo](.\assets\img\chat_pdf_chatGLM_example.png)



🍬介绍

- 实现基于ChatGLM-6B（运行自动下载）下的本地pdf（data/pdf_files）知识问答；
- 采用PDF文本提取方法获得数据信息；
- 利用了PaddlePaddle相关模型来实现高效的检索和排序，返回相关Query段落构建提示信息 `PromptTemplate("请根据以下背景资料回答问题：\n 背景资料：{documents} \n问题：{query}")`
- 支持FAISS搜索引擎来存储和查询段落；
- gradio库用于创建用户友好的界面；

🦎注意

- 如显存有限（RTX 3060 12G）建议最小ChatGLM-6B版本；



## `chatglm6b_qa_research_knowledgebase_competition.py`

难度指数：⭐⭐⭐





🍬介绍





🦎注意

- Embedding模型需要根据文本中英文进行切换，例如，采用`DensePassageRetriever.embed_documents` 得到Embedding 之后`faiss.update_embeddings`方法存储，针对Query采用`DensePassageRetriever.embed_queries`得到Embedding，之后进行faiss检索。



























