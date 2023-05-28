# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import gradio as gr
import glob
from pipelines.document_stores import FAISSDocumentStore, MilvusDocumentStore
import time
from pipelines.utils import (
    convert_files_to_dicts,
    fetch_archive_from_http,
    print_documents,
)
from pipelines.document_stores import ElasticsearchDocumentStore
from pipelines.nodes import (
    CharacterTextSplitter,
    ChatGLMBot,
    DensePassageRetriever,
    ErnieBot,
    ErnieRanker,
    GLMNode,
    PDFToTextConverter,
    PromptTemplate,
)
from pipelines.pipelines import Pipeline

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default='THUDM/chatglm-6b-int4-qe', type=str, help="The ann index name of ANN.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu",
                    help="Select which device to run dense_qa system, defaults to gpu.")
parser.add_argument("--index_name", default='dureader_index', type=str, help="The ann index name of ANN.")
parser.add_argument("--search_engine", choices=['faiss', 'milvus'], default="faiss",
                    help="The type of ANN search engine.")
parser.add_argument("--max_seq_len_query", default=64, type=int,
                    help="The maximum total length of query after tokenization.")
parser.add_argument("--max_seq_len_passage", default=256, type=int,
                    help="The maximum total length of passage after tokenization.")
parser.add_argument("--retriever_batch_size", default=16, type=int,
                    help="The batch size of retriever to extract passage embedding for building ANN index.")
parser.add_argument("--query_embedding_model", default="rocketqa-zh-nano-query-encoder", type=str,
                    help="The query_embedding_model path")
parser.add_argument("--passage_embedding_model", default="rocketqa-zh-nano-query-encoder", type=str,
                    help="The passage_embedding_model path")
parser.add_argument("--params_path", default="checkpoints/model_40/model_state.pdparams", type=str,
                    help="The checkpoint path")
parser.add_argument("--embedding_dim", default=312, type=int, help="The embedding_dim of index")
parser.add_argument('--host', type=str, default="localhost", help='host ip of ANN search engine')
parser.add_argument('--port', type=str, default="8530", help='port of ANN search engine')
parser.add_argument('--embed_title', default=False, type=bool, help="The title to be  embedded into embedding")
parser.add_argument('--model_type', choices=['ernie_search', 'ernie', 'bert', 'neural_search'], default="ernie",
                    help="the ernie model types")
args = parser.parse_args()


# yapf: enable

def get_faiss_retriever(use_gpu):
    faiss_document_store = "faiss_document_store.db"
    if os.path.exists(args.index_name) and os.path.exists(faiss_document_store):
        # connect to existed FAISS Index
        document_store = FAISSDocumentStore.load(args.index_name)
        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model=args.query_embedding_model,
            passage_embedding_model=args.passage_embedding_model,
            params_path=args.params_path,
            output_emb_size=args.embedding_dim if args.model_type in ["ernie_search", "neural_search"] else None,
            max_seq_len_query=args.max_seq_len_query,
            max_seq_len_passage=args.max_seq_len_passage,
            batch_size=args.retriever_batch_size,
            use_gpu=use_gpu,
            embed_title=args.embed_title,
        )
    else:
        doc_dir = "data/dureader_dev"
        dureader_data = "https://paddlenlp.bj.bcebos.com/applications/dureader_dev.zip"

        fetch_archive_from_http(url=dureader_data, output_dir=doc_dir)
        dicts = convert_files_to_dicts(dir_path=doc_dir, split_paragraphs=True, encoding="utf-8")

        if os.path.exists(args.index_name):
            os.remove(args.index_name)
        if os.path.exists(faiss_document_store):
            os.remove(faiss_document_store)

        document_store = FAISSDocumentStore(embedding_dim=args.embedding_dim, faiss_index_factory_str="Flat")
        document_store.write_documents(dicts)

        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model=args.query_embedding_model,
            passage_embedding_model=args.passage_embedding_model,
            params_path=args.params_path,
            output_emb_size=args.embedding_dim if args.model_type in ["ernie_search", "neural_search"] else None,
            max_seq_len_query=args.max_seq_len_query,
            max_seq_len_passage=args.max_seq_len_passage,
            batch_size=args.retriever_batch_size,
            use_gpu=use_gpu,
            embed_title=args.embed_title,
        )

        # update Embedding
        document_store.update_embeddings(retriever)

        # save index
        document_store.save(args.index_name)
    return retriever


def get_milvus_retriever(use_gpu):
    milvus_document_store = "milvus_document_store.db"
    if os.path.exists(milvus_document_store):
        document_store = MilvusDocumentStore(
            embedding_dim=args.embedding_dim,
            host=args.host,
            index=args.index_name,
            port=args.port,
            index_param={"M": 16, "efConstruction": 50},
            index_type="HNSW",
        )
        # connect to existed Milvus Index
        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model=args.query_embedding_model,
            passage_embedding_model=args.passage_embedding_model,
            params_path=args.params_path,
            output_emb_size=args.embedding_dim if args.model_type in ["ernie_search", "neural_search"] else None,
            max_seq_len_query=args.max_seq_len_query,
            max_seq_len_passage=args.max_seq_len_passage,
            batch_size=args.retriever_batch_size,
            use_gpu=use_gpu,
            embed_title=args.embed_title,
        )
    else:
        doc_dir = "data/dureader_dev"
        dureader_data = "https://paddlenlp.bj.bcebos.com/applications/dureader_dev.zip"

        fetch_archive_from_http(url=dureader_data, output_dir=doc_dir)
        dicts = convert_files_to_dicts(dir_path=doc_dir, split_paragraphs=True, encoding="utf-8")
        document_store = MilvusDocumentStore(
            embedding_dim=args.embedding_dim,
            host=args.host,
            index=args.index_name,
            port=args.port,
            index_param={"M": 16, "efConstruction": 50},
            index_type="HNSW",
        )
        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model=args.query_embedding_model,
            passage_embedding_model=args.passage_embedding_model,
            params_path=args.params_path,
            output_emb_size=args.embedding_dim if args.model_type in ["ernie_search", "neural_search"] else None,
            max_seq_len_query=args.max_seq_len_query,
            max_seq_len_passage=args.max_seq_len_passage,
            batch_size=args.retriever_batch_size,
            use_gpu=use_gpu,
            embed_title=args.embed_title,
        )

        document_store.write_documents(dicts)
        # update Embedding
        document_store.update_embeddings(retriever)

    return retriever


class ChatInterface:
    def __init__(self, model_path, args):
        use_gpu = True if args.device == "gpu" else False

        self.glm_node = GLMNode(model_path)
        self.history = []

        if args.search_engine == "milvus":
            self.retriever = get_milvus_retriever(use_gpu)
        else:
            self.retriever = get_faiss_retriever(use_gpu)

        self.ranker = ErnieRanker(model_name_or_path="rocketqa-zh-dureader-cross-encoder", use_gpu=use_gpu)

        self.query_pipeline = Pipeline()
        self.query_pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])
        self.query_pipeline.add_node(component=self.ranker, name="Ranker", inputs=["Retriever"])
        self.query_pipeline.add_node(
            component=PromptTemplate("请根据以下背景资料回答问题：\n 背景资料：{documents} \n问题：{query}"),
            name="Template", inputs=["Ranker"]
        )
        self.query_pipeline.add_node(component=self.glm_node, name="GLMNode", inputs=["Template"])

    def chat(self, message):
        prediction = self.query_pipeline.run(query=message, params={"Retriever": {"top_k": 10}, "Ranker": {"top_k": 2}})
        result = prediction["response"]
        documents = '\n'.join([doc.content for doc in prediction["documents"]])
        self.history.append({"role": "system", "content": message})
        self.history.append({"role": "user", "content": result})
        return [result,documents]

def main():
    chat_interface = ChatInterface(model_path=args.model_path, args=args)
    iface = gr.Interface(
        fn= chat_interface.chat,
        inputs=gr.inputs.Textbox(lines=2, label="Your question"),
        outputs= [gr.outputs.Textbox(label="Search Results"), gr.outputs.Textbox(label="Documents")],
        examples=[
            ["pipeline-ChatGLM:流水线系统(pipeline)构建本地知识库的ChatGLM问答系统实现"],
            ["交流&问答群：835323155"],
        ],
    )
    iface.launch()


if __name__ == "__main__":
    main()
