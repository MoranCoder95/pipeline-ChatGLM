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
from pipelines.nodes import ChatRefHistory

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--pdf_file_path", default='./data/pdf_files', type=str,
                    help="The path of the directory containing PDF files.")
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
parser.add_argument("--chunk_size", default=384, type=int, help="The length of data for indexing by retriever")
args = parser.parse_args()


class ChatInterface:
    def __init__(self, model_path, args):
        use_gpu = True if args.device == "gpu" else False
        self.history = ChatRefHistory()

        self.glm_node = GLMNode(model_path)
        self.ranker = ErnieRanker(model_name_or_path="rocketqa-zh-dureader-cross-encoder", use_gpu=use_gpu)

        pdf_converter = PDFToTextConverter()
        text_splitter = CharacterTextSplitter(separator="\f", chunk_size=args.chunk_size, chunk_overlap=0,
                                              filters=["\n"])
        document_store = FAISSDocumentStore(
            embedding_dim=args.embedding_dim,
            duplicate_documents="skip",
            return_embedding=True,
            faiss_index_factory_str="Flat",
        )
        self.retriever = DensePassageRetriever(
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

        indexing_pipeline = Pipeline()
        indexing_pipeline.add_node(component=pdf_converter, name="pdf_converter", inputs=["File"])
        indexing_pipeline.add_node(component=text_splitter, name="Splitter", inputs=["pdf_converter"])
        indexing_pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Splitter"])
        indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["Retriever"])
        files_paths = glob.glob(args.pdf_file_path + "/*.pdf")
        indexing_pipeline.run(file_paths=files_paths)

        self.query_pipeline = Pipeline()
        self.query_pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])
        self.query_pipeline.add_node(component=self.ranker, name="Ranker", inputs=["Retriever"])
        self.query_pipeline.add_node(
            component=PromptTemplate("请根据以下背景资料回答问题：\n 背景资料：{documents} \n问题：{query}"),
            name="Template", inputs=["Ranker"]
        )
        self.query_pipeline.add_node(component=self.glm_node, name="GLMNode", inputs=["Template"])

    def chat(self, message):
        self.history.add_user_message(message)
        prediction = self.query_pipeline.run(query=message, params={"Retriever": {"top_k": 10}, "Ranker": {"top_k": 2}})
        result = prediction["response"]
        documents = '\n'.join([doc.content for doc in prediction["documents"]])
        self.history.add_ai_message(result,references=documents)
        return self.history

def main():
    chat_interface = ChatInterface(model_path=args.model_path, args=args)
    history = gr.State(initial_value=[])

    def convert_history_for_chatbot(history):
        chatbot_messages = []
        for i in range(0, len(history), 2):  # We are assuming pairs of messages (user, ai).
            user_message = history[i]['content']
            ai_message = history[i + 1]['content']
            references = history[i + 1].get('references')
            if references:
                ai_message += '\n References: \n'
                if isinstance(references, list):
                    for ref in references:
                        ai_message += f'<details> <summary>出处 {os.path.split(ref)[-1]}</summary> \n'
                        ai_message += f'{ref}\n'
                        ai_message += f'</details>'
                else:
                    ai_message += f'<details> <summary>出处 {os.path.split(references)[-1]}</summary> \n'
                    ai_message += f'{references}\n'
                    ai_message += f'</details>'
            chatbot_messages.append([user_message, ai_message])
        return chatbot_messages

    def chat_function(message):
        history = chat_interface.chat(message)

        return convert_history_for_chatbot(history.messages)

    init_message = "Hello! How can I assist you today?"

    with gr.Blocks() as demo:
        radio = gr.Radio(value='gpt-3.5-turbo', choices=['gpt-3.5-turbo', 'gpt-4'], label='models')
        chatbot = gr.Chatbot([["pipeline-ChatGLM:流水线系统(pipeline)构建本地知识库的ChatGLM问答系统实现（交流&问答群：835323155）", "Hello! How can I assist you today?"]],
                             elem_id="chat-box",
                             show_label=False).style(height=750)
        with gr.Row():
            with gr.Column(scale=1):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter",
                ).style(container=False)

        txt.submit(chat_function, [txt], [chatbot])

    demo.queue(concurrency_count=3)\
        .launch(server_name='localhost',
            server_port=7860,
            show_api=False,
            share=False,
            inbrowser=False)



if __name__ == "__main__":
    main()
