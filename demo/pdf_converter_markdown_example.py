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
import glob
import time
import gradio as gr
from tqdm import tqdm
import random
from pipelines.document_stores import LocalFileDocumentStore
from pipelines.document_stores import FAISSDocumentStore
from pipelines.document_stores import ElasticsearchDocumentStore
from pipelines.nodes import (
    BM25Retriever,
    CharacterTextSplitter,
    ChatGLMBot,
    DensePassageRetriever,
    ErnieBot,
    ErnieRanker,
    JoinDocuments,
    MarkdownConverter,
    PromptTemplate,
    TruncatedConversationHistory,
    PDFToTextConverter,
    GLMNode,
    PDFToTextOCRConverter,
    PromptTemplate,
    PDFToTextGROBIDConverter,
    PDFPlumberToTextConverter,
    PDFToTextImgOCRConverter,

)
from pipelines.pipelines import Pipeline
from pipelines.utils import (
    convert_files_to_dicts,
    fetch_archive_from_http,
    print_documents,
)


# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['gpu'], default="gpu",
                    help="Select which device to run dense_qa system, defaults to gpu.")
parser.add_argument("--file_paths", default='./data/pdf_files', type=str, help="The PDF file path.")
parser.add_argument("--markdown_file_paths", default='./data/2023competition_markdown', type=str, help="The markdown file path.")
args = parser.parse_args()


use_gpu = True if args.device == "gpu" else False

def pdf_converter_markdown():
    # pdf_converter = PDFToTextConverter()
    # pdf_converter = PDFToTextOCRConverter()
    # pdf_converter = PDFToTextGROBIDConverter(grobid_host='http://localhost', grobid_port='8070')
    # pdf_converter = PDFPlumberToTextConverter()
    files = glob.glob(args.file_paths + "/*.pdf", recursive=True)
    pdf_converter = PDFToTextConverter()
    document_store = LocalFileDocumentStore(directory_path = './data/2023competition_markdown', index = files)
    converter_pipeline = Pipeline()
    converter_pipeline.add_node(component=pdf_converter, name="PdfConverter", inputs=["File"])
    converter_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["PdfConverter"])
    converter_pipeline.run(file_paths=files)


if __name__ == "__main__":
    pdf_converter_markdown()