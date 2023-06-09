# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 deepset GmbH. All Rights Reserved.
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


import os
import json
from typing import List
from pipelines.document_stores.base import BaseDocumentStore
from typing import Dict, Generator, List, Optional, Set, Union
from pipelines.schema import Document, FilterType, Label


# class LocalFileDocumentStore(BaseDocumentStore):
#     def __init__(self, file_path: str, index: List[str] ):
#         super().__init__()  # 调用父类的初始化方法
#         self.file_path = file_path
#         self.index = index
#
#     def write_documents(self, documents: List[dict], index: Optional[str] = None,):
#         with open(self.file_path, 'a', encoding='utf-8') as file:
#             for doc in documents:
#                 doc_str = json.dumps(doc, ensure_ascii=False)
#                 file.write(doc_str + '/n')
#
#     def get_all_documents(self) -> List[dict]:
#         documents = []
#         with open(self.file_path, 'r', encoding='utf-8') as file:
#             for line in file:
#                 doc = json.loads(line)
#                 documents.append(doc)
#         return documents
#
#     def delete_all_documents(self):
#         os.remove(self.file_path)
#
#     def run(
#         self,
#         documents: List[dict],
#         index: Optional[str] = None,
#         headers: Optional[Dict[str, str]] = None,
#         id_hash_keys: Optional[List[str]] = None,
#     ):
#         index = index or self.index
#         field_map = self._create_document_field_map()
#         doc_objects = [Document.from_dict(d, field_map=field_map, id_hash_keys=id_hash_keys) for d in documents]
#         self.write_documents(documents=doc_objects, index=index, headers=headers)
#         return {}, "output_1"


import os
import json
from typing import List, Dict, Optional


class LocalFileDocumentStore(BaseDocumentStore):
    def __init__(self, directory_path: str, index: List[str]):
        super().__init__()  # Call the initializer of the base class
        self.directory_path = directory_path
        self.index = index

    def write_documents(self, documents: List[dict], index: List[str],headers: Optional[Dict[str, str]] = None,):
        if len(documents) != len(index):
            raise ValueError("The length of documents must be equal to the length of index")

        for i in range(len(documents)):
            doc = documents[i]
            file_name = os.path.basename(index[i])
            file_name, _ = os.path.splitext(file_name)
            file_path = os.path.join(self.directory_path, f"{file_name}.md")
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(doc.content)

    def get_all_documents(self) -> List[dict]:
        documents = []
        for file_name in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    doc = json.loads(line)
                    documents.append(doc)
        return documents

    def delete_all_documents(self):
        for file_name in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, file_name)
            os.remove(file_path)

    def run(
            self,
            documents: List[dict],
            index: Optional[List[str]] = None,
            headers: Optional[Dict[str, str]] = None,
            id_hash_keys: Optional[List[str]] = None,
    ):
        index = index or self.index
        field_map = self._create_document_field_map()
        doc_objects = [Document.from_dict(d, field_map=field_map, id_hash_keys=id_hash_keys) for d in documents]
        self.write_documents(documents=doc_objects, index=index, headers=headers)
        return {}, "output_1"
