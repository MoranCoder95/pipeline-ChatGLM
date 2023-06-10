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
from pipelines.nodes import (
    PDFToTextGROBIDConverter,
)
import os
from tqdm import tqdm

class ChatInterface:
    def __init__(self, input_directory, output_directory, grobid_host='http://localhost', grobid_port='8080'):
        pdf_converter = PDFToTextGROBIDConverter(grobid_host=grobid_host, grobid_port=grobid_port)

        # List all PDF files in the input directory
        pdf_files = [f for f in os.listdir(input_directory) if f.endswith('.pdf')]

        for pdf_file in tqdm(pdf_files, desc="Converting PDFs"):
            # Full path to the PDF file
            pdf_file_path = os.path.join(input_directory, pdf_file)


            converted_text = pdf_converter.convert(file_path=pdf_file_path)

            base_name, file_extension = os.path.splitext(pdf_file)

            output_file_path = os.path.join(output_directory, f"{base_name}.txt")

            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(converted_text[0].get('content'))

def main():
    input_directory = "./data/2023competition_pdf"
    output_directory = "./data/2023competition_txt"

    os.makedirs(output_directory, exist_ok=True)

    ChatInterface(input_directory=input_directory, output_directory=output_directory)


if __name__ == "__main__":
    main()
