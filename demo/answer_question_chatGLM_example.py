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
from pipelines.nodes import GLMNode


# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default='THUDM/chatglm-6b-int4-qe', type=str, help="The ann index name of ANN.")
args = parser.parse_args()
# yapf: enable

class ChatInterface:
    def __init__(self, model_path):
        self.glm_node = GLMNode(model_path)
        self.history = []

    def chat(self, message):
        result, _ = self.glm_node.run(query=message, history=self.history)
        self.history = result["history"]
        return result["response"]
def main():
    chat_interface = ChatInterface(model_path=args.model_path)

    iface = gr.Interface(
        fn=chat_interface.chat,
        inputs=gr.inputs.Textbox(lines=2, label="Your question"),
        outputs=gr.outputs.Textbox(label="GLM's response"),
        examples=[
            ["pipeline-ChatGLM:流水线系统(pipeline)构建本地知识库的ChatGLM问答系统实现"],
            ["交流&问答群：835323155"],
        ],
    )
    iface.launch()


if __name__ == "__main__":
    main()












