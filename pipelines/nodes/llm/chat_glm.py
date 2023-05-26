# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import logging
import torch
from pipelines.nodes.base import BaseComponent
from transformers import AutoTokenizer, AutoModel, AutoConfig

logger = logging.getLogger(__name__)


class GLMNode(BaseComponent):
    """
    The GLMNode class is a subclass of the BaseComponent class, designed to interface with
    the local GLM model for generating AI chatbot responses.
    """
    outgoing_edges = 1

    def __init__(self, model_path=None, device='cuda'):
        """
        Initialize the GLMNode instance with the provided model_path.

        :param model_path: Path to the GLM model.
        :param device: Device to run the model on.
        """
        if model_path is None:
            raise Exception("Please provide model_path.")
        self.device = device
        self.model_path = model_path
        self.model,self.tokenizer = self.load_model(model_path, device)
        self.history_len = 10
        self.max_token = 10000
        self.temperature = 0.8
        self.top_p = 0.9

    def load_model(self, model_path, device):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, config=model_config, trust_remote_code=True).half().cuda()
        model = model.eval()
        return model, tokenizer

    def run(self, query, history=None):
        """
        Generate a response using the GLM model with the given query and optional conversation history.
        Returns the chatbot response and updates the conversation history accordingly.

        :param query: The user's input/query to be sent to the GLM model.
        :param history: A list of dictionaries representing the conversation history,
        """
        logger.info(query)

        # if history is not None:
        #     if len(history) % 2 == 0:
        #         for past_msg in history:
        #             if past_msg["role"] not in ["user", "assistant"]:
        #                 raise ValueError(
        #                     "Invalid history: The `role` in each message in history must be `user` or `assistant`."
        #                 )
        #     else:
        #         raise ValueError("Invalid history: an even number of `messages` is expected!")



        history.append({"role": "user", "content": f"{query}"})

        # Here you may want to convert the history to the format that's compatible with GLM.

        # And here is the place to call the function to generate the reply with GLM model.
        response, _ = self.model.chat(
            self.tokenizer,
            query,
            history=history[-self.history_len:] if self.history_len > 0 else [],
            max_length=self.max_token,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        # Add the chatbot's response to the conversation history
        history.append({"role": "assistant", "content": f"{response}"})

        return {"response": response, "history": history}, "default"
