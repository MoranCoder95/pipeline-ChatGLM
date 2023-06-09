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

from pipelines.nodes.base import BaseComponent


class PromptTemplate(BaseComponent):
    outgoing_edges = 1

    def __init__(self, template):
        self.template = template

    def run(self, query=None, documents=None, history=None):
        if documents is not None:
            documents = [i.content for i in documents]
            context = "".join(documents)
            result = {"documents": context, "query": query}
        elif history is not None:
            chat_history = "\n".join(history)
            question = query
            result = {"chat_history": chat_history, "question": question}
        else:
            raise NotImplementedError("This prompt template is not implemented!")

        return {"query": self.template.format(**result)}, "output_1"

# examples = [
#     {"documents": "巴黎是法国的首都。\n", "query": "哪个国家的首都是巴黎？"},
#     {"documents": "华盛顿是美国的首都。\n", "query": "哪个国家的首都是华盛顿？"}
# ]
#
# template = "以下是一些示例问题和背景资料：\n{example_context}\n现在，请根据以下背景资料回答问题：\n背景资料：{documents}\n问题：{query}"
#
# self.query_pipeline.add_node(
#     component=FewShotPromptTemplate(template, examples),
#     name="FewShotTemplate",
#     inputs=["Ranker"]
# )
class FewShotPromptTemplate(BaseComponent):
    outgoing_edges = 1

    def __init__(self, template, examples):
        self.template = template
        self.examples = examples  # 新增的参数，用于存储示例

    def run(self, query=None, documents=None, history=None):
        # 在填充模板之前，先使用示例来生成一段额外的上下文
        example_context = "\n".join(
            self.template.format(**example) for example in self.examples
        )

        if documents is not None:
            documents = [i.content for i in documents]
            context = "".join(documents)
            result = {"documents": context, "query": query}
        elif history is not None:
            chat_history = "\n".join(history)
            question = query
            result = {"chat_history": chat_history, "question": question}
        else:
            raise NotImplementedError("This prompt template is not implemented!")

        # 在模板中增加示例上下文
        result["example_context"] = example_context

        return {"query": self.template.format(**result)}, "output_1"

# steps = [
#     "首先，把烤箱预热到180度。",
#     "然后，在一个碗里混合面粉、糖和鸡蛋。",
#     "慢慢加入牛奶，直到面糊变得平滑。",
#     "把面糊倒入一个蛋糕模具里。",
#     "把它放入烤箱烤25分钟。",
#     "取出蛋糕，让它冷却，然后享用。"
# ]
#
# template = "以下是制作蛋糕的步骤：\n{step_context}\n如果你有关于这个过程的问题，请问：\n问题：{query}"
#
# self.query_pipeline.add_node(
#     component=CoTPromptTemplate(template, steps),
#     name="CoTTemplate",
#     inputs=["Ranker"]
# )

class CoTPromptTemplate(BaseComponent):
    outgoing_edges = 1

    def __init__(self, template, steps):
        self.template = template
        self.steps = steps

    def run(self, query=None, documents=None, history=None):
        # 在填充模板之前，先使用步骤来生成一段额外的上下文
        step_context = "\n".join(
            f"Step {i+1}: {step}" for i, step in enumerate(self.steps)
        )

        if documents is not None:
            documents = [i.content for i in documents]
            context = "".join(documents)
            result = {"documents": context, "query": query}
        elif history is not None:
            chat_history = "\n".join(history)
            question = query
            result = {"chat_history": chat_history, "question": question}
        else:
            raise NotImplementedError("This prompt template is not implemented!")

        # 在模板中增加步骤上下文
        result["step_context"] = step_context

        return {"query": self.template.format(**result)}, "output_1"
