import json


def generate_prompt(retrieved_result):
    prompt = "鉴于检索到的档内容，请生成以下问题的简洁准确的答案 " + \
        f"\n 检索到的文档内容: {retrieved_result}"
    return prompt
