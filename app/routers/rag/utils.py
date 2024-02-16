def generate_prompt(retrieved_result,question):
    prompt = "根据检索到的档内容作为事实依据，仅在检索到的内容与用户问题相关的情况下,回答用户问题 当问题与检索的内容不相关，请回答检索的内容不相关" + \
                f"\n 用户问题: {question}" + \
                f"\n 检索到的文档内容: {retrieved_result}"
    return prompt
