from typing import List
from transformers import AutoTokenizer, AutoModel
import torch


class SentencesToEmbeddings():
    def __init__(self, model_path='BAAI/bge-base-zh-v1.5') -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path).to(self.device)

    def __call__(self, sentences: List[str]):
        encoded_input = self.tokenizer(sentences, padding=True,
                                       truncation=True, return_tensors='pt')
        encoded_input = encoded_input.to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]

        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.cpu().numpy().tolist()
