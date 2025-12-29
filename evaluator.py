import torch
from bert_score import score
from sentence_transformers import SentenceTransformer


class Evaluator:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        embedding_model_type: str = "BAAI/bge-m3",
        bert_model_type: str = "microsoft/deberta-v3-large",
    ):
        self.embedding_model = SentenceTransformer(
            embedding_model_type, trust_remote_code=True, device=device
        )
        self.bert_model_type = bert_model_type

    def get_embedding(self, text: str) -> torch.Tensor:
        return self.embedding_model.encode(text, convert_to_tensor=True)

    def compare_outputs(self, output_prompted: str, output_steered: str) -> dict:
        """Compare the outputs of the prompted and steered models in terms of BERTScore and cosine similarity."""
        vec_prompted = self.get_embedding(output_prompted)
        vec_steered = self.get_embedding(output_steered)

        similarity_score = torch.nn.functional.cosine_similarity(
            vec_prompted.unsqueeze(0), vec_steered.unsqueeze(0)
        ).item()

        P, R, F1 = score(
            [output_steered],
            [output_prompted],
            model_type=self.bert_model_type,
            verbose=False,
        )

        l2_distance = torch.norm(vec_prompted - vec_steered).item()

        return {
            "cosine_similarity": similarity_score,
            "l2_distance": l2_distance,
            "bert_f1": F1.item(),
        }
