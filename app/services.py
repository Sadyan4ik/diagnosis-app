import torch
import pickle
from model import RNNModel
from utils import clean_text
import torchtext

class DiagnosisService:
    def __init__(self, model_path: str, vocab_path: str, diagnoses: list, max_words: int, device: str):
        self.max_words = max_words
        self.device = device
        self.diagnoses = diagnoses

        # Загрузка ресурсов
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        self.model = RNNModel(
            vocab_size=len(self.vocab),
            embedding_dim=256,
            hidden_dim=128,
            num_classes=len(diagnoses),
            drop_prob=0.4,
            bidir=True,
            seq="gru"
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        # Мапа советов (можно расширить)
        self.advice_map = {d: "Тут должен быть совет для " + d for d in diagnoses}

    def predict_top3(self, text: str):
        words = clean_text(text)
        unk_idx = self.vocab['<unk>']
        pad_idx = self.vocab['<pad>']

        # Конвертация и паддинг
        indices = [self.vocab[w] if w in self.vocab else unk_idx for w in words]
        indices = (indices + [pad_idx] * self.max_words)[:self.max_words]

        input_tensor = torch.tensor(indices).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            top3_probs, top3_indices = torch.topk(probs, 3)

        results = []
        for i in range(3):
            idx = top3_indices[0][i].item()
            name = self.diagnoses[idx]
            results.append({
                "name": name,
                "prob": f"{top3_probs[0][i].item():.2%}",
                "advice": self.advice_map.get(name, "Соблюдайте режим покоя.")
            })
        return results