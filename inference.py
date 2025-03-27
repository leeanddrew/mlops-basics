import torch
from model import ColaModel
from data import DataModule

class ColaPredictor:
    def __init__(self, model_path):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path).to(self.mps_device)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.lables = ["unacceptable", "acceptable"]

    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        input_ids = torch.tensor([processed["input_ids"]]).to(self.mps_device)
        attention_mask = torch.tensor([processed["attention_mask"]]).to(self.mps_device)

        logits = self.model(input_ids, attention_mask)
        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions

if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaPredictor("./models/epoch=0-step=268.ckpt")
    print(predictor.predict(sentence))