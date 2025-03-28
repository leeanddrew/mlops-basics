import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
import wandb
from transformers import AutoModel,AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2",lr=1e-2):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()
        self.validation_step_outputs = {"labels":[],"logits":[]}
        
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=2)
        self.num_classes = 2

        #metrics to log
        self.train_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.val_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.f1_metric = torchmetrics.F1Score(task="binary",num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            task="binary",average="macro",num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            task="binary",average="macro",num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(task="binary",average="micro")
        self.recall_micro_metric = torchmetrics.Precision(task="binary",average="micro")
        
    def forward(self, input_ids, attention_mask, labels=None):
        device = next(self.bert.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask, labels=labels)
        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch["input_ids"],batch["attention_mask"],labels=batch['label'])
        #loss = F.cross_entropy(logits, batch["label"])
        preds = torch.argmax(outputs.logits,1)
        train_acc = self.train_accuracy_metric(preds,batch['label'])
        self.log("train/loss",outputs.loss,prog_bar=True,on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss
    
    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        outputs = self.forward(batch["input_ids"], batch["attention_mask"], labels=batch["label"])
        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True)
        self.log("valid/f1", f1, prog_bar=True)

        self.validation_step_outputs["labels"].append(labels)
        self.validation_step_outputs["logits"].append(labels)

        return {"labels": labels, "logits": outputs.logits}
    
    def on_validation_epoch_end(self):
        labels = torch.cat(self.validation_step_outputs["labels"],dim=0)
        logits = torch.cat(self.validation_step_outputs["logits"],dim=0)
        preds = torch.argmax(logits)

        self.logger.experiment.log(
            {
                "conf":wandb.plot.confusion_matrix(
                    preds=logits.cpu().numpy(), y_true=labels.cpu().numpy()
                )
            }
        )
        self.validation_step_outputs = {"labels":[],"logits":[]} # free memory

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])