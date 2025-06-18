import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels.long())
        return (loss, outputs) if return_outputs else loss

class ModelTrainer:
    def __init__(self, model, train_dataset, val_dataset, output_dir, epochs, batch_size):
        self.model = model
        self.output_dir = output_dir

        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )

        self.trainer = CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

    def train(self):
        self.trainer.train()
