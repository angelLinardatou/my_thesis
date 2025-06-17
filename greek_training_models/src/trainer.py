import torch
from transformers import Trainer, TrainingArguments

class ModelTrainer:
    """Handle HuggingFace model fine-tuning."""

    def __init__(self, model, train_dataset, eval_dataset, output_dir, epochs=3, batch_size=8):
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_dir=output_dir / "logs",
            logging_strategy="epoch",
            report_to="none"  # disables wandb etc.
        )
        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=None  # will set later if needed
        )

    def train(self):
        """Run the fine-tuning process."""
        self.trainer.train()

    def save_model(self, save_path):
        """Save the fine-tuned model."""
        self.trainer.save_model(save_path)
 
