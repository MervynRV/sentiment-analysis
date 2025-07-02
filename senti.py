import pandas as pd
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset, random_split
from sklearn.metrics import accuracy_score, f1_score
import logging
from typing import Tuple, Dict, List
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiasDataset(Dataset):
    """Custom dataset class for handling text classification data."""
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_dataset(filepath: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load and preprocess dataset."""
    try:
        df = pd.read_csv(filepath)
        print("Dataset columns:", df.columns)

        # Ensure correct column names
        df = df.rename(columns={'Feedback Text': 'text', 'Bias Indicator': 'label'})

        if 'text' not in df.columns or 'label' not in df.columns:
            raise KeyError("Dataset must contain 'text' and 'label' columns.")

        # Clean and prepare data
        df = df.dropna()
        df['label'] = df['label'].astype(str)
        unique_labels = df['label'].unique().tolist()
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        df['label'] = df['label'].map(label_map)

        return df, label_map

    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def compute_metrics(pred) -> Dict[str, float]:
    """Compute evaluation metrics."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted')
    }

def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    num_labels: int
) -> Tuple[BertForSequenceClassification, Trainer]:
    """Train the model."""
    try:
        # Initialize model
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_labels
        )

        # Training configuration
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            report_to='none',
            remove_unused_columns=False
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        # Train model
        trainer.train()

        return model, trainer

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def predict_bias(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    text: str,
    label_map: Dict[str, int]
) -> str:
    """Predict bias for given text."""
    try:
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = torch.argmax(outputs.logits, dim=-1).item()
        reverse_label_map = {v: k for k, v in label_map.items()}
        return reverse_label_map[prediction]

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return "error"

def main():
    """Main execution function."""
    try:
        # Configuration
        filepath = '/content/feedback.csv'

        # Initialize tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load and preprocess data
        logger.info("Loading dataset...")
        df, label_map = load_dataset(filepath)

        # Create dataset
        dataset = BiasDataset(
            texts=df['text'].tolist(),
            labels=df['label'].tolist(),
            tokenizer=tokenizer
        )

        # Split dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size]
        )

        # Train model
        logger.info("Training model...")
        model, trainer = train_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_labels=len(label_map)
        )

        # Interactive prediction
        logger.info("Entering prediction mode. Type 'exit' to quit.")
        while True:
            try:
                user_input = input("\nEnter text to analyze: ").strip()

                if user_input.lower() == 'exit':
                    break

                if not user_input:
                    print("Please enter some text.")
                    continue

                prediction = predict_bias(model, tokenizer, user_input, label_map)
                print(f"Detected bias type: {prediction}")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error during interaction: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        return 1

    return 0

if __name__ == '__main__':
    main()
