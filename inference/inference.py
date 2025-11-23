import torch
import argparse
import yaml
import pandas as pd
from typing import Union, List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import warnings

warnings.filterwarnings(action="ignore")


class SpamClassifier:
    def __init__(self, config_path: str = "inference_config.yaml"):
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def _setup_device(self) -> str:
        device_config = self.config["model"]["device"]

        if device_config == "cuda" and torch.cuda.is_available():
            device = "cuda"
            if self.config["output"]["verbose"]:
                print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif device_config == "mps" and torch.backends.mps.is_available():
            device = "mps"
            if self.config["output"]["verbose"]:
                print("Using Apple Silicon GPU (MPS)")
        else:
            device = "cpu"
            if self.config["output"]["verbose"]:
                print("Using CPU")

        return device

    def _load_model(self):
        base_model_name = self.config["model"]["base_model_name"]
        adapter_path = self.config["model"]["adapter_path"]

        if self.config["output"]["verbose"]:
            print(f"\nLoading model components...")
            print(f"Base model: {base_model_name}")
            print(f"LoRA adapter: {adapter_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=2, problem_type="single_label_classification"
        )

        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.to(self.device)
        self.model.eval()

        if self.config["output"]["verbose"]:
            print("Model loaded successfully!\n")

    def _prepare_input(self, text: str) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.config["model"]["max_length"],
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
        }

    def predict_single(self, text: str) -> Dict[str, Union[str, float]]:
        inputs = self._prepare_input(text)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[prediction].item()

        label_map = self.config["output"]["labels"]
        result = {
            "prediction": prediction,
            "label": label_map[str(prediction)],
            "confidence": confidence,
        }

        if self.config["inference"]["return_probabilities"]:
            result["probabilities"] = {
                label_map["0"]: probabilities[0].item(),
                label_map["1"]: probabilities[1].item(),
            }

        return result

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        results = []
        batch_size = self.config["inference"]["batch_size"]

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_results = [self.predict_single(text) for text in batch_texts]
            results.extend(batch_results)

            if self.config["output"]["verbose"]:
                print(
                    f"Processed {min(i + batch_size, len(texts))}/{len(texts)} emails",
                    end="\r",
                )

        if self.config["output"]["verbose"]:
            print()

        return results

    def predict_from_file(
        self, file_path: str, text_column: str = "text"
    ) -> pd.DataFrame:
        df = pd.read_csv(file_path)

        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found in CSV. Available columns: {df.columns.tolist()}"
            )

        texts = df[text_column].astype(str).tolist()
        predictions = self.predict_batch(texts)

        df["predicted_label"] = [p["label"] for p in predictions]
        df["confidence"] = [p["confidence"] for p in predictions]

        if self.config["inference"]["return_probabilities"]:
            df["prob_ham"] = [p["probabilities"]["HAM"] for p in predictions]
            df["prob_spam"] = [p["probabilities"]["SPAM"] for p in predictions]

        return df


def main():
    parser = argparse.ArgumentParser(
        description="Spam Email Classification using LoRA Fine-tuned Transformers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
              # Predict single email
              python inference.py --text "Subject: Win $1000 now! Click here!"
              
              # Predict from file
              python inference.py --input_file emails.csv
              
              # Use custom config
              python inference.py --config my_config.yaml --text "Your email"  
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="inference_config.yaml",
        help="Path to inference configuration file",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Single email text to classify")
    input_group.add_argument(
        "--input_file", type=str, help="Path to CSV file with emails"
    )
    parser.add_argument("--base_model", type=str, help="Override base model name")
    parser.add_argument("--adapter_path", type=str, help="Override adapter path")
    parser.add_argument(
        "--output_file", type=str, help="Path to save predictions (CSV format)"
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column name for email text in CSV (default: text)",
    )
    args = parser.parse_args()

    classifier = SpamClassifier(config_path=args.config)

    if args.base_model:
        classifier.config["model"]["base_model_name"] = args.base_model
    if args.adapter_path:
        classifier.config["model"]["adapter_path"] = args.adapter_path
        classifier._load_model()

    if args.text:
        print("=" * 60)
        print("SPAM EMAIL CLASSIFICATION")
        print("=" * 60)
        print(f"\nInput Email:\n{args.text}\n")

        result = classifier.predict_single(args.text)

        print("-" * 60)
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")

        if "probabilities" in result:
            print(f"\nProbability Breakdown:")
            for label, prob in result["probabilities"].items():
                print(f"{label}: {prob:.2%}")

        print("=" * 60)

    elif args.input_file:
        print(f"\nLoading emails from: {args.input_file}")
        results_df = classifier.predict_from_file(
            args.input_file, text_column=args.text_column
        )
        print(f"\nProcessed {len(results_df)} emails")
        print(f"\nClassification Summary:")
        print(results_df["predicted_label"].value_counts())
        print(f"\nAverage Confidence: {results_df['confidence'].mean():.2%}")

        if args.output_file:
            results_df.to_csv(args.output_file, index=False)
            print(f"\nResults saved to: {args.output_file}")
        else:
            print(f"\nSample Predictions:")
            display_cols = ["predicted_label", "confidence"]
            if "prob_ham" in results_df.columns:
                display_cols.extend(["prob_ham", "prob_spam"])
            print(results_df[display_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
