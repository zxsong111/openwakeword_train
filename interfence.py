import os
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import openwakeword
from tqdm import tqdm
import soundfile as sf
import shutil
import uuid
from datetime import datetime


def get_wav_files(directory: str) -> List[str]:
    """Return a list of .wav files in a directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".wav")]


def compute_basic_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """Compute correct, incorrect counts and accuracy."""
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    incorrect = len(y_true) - correct
    accuracy = correct / len(y_true) if y_true else 0.0
    return {"correct": correct, "incorrect": incorrect, "accuracy": accuracy}


def compute_full_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Compute full classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }


def predict_samples(oww_model: openwakeword.Model, files: List[str], model_name: str, label: int, threshold: float,error_dir: str, verbose: bool = True):
    """Predict a list of samples and return true and predicted labels."""
    y_true, y_pred = [], []
    for f in tqdm(files, desc=f"Predicting {'positive' if label else 'negative'} samples", disable=not verbose):
        result = oww_model.predict_clip(f)
        pred = 0
        for frame in result:
            score = frame[model_name]
            if score > threshold:
                pred = 1
                break
        y_true.append(label)
        y_pred.append(pred)
        # 如果预测错误，复制文件到错误文件夹
        if pred != label:
            target_subdir = "false_negative" if label == 1 else "false_positive"
            target_dir = os.path.join(error_dir, target_subdir)
            
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy(f, target_dir)

    return y_true, y_pred


def evaluate_wakeword_model(model_path: str,
                            pos_dir: str,
                            neg_dir: str,
                            threshold: float = 0.5,
                            error_output_dir: str = "./error_samples",
                            inference_framework:str="tflite",
                            verbose: bool = True) -> Dict:
    """Evaluate a wakeword model with simplified metrics for positive/negative samples and full metrics overall."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    if model_path.split('.')[-1] != inference_framework:
        raise ValueError(f"Model extension does not match the specified inference framework: {inference_framework}")
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    oww_model = openwakeword.Model(wakeword_models=[model_path])

    pos_files = get_wav_files(pos_dir)
    neg_files = get_wav_files(neg_dir)

    if verbose:
        print(f"\n🔍 Evaluating model: {model_name}")
        print(f"Positive samples: {len(pos_files)} | Negative samples: {len(neg_files)} | Threshold: {threshold}")

    pos_true, pos_pred = predict_samples(oww_model, pos_files, model_name, label=1, threshold=threshold, error_dir=os.path.join(error_output_dir, "positive"), verbose=verbose)
    neg_true, neg_pred = predict_samples(oww_model, neg_files, model_name, label=0, threshold=threshold, error_dir=os.path.join(error_output_dir, "negative"), verbose=verbose)

    overall_true = pos_true + neg_true
    overall_pred = pos_pred + neg_pred

    metrics = {
        "model": model_name,
        "threshold": threshold,
        "positive": compute_basic_metrics(pos_true, pos_pred),
        "negative": compute_basic_metrics(neg_true, neg_pred),
        "overall": compute_full_metrics(overall_true, overall_pred)
    }

    if verbose:
        print("\n===== 📊 Evaluation Report =====")
        for category in ["positive", "negative"]:
            print(f"\n--- {category.capitalize()} Samples ---")
            for k, v in metrics[category].items():
                if isinstance(v, float):
                    print(f"{k.capitalize():<10}: {v:.4f}")
                else:
                    print(f"{k.capitalize():<10}: {v}")

        print(f"\n--- Overall Samples ---")
        for k, v in metrics["overall"].items():
            print(f"{k.capitalize():<10}: {v:.4f}")

    return metrics






if __name__ == "__main__":
    pos_dataset_dir = "./dataset/hi_aldelo_pcm/positive_test"
    neg_dataset_dir = "./cartesia_hi_adela_test"
    model_path="./web/model/rnn/hi_aldelo_15000_adela.tflite"
    # 生成当前时间字符串，例如 2025-10-24_15-32-45
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    error_output_dir=f"./dataset/interfence/error_{timestamp}"
    result = evaluate_wakeword_model(
        model_path=model_path,
        inference_framework=model_path.split('.')[-1],
        pos_dir=pos_dataset_dir,
        neg_dir=neg_dataset_dir,
        threshold=0.5,
        error_output_dir=error_output_dir,
    )

    print("\n===== ✅ Final Summary =====")
    print(result)