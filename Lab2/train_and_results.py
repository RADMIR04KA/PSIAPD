# ============================================================
# TRAINING SCRIPT FOR DOWNLOADED HOWTO100M HEALTH VIDEOS
#
# This script DOES NOT download videos.
# It uses:
#   howto100m_project/outputs/downloaded_records.csv
#   howto100m_project/videos/*.mp4
#
# Pipeline:
# 1. Load downloaded records
# 2. Filter classes with too few samples
# 3. Extract frames from videos
# 4. Extract VideoMAE embeddings
# 5. Train Logistic Regression classifier
# 6. Print metrics
# 7. Save model, features, labels, plots
# ============================================================

import os
import gc
import json
import random
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from transformers import AutoImageProcessor, VideoMAEModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    top_k_accuracy_score,
    classification_report,
)

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path("howto100m_project")
VIDEO_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "outputs"
TRAIN_OUTPUT_DIR = OUTPUT_DIR / "training_results"

DOWNLOADED_RECORDS_CSV = OUTPUT_DIR / "downloaded_records.csv"

TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "MCG-NJU/videomae-base"

NUM_FRAMES = 16
FRAME_SIZE = 160

# Classes with fewer samples than this will be removed before training.
# With your current distribution, 20 is recommended.
MIN_SAMPLES_PER_CLASS = 20

TEST_SIZE = 0.25
RANDOM_STATE = 42

FEATURES_NPY = TRAIN_OUTPUT_DIR / "howto100m_videomae_features.npy"
LABELS_NPY = TRAIN_OUTPUT_DIR / "howto100m_labels.npy"
USED_RECORDS_CSV = TRAIN_OUTPUT_DIR / "howto100m_used_records.csv"

CLASSIFIER_PKL = TRAIN_OUTPUT_DIR / "howto100m_videomae_logistic_regression.pkl"
LABEL_ENCODER_PKL = TRAIN_OUTPUT_DIR / "howto100m_label_encoder.pkl"

METRICS_JSON = TRAIN_OUTPUT_DIR / "metrics.json"
CLASS_REPORT_TXT = TRAIN_OUTPUT_DIR / "classification_report.txt"
CONFUSION_MATRIX_PNG = TRAIN_OUTPUT_DIR / "confusion_matrix.png"
ERROR_ANALYSIS_CSV = TRAIN_OUTPUT_DIR / "error_analysis.csv"

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# ============================================================
# ENVIRONMENT
# ============================================================

def check_environment():
    print("========== ENVIRONMENT ==========")
    print("Working directory:", Path.cwd().resolve())
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    else:
        print("WARNING: CUDA is not available. Training will be very slow on CPU.")

    print("Video directory:", VIDEO_DIR.resolve())
    print("Records CSV:", DOWNLOADED_RECORDS_CSV.resolve())
    print("Training outputs:", TRAIN_OUTPUT_DIR.resolve())


# ============================================================
# DATA LOADING
# ============================================================

def load_downloaded_records():
    if not DOWNLOADED_RECORDS_CSV.exists():
        raise FileNotFoundError(
            f"downloaded_records.csv not found: {DOWNLOADED_RECORDS_CSV.resolve()}"
        )

    df = pd.read_csv(DOWNLOADED_RECORDS_CSV)

    required_columns = ["video_id", "video_path", "label"]

    for col in required_columns:
        if col not in df.columns:
            raise RuntimeError(f"Required column missing in downloaded_records.csv: {col}")

    valid_rows = []

    for _, row in df.iterrows():
        video_path = Path(str(row["video_path"]))

        if not video_path.exists():
            alt_path = VIDEO_DIR / f"{row['video_id']}.mp4"
            if alt_path.exists():
                row["video_path"] = str(alt_path)
                video_path = alt_path

        if video_path.exists() and video_path.stat().st_size > 50_000:
            valid_rows.append(row)

    df = pd.DataFrame(valid_rows)

    if len(df) == 0:
        raise RuntimeError("No valid video files found.")

    print("\n========== DATASET LOADED ==========")
    print("Valid video records:", len(df))
    print("\nOriginal class distribution:")
    print(df["label"].value_counts())

    return df


def filter_classes(df):
    counts = df["label"].value_counts()
    keep_labels = counts[counts >= MIN_SAMPLES_PER_CLASS].index.tolist()

    filtered_df = df[df["label"].isin(keep_labels)].copy()
    filtered_df = filtered_df.reset_index(drop=True)

    print("\n========== CLASS FILTERING ==========")
    print("MIN_SAMPLES_PER_CLASS:", MIN_SAMPLES_PER_CLASS)
    print("Classes kept:", keep_labels)
    print("Records after filtering:", len(filtered_df))

    print("\nFiltered class distribution:")
    print(filtered_df["label"].value_counts())

    if len(keep_labels) < 2:
        raise RuntimeError(
            "Need at least 2 classes after filtering. "
            "Lower MIN_SAMPLES_PER_CLASS or download more videos."
        )

    return filtered_df


# ============================================================
# FRAME EXTRACTION
# ============================================================

def extract_frames(video_path, num_frames=16, target_size=160):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("Video has zero frames.")

    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    needed = set(frame_indices.tolist())

    frames = []
    current_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if current_index in needed:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (target_size, target_size))
            frames.append(frame)

        current_index += 1

        if len(frames) >= num_frames:
            break

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames extracted.")

    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.array(frames, dtype=np.uint8)


# ============================================================
# VIDEOMAE FEATURE EXTRACTION
# ============================================================

def load_videomae():
    print("\n========== LOADING VIDEOMAE ==========")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"

    print("Device:", device)
    print("FP16:", use_fp16)

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=False)
    model = VideoMAEModel.from_pretrained(MODEL_NAME).to(device)

    if use_fp16:
        model = model.half()

    model.eval()

    print("VideoMAE loaded:", MODEL_NAME)

    return processor, model, device, use_fp16


def get_videomae_features(video_path, processor, model, device, use_fp16):
    frames = extract_frames(
        video_path=video_path,
        num_frames=NUM_FRAMES,
        target_size=FRAME_SIZE,
    )

    inputs = processor(
        list(frames),
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    if use_fp16:
        inputs = {
            key: value.half() if torch.is_floating_point(value) else value
            for key, value in inputs.items()
        }

    with torch.inference_mode():
        outputs = model(**inputs)

    features = outputs.last_hidden_state.mean(dim=1).squeeze(0).float().cpu().numpy()

    del frames
    del inputs
    del outputs

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    return features.astype(np.float32)


def extract_all_features(df):
    print("\n========== FEATURE EXTRACTION ==========")

    processor, model, device, use_fp16 = load_videomae()

    X = []
    used_rows = []
    skipped = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_path = row["video_path"]
        label = row["label"]

        try:
            feature_vector = get_videomae_features(
                video_path=video_path,
                processor=processor,
                model=model,
                device=device,
                use_fp16=use_fp16,
            )

            X.append(feature_vector)
            used_rows.append(row.to_dict())

            if len(X) % 25 == 0:
                np.save(FEATURES_NPY, np.array(X, dtype=np.float32))
                pd.DataFrame(used_rows).to_csv(USED_RECORDS_CSV, index=False)
                print(f"\nSaved partial features: {len(X)} videos")

        except Exception as e:
            skipped.append({
                "video_path": video_path,
                "label": label,
                "reason": str(e),
            })
            print("\nSkipped:", video_path)
            print("Reason:", e)

    X = np.array(X, dtype=np.float32)
    used_df = pd.DataFrame(used_rows)

    if len(X) == 0:
        raise RuntimeError("No features extracted.")

    np.save(FEATURES_NPY, X)
    used_df.to_csv(USED_RECORDS_CSV, index=False)

    skipped_csv = TRAIN_OUTPUT_DIR / "skipped_videos.csv"
    pd.DataFrame(skipped).to_csv(skipped_csv, index=False)

    print("\n========== FEATURE EXTRACTION DONE ==========")
    print("Features shape:", X.shape)
    print("Used videos:", len(used_df))
    print("Skipped videos:", len(skipped))
    print("Saved features:", FEATURES_NPY.resolve())
    print("Saved used records:", USED_RECORDS_CSV.resolve())
    print("Saved skipped records:", skipped_csv.resolve())

    return X, used_df


# ============================================================
# TRAINING + EVALUATION
# ============================================================

def train_and_evaluate(X, used_df):
    print("\n========== TRAINING ==========")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(used_df["label"].values)

    np.save(LABELS_NPY, y)
    joblib.dump(label_encoder, LABEL_ENCODER_PKL)

    print("Classes:", list(label_encoder.classes_))

    unique_labels, label_counts = np.unique(y, return_counts=True)

    print("\nFinal class counts:")
    for label_id, count in zip(unique_labels, label_counts):
        class_name = label_encoder.inverse_transform([label_id])[0]
        print(f"{class_name}: {count}")

    can_stratify = np.all(label_counts >= 2)

    X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
        X,
        y,
        np.arange(len(y)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if can_stratify else None,
    )

    print("\nTrain X:", X_train.shape)
    print("Validation X:", X_val.shape)

    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
    )

    clf.fit(X_train, y_train)

    print("Classifier trained.")

    pred = clf.predict(X_val)
    proba = clf.predict_proba(X_val)

    accuracy = accuracy_score(y_val, pred)

    precision_macro = precision_score(y_val, pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_val, pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_val, pred, average="macro", zero_division=0)

    precision_weighted = precision_score(y_val, pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_val, pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_val, pred, average="weighted", zero_division=0)

    top2_acc = top_k_accuracy_score(
        y_val,
        proba,
        k=2,
        labels=clf.classes_,
    ) if len(clf.classes_) >= 2 else None

    top3_acc = top_k_accuracy_score(
        y_val,
        proba,
        k=3,
        labels=clf.classes_,
    ) if len(clf.classes_) >= 3 else None

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "top2_accuracy": float(top2_acc) if top2_acc is not None else None,
        "top3_accuracy": float(top3_acc) if top3_acc is not None else None,
        "num_total_samples": int(len(y)),
        "num_train_samples": int(len(y_train)),
        "num_validation_samples": int(len(y_val)),
        "classes": list(label_encoder.classes_),
    }

    print("\n========== METRICS ==========")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision macro:    {precision_macro:.4f}")
    print(f"Recall macro:       {recall_macro:.4f}")
    print(f"F1-score macro:     {f1_macro:.4f}")
    print(f"Precision weighted: {precision_weighted:.4f}")
    print(f"Recall weighted:    {recall_weighted:.4f}")
    print(f"F1-score weighted:  {f1_weighted:.4f}")

    if top2_acc is not None:
        print(f"Top-2 Accuracy:     {top2_acc:.4f}")

    if top3_acc is not None:
        print(f"Top-3 Accuracy:     {top3_acc:.4f}")

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    target_names = list(label_encoder.classes_)

    report = classification_report(
        y_val,
        pred,
        labels=clf.classes_,
        target_names=target_names,
        zero_division=0,
    )

    print("\n========== CLASSIFICATION REPORT ==========")
    print(report)

    with open(CLASS_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)

    save_confusion_matrix(y_val, pred, label_encoder, clf.classes_)

    save_error_analysis(
        used_df=used_df,
        val_idx=val_idx,
        y_val=y_val,
        pred=pred,
        proba=proba,
        label_encoder=label_encoder,
        clf=clf,
    )

    joblib.dump(clf, CLASSIFIER_PKL)

    print("\n========== SAVED FILES ==========")
    print("Classifier:", CLASSIFIER_PKL.resolve())
    print("Label encoder:", LABEL_ENCODER_PKL.resolve())
    print("Features:", FEATURES_NPY.resolve())
    print("Labels:", LABELS_NPY.resolve())
    print("Metrics:", METRICS_JSON.resolve())
    print("Classification report:", CLASS_REPORT_TXT.resolve())
    print("Confusion matrix:", CONFUSION_MATRIX_PNG.resolve())
    print("Error analysis:", ERROR_ANALYSIS_CSV.resolve())

    return clf, label_encoder


def save_confusion_matrix(y_val, pred, label_encoder, labels_present):
    target_names = [
        label_encoder.inverse_transform([i])[0]
        for i in labels_present
    ]

    cm = confusion_matrix(
        y_val,
        pred,
        labels=labels_present,
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=target_names,
        yticklabels=target_names,
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG, dpi=150)
    plt.show()

    print("Saved confusion matrix:", CONFUSION_MATRIX_PNG.resolve())


def save_error_analysis(used_df, val_idx, y_val, pred, proba, label_encoder, clf):
    rows = []

    for local_i, original_idx in enumerate(val_idx):
        true_id = y_val[local_i]
        pred_id = pred[local_i]

        true_label = label_encoder.inverse_transform([true_id])[0]
        pred_label = label_encoder.inverse_transform([pred_id])[0]

        sample_proba = proba[local_i]

        top_items = []

        for class_id, probability in zip(clf.classes_, sample_proba):
            class_name = label_encoder.inverse_transform([class_id])[0]
            top_items.append((class_name, float(probability)))

        top_items = sorted(top_items, key=lambda x: x[1], reverse=True)

        row = used_df.iloc[original_idx].to_dict()

        row.update({
            "true_label": true_label,
            "predicted_label": pred_label,
            "is_correct": bool(true_label == pred_label),
            "top1_label": top_items[0][0] if len(top_items) > 0 else None,
            "top1_probability": top_items[0][1] if len(top_items) > 0 else None,
            "top2_label": top_items[1][0] if len(top_items) > 1 else None,
            "top2_probability": top_items[1][1] if len(top_items) > 1 else None,
            "top3_label": top_items[2][0] if len(top_items) > 2 else None,
            "top3_probability": top_items[2][1] if len(top_items) > 2 else None,
        })

        rows.append(row)

    error_df = pd.DataFrame(rows)
    error_df.to_csv(ERROR_ANALYSIS_CSV, index=False)

    wrong_df = error_df[error_df["is_correct"] == False]

    print("\n========== ERROR ANALYSIS ==========")
    print("Validation samples:", len(error_df))
    print("Wrong predictions:", len(wrong_df))
    print("Correct predictions:", len(error_df) - len(wrong_df))

    if len(wrong_df) > 0:
        print("\nFirst wrong predictions:")
        cols = [
            "video_id",
            "true_label",
            "predicted_label",
            "top1_probability",
            "top2_label",
            "top2_probability",
            "video_path",
        ]

        existing_cols = [c for c in cols if c in wrong_df.columns]
        print(wrong_df[existing_cols].head(10).to_string(index=False))


# ============================================================
# MAIN
# ============================================================

def main():
    check_environment()

    df = load_downloaded_records()
    df = filter_classes(df)

    X, used_df = extract_all_features(df)

    train_and_evaluate(X, used_df)




if __name__ == "__main__":
    main()
