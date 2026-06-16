# submission.py
import os
import torch
import pandas as pd
from sentiment_analysis_experiments import TextDataset, collate_fn, AutoTokenizer, AutoModelForSequenceClassification, predict_test

def main():
    # ===================== 默认参数 =====================
    train_csv = "train.csv"
    test_csv = "test.csv"
    sample_csv = "sample_submission.csv"
    out_csv = "submission.csv"
    model_name = "roberta-base"
    batch_size = 16
    max_len = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型路径
    save_path = f"./saved_models/{model_name}_best.pt"
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Best model not found: {save_path}")

    # ===================== 加载测试集 =====================
    df_test = pd.read_csv(test_csv)
    df_sample = pd.read_csv(sample_csv)
    test_texts = df_test["content"].fillna("").tolist()

    test_ds = TextDataset(test_texts)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collate = lambda b: collate_fn(b, tokenizer, max_len)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # ===================== 加载模型 =====================
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)

    # ===================== 预测 =====================
    preds = predict_test(model, test_loader, device)

    # ===================== 保存 submission =====================
    submission = pd.DataFrame({"id": df_test["id"], "label": preds})
    submission.to_csv(out_csv, index=False)
    print(f"Submission saved to {out_csv}")

    # ===================== 测试准确率 =====================
    df_pred = submission.sort_values("id")
    df_true = df_sample.sort_values("id")
    from sklearn.metrics import accuracy_score
    test_acc = accuracy_score(df_true["label"], df_pred["label"])
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
