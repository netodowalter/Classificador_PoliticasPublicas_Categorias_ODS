import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "sadickam/sdgBERT"   # SDG 1–16
INPUT_FILE = "phrases.xlsx"
OUTPUT_FILE = "phrases_classifiedSDG_EN.xlsx"

TEXT_COLUMN = "phrase"
CODE_COLUMN = "code"

MAX_LEN = 256
THRESHOLD = 0.25

SDG_LABELS = [f"SDG {i}" for i in range(1, 17)]


def main():

    df = pd.read_excel(INPUT_FILE)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Coluna '{TEXT_COLUMN}' não encontrada.")

    if CODE_COLUMN not in df.columns:
        raise ValueError(f"Coluna '{CODE_COLUMN}' não encontrada.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    all_scores = []

    with torch.no_grad():

        for text in df[TEXT_COLUMN].fillna("").astype(str).tolist():

            inputs = tokenizer(
                text,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).detach().cpu().numpy()

            all_scores.append(probs)

    scores_df = pd.DataFrame(all_scores, columns=SDG_LABELS)

    top1 = []
    top2 = []
    top3 = []

    for row in scores_df.values:

        order = row.argsort()[::-1]

        # 1º
        if row[order[0]] > THRESHOLD:
            top1.append(SDG_LABELS[order[0]])
        else:
            top1.append("None_indeterminate")

        # 2º
        if row[order[1]] > THRESHOLD:
            top2.append(SDG_LABELS[order[1]])
        else:
            top2.append("None_indeterminate")

        # 3º
        if row[order[2]] > THRESHOLD:
            top3.append(SDG_LABELS[order[2]])
        else:
            top3.append("None_indeterminate")

    out_df = pd.DataFrame()
    out_df["code"] = df[CODE_COLUMN]

    # SDG 1..16
    for c in SDG_LABELS:
        out_df[c] = scores_df[c]

    out_df["1_most_similar"] = top1
    out_df["2_most_similar"] = top2
    out_df["3_most_similar"] = top3

    out_df.to_excel(OUTPUT_FILE, index=False)

    print("Arquivo gerado:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
