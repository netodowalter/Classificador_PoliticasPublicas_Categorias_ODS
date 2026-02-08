import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "jonas/bert-base-uncased-finetuned-sdg-Mar23"
INPUT_FILE = "phrases.xlsx"
OUTPUT_FILE = "phrases_classifiedSDG_EN_jonas.xlsx"

TEXT_COLUMN = "phrase"
CODE_COLUMN = "code"

MAX_LEN = 256
THRESHOLD = 0.25  # sugestão inicial (ajuste conforme sua base)

# O modelo tem 16 labels, com esta rotulagem declarada no card (índice -> SDG):
# 0:'1', 1:'10', 2:'11', 3:'12', 4:'13', 5:'14', 6:'15', 7:'16',
# 8:'2', 9:'3', 10:'4', 11:'5', 12:'6', 13:'7', 14:'8', 15:'9'
# Fonte: model card. :contentReference[oaicite:1]{index=1}
ID_TO_SDGNUM = {
    0: 1,  1: 10, 2: 11, 3: 12, 4: 13, 5: 14, 6: 15, 7: 16,
    8: 2,  9: 3, 10: 4, 11: 5, 12: 6, 13: 7, 14: 8, 15: 9
}

SDG_COLS = [f"SDG {i}" for i in range(1, 17)]  # SDG 1..SDG 16


def main():
    df = pd.read_excel(INPUT_FILE)

    if CODE_COLUMN not in df.columns:
        raise ValueError(f"Coluna '{CODE_COLUMN}' não encontrada em {INPUT_FILE}.")
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Coluna '{TEXT_COLUMN}' não encontrada em {INPUT_FILE}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    # vamos construir uma lista de dicts: { "SDG 1": score, ..., "SDG 16": score }
    rows_scores = []
    top1_list, top2_list, top3_list = [], [], []

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
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).detach().cpu().numpy()  # len=16

            # mapeia probs (index -> SDG num -> coluna)
            score_map = {col: 0.0 for col in SDG_COLS}
            for idx, p in enumerate(probs):
                sdgnum = ID_TO_SDGNUM[idx]
                score_map[f"SDG {sdgnum}"] = float(p)

            rows_scores.append(score_map)

            # ranking top-3 sobre as colunas SDG 1..16
            ordered = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)

            def pick_or_none(k: int) -> str:
                label, score = ordered[k]
                return label if score > THRESHOLD else "None_indeterminate"

            top1_list.append(pick_or_none(0))
            top2_list.append(pick_or_none(1))
            top3_list.append(pick_or_none(2))

    scores_df = pd.DataFrame(rows_scores, columns=SDG_COLS)

    out_df = pd.DataFrame()
    out_df["code"] = df[CODE_COLUMN]

    # SDG 1..16
    for c in SDG_COLS:
        out_df[c] = scores_df[c]

    out_df["1_most_similar"] = top1_list
    out_df["2_most_similar"] = top2_list
    out_df["3_most_similar"] = top3_list

    out_df.to_excel(OUTPUT_FILE, index=False)
    print("Arquivo gerado:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
