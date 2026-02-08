#!/usr/bin/env python
# -*- coding: utf-8 -*-

# classifier_odsbahia-ptbr.py
# --- lê phrases.xlsx, classifica com o modelo ODSBahia e gera phrases_classifiedODS.xlsx ---

import argparse
import os
from typing import List

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "odsbahia/odsbahia-ptbr"
DEFAULT_MAX_LEN = 128
DEFAULT_BATCH_SIZE = 32
DEFAULT_CODE_COL = "code"
DEFAULT_TEXT_COL = "phrase"
DEFAULT_INPUT = "phrases.xlsx"
DEFAULT_OUTPUT = "phrases_classifiedODS.xlsx"

THRESHOLD = 0.25  # usado para preencher 1/2/3_most_similar; abaixo disso => None_indeterminate


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def classify_batch(texts: List[str], tokenizer, model, device, max_len: int):
    if not texts:
        return []

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        # multi-label -> sigmoid por classe
        probs = torch.sigmoid(logits)

    return probs.detach().cpu().numpy().tolist()


def run(input_path, code_column, text_column, output_path, batch_size, max_len):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

    print(f"Lendo: {input_path}")
    df = pd.read_excel(input_path)

    for col in (code_column, text_column):
        if col not in df.columns:
            raise ValueError(
                f"Coluna '{col}' não existe. Colunas disponíveis: {list(df.columns)}"
            )

    # ==============================
    # GERAR code SEQUENCIAL QUANDO VIER EM BRANCO
    # ==============================
    raw_codes = df[code_column].astype(str).tolist()
    codes = []
    counter = 1
    for c in raw_codes:
        c_strip = c.strip()
        if c_strip == "" or c_strip.lower() == "nan":
            codes.append(str(counter))
            counter += 1
        else:
            codes.append(c_strip)

    texts = df[text_column].fillna("").astype(str).tolist()

    tokenizer, model, device = load_model()
    id2label = model.config.id2label
    num_labels = model.config.num_labels

    # tenta primeiro índice int; se der KeyError, cai pra string
    try:
        label_order = [id2label[i] for i in range(num_labels)]
    except KeyError:
        label_order = [id2label[str(i)] for i in range(num_labels)]

    print("Labels do modelo na ordem dos logits:")
    print(label_order)

    print(f"Classificando {len(texts)} phrases...")
    all_scores = []

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        all_scores.extend(
            classify_batch(batch, tokenizer, model, device, max_len)
        )
        print(f"  {end}/{len(texts)}")

    # DataFrame com scores por label (20 ODS)
    scores_df = pd.DataFrame(all_scores, columns=label_order)

    # ===== Top-3 com threshold =====
    top1, top2, top3 = [], [], []

    for row in scores_df.values:
        # ordem decrescente
        order = row.argsort()[::-1]

        def pick(k: int) -> str:
            idx = int(order[k])
            score = float(row[idx])
            return label_order[idx] if score > THRESHOLD else "None_indeterminate"

        top1.append(pick(0))
        top2.append(pick(1))
        top3.append(pick(2))

    # ===== Monta output no formato desejado =====
    out_df = pd.DataFrame()
    out_df[code_column] = codes

    # colunas ODS (20)
    for col in label_order:
        out_df[col] = scores_df[col]

    # colunas finais (igual aos outros)
    out_df["1_most_similar"] = top1
    out_df["2_most_similar"] = top2
    out_df["3_most_similar"] = top3

    out_path = output_path if output_path else DEFAULT_OUTPUT
    print(f"Salvando -> {out_path}")
    out_df.to_excel(out_path, index=False)
    print("Done.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--code-column", default=DEFAULT_CODE_COL)
    p.add_argument("--text-column", default=DEFAULT_TEXT_COL)
    p.add_argument("--output")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.input, a.code_column, a.text_column, a.output, a.batch_size, a.max_len)
