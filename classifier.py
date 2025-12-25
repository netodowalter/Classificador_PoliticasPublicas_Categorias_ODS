import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)

categories_df = pd.read_excel("categories.xlsx")
categories_codes = categories_df['code'].tolist()
categories_description = categories_df['description'].tolist()
categories_embeddings = model.encode(categories_description, convert_to_tensor=True, device=device)

df = pd.read_excel("phrases.xlsx")

if "code" not in df.columns:
    raise ValueError("A planilha precisa ter a coluna 'code'.")

codes = df["code"].astype(str).tolist()
new_codes = []
counter = 1

for c in codes:
    c_strip = c.strip()
    if c_strip == "" or c_strip.lower() == "nan":
        new_codes.append(str(counter))
        counter += 1
    else:
        new_codes.append(c_strip)

codes = new_codes

phrases = df["phrase"].fillna("").astype(str).tolist()

resultados = []

for code, phrase in zip(codes, phrases):
    phrase_embedding = model.encode(phrase, convert_to_tensor=True, device=device)
    similaridades = util.pytorch_cos_sim(phrase_embedding, categories_embeddings)[0]
    
    similaridade_dict = {
        categories_codes[i]: round(similaridades[i].item(), 4)
        for i in range(len(categories_codes))
    }
    
    max_category = max(similaridade_dict, key=similaridade_dict.get)
    max_value = similaridade_dict[max_category]
    
    categories_classificado = (
        max_category if max_value >= 0.7 else "None_indeterminate"
    )
    
    resultados.append({
        "code": code,
        **similaridade_dict,
        "most_similar": categories_classificado,
        "maximum_similarity": max_value
    })

df_resultado = pd.DataFrame(resultados)
df_resultado.to_excel("phrases_classified.xlsx", index=False)
print("Classification completed! Results saved in the same folder")
