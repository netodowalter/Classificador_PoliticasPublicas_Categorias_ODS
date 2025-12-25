# 🧩 Classificador PPA — Similaridade & ODS

Ferramenta para análise automática de frases de programas de políticas públicas:  
classificação semântica e mapeamento para Objetivos de Desenvolvimento Sustentável (ODS).

Dois classificadores independentes:

| Script | Método | Finalidade |
|--------|--------|------------|
| `classifier.py` | Similaridade semântica (Sentence Transformers) | Classificação conforme categorias definidas pelo usuário |
| `classifier_odsbahia-ptbr.py` | Modelo pré-treinado ODSBahia | Mapeamento automático de frases para ODS |

---

## 🚀 Como usar

Tudo pode ser feito executando o arquivo:

classifica.bat

markdown
Copiar código

Este comando:

1. Cria ou ativa o ambiente virtual `classificador_env`
2. Instala automaticamente: PyTorch (GPU se disponível), Transformers, Sentence-Transformers, Pandas, OpenPyXL
3. Executa o **classificador de categorias** (`classifier.py`)

Se as planilhas **`categories.xlsx`** e **`phrases.xlsx`** estiverem preenchidas, o processo roda sem configurações adicionais.

📌 Para executar o classificador de ODS manualmente:

python classifier_odsbahia-ptbr.py

yaml
Copiar código

---

## 📊 `classifier.py` — Similaridade com categorias customizadas

### Entradas necessárias

#### `categories.xlsx`  
Obrigatória, com duas colunas:

| code | description |
|------|-------------|
| SAUDE | Ações relativas ao acesso e qualidade da saúde |
| EDU | Políticas de educação, formação e aprendizagem |

📌 Boas práticas

- `code`: curto, sem espaços (`SAUDE`, `EDU`, `MEIOAMBIENTE`)  
- `description`: texto representativo da categoria  
  - excludente e exaustivo em relação às demais
  - todas no **mesmo nível de análise**
  - descreve o conceito, não lista de exemplos

#### `phrases.xlsx`  
Mesma estrutura para ambos os scripts:

| code | phrase |
|------|--------|
| P001 | Construção e ampliação de UBS em municípios vulneráveis |
| P001 | Formação de profissionais de saúde da família |
|      | Implantação de energia solar comunitária |

📌 Se `code` estiver vazio → é atribuído sequencialmente: `"1"`, `"2"`, `"3"`...

---

### Saída gerada

phrases_classified.xlsx

yaml
Copiar código

| code | SAUDE | EDU | ... | most_similar_category | maximum_similarity |
|------|-------|-----|-----|----------------------|------------------:|

- Similaridade medida por *cosine similarity*
- Se `maximum_similarity < 0.7` ➜ classifica como **None_indeterminate**

---

## 🌍 `classifier_odsbahia-ptbr.py` — Classificador ODS

Não usa `categories.xlsx`  
As 20 classes já vêm no modelo (Hugging Face)

### Entrada

`phrases.xlsx` (mesma estrutura anterior)

### Saída

phrases_classifiedODS.xlsx

markdown
Copiar código

| code | ODS1 | ODS2 | … | ODS20 | most_similar | maximum_similarity |
|------|------|------|---|-------|--------------|-------------------:|

- Classificação **multi-label** com scores contínuos (0 a 1)
- Se `maximum_similarity ≥ 0.7` → label do ODS
- Se não → `"None_indeterminate"`

📌 O modelo é baixado automaticamente na primeira execução  
Depois funciona **offline**

---

## 🧠 Modelos usados

| Classificador | Modelo | Fonte |
|--------------|--------|------|
| Categorias | `sentence-transformers/all-mpnet-base-v2` | UKP Lab / Hugging Face |
| ODS | `odsbahia/odsbahia-ptbr` | ODS Bahia / Hugging Face |

Agradecimento especial ao time responsável pelo modelo **ODS Bahia** 💚  
🔗 https://huggingface.co/odsbahia/odsbahia-ptbr

---

## 📦 Dependências

Instaladas automaticamente via `.bat`:

- Python 3.10+
- Torch (CUDA se disponível)
- Transformers
- Sentence-Transformers
- Pandas
- OpenPyXL

`requirements.txt` mínimo:

```txt
pandas>=1.3
sentence-transformers>=2.2
openpyxl>=3.0
transformers>=4.38
📌 Torch não deve estar no requirements para evitar instalação errada (CPU only)

📂 Estrutura sugerida do repositório
Copiar código
.
├── classifier.py
├── classifier_odsbahia-ptbr.py
├── classifica.bat
├── requirements.txt
├── phrases.xlsx
├── categories.xlsx
├── phrases_classified.xlsx
└── phrases_classifiedODS.xlsx
🛣️ Roadmap
Agregação automática por programa (via padrão em code)

Dashboards de visualização (ODS / categorias / distribuição)

Modo offline completo (modelo local)

Testes automatizados e logs mais detalhados

✍️ Autor
Walter Desiderá
Pesquisador da Diretoria de Estudos Internacionais do IPEA

🤝 Contribuições
Sugestões, melhorias e correções são muito bem-vindas!
Abra uma issue ou envie um pull request 🙌

