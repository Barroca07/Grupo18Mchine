# 🎓 Grupo 18 – Machine Learning
## Evasão no ensino superior: quem desiste do curso?

Projeto de Machine Learning que identifica se um estudante do ensino superior vai **desistir do curso** ou **se formar**, usando características pessoais, socioeconômicas e acadêmicas.

---

## 📌 Sobre o projeto

O objetivo é desenvolver modelos de classificação capazes de prever a evasão com base em informações como:

- Idade na matrícula, estado civil, sexo e migração;
- Bolsa de estudos, dívidas e mensalidades em dia;
- Curso, período (diurno ou noturno) e qualificação prévia;
- Disciplinas matriculadas, avaliadas e aprovadas no 1º e 2º semestres;
- Notas do 1º e 2º semestres;
- Indicadores econômicos (desemprego, inflação e PIB).

**Tipo de problema:** classificação binária.

**Variável alvo:** `Evasao` (Desistente ou Graduado).

**Quantidade de registros:** 4.424 alunos na base original; 3.630 usados no modelo (2.209 graduados e 1.421 desistentes). Os 794 alunos "Matriculado" ficam de fora porque ainda não têm desfecho.

---

## 🧭 Roteiro da apresentação

| # | Pergunta | Resposta |
|---|----------|----------|
| 1 | Qual o dataset? | Evasão no ensino superior (Portugal, 2021): 4.424 alunos e 28 colunas |
| 2 | Análise exploratória | 39% desistem; quem desiste é mais velho e tem notas menores |
| 3 | Pré-processamento | Mediana ou moda, padronização, One-Hot e divisão 70/30 estratificada |
| 4 | O que querem prever? | Se o aluno vai desistir ou se formar |
| 5 | Modelo escolhido | Regressão Logística |
| 6 | Métrica de avaliação | Acurácia, com precision, recall, F1 e matriz de confusão |

---

## 🛠️ Tecnologias utilizadas

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📊 Etapas do projeto

1. Carregamento e descrição do dataset;
2. Análise exploratória dos dados (EDA) com gráficos e estatísticas por grupo;
3. Pré-processamento com `Pipeline` e `ColumnTransformer`;
4. Divisão entre treinamento e teste (70% / 30%, estratificada);
5. Treinamento dos modelos;
6. Avaliação dos resultados;
7. Validação cruzada (5 partes);
8. Comparação de diferentes valores de `random_state`.

### Pré-processamento

- Seleção: apenas alunos Graduado ou Desistente;
- Valores ausentes: mediana nas numéricas e moda nas categóricas (a base não tem ausentes, mas o pipeline os trata por segurança);
- Variáveis numéricas (16): padronização com `StandardScaler`;
- Variáveis categóricas (11): `OneHotEncoder`;
- Divisão: 2.541 alunos para treino e 1.089 para teste, com `random_state=42`.

---

## 🤖 Modelos utilizados

**Regressão Logística** (modelo escolhido)
Estima a probabilidade de o aluno desistir. Teve a maior acurácia no teste e na validação cruzada.

**Árvore de Decisão** (`max_depth=4`)
Aprende regras de classificação a partir dos dados e serve como comparação.

---

## 📈 Avaliação

Métrica principal: **acurácia**. Como "Desistente" é a classe minoritária, também são avaliados precision, recall e F1-score dela, além da matriz de confusão.

### Resultados (`random_state=42`)

| Modelo | Acurácia | Precision | Recall | F1 |
|--------|----------|-----------|--------|----|
| Regressão Logística | 91,00% | 0,93 | 0,83 | 0,88 |
| Árvore de Decisão | 88,98% | 0,92 | 0,78 | 0,85 |
| Sempre prever "Graduado" (referência) | 60,85% | – | – | – |

Precision, recall e F1 são da classe Desistente.

### Estabilidade

| Modelo | Validação cruzada (5 partes) | 8 valores de `random_state` (média, mínima e máxima) |
|--------|------------------------------|------------------------------------------------------|
| Regressão Logística | 91,60% | 91,33% (90,45% a 92,19%) |
| Árvore de Decisão | 88,07% | 88,03% (87,24% a 88,98%) |

Valores de `random_state` testados: 1, 10, 20, 42, 50, 100, 123 e 200.

---

## 📦 Instalação das dependências

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## ▶️ Como executar

Clone o repositório:

```bash
git clone https://github.com/Barroca07/Grupo18Machine.git
```

Entre na pasta:

```bash
cd Grupo18Machine
```

Execute o projeto:

```bash
python Trabalho_Evasao.py
```

No Windows, também pode utilizar:

```bash
py Trabalho_Evasao.py
```

Para salvar os gráficos na pasta `figs` (além de exibi-los), defina a variável `SALVAR_FIGURAS=1` antes de executar:

```bash
SALVAR_FIGURAS=1 python Trabalho_Evasao.py
```

> Observação: é necessário ter Python instalado e conexão com a internet para carregar a base de dados.

---

## 🗂️ Estrutura do projeto

```text
Grupo18Machine/
│
├── Trabalho_Evasao.py
│
└── README.md
```

---

## 📚 Fonte dos dados

**Predict Students' Dropout and Academic Success**, de Realinho, Vieira Martins, Machado e Baptista (2021), Instituto Politécnico de Portalegre, Portugal. UCI Machine Learning Repository, licença CC BY 4.0.

Versão em português usada no curso de classificação da Alura:
`https://raw.githubusercontent.com/alura-cursos/classificacao_multiclasse/main/Dados/dados_estudantes.csv`

---

## ⚠️ Observações

- As variáveis do 1º e 2º semestres só existem depois que o aluno cursou esses períodos. O modelo funciona como um **alerta de evasão após o 1º ou 2º semestre**, e não como previsão feita na matrícula.
- Os alunos "Matriculado" foram excluídos. Um próximo passo seria incluí-los em um modelo de três classes.
- Versão anterior do trabalho: "Alunos da UNIRIO: quem trabalha?", com a base QE.csv do DATAUNIRIO (95 estudantes). Por ser uma base pequena, seus resultados variavam bastante conforme a divisão dos dados.

---

Projeto acadêmico – Grupo 18
Tema: Educação | Machine Learning