
# 🧠 Obesity FastCheck

**Obesity FastCheck** é uma aplicação web interativa que utiliza **Machine Learning** para prever o nível de obesidade e **Inteligência Artificial Generativa (Google Gemini)** para fornecer uma análise educativa e personalizada com base nos dados do usuário.

A aplicação é construída inteiramente em Python e Streamlit, oferecendo uma experiência unificada sem a necessidade de uma API externa.

---

## ✨ Visão Geral

Este projeto foi redesenhado para ser uma ferramenta "all-in-one" que entrega:

- 🎯 Predição de Nível de Obesidade: Utiliza um modelo Scikit-learn treinado para classificar o perfil do usuário em 7 categorias de peso.

- 🤖 Análise com IA Generativa: Conecta-se diretamente à API do Google Gemini para criar uma análise de saúde personalizada e educativa com base nos resultados.

- 📊 Indicadores de Saúde: Calcula e explica métricas importantes como IMC, pontuação de estilo devida, balanço de atividades e mais.

- 🚀 Simplicidade e Performance: Roda como uma aplicação única no Streamlit Community Cloud, garantindo facilidade no deploy e manutenção.


---

## 🧱 Estrutura do Projeto

```bash
obesityfastcheck/
├── train/             # Script de treinamento do modelo
│   ├── train.py
│   └── requirements.txt
│
├── models/             # Modelo treinado
│   ├── obesity_model.joblib
│
├── shared/            # Módulos de engenharia de features
│   ├── utils.py
│
├── Obesity.csv        # Base de dados original
├── app.py             # Streamlit integrado com ML e IA
├── data_model.ipynb   # arquivo de testes e análises gerais
├── requirements.txt           
└── README.md

```

---

## 🧠 Indicadores Calculados

| Indicador             | Interpretação                                                                 |
|-----------------------|------------------------------------------------------------------------------|
| **IMC**               | Classifica o peso de acordo com a altura. Ideal: entre 18.5 e 24.9            |
| **LifestyleScore**    | De 0 a 4. Quanto maior, melhor: considera não fumar, controle calórico, etc. |
| **HealthyMealRatio**  | Proporção de refeições com vegetais. Ideal > 0.4                              |
| **ActivityBalance**   | Diferença entre tempo ativo e tempo em telas. Positivo é desejável            |
| **TransportType**     | Classificação do transporte: *active*, *neutral*, *sedentary*                 |

---

## ⚙️ Como Executar Localmente

### 1. Clonar o repositório

```bash
git clone https://github.com/luishrufino/obesityfastcheck.git
cd obesityfastcheck
```

### 2. Treinar o modelo

```bash
cd train
python train.py
```

O modelo será salvo em `models/obesity_model.joblib`.

### 3. Configurar o Ambiente Virtual e Instalar Dependências

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar o ambiente (Windows)
venv\Scripts\activate
# Ativar o ambiente (Linux/Mac)
source venv/bin/activate

# Instalar as dependências
pip install -r requirements.txt
```

### 4. Configurar a Chave de API do Google

- Crie um arquivo chamado secrets.toml dentro de uma pasta .streamlit:
  ```bash
  mkdir .streamlit
  touch .streamlit/secrets.toml
  ```
- Adicione sua chave de API ao arquivo secrets.toml neste formato:
  ```bash
  # .streamlit/secrets.toml
  GOOGLE_API_KEY="SUA_CHAVE_DE_API_AQUI"
  ```

### 5. Rodar a Aplicação Streamlit

```bash
streamlit run app.py
```

---
## 🌐 Deploy no Streamlit Community Cloud

O deploy desta aplicação é muito simples:
1. Fork este repositório para a sua conta do **GitHub**.
2. Acesse o Streamlit Community Cloud: <https://streamlit.io/cloud>.
3. Clique em **"New app"** e conecte seu repositório do GitHub.
4. Selecione o repositório **obesityfastcheck e o arquivo app.py**.
5. Vá para a seção **"Advanced settings..."** e adicione seus "Secrets". O conteúdo será o mesmo do seu arquivo secrets.toml:
```bash
GOOGLE_API_KEY="SUA_CHAVE_DE_API_AQUI"
```

6. Clique em "Deploy!". Sua aplicação estará online em poucos minutos.

---

## 📌 Observações

- Esta aplicação **não substitui diagnóstico médico**.
- O modelo foi treinado com fins educativos para auxiliar na conscientização sobre hábitos de saúde.

---

## 👨‍💻 Autor

Desenvolvido por **Luis Rufino**, Analista de Dados.  
Este projeto integra conceitos de machine learning, análise de dados e deploy de aplicações interativas.

