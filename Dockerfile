# 1. Usar uma imagem base oficial do Python (versão slim para ser leve e rápida)
FROM python:3.9-slim

# 2. Definir o diretório de trabalho dentro do container
WORKDIR /app

# 3. Instalar dependências de sistema (necessárias para algumas libs de stats/math)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiar o requirements.txt primeiro (otimiza o cache do Docker)
COPY requirements.txt .

# 5. Instalar as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiar o restante dos arquivos do projeto (seu código, modelos pkl, etc)
COPY . .

# 7. Expor a porta que o Streamlit usa por padrão
EXPOSE 8501

# 8. Healthcheck: Um sinal de senioridade. Diz se o app está saudável.
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 9. Comando para rodar a aplicação quando o container iniciar
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]