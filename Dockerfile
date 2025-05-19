FROM python:3.12-slim

WORKDIR /app/scripts

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copia os arquivos necessários
COPY requirements.txt .
COPY ./scripts/app.py ./scripts/app.py
COPY ./scripts/runs/detect/train/weights/best.pt ./scripts/runs/detect/train/weights/best.pt
COPY ./imgs/logo.png ./imgs/  

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]