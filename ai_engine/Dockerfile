# Usamos una imagen oficial de Python ligera
FROM python:3.10-slim

# Damos permisos para evitar problemas de usuario en Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos las dependencias primero para aprovechar la caché de Docker
COPY --chown=user ./requirements.txt requirements.txt

# Instalamos las librerías científicas y FastAPI
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Descargamos los modelos de lenguaje de Spacy (Español e Inglés)
RUN python -m spacy download es_core_news_sm
RUN python -m spacy download en_core_web_sm

# Copiamos el resto de tu código de backend
COPY --chown=user . /app

# Hugging Face Spaces expone OBLIGATORIAMENTE el puerto 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]