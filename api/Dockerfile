FROM mambaorg/micromamba:1.5.6

ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV MAMBA_ENV_NAME=tf-gpu

USER root

# Instala dependencias del sistema (solo si necesitas compilar algo como blis o bitsandbytes)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar entorno
COPY environment.yml .

# Crear entorno conda
RUN micromamba create -y -n ${MAMBA_ENV_NAME} -f environment.yml && \
    micromamba clean --all --yes

# Activar entorno en el shell
SHELL ["micromamba", "run", "-n", "tf-gpu", "/bin/bash", "-c"]

# Reinstalar numpy con pip y reforzar dependencias para evitar conflictos (por seguridad)
RUN pip install --no-cache-dir "numpy<2.0" "faiss-cpu" --force-reinstall

# Copiar tu proyecto
COPY . /app

# Evitar añadir archivos pesados innecesarios (ver más abajo para .dockerignore)
EXPOSE 8000

# Comando de arranque de Django
CMD ["micromamba", "run", "-n", "tf-gpu", "python", "manage.py", "runserver", "0.0.0.0:8000"]

