# Usa Node LTS como base
FROM node:18

# Establece el directorio de trabajo
WORKDIR /app

# Copia solo los archivos necesarios para instalar dependencias
COPY package.json package-lock.json* ./

# Instala las dependencias primero para aprovechar la caché de Docker
RUN npm install

# Copia el resto del código después (así no invalida la capa anterior si solo cambias código)
COPY . .

# Expone el puerto para desarrollo
EXPOSE 3000

# Inicia la app en modo desarrollo
CMD ["npm", "start"]
