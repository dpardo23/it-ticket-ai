# IT Ticket AI — Pipeline NLP & Triaje Automatizado 🚀

[![GitHub Release](https://img.shields.io/github/v/release/dpardo-bo/it-ticket-ai?color=EF4444&style=for-the-badge)](https://github.com/dpardo-bo/it-ticket-ai/releases)
[![Rust](https://img.shields.io/badge/Rust-2021_Edition-000000?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Next.js](https://img.shields.io/badge/Next.js_14+-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

IT Ticket AI es una plataforma de grado empresarial diseñada para optimizar la gestión de incidentes tecnológicos mediante el Procesamiento de Lenguaje Natural (NLP). El sistema actúa como un despachador inteligente híbrido que intercepta descripciones técnicas crudas de problemas de TI, analiza su carga semántica y determina instantáneamente el departamento de enrutamiento óptimo, mitigando cuellos de botella organizacionales.

Construido bajo una arquitectura monorepo híbrida, el núcleo matemático predictivo está desarrollado en Python, empaquetado en una aplicación de escritorio nativa multiplataforma ultraligera mediante **Tauri v2**, y dotado de una interfaz dinámica reactiva en **Next.js**.

---

## 🎯 Objetivos del Proyecto

* **Clasificación Autónoma:** Sustituir el triaje manual de tickets de soporte técnico por un modelo algorítmico probabilístico de latencia ultrabaja.
* **Aprendizaje Dinámico en Línea:** Implementar un lazo de retroalimentación MLOps continuo (*Human-in-the-Loop*) que permita al modelo adaptar sus pesos vectoriales en caliente sin requerir paradas de servicio.
* **Distribución Eficiente:** Producir ejecutables de escritorio nativos que consuman una fracción de memoria RAM en comparación con frameworks convencionales basados en Chromium (como Electron).
* **Auditoría de Datos:** Proveer herramientas visuales avanzadas (Matrices de Confusión, Diccionarios TF-IDF y análisis de F1-Score) para evaluar el rendimiento de la IA ante cargas masivas de datos.

---

## 🛠️ Arquitectura Teconológica

El proyecto se divide en capas perfectamente desacopladas:

1.  **Núcleo de Inteligencia Artificial (`ai_engine`):** Pipeline NLP funcional desarrollado en Python. Utiliza vectorización de texto basada en TF-IDF (Term Frequency - Inverse Document Frequency) y un clasificador lineal `SGDClassifier` optimizado mediante descenso de gradiente estocástico para habilitar aprendizaje incremental (`partial_fit`).
2.  **Capa de Presentación (`ui`):** Dashboard analítico programado en Next.js utilizando TypeScript, Tailwind CSS y componentes primitivos optimizados de Radix y Framer Motion para las animaciones del pipeline.
3.  **Contenedor Nativo (`src-tauri`):** Núcleo en Rust que se comunica con el sistema operativo y Next.js mediante un puente síncrono binario (*IPC Bridge*), compilado con optimizaciones extremas de producción (`lto = true`, `opt-level = 3`).
4.  **Persistencia Relacional:** Almacenamiento distribuido administrado mediante PostgreSQL en Supabase para el registro histórico de datasets y logs de feedback humano.

---

## 🖥️ Módulos de la Aplicación

La interfaz de usuario se divide en tres centros de control principales dentro del panel analítico:

### 1. Panel de Análisis de Motor (Pipeline NLP)
Diseñado para la ingesta manual de incidentes en tiempo real.
* **Heurística Anti-Basura:** Filtra cadenas de texto irrelevantes o insuficientes mediante análisis de longitud de tokens semánticos lógicos antes de invocar al clasificador.
* **Pipeline de Lematización:** Desglosa visualmente la transformación del texto crudo ingresado por el usuario hacia su equivalente lematizado y limpio de *stopwords*.
* **Softmax Probabilístico:** Despliega un gráfico termómetro dinámico con el desglose porcentual de confianza asignado a cada departamento de TI disponible.
* **Módulo de Feedback Humano:** Si la IA comete un error, un administrador puede ingresar el departamento correcto, forzando un ajuste inmediato en la matriz de pesos del modelo.

### 2. Auditoría de IA (Lotes de Entrenamiento)
Fase científica orientada a administradores de datos y especialistas de soporte.
* **Validación Cruzada K-Fold:** Permite cargar archivos CSV masivos con históricos etiquetados para calcular el F1-Score Ponderado y la Exactitud (*Accuracy*) global del modelo.
* **Matriz de Confusión Interactiva:** Mapa de calor de intensidad que expone visualmente los falsos positivos y desviaciones de clasificación entre departamentos reales vs predicciones.
* **Diccionario Vectorial:** Inspección en tiempo real de los términos léxicos técnicos con mayor peso específico dentro del espacio vectorial del vocabulario entrenado.

### 3. Inferencia Masiva Ciega
Herramienta de procesamiento masivo en tiempo real para operaciones a gran escala.
* **Clasificación Asíncrona Masiva:** Permite subir un CSV con miles de tickets técnicos sin clasificar. El sistema procesa, limpia y predice el destino de toda la carga en pocos milisegundos.
* **Gráficos de Demanda:** Genera una analítica de distribución instantánea que revela qué áreas de infraestructura están colapsando o recibiendo mayor volumen de incidentes.

---

## ⚙️ Requisitos de Instalación (Desarrolladores)

Si deseas clonar el repositorio para auditar el código o extender el sistema, tu entorno local debe contar con las siguientes dependencias globales:

### 1. Entorno Frontend & Escritorio
* **Node.js:** Versión 20 o superior LTS.
* **PNPM:** Administrador de paquetes de Node de alto rendimiento (Versión 9+).
* **Rust & Cargo:** Cadena de herramientas de Rust estable (Versión 1.77.2+) para compilar el motor de Tauri.

### 2. Dependencias de Sistema (Solo Linux / Ubuntu)
Para compilar las interfaces gráficas nativas de Tauri en sistemas basados en Debian/Ubuntu, debes ejecutar el siguiente comando en tu terminal:
```bash
sudo apt-get update && sudo apt-get install -y libwebkit2gtk-4.1-dev build-essential curl wget file libxdo-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev
