# IT Ticket AI

**Plataforma de triaje automatizado de incidentes tecnológicos mediante Procesamiento de Lenguaje Natural**

[![Version](https://img.shields.io/badge/version-v1.1.0-0078D4?style=flat-square)](https://github.com/dpardo-bo/it-ticket-ai/releases/tag/v1.1.0)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-000000?style=flat-square&logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![Tauri](https://img.shields.io/badge/Tauri-v2-FFC131?style=flat-square&logo=tauri&logoColor=white)](https://tauri.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22863A?style=flat-square)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey?style=flat-square)](https://github.com/dpardo-bo/it-ticket-ai/releases)

---

IT Ticket AI sustituye el enrutamiento manual de tickets de soporte técnico por un clasificador adaptativo basado en aprendizaje automático. El sistema acepta descripciones en lenguaje natural, aplica un pipeline NLP completo y determina el departamento de resolución óptimo en milisegundos. El modelo puede actualizarse en caliente mediante correcciones humanas sin interrupciones de servicio, y recuperarse automáticamente tras reinicios de contenedor en entornos serverless.

---

## Tabla de Contenidos

- [Características Principales](#características-principales)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Stack Tecnológico](#stack-tecnológico)
- [Requisitos Previos](#requisitos-previos)
- [Instalación](#instalación)
- [Ejecución en Modo Desarrollo](#ejecución-en-modo-desarrollo)
- [Distribución para Usuarios Finales](#distribución-para-usuarios-finales)
- [Referencia de la API](#referencia-de-la-api)
- [Changelog](#changelog)
- [Licencia](#licencia)
- [Autor](#autor)

---

## Características Principales

### Clasificación con Escudo OOD (Out-of-Domain)

El motor rechaza entradas sin contexto técnico válido antes de invocar al clasificador. Si menos del 15% de los tokens del ticket existen en el vocabulario TF-IDF entrenado, la solicitud se descarta con un diagnóstico semántico. Este mecanismo protege el modelo de entradas maliciosas, datos de envenenamiento (*data poisoning*) y texto completamente fuera de dominio.

### Aprendizaje Incremental (Human-in-the-Loop)

Cuando un administrador corrige una predicción errónea, el modelo actualiza sus pesos vectoriales en tiempo real mediante `partial_fit` sin reentrenar desde cero. Cada 20 correcciones acumuladas, el sistema ejecuta automáticamente un ciclo de reentrenamiento completo sobre el Data Lake histórico.

### Recuperación Automática MLOps

Al reiniciar el servidor, el motor detecta la ausencia de artefactos `.pkl` en disco, descarga el Data Lake completo desde Supabase y reconstruye el clasificador sin intervención manual. Este comportamiento es especialmente relevante en entornos serverless con almacenamiento efímero.

### Inserción Masiva con Deduplicación

El endpoint `/api/batch` procesa archivos CSV de cualquier tamaño, aplica el pipeline NLP a cada registro y persiste únicamente los tickets nuevos en el Data Lake. La deduplicación se realiza en dos fases: primero dentro del lote entrante y luego contra el índice de la base de datos en chunks de 200 filas.

### Auditoría Analítica del Modelo

Panel de métricas con historial de sesiones de entrenamiento: F1-Score ponderado, Accuracy global, Matriz de Confusión interactiva y análisis del diccionario TF-IDF ordenado por peso vectorial. Permite evaluar el rendimiento ante cargas masivas de datos.

---

## Arquitectura del Sistema

El proyecto adopta una arquitectura monorepo con tres capas desacopladas:

```
it-ticket-ai/
├── ai_engine/               # Núcleo de IA y API REST
│   ├── api/
│   │   └── main.py          # Endpoints FastAPI, lógica MLOps y ciclo de vida del modelo
│   └── core/
│       ├── engine.py        # Clasificador SGDClassifier + vectorizador TF-IDF
│       ├── nlp.py           # Pipeline de lematización y limpieza con spaCy
│       ├── storage.py       # Capa de acceso a datos (Supabase)
│       └── db.py            # Cliente Supabase
└── ui/                      # Frontend y contenedor de escritorio
    ├── components/          # Componentes React reutilizables
    ├── pages/               # Vistas Next.js
    └── src-tauri/           # Runtime nativo en Rust (Tauri v2)
```

**Capa de IA (`ai_engine`)**
Pipeline NLP en Python. Aplica tokenización, lematización y eliminación de stopwords con spaCy, vectoriza el texto resultante con TF-IDF y clasifica mediante `SGDClassifier` (scikit-learn). El clasificador admite `partial_fit` para aprendizaje online sin reentrenamiento completo.

**Capa de Presentación (`ui`)**
Dashboard analítico construido con Next.js y TypeScript. Utiliza Tailwind CSS para estilos, Radix UI para componentes accesibles y Framer Motion para las visualizaciones animadas del pipeline.

**Contenedor Nativo (`src-tauri`)**
Núcleo en Rust que empaqueta la aplicación web como ejecutable de escritorio nativo multiplataforma mediante Tauri v2. Consume significativamente menos memoria que alternativas basadas en Chromium.

**Persistencia (`Supabase / PostgreSQL`)**

| Tabla | Contenido |
|---|---|
| `dataset_master` | Data Lake de tickets de entrenamiento |
| `feedback_log` | Correcciones manuales de usuarios |
| `training_log` | Historial de sesiones MLOps con métricas |
| `solution_votes` | Base de conocimiento comunitaria de soluciones |

---

## Stack Tecnológico

| Capa | Tecnología | Version |
|---|---|---|
| Machine Learning | scikit-learn (SGDClassifier + TF-IDF) | 1.4+ |
| NLP | spaCy | 3.x |
| API Backend | FastAPI + Uvicorn | 0.111+ |
| Validacion de datos | Pydantic | 2.x |
| Frontend | Next.js + TypeScript | 14+ |
| Estilos | Tailwind CSS | 3.x |
| Animaciones | Framer Motion | 11+ |
| Runtime de escritorio | Tauri v2 (Rust) | 2.x |
| Base de datos | PostgreSQL via Supabase | 15+ |
| CI/CD | GitHub Actions | — |
| Empaquetado | MSI / DMG / DEB / AppImage | — |

---

## Requisitos Previos

### Windows

- [Node.js](https://nodejs.org/) 20 LTS o superior
- [PNPM](https://pnpm.io/) 9+

  ```powershell
  npm install -g pnpm
  ```

- [Python](https://www.python.org/downloads/) 3.10 o superior. Durante la instalación, marcar la opcion **"Add Python to PATH"**.
- [Rust](https://rustup.rs/): descargar y ejecutar `rustup-init.exe` desde el sitio oficial.
- **Microsoft C++ Build Tools**: instalar desde [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) seleccionando el componente _"Desarrollo para escritorio con C++"_.
- **WebView2 Runtime**: preinstalado en Windows 11. En Windows 10, descargar el _Evergreen Bootstrapper_ desde [la pagina oficial de Microsoft Edge WebView2](https://developer.microsoft.com/en-us/microsoft-edge/webview2/).

### macOS

- [Node.js](https://nodejs.org/) 20 LTS o superior. Recomendado via `nvm` o Homebrew:

  ```bash
  brew install node
  ```

- [PNPM](https://pnpm.io/) 9+:

  ```bash
  npm install -g pnpm
  ```

- [Python](https://www.python.org/downloads/) 3.10 o superior.
- Xcode Command Line Tools:

  ```bash
  xcode-select --install
  ```

- [Rust](https://rustup.rs/):

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source "$HOME/.cargo/env"
  ```

### Linux (Ubuntu / Debian)

- [Node.js](https://nodejs.org/) 20 LTS o superior.
- [PNPM](https://pnpm.io/) 9+:

  ```bash
  npm install -g pnpm
  ```

- [Python](https://www.python.org/downloads/) 3.10 o superior.
- [Rust](https://rustup.rs/):

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source "$HOME/.cargo/env"
  ```

- Dependencias del sistema requeridas por Tauri:

  ```bash
  sudo apt-get update && sudo apt-get install -y \
    libwebkit2gtk-4.1-dev \
    build-essential \
    curl \
    wget \
    file \
    libxdo-dev \
    libssl-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev
  ```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/dpardo-bo/it-ticket-ai.git
cd it-ticket-ai
```

### 2. Configurar el servidor de IA

Se recomienda utilizar un entorno virtual para aislar las dependencias del proyecto.

**Windows**

```bat
cd ai_engine
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**

```bash
cd ai_engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configurar el frontend

```bash
cd ui
pnpm install
```

---

## Ejecucion en Modo Desarrollo

Se requieren dos terminales simultaneas.

### Terminal 1 — Servidor de IA

**Windows**

```bat
cd ai_engine
.\venv\Scripts\activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**macOS / Linux**

```bash
cd ai_engine
source venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

El servidor expone la API en `http://localhost:8000`. La documentacion interactiva de FastAPI esta disponible en `http://localhost:8000/docs`.

### Terminal 2 — Aplicacion de escritorio

```bash
cd ui
pnpm tauri dev
```

La ventana de la aplicacion se abre automaticamente. El servidor de IA debe estar en ejecucion antes de iniciar la interfaz.

---

## Distribucion para Usuarios Finales

Los instaladores precompilados para cada plataforma estan disponibles en la seccion [Releases](https://github.com/dpardo-bo/it-ticket-ai/releases) de este repositorio.

| Sistema Operativo | Archivo | Instalacion |
|---|---|---|
| Windows | `app_x.x.x_x64.msi` o `app_x.x.x_x64_en-US.exe` | Ejecutar el asistente y seguir los pasos en pantalla |
| macOS | `app_x.x.x_x64.dmg` | Abrir el archivo y arrastrar la aplicacion a la carpeta _Applications_ |
| Linux (Debian/Ubuntu) | `app_x.x.x_amd64.deb` | `sudo dpkg -i app_x.x.x_amd64.deb` |

---

## Referencia de la API

| Metodo | Ruta | Descripcion |
|---|---|---|
| GET | `/` | Estado del servicio y disponibilidad del modelo |
| POST | `/api/predict` | Clasificacion individual con escudo OOD |
| POST | `/api/feedback` | Correccion de prediccion y actualizacion del modelo |
| POST | `/api/batch` | Entrenamiento por lotes desde archivo CSV |
| POST | `/api/batch_predict` | Inferencia masiva sobre CSV sin etiquetas conocidas |
| GET | `/api/stats` | Historial de sesiones de entrenamiento MLOps |
| GET | `/api/solutions` | Soluciones conocidas para un departamento |
| POST | `/api/solutions/vote` | Registrar que una solucion existente funciono |
| POST | `/api/solutions/feedback` | Aportar una nueva solucion a la base de conocimiento |

---

## Changelog

### v1.1.0 — 2026-06-02

#### Nuevas funcionalidades

- **Deteccion OOD vectorial contra data poisoning**: se implemento un filtro de validacion semantica en los endpoints `/api/predict` y `/api/batch_predict`. Los tickets cuyo porcentaje de tokens presentes en el vocabulario TF-IDF sea inferior al 15% son rechazados antes de llegar al clasificador. Protege el modelo frente a entradas maliciosas, texto de envenenamiento de datos o consultas completamente fuera del dominio IT.

- **Recuperacion automatica MLOps en el evento de arranque**: el motor detecta la ausencia de artefactos de modelo en disco al iniciar FastAPI. Si el clasificador o el vectorizador no estan disponibles — situacion tipica tras reinicios de contenedor en Hugging Face Spaces u otros entornos serverless — el sistema descarga el Data Lake desde Supabase y reconstruye el modelo sin intervencion manual. La sesion queda registrada en `training_log` con `trigger_type = "startup"`.

- **Insercion masiva con deduplicacion optimizada**: `bulk_insert_dataset` deduplica los registros del lote entrante en memoria antes de consultar la base de datos. La verificacion contra registros existentes se realiza en chunks de 200 filas aprovechando el indice `idx_dataset_master_text`. La insercion se ejecuta en bloques de 500 filas para evitar saturar el cliente Supabase.

#### Correcciones

- **Umbral semantico OOD ajustado**: el umbral de deteccion de basura fue reducido del 30% al 15%. El valor anterior rechazaba tickets tecnicos validos con vocabulario especializado escaso o descripcion concisa, produciendo falsos positivos en entradas legitimas.

---

### v1.0.9

- Ultima version estable sin las correcciones del umbral semantico ni la capa OOD vectorial.

---

## Licencia

Distribuido bajo la Licencia MIT. Consultar el archivo [LICENSE](LICENSE) para mas detalles.

---

## Autor

**Juan Diego Pardo Pozo**

Entregable de ingenieria de sistemas — Byte Busters S.R.L. © 2026
