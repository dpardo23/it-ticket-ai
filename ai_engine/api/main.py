import time
import io
import pandas as pd
from functools import reduce
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.engine import ITTicketModel
from core.nlp import preprocess_text
from core.storage import insert_feedback, fetch_master_dataset, fetch_feedback_count, bulk_insert_dataset

# ==========================================
# 1. Configuración de la App
# ==========================================
app = FastAPI(title="IT Ticket AI - Functional API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia global del Motor de IA (Singleton Funcional)
engine = ITTicketModel()

# ==========================================
# 1.5 Arquitectura MLOps: Arranque Inmortal
# ==========================================
@app.on_event("startup")
def restore_brain():
    """
    Detecta si Hugging Face borró los archivos .pkl temporales por inactividad.
    Si hay amnesia, descarga el Data Lake (Supabase) y reconstruye la red neuronal.
    """
    if not engine.classifier or not engine.vectorizer:
        print("[MLOps] Amnesia temporal detectada. Restaurando cerebro desde Supabase...")
        df = fetch_master_dataset()
        if not df.empty and len(df) >= 5:
            engine.train_batch(df)
            print(f"[MLOps] Cerebro restaurado exitosamente con {len(df)} tickets históricos.")
        else:
            print("[MLOps] Data Lake vacío o insuficiente. Esperando carga inicial de CSV.")

# ==========================================
# 2. Esquemas de Datos (Pydantic)
# ==========================================
class PredictRequest(BaseModel):
    title: str
    description: str

class FeedbackRequest(BaseModel):
    original_text: str
    correct_department: str

class BlindTicketRequest(BaseModel):
    text: str

# ==========================================
# 3. Endpoints
# ==========================================

@app.get("/")
def health_check():
    return {
        "status": "ok", 
        "model_ready": engine.classifier is not None,
        "message": "API Funcional Operativa"
    }

@app.post("/api/predict")
def predict(req: PredictRequest):
    """Fase 3: Inferencia manual con latencia ultrabaja (Funcional)."""
    start = time.time()
    original_text = f"{req.title}\n{req.description}"
    
    # NLP Funcional
    clean_text, tokens = preprocess_text(original_text)
    
    # Heurística Anti-Basura
    if len(tokens) < 2:
        return {"is_garbage": True, "message": "Texto inválido o insuficiente."}
        
    try:
        result = engine.predict_single(clean_text, tokens)
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Modelo no entrenado. Ve a la pestaña Batch y sube un CSV primero.")
        
    latency = int((time.time() - start) * 1000)
    
    # Heurística de Criticidad usando List Comprehensions (Paradigma Funcional Python)
    t_lower = original_text.lower()
    score = sum([3 for w in ["urgente", "caído", "caida", "crítico", "producción"] if w in t_lower]) + \
            sum([1 for w in ["lento", "error", "falla", "problema"] if w in t_lower])
            
    # Asignación declarativa
    level = "Nivel 3 (Crítico)" if score >= 4 else "Nivel 2 (Especializado)" if score >= 2 else "Nivel 1 (Triaje Directo)"
    
    return {
        "winner": result["winner"],
        "probabilities": result["probabilities"],
        "tokens": result["tokens"],
        "latency": latency,
        "level": level,
        "originalText": original_text,
        "cleanText": result["cleanText"],
        "topTfidf": result["topTfidf"],
        "is_garbage": False
    }

@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    """Aprendizaje Continuo con Expansión Orgánica y Guardia Anti-Cold-Start."""
    insert_feedback(req.original_text, req.correct_department)
    
    clean_txt, _ = preprocess_text(req.original_text)
    learning_status = engine.learn_online(clean_txt, req.correct_department)
    
    retrained = False
    
    # Función lambda para evaluar si es seguro reentrenar (Previene el error de K-Fold)
    is_safe_to_train = lambda data_frame: not data_frame.empty and len(data_frame) >= 5
    
    if learning_status == "NEEDS_RETRAIN":
        df = fetch_master_dataset()
        if is_safe_to_train(df):
            engine.train_batch(df)
            retrained = True
    else:
        count = fetch_feedback_count()
        if count > 0 and count % 20 == 0:
            df = fetch_master_dataset()
            if is_safe_to_train(df):
                engine.train_batch(df)
                retrained = True
            
    return {
        "success": True,
        "learnedImmediately": learning_status == "LEARNED",
        "retrainedBatch": retrained,
        "message": "Feedback consolidado. La IA se ha adaptado orgánicamente."
    }

@app.post("/api/batch")
async def batch(file: UploadFile = File(...)):
    """Fase 2: Auditoría y Entrenamiento Masivo usando Map/Filter."""
    start = time.time()
    
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo CSV inválido.")

    if "text" not in df.columns or "department" not in df.columns:
        if "titulo" in df.columns and "descripcion" in df.columns and "departamento" in df.columns:
            df["text"] = df["titulo"].astype(str) + " " + df["descripcion"].astype(str)
            df["department"] = df["departamento"]
        else:
            raise HTTPException(status_code=400, detail="El CSV debe tener las columnas 'text' y 'department'.")

    df["text"] = df["text"].fillna("")
    records = df.to_dict('records')
    
    # 1. Map: Preprocesar todos los registros (Lazy Evaluation)
    preprocess_mapper = lambda r: {
        "raw_dept": r.get("department"), 
        "nlp": preprocess_text(str(r.get("text", "")))
    }
    processed_records = list(map(preprocess_mapper, records))
    
    # 2. Filter: Descartar basura y nulos
    is_valid = lambda item: len(item["nlp"][1]) >= 2 and pd.notna(item["raw_dept"])
    valid_records = list(filter(is_valid, processed_records))
    rejected = len(records) - len(valid_records)
    
    if not valid_records:
        raise HTTPException(status_code=400, detail="El CSV no contenía tickets válidos tras la limpieza NLP.")
    
    # 3. Map: Formatear para el DataFrame final
    format_mapper = lambda item: {"text": item["nlp"][0], "department": item["raw_dept"]}
    clean_data = list(map(format_mapper, valid_records))
    
    # MLOps: Inserción en el Data Lake (Supabase) evitando duplicados
    bulk_insert_dataset(clean_data)
    
    # Entrenamiento matemático del modelo
    clean_df = pd.DataFrame(clean_data)
    train_stats = engine.train_batch(clean_df)
    
    speed = int((time.time() - start) * 1000)
    dist = clean_df["department"].value_counts().to_dict()
    
    return {
        "totalTickets": len(df),
        "processedCount": len(clean_df),
        "rejectedCount": rejected,
        "f1Score": train_stats["f1Score"],
        "accuracy": train_stats["accuracy"],
        "bestModelName": train_stats["bestModelName"],
        "optimalAlpha": "N/A (Log Loss)",
        "confusionMatrix": train_stats["confusionMatrix"],
        "labels": train_stats["labels"],
        "departmentDistribution": dist,
        "globalTfidf": train_stats["globalTfidf"],
        "speed": speed
    }

@app.post("/api/batch_predict")
async def batch_predict(file: UploadFile = File(...)):
    """Fase 1.6: Inferencia Masiva Ciega usando Map/Reduce."""
    start_time = time.time()
    
    if not engine.classifier or not engine.vectorizer:
        raise HTTPException(status_code=400, detail="La IA no ha sido entrenada. Sube primero un dataset en Lotes.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo CSV inválido.")

    if "text" not in df.columns:
        if "titulo" in df.columns and "descripcion" in df.columns:
            df["text"] = df["titulo"].astype(str) + " " + df["descripcion"].astype(str)
        else:
            raise HTTPException(status_code=400, detail="El CSV debe contener 'text' o 'titulo' y 'descripcion'.")

    df["text"] = df["text"].fillna("")
    records = df.to_dict('records')
    
    # 1. Función Pura para inferir un ticket individual
    def infer_ticket(idx: int, raw_text: str) -> dict:
        clean_txt, tokens = preprocess_text(raw_text)
        text_preview = raw_text[:120] + "..." if len(raw_text) > 120 else raw_text
        
        if len(tokens) < 2:
            return {
                "id": idx + 1,
                "text_original": text_preview,
                "predicted_department": "Rechazado (Basura)",
                "confidence": "0.0%"
            }
            
        X_vec = engine.vectorizer.transform([clean_txt])
        probas = engine.classifier.predict_proba(X_vec)[0]
        max_idx = probas.argmax()
        
        return {
            "id": idx + 1,
            "text_original": text_preview,
            "predicted_department": engine.classifier.classes_[max_idx],
            "confidence": f"{round(float(probas[max_idx]) * 100, 1)}%"
        }

    # 2. Map: Proyectar la inferencia sobre toda la colección de registros
    predictions = list(map(lambda item: infer_ticket(item[0], str(item[1].get("text", ""))), enumerate(records)))
    
    # 3. Reduce: Agrupar la distribución de frecuencias sin mutar variables externas
    distribution = reduce(
        lambda acc, p: {**acc, p["predicted_department"]: acc.get(p["predicted_department"], 0) + 1},
        predictions,
        {}
    )

    speed = int((time.time() - start_time) * 1000)
    
    return {
        "totalTickets": len(df),
        "speed": speed,
        "departmentDistribution": distribution,
        "predictions": predictions
    }