import time
import io
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.engine import ITTicketModel
from core.nlp import preprocess_text
from core.storage import insert_feedback, fetch_master_dataset, fetch_feedback_count

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
    """Fase 3: Inferencia manual con latencia ultrabaja."""
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
    
    # Heurística de Criticidad simple basada en keywords
    t = original_text.lower()
    score = sum([3 for w in ["urgente", "caído", "caida", "crítico", "producción"] if w in t]) + \
            sum([1 for w in ["lento", "error", "falla", "problema"] if w in t])
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
    """Aprendizaje Continuo con Expansión Orgánica."""
    insert_feedback(req.original_text, req.correct_department)
    
    clean_txt, _ = preprocess_text(req.original_text)
    
    learning_status = engine.learn_online(clean_txt, req.correct_department)
    
    retrained = False
    
    if learning_status == "NEEDS_RETRAIN":
        df = fetch_master_dataset()
        if not df.empty:
            engine.train_batch(df)
            retrained = True
    else:
        count = fetch_feedback_count()
        if count > 0 and count % 20 == 0:
            df = fetch_master_dataset()
            if not df.empty:
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
    """Fase 2: Auditoría, K-Fold y Entrenamiento Masivo por CSV."""
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
    
    clean_data = []
    rejected = 0
    
    for _, row in df.iterrows():
        c_text, toks = preprocess_text(str(row["text"]))
        if len(toks) >= 2 and pd.notna(row["department"]):
            clean_data.append({"text": c_text, "department": row["department"]})
        else:
            rejected += 1
            
    if not clean_data:
        raise HTTPException(status_code=400, detail="El CSV no contenía tickets válidos tras la limpieza NLP.")
        
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
    """
    Fase 1.6: Inferencia Masiva Ciega.
    Recibe un CSV sin departamentos, limpia cada ticket y predice su clase en milisegundos.
    """
    start_time = time.time()
    
    if not engine.classifier or not engine.vectorizer:
        raise HTTPException(status_code=400, detail="La IA no ha sido entrenada. Sube primero un dataset en la pestaña de Lotes.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo CSV inválido.")

    if "text" not in df.columns:
        if "titulo" in df.columns and "descripcion" in df.columns:
            df["text"] = df["titulo"].astype(str) + " " + df["descripcion"].astype(str)
        else:
            raise HTTPException(status_code=400, detail="El CSV debe contener una columna 'text' o 'titulo' y 'descripcion'.")

    df["text"] = df["text"].fillna("")
    
    predictions = []
    distribution = {}
    
    for idx, row in df.iterrows():
        raw_text = str(row["text"])
        clean_txt, tokens = preprocess_text(raw_text)
        
        if len(tokens) < 2:
            pred_dept = "Rechazado (Basura)"
            prob = 0.0
        else:
            X_vec = engine.vectorizer.transform([clean_txt])
            probas = engine.classifier.predict_proba(X_vec)[0]
            max_idx = probas.argmax()
            pred_dept = engine.classifier.classes_[max_idx]
            prob = round(float(probas[max_idx]), 4)
            
        distribution[pred_dept] = distribution.get(pred_dept, 0) + 1
        
        predictions.append({
            "id": idx + 1,
            "text_original": raw_text[:120] + "..." if len(raw_text) > 120 else raw_text,
            "predicted_department": pred_dept,
            "confidence": f"{prob * 100:.1f}%"
        })

    speed = int((time.time() - start_time) * 1000)
    
    return {
        "totalTickets": len(df),
        "speed": speed,
        "departmentDistribution": distribution,
        "predictions": predictions
    }