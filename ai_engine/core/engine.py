"""
Motor de clasificación de tickets IT basado en SGDClassifier y TF-IDF.

Define la clase ITTicketModel, que encapsula el ciclo de vida completo
del modelo: carga desde disco, entrenamiento por lotes, inferencia individual
y aprendizaje online incremental.
Llamado por: api.main.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Umbral a partir del cual se usa split 80/20 en lugar de 5-fold CV.
# Para datasets grandes, la CV requiere entrenar el modelo 5 veces extra;
# un split estratificado es estadísticamente equivalente y evita ese costo.
_LARGE_DATASET_THRESHOLD = 20_000

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "sgd_model.pkl")


class ITTicketModel:
    """
    Motor de IA para clasificación de tickets IT en departamentos.

    Gestiona el vectorizador TF-IDF y el clasificador SGD como artefactos
    persistidos en disco. Expone métodos para entrenamiento completo,
    predicción con probabilidades y aprendizaje incremental sin reentrenar.
    Instanciada como singleton en api.main al arrancar la aplicación.
    """

    def __init__(self):
        """
        Carga vectorizador y clasificador desde disco si existen.

        Deriva la lista de departamentos directamente de las clases
        almacenadas en el clasificador para evitar valores hardcodeados.
        """
        self.vectorizer = self._load_artifact(VECTORIZER_PATH)
        self.classifier = self._load_artifact(MODEL_PATH)
        self.departments = list(self.classifier.classes_) if self.classifier else []

    def _load_artifact(self, path: str):
        return joblib.load(path) if os.path.exists(path) else None

    def _save_artifacts(self) -> None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self.vectorizer, VECTORIZER_PATH)
        joblib.dump(self.classifier, MODEL_PATH)

    def train_batch(self, df: pd.DataFrame) -> dict:
        """
        Entrena el modelo completo desde cero con un DataFrame de tickets.

        Construye un TfidfVectorizer (max 12 000 features, bigramas),
        calcula pesos de clase balanceados matemáticamente y ajusta un
        SGDClassifier con Log Loss. Evalúa rendimiento con validación
        cruzada estratificada de 5 pliegues antes del ajuste final.
        Persiste ambos artefactos en disco y retorna métricas de calidad.

        Retorna un diccionario con f1Score, accuracy, avgConfidence,
        confusionMatrix, labels, bestModelName y globalTfidf.
        Llamado por: api.main (startup, batch, feedback).
        """
        if df.empty:
            raise ValueError("Dataset vacío.")

        texts = df["text"].tolist()
        labels = df["department"].tolist()

        self.departments = sorted(list(set(labels)))

        weights = compute_class_weight(class_weight="balanced", classes=np.array(self.departments), y=labels)
        class_weight_dict = dict(zip(self.departments, weights))

        self.vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(texts)

        self.classifier = SGDClassifier(loss="log_loss", random_state=42, class_weight=class_weight_dict)

        if len(texts) >= _LARGE_DATASET_THRESHOLD:
            # Con datasets grandes (≥20k), un split estratificado 80/20 produce
            # métricas igualmente representativas sin necesidad de entrenar el
            # modelo 5 veces extra como haría la validación cruzada.
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, labels, test_size=0.2, random_state=42, stratify=labels
            )
            _eval_clf = SGDClassifier(loss="log_loss", random_state=42, class_weight=class_weight_dict)
            _eval_clf.fit(X_tr, y_tr)
            y_pred_eval = _eval_clf.predict(X_te)
            f1 = f1_score(y_te, y_pred_eval, average="weighted")
            acc = accuracy_score(y_te, y_pred_eval)
            cm = confusion_matrix(y_te, y_pred_eval, labels=self.departments)
        else:
            # Para datasets pequeños (<20k), 5-fold CV ofrece una estimación
            # de generalización más robusta al evaluar sobre el 100% de los datos.
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            y_pred_cv = cross_val_predict(self.classifier, X, labels, cv=cv, n_jobs=-1)
            f1 = f1_score(labels, y_pred_cv, average="weighted")
            acc = accuracy_score(labels, y_pred_cv)
            cm = confusion_matrix(labels, y_pred_cv, labels=self.departments)

        # Entrenamiento final sobre la totalidad del dataset, independientemente
        # del método de evaluación usado arriba.
        self.classifier.fit(X, labels)
        self._save_artifacts()

        proba_matrix = self.classifier.predict_proba(X)
        avg_confidence = float(np.mean(proba_matrix.max(axis=1)))

        avg_scores = np.asarray(X.mean(axis=0)).ravel()
        top_indices = avg_scores.argsort()[-15:][::-1]
        feature_names = self.vectorizer.get_feature_names_out()

        global_tfidf = [
            {"term": feature_names[i], "weight": round(float(avg_scores[i]), 6)}
            for i in top_indices if avg_scores[i] > 0
        ]

        return {
            "f1Score": float(f1),
            "accuracy": float(acc),
            "avgConfidence": avg_confidence,
            "confusionMatrix": cm.tolist(),
            "labels": self.departments,
            "bestModelName": "SGD Classifier (Log Loss)",
            "globalTfidf": global_tfidf
        }

    def predict_single(self, clean_text: str, original_tokens: list[str]) -> dict:
        """
        Clasifica un ticket preprocesado y devuelve probabilidades por departamento.

        Vectoriza el texto limpio con el TF-IDF entrenado, obtiene las
        probabilidades de todas las clases y extrae los 5 términos con mayor
        peso TF-IDF para explicabilidad.

        Retorna un diccionario con winner, probabilities (normalizadas 0-1),
        tokens, topTfidf y cleanText.
        Llamado por: api.main.predict, api.main.batch_predict (vía infer_ticket).
        """
        if not self.vectorizer or not self.classifier:
            raise RuntimeError("El modelo no está entrenado.")

        X_vec = self.vectorizer.transform([clean_text])
        probas = self.classifier.predict_proba(X_vec)[0]

        results = [
            {"name": class_name, "value": round(float(prob) * 100, 2)}
            for class_name, prob in zip(self.classifier.classes_, probas)
        ]
        results.sort(key=lambda x: x["value"], reverse=True)
        winner = results[0]["name"]

        tfidf_scores = X_vec.toarray()[0]
        top_indices = tfidf_scores.argsort()[-5:][::-1]
        feature_names = self.vectorizer.get_feature_names_out()

        top_tfidf = [
            {"term": feature_names[i], "weight": round(float(tfidf_scores[i]), 6)}
            for i in top_indices if tfidf_scores[i] > 0
        ]

        norm_probas = {r["name"]: r["value"] / 100.0 for r in results}

        return {
            "winner": winner,
            "probabilities": norm_probas,
            "tokens": original_tokens,
            "topTfidf": top_tfidf,
            "cleanText": clean_text
        }

    def learn_online(self, clean_text: str, correct_department: str) -> str:
        """
        Actualiza el clasificador de forma incremental con un ejemplo corregido.

        Usa partial_fit del SGDClassifier para ajustar pesos sin reentrenar
        desde cero. Si el departamento corregido no existe en el espacio de
        clases conocido, no puede expandir la matriz y señaliza reentrenamiento
        completo al llamador.

        Retorna uno de tres literales:
        - "LEARNED": los pesos se ajustaron y los artefactos se persistieron.
        - "NEEDS_RETRAIN": el departamento es nuevo; requiere train_batch.
        - "ERROR": el modelo no está inicializado.
        Llamado por: api.main.feedback.
        """
        if not self.vectorizer or not self.classifier:
            return "ERROR"

        if correct_department not in self.departments:
            return "NEEDS_RETRAIN"

        X_vec = self.vectorizer.transform([clean_text])
        self.classifier.partial_fit(X_vec, [correct_department], classes=self.departments)
        self._save_artifacts()
        return "LEARNED"
