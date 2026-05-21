import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "sgd_model.pkl")

class ITTicketModel:
    def __init__(self):
        self.vectorizer = self._load_artifact(VECTORIZER_PATH)
        self.classifier = self._load_artifact(MODEL_PATH)
        # Los departamentos ya no están hardcodeados. Nacen del modelo entrenado.
        self.departments = list(self.classifier.classes_) if self.classifier else []

    def _load_artifact(self, path: str):
        return joblib.load(path) if os.path.exists(path) else None

    def _save_artifacts(self) -> None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self.vectorizer, VECTORIZER_PATH)
        joblib.dump(self.classifier, MODEL_PATH)

    def train_batch(self, df: pd.DataFrame) -> dict:
        if df.empty:
            raise ValueError("Dataset vacío.")

        texts = df["text"].tolist()
        labels = df["department"].tolist()

        # Aprendizaje orgánico: la IA descubre los departamentos desde los datos
        self.departments = sorted(list(set(labels)))

        self.vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(texts)

        self.classifier = SGDClassifier(loss="log_loss", random_state=42, class_weight="balanced")

        # Validación cruzada K-Fold para evitar overfitting
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_cv = cross_val_predict(self.classifier, X, labels, cv=cv, n_jobs=-1)

        f1 = f1_score(labels, y_pred_cv, average="weighted")
        acc = accuracy_score(labels, y_pred_cv)
        cm = confusion_matrix(labels, y_pred_cv, labels=self.departments)

        self.classifier.fit(X, labels)
        self._save_artifacts()

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
            "confusionMatrix": cm.tolist(),
            "labels": self.departments,
            "bestModelName": "SGD Classifier (Log Loss)",
            "globalTfidf": global_tfidf
        }

    def predict_single(self, clean_text: str, original_tokens: list[str]) -> dict:
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
        Retorna:
        - "LEARNED": Si ajustó pesos en vivo.
        - "NEEDS_RETRAIN": Si el departamento es NUEVO y requiere reconstruir la matriz.
        - "ERROR": Si no hay modelo.
        """
        if not self.vectorizer or not self.classifier:
            return "ERROR"

        # Si ingresas un departamento completamente nuevo, el cerebro debe expandirse
        if correct_department not in self.departments:
            return "NEEDS_RETRAIN"

        X_vec = self.vectorizer.transform([clean_text])
        self.classifier.partial_fit(X_vec, [correct_department], classes=self.departments)
        self._save_artifacts()
        return "LEARNED"