// ui/types/ai.ts

export interface SingleInferenceResult {
    winner: string;
    probabilities: Record<string, number>;
    tokens: string[];
    latency: number;
    level: string;
    originalText: string;
    cleanText: string;
    topTfidf: Array<{ term: string; weight: number }>;
    isGarbage?: boolean;
    garbageMessage?: string;
}

export interface BatchInferenceResult {
    totalTickets: number;
    processedCount: number;
    rejectedCount: number;
    f1Score: number;
    accuracy: number;
    bestModelName: string;
    optimalAlpha: number | null;
    confusionMatrix: number[][];
    labels: string[];
    departmentDistribution: Record<string, number>;
    globalTfidf: Array<{ term: string; weight: number }>;
    speed: number;
}

export interface BlindPrediction {
    id: number;
    text_original: string;
    predicted_department: string;
    confidence: string;
}

export interface BlindBatchResult {
    totalTickets: number;
    speed: number;
    departmentDistribution: Record<string, number>;
    predictions: BlindPrediction[];
}

/**
 * Solución conocida para un departamento, con metadatos de relevancia comunitaria.
 *
 * voteCount indica cuántos perfiles confirmaron que esta solución funcionó.
 * source distingue si fue extraída del Data Lake de entrenamiento o aportada
 * directamente por un perfil como retroalimentación de solución.
 */
export interface SolutionItem {
    solution: string;
    voteCount?: number;
    source?: 'dataset' | 'vote' | 'user_feedback';
}

export interface TrainingLogEntry {
    id: number;
    created_at: string;
    trigger_type: 'csv_upload' | 'feedback_retrain' | 'startup';
    record_count: number;
    department_count: number;
    f1_score: number;
    accuracy: number;
    avg_confidence: number | null;
}

export interface StatsResult {
    history: TrainingLogEntry[];
    totalSessions: number;
    latestF1: number | null;
    latestAccuracy: number | null;
    latestConfidence: number | null;
    totalRecords: number;
}