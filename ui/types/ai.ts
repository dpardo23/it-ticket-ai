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