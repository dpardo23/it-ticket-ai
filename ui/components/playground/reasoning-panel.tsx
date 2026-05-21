'use client'

import React, { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { BrainCircuit, Gauge, Layers, CheckCircle2, AlertTriangle, BarChart3, Cpu, Loader2, Send, Thermometer, FileText, Wand2, Zap } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { SingleInferenceResult, BatchInferenceResult, BlindBatchResult } from '@/types/ai'

interface ReasoningPanelProps {
    activeTab: 'manual' | 'batch' | 'predict'
    manualResult: SingleInferenceResult | null
    batchResult: BatchInferenceResult | null
    blindResult: BlindBatchResult | null
    isProcessing: boolean
    onFeedback: (originalText: string, correctDepartment: string) => void
}

function SkeletonBlock({ className }: { className: string }) {
    return <div className={`animate-pulse rounded-lg bg-muted/60 ${className}`} />
}

function safeNumber(n: any): number | null {
    const num = Number(n)
    if (Number.isNaN(num) || !Number.isFinite(num)) return null
    return num
}

function Heatmap({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
    const maxValue = useMemo(() => {
        let max = 0
        for (const row of matrix) {
            for (const val of row) max = Math.max(max, val)
        }
        return max
    }, [matrix])

    return (
        <div className="overflow-auto max-h-[320px] rounded-lg border border-border/60 bg-background">
            <div className="min-w-[640px]">
                <div className="grid" style={{ gridTemplateColumns: `180px repeat(${labels.length}, 1fr)` }}>
                    <div className="p-2 text-[11px] font-bold text-muted-foreground border-b border-border/60 bg-muted/50 sticky top-0 backdrop-blur">
                        Real \ Predicción
                    </div>
                    {labels.map((l) => (
                        <div key={l} className="p-2 text-[11px] font-bold text-muted-foreground border-b border-border/60 bg-muted/50 sticky top-0 backdrop-blur truncate">
                            {l}
                        </div>
                    ))}
                    {matrix.map((row, i) => (
                        <React.Fragment key={i}>
                            <div className="p-2 text-[11px] font-bold border-b border-border/60 bg-card text-foreground truncate">
                                {labels[i]}
                            </div>
                            {row.map((val, j) => {
                                const intensity = maxValue > 0 ? val / maxValue : 0
                                return (
                                    <div
                                        key={j}
                                        className="p-2 border-b border-border/60 text-[11px] font-bold text-center transition-colors duration-300"
                                        style={{ background: `rgba(239,68,68,${intensity * 0.8})` }}
                                    >
                                        <span className={intensity > 0.4 ? "text-white" : "text-foreground"}>
                                            {val}
                                        </span>
                                    </div>
                                )
                            })}
                        </React.Fragment>
                    ))}
                </div>
            </div>
        </div>
    )
}

export function ReasoningPanel({ activeTab, manualResult, batchResult, blindResult, isProcessing, onFeedback }: ReasoningPanelProps) {
    const [feedbackDept, setFeedbackDept] = useState('')

    // ==========================================
    // CÁLCULOS MANUAL
    // ==========================================
    const originalText = useMemo(() => {
        if (!manualResult) return ''
        if (manualResult.originalText) return manualResult.originalText
        if (manualResult.tokens?.length) return manualResult.tokens.join(' ')
        return ''
    }, [manualResult])

    const cleanText = useMemo(() => {
        if (!manualResult) return null
        return manualResult.cleanText ?? null
    }, [manualResult])

    const sortedProbs = useMemo(() => {
        if (!manualResult?.probabilities) return []
        return Object.entries(manualResult.probabilities)
            .map(([k, v]) => [k, safeNumber(v)] as [string, number | null])
            .filter(([, v]) => v !== null)
            .map(([k, v]) => [k, v as number] as [string, number])
            .sort((a, b) => b[1] - a[1])
    }, [manualResult])

    const winnerScore = useMemo(() => {
        if (!manualResult?.winner || !sortedProbs.length) return null
        const found = sortedProbs.find(([label]) => label === manualResult.winner)
        return found ? found[1] : null
    }, [manualResult, sortedProbs])

    const confidenceLabel = useMemo(() => {
        if (winnerScore === null) return 'Sin datos'
        if (winnerScore >= 0.75) return 'Alta confianza'
        if (winnerScore >= 0.5) return 'Confianza media'
        return 'Baja confianza'
    }, [winnerScore])

    const handleSendFeedback = () => {
        if (!feedbackDept.trim() || !originalText.trim()) return
        onFeedback(originalText, feedbackDept.trim())
        setFeedbackDept('')
    }

    // ==========================================
    // CÁLCULOS BATCH (ENTRENAMIENTO)
    // ==========================================
    const distributionSorted = useMemo(() => {
        if (!batchResult?.departmentDistribution) return []
        return Object.entries(batchResult.departmentDistribution).sort((a, b) => b[1] - a[1])
    }, [batchResult])

    const maxDist = useMemo(() => {
        if (!distributionSorted.length) return 0
        return distributionSorted[0][1]
    }, [distributionSorted])


    // ==========================================
    // CÁLCULOS INFERENCIA CIEGA (NUEVO)
    // ==========================================
    const predictDistSorted = useMemo(() => {
        if (!blindResult?.departmentDistribution) return []
        return Object.entries(blindResult.departmentDistribution).sort((a, b) => b[1] - a[1])
    }, [blindResult])

    const predictMaxDist = useMemo(() => {
        if (!predictDistSorted.length) return 0
        return predictDistSorted[0][1]
    }, [predictDistSorted])

    const avgConfidence = useMemo(() => {
        if (!blindResult?.predictions.length) return 0
        const total = blindResult.predictions.reduce((acc, p) => acc + parseFloat(p.confidence), 0)
        return total / blindResult.predictions.length
    }, [blindResult])


    return (
        <div className="h-full bg-card border border-border/60 shadow-sm flex flex-col rounded-xl overflow-hidden">
            {/* CABECERA */}
            <div className="h-[52px] px-4 flex items-center justify-between border-b border-border/60 bg-muted/20 text-foreground">
                <div className="flex items-center gap-2 font-semibold tracking-tight text-muted-foreground">
                    <BrainCircuit size={16} className="text-primary" />
                    <span className="text-sm">
                        {activeTab === 'manual' ? 'Análisis del Motor (Pipeline NLP)' : activeTab === 'batch' ? 'Auditoría de IA (Lotes)' : 'Clasificador Masivo (Ciego)'}
                    </span>
                </div>
                {isProcessing && (
                    <span className="text-[11px] flex items-center gap-2 text-primary font-bold tracking-wider uppercase">
                        <Loader2 size={13} className="animate-spin" /> Procesando
                    </span>
                )}
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {/* MANUAL */}
                {activeTab === 'manual' && (
                    <>
                        {isProcessing && !manualResult && (
                            <div className="space-y-3">
                                <div className="grid grid-cols-2 gap-3"><SkeletonBlock className="h-[92px]" /><SkeletonBlock className="h-[92px]" /></div>
                                <SkeletonBlock className="h-[180px]" />
                                <SkeletonBlock className="h-[150px]" />
                            </div>
                        )}

                        {!isProcessing && !manualResult && (
                            <div className="h-full flex flex-col items-center justify-center text-center p-6 text-muted-foreground">
                                <Cpu size={40} className="text-muted/50 mb-3" />
                                <h3 className="font-bold text-foreground">Motor IA a la espera</h3>
                                <p className="text-xs leading-relaxed max-w-[280px] mt-2">
                                    Ingresa un ticket para visualizar el procesamiento del Lenguaje Natural y las probabilidades de predicción.
                                </p>
                            </div>
                        )}

                        {manualResult && manualResult.isGarbage && (
                            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.4, ease: "circOut" }} className="h-full flex flex-col items-center justify-center text-center p-6 space-y-4">
                                <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                                    <AlertTriangle size={32} className="text-primary" />
                                </div>
                                <div>
                                    <h2 className="text-lg font-bold text-foreground">Entrada Rechazada</h2>
                                    <p className="text-sm text-muted-foreground max-w-[320px] mt-2 leading-relaxed">
                                        {manualResult.garbageMessage || 'El texto ingresado carece del contexto técnico necesario o es ilegible para el análisis algorítmico.'}
                                    </p>
                                </div>
                            </motion.div>
                        )}

                        {manualResult && !manualResult.isGarbage && (
                            <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, ease: "circOut" }} className="space-y-4">
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="rounded-lg border border-border/60 bg-card p-3 flex flex-col gap-1 shadow-sm">
                                        <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground">
                                            <CheckCircle2 size={14} className="text-green-500" /> Predicción Final
                                        </div>
                                        <div className="text-lg font-extrabold text-foreground tracking-tight">{manualResult.winner}</div>
                                        <div className="text-[11px] text-muted-foreground">
                                            {winnerScore !== null ? <>{confidenceLabel} • Exactitud: <span className="font-semibold text-foreground">{(winnerScore * 100).toFixed(1)}%</span></> : <span className="text-primary font-semibold">Sin probabilidades</span>}
                                        </div>
                                        <div className="text-[11px] text-muted-foreground">Enrutamiento: <span className="font-semibold text-foreground">{manualResult.level || 'N/A'}</span></div>
                                    </div>
                                    <div className="rounded-lg border border-border/60 bg-card p-3 flex flex-col gap-1 shadow-sm">
                                        <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground">
                                            <Gauge size={14} className="text-primary" /> Latencia (O)
                                        </div>
                                        <div className="text-lg font-extrabold text-foreground tracking-tight">{manualResult.latency} ms</div>
                                        <div className="text-[11px] text-muted-foreground">Inferencia pura online</div>
                                    </div>
                                </div>

                                <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                    <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground mb-3">
                                        <Wand2 size={14} className="text-primary" /> Preprocesamiento (Pipeline NLP)
                                    </div>
                                    <div className="grid grid-cols-2 gap-3">
                                        <div className="rounded-md border border-border/60 bg-muted/30 p-3">
                                            <div className="flex items-center gap-2 text-[11px] font-bold text-muted-foreground mb-2">
                                                <FileText size={12} /> Texto Crudo
                                            </div>
                                            <div className="text-[12px] leading-relaxed whitespace-pre-wrap">{manualResult.originalText || '(No disponible)'}</div>
                                        </div>
                                        <div className="rounded-md border border-border/60 bg-muted/30 p-3">
                                            <div className="flex items-center gap-2 text-[11px] font-bold text-muted-foreground mb-2">
                                                <Wand2 size={12} /> Limpio & Lematizado
                                            </div>
                                            <div className="text-[12px] leading-relaxed whitespace-pre-wrap">{cleanText || '(No disponible)'}</div>
                                        </div>
                                    </div>
                                </div>

                                <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                    <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground mb-3">
                                        <Layers size={14} className="text-primary" /> Array de Tokens
                                    </div>
                                    {manualResult.tokens?.length ? (
                                        <div className="flex flex-wrap gap-1.5">
                                            {manualResult.tokens.slice(0, 80).map((t, i) => (
                                                <span key={i} className="px-2 py-0.5 text-[11px] font-mono rounded-md bg-primary/10 text-primary border border-primary/20">{t}</span>
                                            ))}
                                        </div>
                                    ) : <div className="text-[12px] text-primary font-semibold">Vacío.</div>}
                                </div>

                                <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                    <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground mb-3">
                                        <Thermometer size={14} className="text-primary" /> Desglose de Probabilidades (Softmax)
                                    </div>
                                    {sortedProbs.length ? (
                                        <div className="space-y-3">
                                            {sortedProbs.slice(0, 6).map(([label, prob]) => (
                                                <div key={label} className="space-y-1.5">
                                                    <div className="flex justify-between text-[11px]">
                                                        <span className="font-bold text-foreground truncate max-w-[70%]">{label}</span>
                                                        <span className="text-muted-foreground font-mono">{(prob * 100).toFixed(1)}%</span>
                                                    </div>
                                                    <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden">
                                                        <div className="h-full bg-primary transition-all duration-700 ease-out" style={{ width: `${Math.min(prob * 100, 100)}%` }} />
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    ) : <div className="text-[12px] text-primary">No se generaron probabilidades.</div>}
                                </div>

                                <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                    <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground mb-3">
                                        <Send size={14} className="text-green-500" /> Aprendizaje Continuo (Feedback)
                                    </div>
                                    <div className="text-[11px] text-muted-foreground mb-3">
                                        Si la IA falló, ingresa el departamento correcto para que el modelo ajuste sus pesos (partial_fit).
                                    </div>
                                    <div className="flex gap-2">
                                        <Input
                                            value={feedbackDept}
                                            onChange={(e) => setFeedbackDept(e.target.value)}
                                            placeholder="Ej: Base de Datos"
                                            className="h-9 rounded-md bg-background"
                                            disabled={isProcessing}
                                        />
                                        <Button
                                            onClick={handleSendFeedback}
                                            disabled={isProcessing || feedbackDept.trim().length < 2}
                                            className="h-9 rounded-md bg-green-600 hover:bg-green-500 text-white font-semibold"
                                        >
                                            Entrenar
                                        </Button>
                                    </div>
                                </div>
                            </motion.div>
                        )}
                    </>
                )}

                {/* LOTES (Entrenamiento) */}
                {activeTab === 'batch' && (
                    <>
                        {isProcessing && !batchResult && (
                            <div className="space-y-3">
                                <div className="grid grid-cols-3 gap-3"><SkeletonBlock className="h-[80px]" /><SkeletonBlock className="h-[80px]" /><SkeletonBlock className="h-[80px]" /></div>
                                <SkeletonBlock className="h-[140px]" />
                                <SkeletonBlock className="h-[240px]" />
                            </div>
                        )}

                        {!isProcessing && !batchResult && (
                            <div className="h-full flex flex-col items-center justify-center text-center p-6 text-muted-foreground">
                                <BarChart3 size={40} className="text-muted/50 mb-3" />
                                <h3 className="font-bold text-foreground">Auditoría Vacía</h3>
                                <p className="text-xs leading-relaxed max-w-[280px] mt-2">
                                    Sube un archivo CSV con tickets para evaluar las métricas globales del modelo.
                                </p>
                            </div>
                        )}

                        {batchResult && (
                            <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, ease: "circOut" }} className="space-y-4">
                                <div className="grid grid-cols-3 gap-3">
                                    <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                        <div className="text-[11px] font-semibold text-muted-foreground">F1-Score Ponderado</div>
                                        <div className="text-xl font-extrabold text-foreground">{batchResult.f1Score.toFixed(3)}</div>
                                    </div>
                                    <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                        <div className="text-[11px] font-semibold text-muted-foreground">Exactitud (Accuracy)</div>
                                        <div className="text-xl font-extrabold text-foreground">{batchResult.accuracy.toFixed(3)}</div>
                                    </div>
                                    <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                        <div className="text-[11px] font-semibold text-muted-foreground">Algoritmo Ganador</div>
                                        <div className="text-sm font-bold text-foreground truncate mt-1">{batchResult.bestModelName}</div>
                                    </div>
                                </div>

                                {batchResult.departmentDistribution && (
                                    <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                        <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground mb-4">
                                            <BarChart3 size={14} className="text-primary" /> Distribución de Clases
                                        </div>
                                        <div className="space-y-3">
                                            {distributionSorted.slice(0, 10).map(([dep, count]) => {
                                                const width = maxDist > 0 ? (count / maxDist) * 100 : 0
                                                return (
                                                    <div key={dep} className="space-y-1.5">
                                                        <div className="flex justify-between text-[11px]">
                                                            <span className="font-bold text-foreground truncate max-w-[70%]">{dep}</span>
                                                            <span className="text-muted-foreground font-mono">{count} tickets</span>
                                                        </div>
                                                        <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden">
                                                            <div className="h-full bg-primary transition-all duration-700 ease-out" style={{ width: `${width}%` }} />
                                                        </div>
                                                    </div>
                                                )
                                            })}
                                        </div>
                                    </div>
                                )}

                                {batchResult.confusionMatrix && batchResult.labels && (
                                    <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                        <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground mb-3">
                                            <Thermometer size={14} className="text-primary" /> Matriz de Confusión
                                        </div>
                                        <Heatmap matrix={batchResult.confusionMatrix} labels={batchResult.labels} />
                                    </div>
                                )}

                                {batchResult.globalTfidf?.length > 0 && (
                                    <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                        <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground mb-3">
                                            <Gauge size={14} className="text-primary" /> Diccionario Vectorial TF-IDF
                                        </div>
                                        <div className="flex flex-wrap gap-2">
                                            {batchResult.globalTfidf.slice(0, 40).map((t, i) => (
                                                <span key={i} className="px-2 py-1 text-[10px] font-mono font-medium rounded-md bg-muted text-foreground border border-border/60">
                                                    {t.term} <span className="text-muted-foreground opacity-70">({t.weight.toFixed(2)})</span>
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </motion.div>
                        )}
                    </>
                )}

                {/* INFERENCIA CIEGA (NUEVO) */}
                {activeTab === 'predict' && (
                    <>
                        {isProcessing && !blindResult && (
                            <div className="space-y-3">
                                <div className="grid grid-cols-3 gap-3"><SkeletonBlock className="h-[80px]" /><SkeletonBlock className="h-[80px]" /><SkeletonBlock className="h-[80px]" /></div>
                                <SkeletonBlock className="h-[200px]" />
                                <SkeletonBlock className="h-[300px]" />
                            </div>
                        )}

                        {!isProcessing && !blindResult && (
                            <div className="h-full flex flex-col items-center justify-center text-center p-6 text-muted-foreground">
                                <Zap size={40} className="text-muted/50 mb-3" />
                                <h3 className="font-bold text-foreground">Inferencia Pendiente</h3>
                                <p className="text-xs leading-relaxed max-w-[280px] mt-2">
                                    Sube un CSV sin etiquetas para visualizar la clasificación masiva en milisegundos.
                                </p>
                            </div>
                        )}

                        {blindResult && (
                            <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, ease: "circOut" }} className="space-y-4">

                                {/* MÉTRICAS SUPERIORES */}
                                <div className="grid grid-cols-3 gap-3">
                                    <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                        <div className="text-[11px] font-semibold text-muted-foreground">Tickets Procesados</div>
                                        <div className="text-xl font-extrabold text-foreground">{blindResult.totalTickets}</div>
                                    </div>
                                    <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                        <div className="text-[11px] font-semibold text-muted-foreground">Confianza Media</div>
                                        <div className="text-xl font-extrabold text-foreground">{avgConfidence.toFixed(1)}%</div>
                                    </div>
                                    <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                        <div className="text-[11px] font-semibold text-muted-foreground">Velocidad (O: N)</div>
                                        <div className="text-xl font-extrabold text-foreground">{blindResult.speed} ms</div>
                                    </div>
                                </div>

                                {/* DISTRIBUCIÓN DE CLASES */}
                                {blindResult.departmentDistribution && (
                                    <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                        <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground mb-4">
                                            <BarChart3 size={14} className="text-primary" /> Demanda por Departamento
                                        </div>
                                        <div className="grid grid-cols-2 gap-x-6 gap-y-3">
                                            {predictDistSorted.map(([dep, count]) => {
                                                const width = predictMaxDist > 0 ? (count / predictMaxDist) * 100 : 0
                                                return (
                                                    <div key={dep} className="space-y-1.5">
                                                        <div className="flex justify-between text-[11px]">
                                                            <span className="font-bold text-foreground truncate max-w-[75%]">{dep}</span>
                                                            <span className="text-muted-foreground font-mono">{count}</span>
                                                        </div>
                                                        <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden">
                                                            <div className="h-full bg-primary transition-all duration-700 ease-out" style={{ width: `${width}%` }} />
                                                        </div>
                                                    </div>
                                                )
                                            })}
                                        </div>
                                    </div>
                                )}

                                {/* TABLA DE RESULTADOS */}
                                <div className="rounded-lg border border-border/60 bg-card shadow-sm flex flex-col h-[400px]">
                                    <div className="p-3 border-b border-border/60 bg-muted/20 text-[11px] font-semibold text-muted-foreground flex justify-between uppercase tracking-wider">
                                        <span>ID / Ticket Original</span>
                                        <span>Clasificación IA</span>
                                    </div>
                                    <div className="flex-1 overflow-auto">
                                        {blindResult.predictions.map((p) => (
                                            <div key={p.id} className="p-3 border-b border-border/40 hover:bg-muted/30 transition-colors flex justify-between gap-4 items-center">
                                                <div className="min-w-0 flex-1">
                                                    <div className="text-[10px] text-muted-foreground font-mono mb-0.5">#{p.id}</div>
                                                    <div className="text-[12px] text-foreground truncate" title={p.text_original}>{p.text_original}</div>
                                                </div>
                                                <div className="shrink-0 text-right">
                                                    <div className="text-xs font-bold text-primary">{p.predicted_department}</div>
                                                    <div className="text-[10px] text-muted-foreground font-mono mt-0.5">{p.confidence} de confianza</div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                            </motion.div>
                        )}
                    </>
                )}
            </div>
        </div>
    )
}