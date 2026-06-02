'use client'

import React, { useEffect, useMemo, useState } from 'react'
import { createPortal } from 'react-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'sonner'
import {
    BrainCircuit, Gauge, Layers, CheckCircle2, AlertTriangle, BarChart3,
    Cpu, Loader2, Send, Thermometer, FileText, Wand2, Zap,
    Lightbulb, X, Target, TrendingUp, RefreshCw, Database, Award, ThumbsUp
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { SingleInferenceResult, BatchInferenceResult, BlindBatchResult, SolutionItem, StatsResult, TrainingLogEntry } from '@/types/ai'

/**
 * Panel de resultados del motor de IA (sección derecha del Playground).
 *
 * Renderiza cuatro vistas exclusivas según activeTab: análisis NLP individual,
 * auditoría de entrenamiento por lotes, clasificación masiva ciega y curva de
 * aprendizaje histórica. El modal de soluciones conocidas se monta en
 * document.body vía portal para evitar problemas de z-index.
 *
 * Relaciones: AIPlayground (props y callbacks), LearningCurve (subcomponente
 * SVG interactivo), Heatmap (matriz de confusión), api.main vía callbacks.
 */
/**
 * Contrato de props del panel de resultados.
 *
 * onVoteSolution y onSubmitSolutionFeedback son los puntos de extensión
 * para el sistema de conocimiento colaborativo: el primero registra que
 * una solución existente funcionó; el segundo persiste una nueva solución
 * aportada por el perfil cuando ninguna de las conocidas fue útil.
 */
interface ReasoningPanelProps {
    activeTab: 'manual' | 'batch' | 'predict' | 'stats'
    manualResult: SingleInferenceResult | null
    batchResult: BatchInferenceResult | null
    blindResult: BlindBatchResult | null
    statsResult: StatsResult | null
    isProcessing: boolean
    onFeedback: (originalText: string, correctDepartment: string) => void
    onFetchSolutions: (department: string) => Promise<SolutionItem[]>
    onFetchStats: () => void
    onVoteSolution: (department: string, solution: string) => Promise<void>
    onSubmitSolutionFeedback: (department: string, solution: string) => Promise<boolean>
}

function SkeletonBlock({ className }: { className: string }) {
    return <div className={`animate-pulse rounded-lg bg-muted/60 ${className}`} />
}

function safeNumber(n: unknown): number | null {
    const num = Number(n)
    return Number.isNaN(num) || !Number.isFinite(num) ? null : num
}

function metricColor(val: number): string {
    if (val >= 0.80) return 'text-green-500'
    if (val >= 0.60) return 'text-amber-500'
    return 'text-red-500'
}

function Heatmap({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
    const maxValue = useMemo(() => {
        let max = 0
        for (const row of matrix) for (const val of row) max = Math.max(max, val)
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
                                        className="p-2 border-b border-border/60 text-[11px] font-bold text-center transition-colors duration-100"
                                        style={{ background: `rgba(239,68,68,${intensity * 0.8})` }}
                                    >
                                        <span className={intensity > 0.4 ? 'text-white' : 'text-foreground'}>{val}</span>
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

const TRIGGER_LABEL: Record<string, string> = {
    csv_upload: 'CSV',
    feedback_retrain: 'Feedback',
    startup: 'Startup',
}
const TRIGGER_COLOR: Record<string, string> = {
    csv_upload: 'bg-primary/20 text-primary',
    feedback_retrain: 'bg-green-500/20 text-green-500',
    startup: 'bg-amber-500/20 text-amber-600',
}

const CHART_COLORS = { f1: '#ef4444', acc: '#94a3b8', conf: '#22c55e' } as const

function LearningCurve({ entries }: { entries: TrainingLogEntry[] }) {
    const data = useMemo(() => [...entries].reverse(), [entries])
    const [hoveredIdx, setHoveredIdx] = useState<number | null>(null)

    if (data.length === 0) return null

    const W = 540
    const H = 220
    const PAD = { top: 16, right: 24, bottom: 38, left: 46 }
    const innerW = W - PAD.left - PAD.right
    const innerH = H - PAD.top - PAD.bottom
    const GRID = [0, 0.25, 0.5, 0.75, 1.0]
    const TIP_W = 176

    const xScale = (i: number) =>
        data.length === 1 ? PAD.left + innerW / 2 : PAD.left + (i / (data.length - 1)) * innerW
    const yScale = (v: number) => PAD.top + innerH - v * innerH

    const makePath = (fn: (d: TrainingLogEntry) => number | null) =>
        data.reduce((acc, d, i) => {
            const v = fn(d)
            if (v === null) return acc
            return acc + (acc === '' ? 'M' : 'L') + `${xScale(i).toFixed(1)},${yScale(v).toFixed(1)} `
        }, '').trim()

    const pathF1   = makePath(d => d.f1_score)
    const pathAcc  = makePath(d => d.accuracy)
    const pathConf = makePath(d => d.avg_confidence)

    const h = hoveredIdx !== null ? data[hoveredIdx] : null
    const tipFlipped = hoveredIdx !== null && xScale(hoveredIdx) > W - TIP_W - 12

    return (
        <div className="w-full overflow-x-auto select-none">
            <div className="relative" style={{ minWidth: 300 }}>
                <svg viewBox={`0 0 ${W} ${H}`} className="w-full">

                    {GRID.map(v => (
                        <g key={v}>
                            <line
                                x1={PAD.left} y1={yScale(v)} x2={W - PAD.right} y2={yScale(v)}
                                style={{ stroke: 'hsl(var(--border))' }}
                                strokeWidth={v === 0 ? 1.5 : 1}
                            />
                            <text
                                x={PAD.left - 6} y={yScale(v) + 4} textAnchor="end" fontSize={10}
                                style={{ fill: 'hsl(var(--muted-foreground))' }}
                            >
                                {(v * 100).toFixed(0)}%
                            </text>
                        </g>
                    ))}

                    {pathAcc  && <path d={pathAcc}  fill="none" stroke={CHART_COLORS.acc}  strokeWidth={2}   strokeDasharray="6 4" opacity={0.8} />}
                    {pathConf && <path d={pathConf} fill="none" stroke={CHART_COLORS.conf} strokeWidth={2.5} opacity={0.9} />}
                    {pathF1   && <path d={pathF1}   fill="none" stroke={CHART_COLORS.f1}   strokeWidth={2.5} />}

                    {data.map((d, i) => (
                        <g key={i}
                            onMouseEnter={() => setHoveredIdx(i)}
                            onMouseLeave={() => setHoveredIdx(null)}
                            style={{ cursor: 'crosshair' }}
                        >
                            <rect x={xScale(i) - 18} y={PAD.top} width={36} height={innerH} fill="transparent" />

                            {hoveredIdx === i && (
                                <line
                                    x1={xScale(i)} y1={PAD.top} x2={xScale(i)} y2={PAD.top + innerH}
                                    style={{ stroke: 'hsl(var(--border))' }}
                                    strokeWidth={1} strokeDasharray="4 3"
                                />
                            )}

                            <circle cx={xScale(i)} cy={yScale(d.accuracy)}      r={hoveredIdx === i ? 5 : 3.5} fill={CHART_COLORS.acc} />
                            {d.avg_confidence !== null && (
                                <circle cx={xScale(i)} cy={yScale(d.avg_confidence)} r={hoveredIdx === i ? 5 : 3.5} fill={CHART_COLORS.conf} />
                            )}
                            <circle cx={xScale(i)} cy={yScale(d.f1_score)}       r={hoveredIdx === i ? 6 : 4}   fill={CHART_COLORS.f1} />

                            {(data.length <= 14 || i % Math.ceil(data.length / 12) === 0) && (
                                <text
                                    x={xScale(i)} y={H - 8} textAnchor="middle" fontSize={9}
                                    style={{ fill: 'hsl(var(--muted-foreground))' }}
                                >
                                    #{i + 1}
                                </text>
                            )}
                        </g>
                    ))}
                </svg>

                {h !== null && hoveredIdx !== null && (
                    <div
                        className="absolute top-4 pointer-events-none z-10"
                        style={{
                            left: tipFlipped
                                ? `calc(${(xScale(hoveredIdx) / W) * 100}% - ${TIP_W + 8}px)`
                                : `calc(${(xScale(hoveredIdx) / W) * 100}% + 8px)`,
                        }}
                    >
                        <div
                            className="bg-popover border border-border rounded-lg shadow-xl p-3 text-popover-foreground"
                            style={{ minWidth: TIP_W }}
                        >
                            <div className="flex items-center gap-1.5 mb-0.5">
                                <span className="text-[11px] font-bold text-foreground">Sesión #{hoveredIdx + 1}</span>
                                <span className="text-[10px] text-muted-foreground">— {TRIGGER_LABEL[h.trigger_type] ?? h.trigger_type}</span>
                            </div>
                            <div className="text-[9px] text-muted-foreground mb-2">
                                {new Date(h.created_at).toLocaleString('es-AR', { dateStyle: 'short', timeStyle: 'short' })}
                            </div>
                            <div className="h-px bg-border mb-2" />
                            <div className="space-y-1.5">
                                <div className="flex items-center gap-1.5 text-[10px]">
                                    <span className="inline-block w-2.5 h-2.5 rounded-full shrink-0" style={{ background: CHART_COLORS.f1 }} />
                                    <span className="text-muted-foreground">F1-Score:</span>
                                    <span className="font-bold" style={{ color: CHART_COLORS.f1 }}>{(h.f1_score * 100).toFixed(1)}%</span>
                                </div>
                                <div className="flex items-center gap-1.5 text-[10px]">
                                    <span className="inline-block w-2.5 h-2.5 rounded-full shrink-0" style={{ background: CHART_COLORS.acc }} />
                                    <span className="text-muted-foreground">Accuracy:</span>
                                    <span className="font-bold" style={{ color: CHART_COLORS.acc }}>{(h.accuracy * 100).toFixed(1)}%</span>
                                </div>
                                <div className="flex items-center gap-1.5 text-[10px]">
                                    <span className="inline-block w-2.5 h-2.5 rounded-full shrink-0" style={{ background: CHART_COLORS.conf }} />
                                    <span className="text-muted-foreground">Confianza:</span>
                                    <span className="font-bold" style={{ color: CHART_COLORS.conf }}>
                                        {h.avg_confidence !== null ? `${(h.avg_confidence * 100).toFixed(1)}%` : '—'}
                                    </span>
                                </div>
                            </div>
                            <div className="mt-2 pt-2 border-t border-border text-[9px] text-muted-foreground">
                                {h.record_count.toLocaleString()} tickets · {h.department_count} áreas
                            </div>
                        </div>
                    </div>
                )}
            </div>

            <div className="flex items-center gap-5 mt-2 px-1">
                <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                    <svg width="20" height="8"><line x1="0" y1="4" x2="20" y2="4" stroke={CHART_COLORS.f1} strokeWidth="2.5" /></svg>
                    F1-Score
                </div>
                <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                    <svg width="20" height="8"><line x1="0" y1="4" x2="20" y2="4" stroke={CHART_COLORS.acc} strokeWidth="2" strokeDasharray="5 3" /></svg>
                    Accuracy
                </div>
                <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                    <svg width="20" height="8"><line x1="0" y1="4" x2="20" y2="4" stroke={CHART_COLORS.conf} strokeWidth="2.5" /></svg>
                    Confianza Media
                </div>
            </div>
        </div>
    )
}

export function ReasoningPanel({
    activeTab, manualResult, batchResult, blindResult, statsResult,
    isProcessing, onFeedback, onFetchSolutions, onFetchStats,
    onVoteSolution, onSubmitSolutionFeedback,
}: ReasoningPanelProps) {
    const [feedbackDept, setFeedbackDept] = useState('')
    const [mounted, setMounted] = useState(false)
    const [solutionsOpen, setSolutionsOpen] = useState(false)
    const [solutionsList, setSolutionsList] = useState<SolutionItem[]>([])
    const [loadingSolutions, setLoadingSolutions] = useState(false)
    const [votedSolutions, setVotedSolutions] = useState<Set<string>>(new Set())
    const [userSolution, setUserSolution] = useState('')
    const [submittingFeedback, setSubmittingFeedback] = useState(false)

    useEffect(() => { setMounted(true) }, [])

    const originalText = useMemo(() => {
        if (!manualResult) return ''
        return manualResult.originalText || manualResult.tokens?.join(' ') || ''
    }, [manualResult])

    const sortedProbs = useMemo(() => {
        if (!manualResult?.probabilities) return []
        return Object.entries(manualResult.probabilities)
            .map(([k, v]) => [k, safeNumber(v)] as [string, number | null])
            .filter((e): e is [string, number] => e[1] !== null)
            .sort((a, b) => b[1] - a[1])
    }, [manualResult])

    const winnerScore = useMemo(() => {
        if (!manualResult?.winner || !sortedProbs.length) return null
        return sortedProbs.find(([l]) => l === manualResult.winner)?.[1] ?? null
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

    const handleOpenSolutions = async () => {
        if (!manualResult?.winner) return
        setSolutionsOpen(true)
        setLoadingSolutions(true)
        setSolutionsList([])
        setUserSolution('')
        const solutions = await onFetchSolutions(manualResult.winner)
        setSolutionsList(solutions)
        setLoadingSolutions(false)
    }

    const handleVote = async (solution: string) => {
        if (!manualResult?.winner) return
        setVotedSolutions(prev => new Set(prev).add(solution))
        await onVoteSolution(manualResult.winner, solution)
    }

    const handleSubmitUserSolution = async () => {
        if (!manualResult?.winner || userSolution.trim().length < 10) return
        setSubmittingFeedback(true)
        const ok = await onSubmitSolutionFeedback(manualResult.winner, userSolution.trim())
        if (ok) {
            setSolutionsList(prev => [
                { solution: userSolution.trim(), voteCount: 1, source: 'user_feedback' as const },
                ...prev
            ])
            setUserSolution('')
            toast.success('Solución aportada', { description: 'Visible para otros perfiles con el mismo problema.' })
        } else {
            toast.error('No se pudo enviar la solución', { description: 'Verificá la conexión al backend.' })
        }
        setSubmittingFeedback(false)
    }

    const distributionSorted = useMemo(() => {
        if (!batchResult?.departmentDistribution) return []
        return Object.entries(batchResult.departmentDistribution).sort((a, b) => b[1] - a[1])
    }, [batchResult])

    const maxDist = distributionSorted[0]?.[1] ?? 0

    const perClassMetrics = useMemo(() => {
        const cm = batchResult?.confusionMatrix
        const labels = batchResult?.labels
        if (!cm || !labels) return []
        return labels.map((label, i) => {
            const tp = cm[i]?.[i] ?? 0
            const fp = cm.reduce((s, row, ri) => ri !== i ? s + (row[i] ?? 0) : s, 0)
            const fn = (cm[i] ?? []).reduce((s, v, ci) => ci !== i ? s + v : s, 0)
            const support = (cm[i] ?? []).reduce((s, v) => s + v, 0)
            const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0
            const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0
            const f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0
            return { label, precision, recall, f1, support }
        })
    }, [batchResult])

    const macroAvg = useMemo(() => {
        if (!perClassMetrics.length) return null
        const avg = (arr: number[]) => arr.reduce((s, v) => s + v, 0) / arr.length
        return {
            precision: avg(perClassMetrics.map(m => m.precision)),
            recall: avg(perClassMetrics.map(m => m.recall)),
            f1: avg(perClassMetrics.map(m => m.f1)),
        }
    }, [perClassMetrics])

    const modelInterpretation = useMemo(() => {
        if (!macroAvg) return null
        const f1 = macroAvg.f1
        if (f1 >= 0.90) return { label: 'Modelo Excelente', desc: 'Listo para producción. Rendimiento excepcional en todas las clases.', color: 'text-green-500', bg: 'bg-green-500/10 border-green-500/30' }
        if (f1 >= 0.80) return { label: 'Modelo Bueno', desc: 'Alta fiabilidad. Recomendado para entornos con supervisión.', color: 'text-green-400', bg: 'bg-green-500/10 border-green-400/30' }
        if (f1 >= 0.70) return { label: 'Modelo Aceptable', desc: 'Rendimiento moderado. Considera agregar más datos de entrenamiento.', color: 'text-amber-500', bg: 'bg-amber-500/10 border-amber-500/30' }
        return { label: 'Modelo en Desarrollo', desc: 'Rendimiento bajo. Se necesitan más datos o revisar etiquetas.', color: 'text-red-500', bg: 'bg-red-500/10 border-red-500/30' }
    }, [macroAvg])

    const predictDistSorted = useMemo(() => {
        if (!blindResult?.departmentDistribution) return []
        return Object.entries(blindResult.departmentDistribution).sort((a, b) => b[1] - a[1])
    }, [blindResult])

    const predictMaxDist = predictDistSorted[0]?.[1] ?? 0

    const avgConfidence = useMemo(() => {
        if (!blindResult?.predictions.length) return 0
        return blindResult.predictions.reduce((acc, p) => acc + parseFloat(p.confidence), 0) / blindResult.predictions.length
    }, [blindResult])

    const solutionsPortal = mounted ? createPortal(
        <AnimatePresence>
            {solutionsOpen && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="fixed inset-0 z-[200] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
                    onClick={() => setSolutionsOpen(false)}
                >
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: 16 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 16 }}
                        transition={{ duration: 0.25, ease: 'circOut' }}
                        className="bg-card border border-border/60 rounded-xl shadow-2xl w-full max-w-md max-h-[85vh] flex flex-col overflow-hidden"
                        onClick={(e) => e.stopPropagation()}
                    >
                        {/* Header */}
                        <div className="flex items-center justify-between px-4 py-3 border-b border-border/60 bg-muted/20 shrink-0">
                            <div className="flex items-center gap-2 min-w-0">
                                <Lightbulb size={15} className="text-primary shrink-0" />
                                <div className="min-w-0">
                                    <p className="text-sm font-bold text-foreground">Base de Conocimiento</p>
                                    <p className="text-[10px] text-muted-foreground truncate">
                                        Área: <span className="font-semibold text-primary">{manualResult?.winner}</span>
                                    </p>
                                </div>
                            </div>
                            <button
                                onClick={() => setSolutionsOpen(false)}
                                className="p-1.5 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-foreground shrink-0 ml-2"
                            >
                                <X size={15} />
                            </button>
                        </div>

                        {/* Solutions list */}
                        <div className="flex-1 overflow-y-auto p-4">
                            {loadingSolutions ? (
                                <div className="flex flex-col items-center justify-center py-10 gap-3 text-muted-foreground">
                                    <Loader2 size={24} className="animate-spin text-primary" />
                                    <span className="text-xs">Consultando base de conocimiento...</span>
                                </div>
                            ) : solutionsList.length > 0 ? (
                                <div className="space-y-2.5">
                                    <p className="text-[11px] text-muted-foreground mb-3">
                                        {solutionsList.length} solución{solutionsList.length !== 1 ? 'es' : ''} encontrada{solutionsList.length !== 1 ? 's' : ''} · Las más votadas aparecen primero.
                                    </p>
                                    {solutionsList.map((item, idx) => (
                                        <motion.div
                                            key={idx}
                                            initial={{ opacity: 0, x: -8 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ delay: idx * 0.02 }}
                                            className="flex gap-3 p-3 rounded-lg border border-border/60 bg-muted/20 hover:bg-muted/40 transition-colors"
                                        >
                                            <div className="shrink-0 w-5 h-5 rounded-full bg-primary/15 flex items-center justify-center text-[10px] font-bold text-primary mt-0.5">
                                                {idx + 1}
                                            </div>
                                            <div className="flex-1 min-w-0">
                                                <p className="text-[12px] text-foreground leading-relaxed">{item.solution}</p>
                                                {(item.voteCount ?? 0) > 0 && (
                                                    <p className="text-[10px] text-muted-foreground mt-1">
                                                        <span className="text-green-600 font-semibold">{item.voteCount}</span> perfil{(item.voteCount ?? 0) !== 1 ? 'es' : ''} confirmaron esta solución
                                                    </p>
                                                )}
                                                {item.source === 'user_feedback' && (
                                                    <span className="inline-block mt-1 text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                        Aporte de la comunidad
                                                    </span>
                                                )}
                                            </div>
                                            <button
                                                onClick={() => handleVote(item.solution)}
                                                disabled={votedSolutions.has(item.solution)}
                                                className={`shrink-0 self-start flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-semibold border transition-all ${
                                                    votedSolutions.has(item.solution)
                                                        ? 'bg-green-500/10 text-green-600 border-green-500/20 cursor-default'
                                                        : 'bg-muted hover:bg-green-500/10 text-muted-foreground hover:text-green-600 border-border/60 hover:border-green-500/20'
                                                }`}
                                            >
                                                <ThumbsUp size={10} />
                                                {votedSolutions.has(item.solution) ? 'Votado' : 'Funcionó'}
                                            </button>
                                        </motion.div>
                                    ))}
                                </div>
                            ) : (
                                <div className="flex flex-col items-center justify-center py-8 gap-3 text-center">
                                    <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center">
                                        <Lightbulb size={22} className="text-muted-foreground" />
                                    </div>
                                    <div>
                                        <p className="text-sm font-semibold text-foreground">Sin soluciones registradas</p>
                                        <p className="text-[11px] text-muted-foreground mt-1 max-w-[240px] leading-relaxed">
                                            Sé el primero en aportar una solución para este departamento.
                                        </p>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* User solution form */}
                        <div className="shrink-0 border-t border-border/60 bg-muted/10 p-4 space-y-3">
                            <div>
                                <p className="text-[11px] font-semibold text-foreground">¿Ninguna funcionó?</p>
                                <p className="text-[10px] text-muted-foreground mt-0.5">Aportá tu solución para ayudar a otros perfiles con el mismo problema.</p>
                            </div>
                            <textarea
                                value={userSolution}
                                onChange={(e) => setUserSolution(e.target.value)}
                                placeholder="Describe detalladamente la solución que funcionó..."
                                disabled={submittingFeedback}
                                className="w-full min-h-[72px] resize-none rounded-lg border border-input bg-background px-3 py-2 text-[12px] text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50 transition-all"
                            />
                            <Button
                                onClick={handleSubmitUserSolution}
                                disabled={submittingFeedback || userSolution.trim().length < 10}
                                className="w-full h-8 text-[11px] bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20 rounded-md font-semibold"
                            >
                                {submittingFeedback ? (
                                    <><Loader2 size={11} className="animate-spin mr-1.5" /> Enviando...</>
                                ) : (
                                    <><Send size={11} className="mr-1.5" /> Aportar Solución</>
                                )}
                            </Button>
                        </div>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>,
        document.body
    ) : null

    return (
        <>
            <div className="h-full bg-card border border-border/60 shadow-sm flex flex-col rounded-xl overflow-hidden">
                {/* Header */}
                <div className="min-h-[52px] h-auto py-3 px-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-border/60 bg-muted/20 text-foreground">
                    <div className="flex items-center gap-2 font-semibold tracking-tight text-muted-foreground">
                        <BrainCircuit size={16} className="text-primary shrink-0" />
                        <span className="text-sm">
                            {activeTab === 'manual' ? 'Análisis del Motor (Pipeline NLP)'
                                : activeTab === 'batch' ? 'Auditoría de IA (Lotes)'
                                : activeTab === 'predict' ? 'Clasificador Masivo (Ciego)'
                                : 'Curva de Aprendizaje — Progreso de la IA'}
                        </span>
                    </div>
                    {isProcessing && (
                        <span className="text-[11px] flex items-center gap-2 text-primary font-bold tracking-wider uppercase">
                            <Loader2 size={13} className="animate-spin" /> Procesando
                        </span>
                    )}
                </div>

                <div className="flex-1 overflow-y-auto p-4 space-y-4 flex flex-col">

                    {/* ── MANUAL ─────────────────────────────────────────────────── */}
                    {activeTab === 'manual' && (
                        <>
                            {isProcessing && !manualResult && (
                                <div className="space-y-3">
                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                        <SkeletonBlock className="h-[108px]" />
                                        <SkeletonBlock className="h-[108px]" />
                                    </div>
                                    <SkeletonBlock className="h-[180px]" />
                                    <SkeletonBlock className="h-[150px]" />
                                </div>
                            )}

                            {!isProcessing && !manualResult && (
                                <div className="flex-1 flex flex-col items-center justify-center text-center p-6 text-muted-foreground">
                                    <Cpu size={40} className="text-muted/50 mb-3" />
                                    <h3 className="font-bold text-foreground">Motor IA a la espera</h3>
                                    <p className="text-xs leading-relaxed max-w-[280px] mt-2">
                                        Ingresa un ticket para visualizar el procesamiento del Lenguaje Natural y las probabilidades de predicción.
                                    </p>
                                </div>
                            )}

                            {manualResult?.isGarbage && (
                                <motion.div
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    transition={{ duration: 0.12, ease: 'easeOut' }}
                                    className="flex-1 flex flex-col items-center justify-center text-center p-6 space-y-4"
                                >
                                    <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                                        <AlertTriangle size={32} className="text-primary" />
                                    </div>
                                    <div>
                                        <h2 className="text-lg font-bold text-foreground">Entrada Rechazada</h2>
                                        <p className="text-sm text-muted-foreground max-w-[320px] mt-2 leading-relaxed">
                                            {manualResult.garbageMessage || 'El texto ingresado carece del contexto técnico necesario.'}
                                        </p>
                                    </div>
                                </motion.div>
                            )}

                            {manualResult && !manualResult.isGarbage && (
                                <motion.div
                                    initial={{ opacity: 0, y: 15 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ duration: 0.12, ease: 'easeOut' }}
                                    className="space-y-4"
                                >
                                    {/* Top cards */}
                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                        {/* Winner card */}
                                        <div className="rounded-lg border border-border/60 bg-card p-3 flex flex-col gap-1 shadow-sm">
                                            <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground">
                                                <CheckCircle2 size={14} className="text-green-500" /> Predicción Final
                                            </div>
                                            <div className="text-lg font-extrabold text-foreground tracking-tight">{manualResult.winner}</div>
                                            <div className="text-[11px] text-muted-foreground">
                                                {winnerScore !== null
                                                    ? <>{confidenceLabel} · <span className="font-semibold text-foreground">{(winnerScore * 100).toFixed(1)}%</span></>
                                                    : <span className="text-primary font-semibold">Sin probabilidades</span>}
                                            </div>
                                            <div className="text-[11px] text-muted-foreground">
                                                Enrutamiento: <span className="font-semibold text-foreground">{manualResult.level || 'N/A'}</span>
                                            </div>
                                            <Button
                                                onClick={handleOpenSolutions}
                                                size="sm"
                                                className="mt-2 h-7 text-[11px] bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20 rounded-md w-full"
                                            >
                                                <Lightbulb size={12} className="mr-1.5" />
                                                Ver Soluciones
                                            </Button>
                                        </div>

                                        {/* Latency card */}
                                        <div className="rounded-lg border border-border/60 bg-card p-3 flex flex-col gap-1 shadow-sm">
                                            <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground">
                                                <Gauge size={14} className="text-primary" /> Latencia
                                            </div>
                                            <div className="text-lg font-extrabold text-foreground tracking-tight">{manualResult.latency} ms</div>
                                            <div className="text-[11px] text-muted-foreground">Inferencia pura online</div>
                                        </div>
                                    </div>

                                    {/* NLP Pipeline */}
                                    <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                        <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground mb-3">
                                            <Wand2 size={14} className="text-primary" /> Preprocesamiento (Pipeline NLP)
                                        </div>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                            <div className="rounded-md border border-border/60 bg-muted/30 p-3">
                                                <div className="flex items-center gap-2 text-[11px] font-bold text-muted-foreground mb-2">
                                                    <FileText size={12} /> Texto Crudo
                                                </div>
                                                <div className="text-[12px] leading-relaxed whitespace-pre-wrap">{manualResult.originalText || '(No disponible)'}</div>
                                            </div>
                                            <div className="rounded-md border border-border/60 bg-muted/30 p-3">
                                                <div className="flex items-center gap-2 text-[11px] font-bold text-muted-foreground mb-2">
                                                    <Wand2 size={12} /> Limpio &amp; Lematizado
                                                </div>
                                                <div className="text-[12px] leading-relaxed whitespace-pre-wrap">{manualResult.cleanText || '(No disponible)'}</div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Tokens */}
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

                                    {/* Softmax — TOP 3 */}
                                    <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                        <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground mb-3">
                                            <Thermometer size={14} className="text-primary" /> Probabilidades Softmax (Top 3)
                                        </div>
                                        {sortedProbs.length ? (
                                            <div className="space-y-3">
                                                {sortedProbs.slice(0, 3).map(([label, prob], idx) => (
                                                    <div key={label} className="space-y-1.5">
                                                        <div className="flex justify-between text-[11px] items-center">
                                                            <div className="flex items-center gap-1.5 min-w-0">
                                                                {idx === 0 && (
                                                                    <span className="shrink-0 text-[9px] font-bold bg-primary/15 text-primary px-1.5 py-0.5 rounded">1°</span>
                                                                )}
                                                                <span className="font-bold text-foreground truncate">{label}</span>
                                                            </div>
                                                            <span className="text-muted-foreground font-mono shrink-0 ml-2">{(prob * 100).toFixed(1)}%</span>
                                                        </div>
                                                        <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden">
                                                            <div
                                                                className={`h-full transition-all duration-200 ease-out ${idx === 0 ? 'bg-primary' : 'bg-primary/40'}`}
                                                                style={{ width: `${Math.min(prob * 100, 100)}%` }}
                                                            />
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        ) : (
                                            <div className="text-[12px] text-primary">No se generaron probabilidades.</div>
                                        )}
                                    </div>

                                    {/* Feedback */}
                                    <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                        <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground mb-3">
                                            <Send size={14} className="text-green-500" /> Aprendizaje Continuo (Feedback)
                                        </div>
                                        <div className="text-[11px] text-muted-foreground mb-3">
                                            Si la IA falló, ingresa el departamento correcto para ajustar los pesos (partial_fit).
                                        </div>
                                        <div className="flex flex-col sm:flex-row gap-2">
                                            <Input
                                                value={feedbackDept}
                                                onChange={(e) => setFeedbackDept(e.target.value)}
                                                placeholder="Ej: Base de Datos"
                                                className="h-9 rounded-md bg-background w-full"
                                                disabled={isProcessing}
                                            />
                                            <Button
                                                onClick={handleSendFeedback}
                                                disabled={isProcessing || feedbackDept.trim().length < 2}
                                                className="h-9 rounded-md bg-green-600 hover:bg-green-500 text-white font-semibold w-full sm:w-auto"
                                            >
                                                Entrenar
                                            </Button>
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                        </>
                    )}

                    {/* ── LOTES ──────────────────────────────────────────────────── */}
                    {activeTab === 'batch' && (
                        <>
                            {isProcessing && !batchResult && (
                                <div className="space-y-3">
                                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                                        <SkeletonBlock className="h-[80px]" />
                                        <SkeletonBlock className="h-[80px]" />
                                        <SkeletonBlock className="h-[80px]" />
                                    </div>
                                    <SkeletonBlock className="h-[140px]" />
                                    <SkeletonBlock className="h-[240px]" />
                                </div>
                            )}

                            {!isProcessing && !batchResult && (
                                <div className="flex-1 flex flex-col items-center justify-center text-center p-6 text-muted-foreground">
                                    <BarChart3 size={40} className="text-muted/50 mb-3" />
                                    <h3 className="font-bold text-foreground">Auditoría Vacía</h3>
                                    <p className="text-xs leading-relaxed max-w-[280px] mt-2">
                                        Sube un archivo CSV con tickets para evaluar las métricas globales del modelo.
                                    </p>
                                </div>
                            )}

                            {batchResult && (
                                <motion.div
                                    initial={{ opacity: 0, y: 15 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ duration: 0.12, ease: 'easeOut' }}
                                    className="space-y-4"
                                >
                                    {/* Top metrics */}
                                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
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

                                    {/* Distribution */}
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
                                                                <div className="h-full bg-primary transition-all duration-200 ease-out" style={{ width: `${width}%` }} />
                                                            </div>
                                                        </div>
                                                    )
                                                })}
                                            </div>
                                        </div>
                                    )}

                                    {/* Confusion matrix */}
                                    {batchResult.confusionMatrix && batchResult.labels && (
                                        <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                            <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground mb-3">
                                                <Thermometer size={14} className="text-primary" /> Matriz de Confusión
                                            </div>
                                            <Heatmap matrix={batchResult.confusionMatrix} labels={batchResult.labels} />
                                        </div>
                                    )}

                                    {/* ── METRICS PANEL ─────────────────────────────────── */}
                                    {perClassMetrics.length > 0 && macroAvg && (
                                        <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                            <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground mb-3">
                                                <Target size={14} className="text-primary" /> Análisis de Confianza por Departamento
                                            </div>

                                            {/* Interpretation badge */}
                                            {modelInterpretation && (
                                                <div className={`mb-4 p-3 rounded-lg border ${modelInterpretation.bg} flex items-start gap-3`}>
                                                    <TrendingUp size={15} className={`${modelInterpretation.color} shrink-0 mt-0.5`} />
                                                    <div className="min-w-0">
                                                        <p className={`text-xs font-bold ${modelInterpretation.color}`}>{modelInterpretation.label}</p>
                                                        <p className="text-[11px] text-muted-foreground mt-0.5">{modelInterpretation.desc}</p>
                                                        <p className="text-[11px] text-muted-foreground mt-0.5">
                                                            F1 Macro: <span className={`font-semibold ${modelInterpretation.color}`}>{(macroAvg.f1 * 100).toFixed(1)}%</span>
                                                            {' '}· Precisión: <span className="font-semibold text-foreground">{(macroAvg.precision * 100).toFixed(1)}%</span>
                                                            {' '}· Recall: <span className="font-semibold text-foreground">{(macroAvg.recall * 100).toFixed(1)}%</span>
                                                        </p>
                                                    </div>
                                                </div>
                                            )}

                                            {/* Per-class table */}
                                            <div className="overflow-auto max-h-[300px] rounded-lg border border-border/60">
                                                <table className="w-full min-w-[480px] text-[11px]">
                                                    <thead className="sticky top-0 bg-muted/50 backdrop-blur">
                                                        <tr>
                                                            <th className="text-left p-2 font-bold text-muted-foreground">Departamento</th>
                                                            <th className="text-center p-2 font-bold text-muted-foreground">Precisión</th>
                                                            <th className="text-center p-2 font-bold text-muted-foreground">Recall</th>
                                                            <th className="text-center p-2 font-bold text-muted-foreground">F1</th>
                                                            <th className="text-center p-2 font-bold text-muted-foreground">Muestras</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {perClassMetrics.map((m, i) => (
                                                            <tr key={i} className="border-t border-border/40 hover:bg-muted/20 transition-colors">
                                                                <td className="p-2 font-semibold text-foreground truncate max-w-[160px]">{m.label}</td>
                                                                <td className="p-2 text-center font-mono font-bold">
                                                                    <span className={metricColor(m.precision)}>{(m.precision * 100).toFixed(0)}%</span>
                                                                </td>
                                                                <td className="p-2 text-center font-mono font-bold">
                                                                    <span className={metricColor(m.recall)}>{(m.recall * 100).toFixed(0)}%</span>
                                                                </td>
                                                                <td className="p-2 text-center font-mono font-bold">
                                                                    <span className={metricColor(m.f1)}>{(m.f1 * 100).toFixed(0)}%</span>
                                                                </td>
                                                                <td className="p-2 text-center text-muted-foreground font-mono">{m.support}</td>
                                                            </tr>
                                                        ))}
                                                        <tr className="border-t-2 border-border/60 bg-muted/30">
                                                            <td className="p-2 font-extrabold text-foreground">Macro-Avg</td>
                                                            <td className="p-2 text-center font-mono font-extrabold">
                                                                <span className={metricColor(macroAvg.precision)}>{(macroAvg.precision * 100).toFixed(0)}%</span>
                                                            </td>
                                                            <td className="p-2 text-center font-mono font-extrabold">
                                                                <span className={metricColor(macroAvg.recall)}>{(macroAvg.recall * 100).toFixed(0)}%</span>
                                                            </td>
                                                            <td className="p-2 text-center font-mono font-extrabold">
                                                                <span className={metricColor(macroAvg.f1)}>{(macroAvg.f1 * 100).toFixed(0)}%</span>
                                                            </td>
                                                            <td className="p-2 text-center text-muted-foreground font-mono">—</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    )}

                                    {/* TF-IDF dictionary */}
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

                    {/* ── INFERENCIA CIEGA ────────────────────────────────────────── */}
                    {activeTab === 'predict' && (
                        <>
                            {isProcessing && !blindResult && (
                                <div className="space-y-3">
                                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                                        <SkeletonBlock className="h-[80px]" />
                                        <SkeletonBlock className="h-[80px]" />
                                        <SkeletonBlock className="h-[80px]" />
                                    </div>
                                    <SkeletonBlock className="h-[200px]" />
                                    <SkeletonBlock className="h-[300px]" />
                                </div>
                            )}

                            {!isProcessing && !blindResult && (
                                <div className="flex-1 flex flex-col items-center justify-center text-center p-6 text-muted-foreground">
                                    <Zap size={40} className="text-muted/50 mb-3" />
                                    <h3 className="font-bold text-foreground">Inferencia Pendiente</h3>
                                    <p className="text-xs leading-relaxed max-w-[280px] mt-2">
                                        Sube un CSV sin etiquetas para visualizar la clasificación masiva en milisegundos.
                                    </p>
                                </div>
                            )}

                            {blindResult && (
                                <motion.div
                                    initial={{ opacity: 0, y: 15 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ duration: 0.12, ease: 'easeOut' }}
                                    className="space-y-4"
                                >
                                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                                        <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                            <div className="text-[11px] font-semibold text-muted-foreground">Tickets Procesados</div>
                                            <div className="text-xl font-extrabold text-foreground">{blindResult.totalTickets}</div>
                                        </div>
                                        <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                            <div className="text-[11px] font-semibold text-muted-foreground">Confianza Media</div>
                                            <div className="text-xl font-extrabold text-foreground">{avgConfidence.toFixed(1)}%</div>
                                        </div>
                                        <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                            <div className="text-[11px] font-semibold text-muted-foreground">Velocidad</div>
                                            <div className="text-xl font-extrabold text-foreground">{blindResult.speed} ms</div>
                                        </div>
                                    </div>

                                    {blindResult.departmentDistribution && (
                                        <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                            <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground mb-4">
                                                <BarChart3 size={14} className="text-primary" /> Demanda por Departamento
                                            </div>
                                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3">
                                                {predictDistSorted.map(([dep, count]) => {
                                                    const width = predictMaxDist > 0 ? (count / predictMaxDist) * 100 : 0
                                                    return (
                                                        <div key={dep} className="space-y-1.5">
                                                            <div className="flex justify-between text-[11px]">
                                                                <span className="font-bold text-foreground truncate max-w-[75%]">{dep}</span>
                                                                <span className="text-muted-foreground font-mono">{count}</span>
                                                            </div>
                                                            <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden">
                                                                <div className="h-full bg-primary transition-all duration-200 ease-out" style={{ width: `${width}%` }} />
                                                            </div>
                                                        </div>
                                                    )
                                                })}
                                            </div>
                                        </div>
                                    )}

                                    <div className="rounded-lg border border-border/60 bg-card shadow-sm flex flex-col h-[400px]">
                                        <div className="p-3 border-b border-border/60 bg-muted/20 text-[11px] font-semibold text-muted-foreground flex justify-between uppercase tracking-wider shrink-0">
                                            <span>ID / Ticket Original</span>
                                            <span>Clasificación IA</span>
                                        </div>
                                        <div className="flex-1 overflow-auto">
                                            {blindResult.predictions.map((p) => (
                                                <div key={p.id} className="p-3 border-b border-border/40 hover:bg-muted/30 transition-colors flex flex-col sm:flex-row justify-between gap-2 sm:gap-4 items-start sm:items-center">
                                                    <div className="min-w-0 flex-1 w-full">
                                                        <div className="text-[10px] text-muted-foreground font-mono mb-0.5">#{p.id}</div>
                                                        <div className="text-[12px] text-foreground line-clamp-2 sm:truncate" title={p.text_original}>{p.text_original}</div>
                                                    </div>
                                                    <div className="shrink-0 text-left sm:text-right w-full sm:w-auto">
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
                    {/* ── IA STATS ──────────────────────────────────────────────── */}
                    {activeTab === 'stats' && (
                        <>
                            {!statsResult && !isProcessing && (
                                <div className="flex-1 flex flex-col items-center justify-center text-center p-6 text-muted-foreground">
                                    <TrendingUp size={40} className="text-muted/50 mb-3" />
                                    <h3 className="font-bold text-foreground">Curva de Aprendizaje</h3>
                                    <p className="text-xs leading-relaxed max-w-[280px] mt-2 mb-4">
                                        Carga el historial para ver cómo mejora la IA con cada dataset y feedback subido.
                                    </p>
                                    <Button
                                        onClick={onFetchStats}
                                        className="h-9 px-5 rounded-lg bg-primary hover:bg-primary/90 text-primary-foreground font-semibold"
                                    >
                                        <RefreshCw size={14} className="mr-2" /> Cargar Estadísticas
                                    </Button>
                                </div>
                            )}

                            {isProcessing && !statsResult && (
                                <div className="space-y-3">
                                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                                        {[0,1,2,3].map(i => <SkeletonBlock key={i} className="h-[72px]" />)}
                                    </div>
                                    <SkeletonBlock className="h-[200px]" />
                                    <SkeletonBlock className="h-[200px]" />
                                </div>
                            )}

                            {statsResult && (
                                <motion.div
                                    initial={{ opacity: 0, y: 15 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ duration: 0.12, ease: 'easeOut' }}
                                    className="space-y-4"
                                >
                                    {/* Refresh button */}
                                    <div className="flex justify-end">
                                        <Button
                                            onClick={onFetchStats}
                                            disabled={isProcessing}
                                            size="sm"
                                            className="h-7 px-3 text-[11px] bg-muted hover:bg-muted/80 text-foreground border border-border/60 rounded-md"
                                        >
                                            <RefreshCw size={11} className="mr-1.5" /> Actualizar
                                        </Button>
                                    </div>

                                    {/* KPI cards */}
                                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                                        <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                            <div className="text-[10px] font-semibold text-muted-foreground mb-1">F1-Score Actual</div>
                                            <div className={`text-xl font-extrabold ${statsResult.latestF1 !== null ? metricColor(statsResult.latestF1) : 'text-muted-foreground'}`}>
                                                {statsResult.latestF1 !== null ? `${(statsResult.latestF1 * 100).toFixed(1)}%` : '—'}
                                            </div>
                                        </div>
                                        <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                            <div className="text-[10px] font-semibold text-muted-foreground mb-1">Exactitud</div>
                                            <div className={`text-xl font-extrabold ${statsResult.latestAccuracy !== null ? metricColor(statsResult.latestAccuracy) : 'text-muted-foreground'}`}>
                                                {statsResult.latestAccuracy !== null ? `${(statsResult.latestAccuracy * 100).toFixed(1)}%` : '—'}
                                            </div>
                                        </div>
                                        <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                            <div className="text-[10px] font-semibold text-muted-foreground mb-1">Confianza Media</div>
                                            <div className={`text-xl font-extrabold ${statsResult.latestConfidence !== null ? metricColor(statsResult.latestConfidence) : 'text-muted-foreground'}`}>
                                                {statsResult.latestConfidence !== null ? `${(statsResult.latestConfidence * 100).toFixed(1)}%` : '—'}
                                            </div>
                                        </div>
                                        <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                            <div className="flex items-center gap-1.5 text-[10px] font-semibold text-muted-foreground mb-1">
                                                <Database size={11} /> Tickets
                                            </div>
                                            <div className="text-xl font-extrabold text-foreground">
                                                {statsResult.totalRecords.toLocaleString()}
                                            </div>
                                            <div className="text-[10px] text-muted-foreground">{statsResult.totalSessions} sesiones</div>
                                        </div>
                                    </div>

                                    {/* Learning curve chart */}
                                    {statsResult.history.length > 0 && (
                                        <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                            <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground mb-3">
                                                <TrendingUp size={14} className="text-primary" /> Curva de Aprendizaje
                                            </div>
                                            {statsResult.history.length === 1 ? (
                                                <div className="text-[11px] text-muted-foreground text-center py-4">
                                                    Solo 1 sesión registrada. La curva se mostrará a partir de la segunda.
                                                </div>
                                            ) : (
                                                <LearningCurve entries={statsResult.history} />
                                            )}
                                        </div>
                                    )}

                                    {/* Session history table */}
                                    {statsResult.history.length > 0 && (
                                        <div className="rounded-lg border border-border/60 bg-card p-3 shadow-sm">
                                            <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground mb-3">
                                                <Award size={14} className="text-primary" /> Historial de Sesiones
                                            </div>
                                            <div className="overflow-auto max-h-[260px] rounded-lg border border-border/60">
                                                <table className="w-full min-w-[480px] text-[11px]">
                                                    <thead className="sticky top-0 bg-muted/50 backdrop-blur">
                                                        <tr>
                                                            <th className="text-left p-2 font-bold text-muted-foreground">#</th>
                                                            <th className="text-left p-2 font-bold text-muted-foreground">Fecha</th>
                                                            <th className="text-left p-2 font-bold text-muted-foreground">Origen</th>
                                                            <th className="text-center p-2 font-bold text-muted-foreground">Tickets</th>
                                                            <th className="text-center p-2 font-bold text-muted-foreground">F1</th>
                                                            <th className="text-center p-2 font-bold text-muted-foreground">Acc</th>
                                                            <th className="text-center p-2 font-bold text-muted-foreground">Conf</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {statsResult.history.map((entry, i) => (
                                                            <tr key={entry.id} className="border-t border-border/40 hover:bg-muted/20 transition-colors">
                                                                <td className="p-2 text-muted-foreground font-mono">{statsResult.totalSessions - i}</td>
                                                                <td className="p-2 text-muted-foreground font-mono text-[10px]">
                                                                    {new Date(entry.created_at).toLocaleString('es-AR', { dateStyle: 'short', timeStyle: 'short' })}
                                                                </td>
                                                                <td className="p-2">
                                                                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${TRIGGER_COLOR[entry.trigger_type] ?? 'bg-muted text-muted-foreground'}`}>
                                                                        {TRIGGER_LABEL[entry.trigger_type] ?? entry.trigger_type}
                                                                    </span>
                                                                </td>
                                                                <td className="p-2 text-center font-mono text-foreground">{entry.record_count.toLocaleString()}</td>
                                                                <td className={`p-2 text-center font-mono font-bold ${metricColor(entry.f1_score)}`}>
                                                                    {(entry.f1_score * 100).toFixed(1)}%
                                                                </td>
                                                                <td className={`p-2 text-center font-mono font-bold ${metricColor(entry.accuracy)}`}>
                                                                    {(entry.accuracy * 100).toFixed(1)}%
                                                                </td>
                                                                <td className={`p-2 text-center font-mono font-bold ${entry.avg_confidence !== null ? metricColor(entry.avg_confidence) : 'text-muted-foreground'}`}>
                                                                    {entry.avg_confidence !== null ? `${(entry.avg_confidence * 100).toFixed(1)}%` : '—'}
                                                                </td>
                                                            </tr>
                                                        ))}
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    )}

                                    {statsResult.history.length === 0 && (
                                        <div className="flex flex-col items-center justify-center py-10 gap-3 text-center text-muted-foreground">
                                            <TrendingUp size={28} className="opacity-40" />
                                            <p className="text-sm font-semibold text-foreground">Sin sesiones registradas</p>
                                            <p className="text-[11px] max-w-[240px] leading-relaxed">
                                                Sube un CSV en la pestaña Lotes para registrar la primera sesión de entrenamiento.
                                            </p>
                                        </div>
                                    )}
                                </motion.div>
                            )}
                        </>
                    )}

                </div>
            </div>

            {/* Solutions modal rendered via portal at document.body */}
            {solutionsPortal}
        </>
    )
}
