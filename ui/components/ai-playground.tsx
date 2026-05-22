'use client'

import React, { useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'sonner'

import { InputPanel } from './playground/input-panel'
import { ReasoningPanel } from './playground/reasoning-panel'
import { TerminalPanel } from './playground/terminal-panel'

import type { SingleInferenceResult, BatchInferenceResult, BlindBatchResult } from '@/types/ai'

// Base URL dinámico para apuntar al backend.
// Usa la variable de entorno NEXT_PUBLIC_API_BASE si está definida,
// si no, usa el valor por defecto solicitado.
const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'https://dpardo-it-ticket-ai-backend.hf.space'

export default function AIPlayground() {
    const [activeTab, setActiveTab] = useState<'manual' | 'batch' | 'predict'>('manual')
    const [isProcessing, setIsProcessing] = useState(false)

    const [manualResult, setManualResult] = useState<SingleInferenceResult | null>(null)
    const [batchResult, setBatchResult] = useState<BatchInferenceResult | null>(null)
    const [blindResult, setBlindResult] = useState<BlindBatchResult | null>(null)

    const [logs, setLogs] = useState<string[]>([])
    const terminalRef = useRef<HTMLDivElement>(null!)

    const pushLog = (msg: string) => {
        setLogs((prev) => [...prev, `[INFO] ${new Date().toLocaleTimeString()} - ${msg}`])
    }

    const pushSuccess = (msg: string) => {
        setLogs((prev) => [...prev, `[ÉXITO] ${new Date().toLocaleTimeString()} - ${msg}`])
    }

    const pushError = (msg: string) => {
        setLogs((prev) => [...prev, `[ERROR] ${new Date().toLocaleTimeString()} - ${msg}`])
    }

    // ==============================
    // MANUAL INFERENCE
    // ==============================
    const handleManualSubmit = async (title: string, description: string) => {
        setIsProcessing(true)
        setManualResult(null)

        pushLog('Iniciando inferencia manual...')
        pushLog('Preprocesando texto (tokenización + limpieza)...')

        try {
            const start = performance.now()

            const res = await fetch(`${API_BASE}/api/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title, description }),
            })

            const data = await res.json()
            const end = performance.now()

            if (!res.ok) {
                pushError(data.detail || 'Error desconocido en backend.')
                toast.error(data.detail || 'Error en inferencia manual', { id: 'manual' })
                return
            }

            if (data.is_garbage) {
                pushError('Ticket rechazado por heurística de basura.')
                setManualResult({
                    isGarbage: true,
                    garbageMessage: data.message,
                    winner: '', probabilities: {}, tokens: [], latency: 0, level: '', originalText: '', cleanText: '', topTfidf: []
                })
                return
            }

            const safeProbabilities: Record<string, number> = data.probabilities ?? {}
            const safeTokens: string[] = data.tokens ?? []
            const safeTopTfidf: Array<{ term: string; weight: number }> = data.topTfidf ?? data.top_tfidf ?? []

            const result: SingleInferenceResult = {
                winner: data.winner ?? 'N/A',
                probabilities: safeProbabilities,
                tokens: safeTokens,
                latency: data.latency ?? Math.round(end - start),
                level: data.level ?? 'N/A',
                originalText: data.originalText ?? data.original_text ?? `${title}\n${description}`,
                cleanText: data.cleanText ?? data.cleaned_text ?? '',
                topTfidf: safeTopTfidf,
            }

            setManualResult(result)

            pushSuccess(`Inferencia completada. Ganador: ${result.winner}`)
            pushLog(`Latencia: ${result.latency}ms`)

        } catch (err: any) {
            pushError(`Fallo conexión backend: ${err.message}`)
            toast.error('No se pudo conectar al backend.', { id: 'manual' })
        } finally {
            setIsProcessing(false)
        }
    }

    // ==============================
    // FEEDBACK
    // ==============================
    const handleFeedback = async (originalText: string, correctDepartment: string) => {
        setIsProcessing(true)
        pushLog(`Registrando feedback humano -> ${correctDepartment}`)

        toast.loading('Enviando feedback...', { id: 'feedback' })

        try {
            const res = await fetch(`${API_BASE}/api/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    original_text: originalText,
                    correct_department: correctDepartment,
                }),
            })

            const data = await res.json()

            if (!res.ok) {
                pushError(data.detail || 'Error desconocido en feedback.')
                toast.error(data.detail || 'Error registrando feedback', { id: 'feedback' })
                return
            }

            pushSuccess('Feedback guardado correctamente.')
            if (data.learnedImmediately) pushSuccess('Aprendizaje incremental aplicado (SGD partial_fit).')
            if (data.retrainedBatch) pushSuccess('Reentrenamiento de lotes automático ejecutado.')

            toast.success('Feedback registrado', {
                id: 'feedback',
                description: data.message || 'Dataset actualizado correctamente',
            })
        } catch (err: any) {
            pushError(`Fallo feedback: ${err.message}`)
            toast.error('No se pudo enviar feedback.', { id: 'feedback' })
        } finally {
            setIsProcessing(false)
        }
    }

    // ==============================
    // BATCH CSV (ENTRENAMIENTO)
    // ==============================
    const handleBatchUpload = async (file: File) => {
        setIsProcessing(true)
        setBatchResult(null)

        pushLog(`Cargando CSV lotes: ${file.name}`)
        pushLog('Validando estructura del CSV...')
        pushLog('Ejecutando clasificación masiva...')

        try {
            const start = performance.now()

            const formData = new FormData()
            formData.append('file', file)

            const res = await fetch(`${API_BASE}/api/batch`, {
                method: 'POST',
                body: formData,
            })

            const data = await res.json()
            const end = performance.now()

            if (!res.ok) {
                pushError(data.detail || 'Error en lotes desconocido.')
                toast.error(data.detail || 'Error procesando lotes', { id: 'batch' })
                return
            }

            const result: BatchInferenceResult = {
                totalTickets: data.totalTickets ?? 0,
                processedCount: data.processedCount ?? 0,
                rejectedCount: data.rejectedCount ?? 0,
                f1Score: data.f1Score ?? 0,
                accuracy: data.accuracy ?? 0,
                bestModelName: data.bestModelName ?? 'N/A',
                optimalAlpha: data.optimalAlpha ?? null,
                confusionMatrix: data.confusionMatrix ?? [],
                labels: data.labels ?? [],
                departmentDistribution: data.departmentDistribution ?? {},
                globalTfidf: data.globalTfidf ?? [],
                speed: data.speed ?? Math.round(end - start),
            }

            setBatchResult(result)

            pushSuccess(`Lotes completados: ${result.processedCount} tickets procesados.`)
            pushLog(`Modelo ganador: ${result.bestModelName}`)
            pushLog(`F1-score: ${result.f1Score.toFixed(3)}`)
            pushLog(`Tiempo total: ${result.speed}ms`)

        } catch (err: any) {
            pushError(`Fallo en lotes: ${err.message}`)
            toast.error('No se pudo conectar al backend.', { id: 'batch' })
        } finally {
            setIsProcessing(false)
        }
    }

    // ==============================
    // BLIND PREDICT (INFERENCIA CIEGA)
    // ==============================
    const handleBlindUpload = async (file: File) => {
        setIsProcessing(true)
        setBlindResult(null)

        pushLog(`Iniciando clasificación ciega masiva: ${file.name}`)

        try {
            const formData = new FormData()
            formData.append('file', file)

            const res = await fetch(`${API_BASE}/api/batch_predict`, {
                method: 'POST',
                body: formData,
            })

            const data = await res.json()

            if (!res.ok) {
                pushError(data.detail || 'Error procesando inferencia ciega.')
                toast.error(data.detail || 'Error en inferencia masiva')
                return
            }

            setBlindResult(data as BlindBatchResult)

            pushSuccess(`Clasificación ciega de ${data.totalTickets} tickets completada.`)
            pushLog(`Velocidad de procesamiento: ${data.speed}ms`)

        } catch (err: any) {
            pushError(`Fallo en inferencia masiva: ${err.message}`)
            toast.error('No se pudo conectar al backend.')
        } finally {
            setIsProcessing(false)
        }
    }

    return (
        <div className="w-full h-full flex flex-col overflow-y-auto lg:overflow-hidden p-2 lg:p-0">
            <motion.div
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, ease: "circOut" }}
                className="flex flex-col lg:grid lg:grid-cols-12 gap-4 h-auto lg:h-full lg:overflow-hidden"
            >
                <div className="w-full lg:col-span-4 h-auto lg:h-full min-h-[500px] lg:min-h-0">
                    <InputPanel
                        activeTab={activeTab}
                        setActiveTab={setActiveTab}
                        isProcessing={isProcessing}
                        onManualSubmit={handleManualSubmit}
                        onBatchUpload={handleBatchUpload}
                        onBlindUpload={handleBlindUpload}
                    />
                </div>

                <div className="w-full lg:col-span-8 h-auto lg:h-full flex flex-col gap-4 lg:overflow-hidden">
                    <div className="flex-1 min-h-[600px] lg:min-h-0">
                        <AnimatePresence mode="wait">
                            <motion.div
                                key={activeTab}
                                initial={{ opacity: 0, scale: 0.98, filter: "blur(4px)" }}
                                animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
                                exit={{ opacity: 0, scale: 0.98, filter: "blur(4px)" }}
                                transition={{ duration: 0.3, ease: "circOut" }}
                                className="h-full"
                            >
                                <ReasoningPanel
                                    activeTab={activeTab}
                                    manualResult={manualResult}
                                    batchResult={batchResult}
                                    blindResult={blindResult}
                                    isProcessing={isProcessing}
                                    onFeedback={handleFeedback}
                                />
                            </motion.div>
                        </AnimatePresence>
                    </div>

                    <div className="h-[300px] lg:h-[180px] shrink-0">
                        <TerminalPanel logs={logs} isProcessing={isProcessing} terminalRef={terminalRef} />
                    </div>
                </div>
            </motion.div>
        </div>
    )
}