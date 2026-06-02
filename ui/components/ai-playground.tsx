'use client'

import React, { useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'sonner'

import { InputPanel } from './playground/input-panel'
import { ReasoningPanel } from './playground/reasoning-panel'
import { TerminalPanel } from './playground/terminal-panel'

import type { SingleInferenceResult, BatchInferenceResult, BlindBatchResult, SolutionItem, StatsResult } from '@/types/ai'

/** Resuelto en build-time. En desarrollo apunta al backend local (puerto 7860). */
const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'https://dpardo-it-ticket-ai-backend.hf.space'

/**
 * Orquestador principal del Playground de IA.
 *
 * Mantiene el estado de resultados (manual, batch, ciego, stats) y los logs
 * del terminal. Realiza todas las llamadas HTTP al backend FastAPI y delega
 * la presentación a InputPanel, ReasoningPanel y TerminalPanel.
 *
 * Relaciones: InputPanel (acciones de entrada), ReasoningPanel (visualización
 * de resultados), TerminalPanel (logs en tiempo real), api.main (HTTP).
 */
export default function AIPlayground() {
    const [activeTab, setActiveTab] = useState<'manual' | 'batch' | 'predict' | 'stats'>('manual')
    const [isProcessing, setIsProcessing] = useState(false)

    const [manualResult, setManualResult] = useState<SingleInferenceResult | null>(null)
    const [batchResult, setBatchResult] = useState<BatchInferenceResult | null>(null)
    const [blindResult, setBlindResult] = useState<BlindBatchResult | null>(null)
    const [statsResult, setStatsResult] = useState<StatsResult | null>(null)

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

    /**
     * Envía el ticket manual al endpoint /api/predict y procesa la respuesta.
     * Aplica la guardia de basura (is_garbage) antes de poblar manualResult.
     */
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

    /**
     * Envía la corrección del perfil al endpoint /api/feedback.
     * El backend aplica partial_fit inmediato o lanza reentrenamiento completo
     * si el departamento corregido es nuevo en el espacio de clases.
     */
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

    /**
     * Sube un CSV de entrenamiento a /api/batch, entrena el modelo y persiste
     * la sesión en training_log vía insert_training_log en el backend.
     */
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

    /** Consulta /api/solutions para obtener soluciones conocidas del departamento predicho. */
    const handleFetchSolutions = async (department: string): Promise<SolutionItem[]> => {
        try {
            const res = await fetch(`${API_BASE}/api/solutions?department=${encodeURIComponent(department)}`)
            if (!res.ok) return []
            const data = await res.json()
            return data.solutions ?? []
        } catch {
            return []
        }
    }

    /**
     * Consulta /api/stats y carga el historial de sesiones de entrenamiento.
     * Muestra un toast descriptivo en caso de 404 (backend desactualizado)
     * o error de conexión.
     */
    const handleFetchStats = async () => {
        setIsProcessing(true)
        pushLog('Consultando historial de entrenamiento...')
        try {
            const res = await fetch(`${API_BASE}/api/stats`)

            if (res.status === 404) {
                pushError('El backend no tiene el endpoint /api/stats. Iniciá el backend local o redesplegá en HF Space.')
                toast.error('Backend desactualizado', {
                    description: 'El servidor no tiene /api/stats. Corré el backend local en puerto 7860.',
                    duration: 6000,
                })
                return
            }

            if (!res.ok) {
                const data = await res.json().catch(() => ({}))
                pushError(data.detail || `Error HTTP ${res.status}`)
                toast.error('Error al cargar estadísticas', { description: data.detail || `HTTP ${res.status}` })
                return
            }

            const data = await res.json()
            setStatsResult(data as StatsResult)
            pushSuccess(`Historial cargado: ${data.totalSessions} sesiones de entrenamiento.`)
        } catch (err: any) {
            pushError(`Sin conexión al backend (${API_BASE}): ${err.message}`)
            toast.error('Backend no disponible', {
                description: `No se puede conectar a ${API_BASE}. Verificá que el servidor esté corriendo.`,
                duration: 6000,
            })
        } finally {
            setIsProcessing(false)
        }
    }

    React.useEffect(() => {
        if (activeTab === 'stats') handleFetchStats()
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [activeTab])

    /**
     * Registra que una solución del modal fue útil para el perfil actual.
     * Llama a /api/solutions/vote e incrementa su relevancia en la base de conocimiento.
     * La operación es silenciosa — un fallo de red no interrumpe la experiencia.
     */
    const handleVoteSolution = async (department: string, solution: string) => {
        try {
            await fetch(`${API_BASE}/api/solutions/vote`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ department, solution }),
            })
        } catch { /* best-effort */ }
    }

    /**
     * Persiste una solución aportada por el perfil en /api/solutions/feedback.
     * Retorna true si el servidor aceptó el aporte, false en caso de error de red.
     * La solución queda disponible inmediatamente para otros perfiles del mismo departamento.
     */
    const handleSubmitSolutionFeedback = async (department: string, solution: string): Promise<boolean> => {
        try {
            const res = await fetch(`${API_BASE}/api/solutions/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ department, solution }),
            })
            return res.ok
        } catch {
            return false
        }
    }

    /** Sube un CSV sin etiquetas a /api/batch_predict y obtiene clasificación masiva ciega. */
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
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2, ease: "easeOut" }}
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
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                transition={{ duration: 0.1, ease: "linear" }}
                                className="h-full"
                            >
                                <ReasoningPanel
                                    activeTab={activeTab}
                                    manualResult={manualResult}
                                    batchResult={batchResult}
                                    blindResult={blindResult}
                                    statsResult={statsResult}
                                    isProcessing={isProcessing}
                                    onFeedback={handleFeedback}
                                    onFetchSolutions={handleFetchSolutions}
                                    onFetchStats={handleFetchStats}
                                    onVoteSolution={handleVoteSolution}
                                    onSubmitSolutionFeedback={handleSubmitSolutionFeedback}
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