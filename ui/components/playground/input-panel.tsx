"use client"

import React, { useMemo, useRef, useState } from "react"
import { motion } from "framer-motion"
import { UploadCloud, FileText, Loader2, PlayCircle, ShieldAlert, Keyboard, Zap, TrendingUp } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { toast } from "sonner"

/**
 * Panel de entrada del Playground (sección izquierda).
 *
 * Expone cuatro tabs: redacción de ticket manual, carga de CSV de entrenamiento
 * (batch), carga de CSV para inferencia ciega y descripción del panel de stats.
 * Valida formato CSV (extensión) y delega todo el procesamiento al padre
 * AIPlayground mediante callbacks, sin lógica de negocio propia.
 *
 * Relaciones: AIPlayground (onManualSubmit, onBatchUpload, onBlindUpload,
 * setActiveTab). El mismo fileInputRef sirve para batch e inferencia ciega.
 */
interface InputPanelProps {
    activeTab: "manual" | "batch" | "predict" | "stats"
    setActiveTab: (tab: "manual" | "batch" | "predict" | "stats") => void
    isProcessing: boolean
    onManualSubmit: (title: string, description: string) => void
    onBatchUpload: (file: File) => void
    onBlindUpload: (file: File) => void
}

export function InputPanel({
    activeTab,
    setActiveTab,
    isProcessing,
    onManualSubmit,
    onBatchUpload,
    onBlindUpload,
}: InputPanelProps) {
    const [title, setTitle] = useState("")
    const [description, setDescription] = useState("")
    const [csvFile, setCsvFile] = useState<File | null>(null)

    const fileInputRef = useRef<HTMLInputElement>(null)

    const maxDescLength = 800
    const minDescLength = 15
    const descCount = description.length

    const descValid = useMemo(() => {
        return descCount >= minDescLength
    }, [descCount])

    const handleManualClick = () => {
        if (isProcessing) return

        if (title.trim().length < 3) {
            toast.warning("El título es muy corto", { description: "Mínimo recomendado: 3 caracteres." })
            return
        }

        if (!descValid) {
            toast.warning("Descripción insuficiente", { description: `Mínimo recomendado: ${minDescLength} caracteres.` })
            return
        }

        onManualSubmit(title.trim(), description.trim())
    }

    const handleFileSelect = (file: File) => {
        if (!file) return
        if (!file.name.toLowerCase().endsWith(".csv")) {
            toast.error("Solo se permiten archivos CSV")
            return
        }
        setCsvFile(file)
    }

    const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault()
        if (isProcessing) return
        const file = e.dataTransfer.files?.[0]
        if (!file) return
        handleFileSelect(file)
    }

    const handleBrowseClick = () => {
        if (isProcessing) return
        fileInputRef.current?.click()
    }

    return (
        <div className="h-full bg-card border border-border/60 shadow-sm flex flex-col rounded-xl overflow-hidden">
            {/* CABECERA */}
            <div className="min-h-[52px] h-auto py-3 px-4 flex flex-col sm:flex-row items-center justify-between gap-3 border-b border-border/60 bg-muted/20 text-foreground">
                <div className="flex items-center gap-2 font-semibold tracking-tight text-muted-foreground w-full sm:w-auto justify-center sm:justify-start">
                    <Keyboard size={16} className="text-primary" />
                    <span className="text-sm">Panel de Entrada</span>
                </div>
                <div className="flex flex-wrap justify-center items-center bg-muted p-1 rounded-lg border border-border/40 w-full sm:w-auto">
                    <button
                        onClick={() => setActiveTab("manual")}
                        disabled={isProcessing}
                        className={`flex-1 sm:flex-none px-3 py-1 text-xs font-medium rounded-md transition-all duration-150 ${activeTab === "manual" ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"}`}
                    >
                        Manual
                    </button>
                    <button
                        onClick={() => setActiveTab("batch")}
                        disabled={isProcessing}
                        className={`flex-1 sm:flex-none px-3 py-1 text-xs font-medium rounded-md transition-all duration-150 ${activeTab === "batch" ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"}`}
                    >
                        Lotes
                    </button>
                    <button
                        onClick={() => setActiveTab("predict")}
                        disabled={isProcessing}
                        className={`flex-1 sm:flex-none px-3 py-1 text-xs font-medium rounded-md transition-all duration-150 ${activeTab === "predict" ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"}`}
                    >
                        Inferencia
                    </button>
                    <button
                        onClick={() => setActiveTab("stats")}
                        disabled={isProcessing}
                        className={`flex-1 sm:flex-none px-3 py-1 text-xs font-medium rounded-md transition-all duration-150 ${activeTab === "stats" ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"}`}
                    >
                        IA Stats
                    </button>
                </div>
            </div>

            {/* CUERPO */}
            <div className={`flex-1 overflow-y-auto p-4 space-y-4 transition-opacity duration-150 ${isProcessing ? "pointer-events-none opacity-60" : ""}`}>

                {/* TICKET MANUAL */}
                {activeTab === "manual" && (
                    <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.12, ease: "easeOut" }} className="space-y-4">
                        <div className="rounded-lg border border-border/60 bg-card p-3 space-y-3 shadow-sm">
                            <div className="text-xs font-semibold text-muted-foreground flex items-center gap-2">
                                <FileText size={14} className="text-primary" />
                                Redacción de Ticket
                            </div>

                            <div className="space-y-2">
                                <label className="text-xs font-semibold text-foreground">Título</label>
                                <Input
                                    value={title}
                                    onChange={(e) => setTitle(e.target.value)}
                                    placeholder="Ej: Error de conexión VPN"
                                    className="h-10 rounded-lg bg-background w-full"
                                    disabled={isProcessing}
                                />
                            </div>

                            <div className="space-y-2">
                                <label className="text-xs font-semibold text-foreground">Descripción</label>
                                <textarea
                                    value={description}
                                    onChange={(e) => {
                                        if (e.target.value.length <= maxDescLength) setDescription(e.target.value)
                                    }}
                                    placeholder="Describe el problema técnico con detalle..."
                                    disabled={isProcessing}
                                    className="w-full min-h-[160px] resize-none rounded-lg border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary/50 transition-all"
                                />

                                <div className="flex justify-between text-[11px] text-muted-foreground">
                                    <span>Mínimo: {minDescLength} caracteres</span>
                                    <span className={`font-semibold ${descValid ? "text-primary" : "text-red-500"}`}>
                                        {descCount}/{maxDescLength}
                                    </span>
                                </div>

                                <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden">
                                    <div
                                        className="h-full bg-primary transition-all duration-150 ease-out"
                                        style={{ width: `${(descCount / maxDescLength) * 100}%` }}
                                    />
                                </div>
                            </div>
                        </div>

                        <Button
                            onClick={handleManualClick}
                            disabled={isProcessing}
                            className="w-full h-11 rounded-lg bg-primary hover:bg-primary/90 text-primary-foreground font-semibold shadow-md transition-all duration-150"
                        >
                            {isProcessing ? (
                                <><Loader2 size={16} className="animate-spin mr-2" /> Analizando NLP...</>
                            ) : (
                                <><PlayCircle size={16} className="mr-2" /> Analizar Ticket</>
                            )}
                        </Button>

                        <div className="rounded-lg border border-border/60 bg-muted/30 p-3 text-[11px] text-muted-foreground leading-relaxed flex gap-2">
                            <ShieldAlert size={14} className="text-primary shrink-0 mt-0.5" />
                            <div>
                                Evita descripciones vacías o irrelevantes. El sistema rechazará "basura" mediante la heurística NLP.
                            </div>
                        </div>
                    </motion.div>
                )}

                {/* SUBIDA POR LOTES (Entrenamiento) */}
                {activeTab === "batch" && (
                    <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.12, ease: "easeOut" }} className="space-y-4">
                        <div className="rounded-lg border border-border/60 bg-card p-3 space-y-2 shadow-sm">
                            <div className="text-xs font-semibold text-muted-foreground flex items-center gap-2">
                                <UploadCloud size={14} className="text-primary" />
                                Entrenamiento K-Fold (CSV)
                            </div>
                            <div className="text-[11px] text-muted-foreground leading-relaxed">
                                Columnas requeridas: <b className="text-foreground">text, department</b>.
                                <br />
                                Columna opcional: <b className="text-foreground">solution</b> (o <b className="text-foreground">solucion</b>) — persiste soluciones conocidas por área en el Data Lake.
                            </div>
                        </div>

                        <div
                            onDragOver={(e) => e.preventDefault()}
                            onDrop={handleDrop}
                            className="rounded-lg border-2 border-dashed border-border/60 bg-muted/20 hover:bg-muted/40 transition-colors duration-300 p-6 flex flex-col items-center justify-center text-center cursor-pointer h-[200px]"
                            onClick={handleBrowseClick}
                        >
                            <UploadCloud size={36} className="text-primary mb-3" />
                            <div className="text-sm font-bold text-foreground break-all px-2">
                                {csvFile ? csvFile.name : "Selecciona o arrastra tu CSV"}
                            </div>
                            <div className="text-xs text-muted-foreground mt-1">
                                Procesamiento optimizado para miles de tickets
                            </div>
                        </div>

                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".csv"
                            className="hidden"
                            onChange={(e) => {
                                const f = e.target.files?.[0]
                                if (!f) return
                                handleFileSelect(f)
                            }}
                            disabled={isProcessing}
                        />

                        <Button
                            onClick={() => {
                                if (!csvFile) {
                                    toast.warning("Sube un archivo primero")
                                    return
                                }
                                onBatchUpload(csvFile)
                            }}
                            disabled={isProcessing || !csvFile}
                            className="w-full h-11 rounded-lg bg-primary hover:bg-primary/90 text-primary-foreground font-semibold shadow-md transition-all duration-150"
                        >
                            {isProcessing ? (
                                <><Loader2 size={16} className="animate-spin mr-2" /> Entrenando IA...</>
                            ) : (
                                <><PlayCircle size={16} className="mr-2" /> Ejecutar Entrenamiento</>
                            )}
                        </Button>
                    </motion.div>
                )}

                {/* IA STATS */}
                {activeTab === "stats" && (
                    <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.12, ease: "easeOut" }} className="space-y-4">
                        <div className="rounded-lg border border-border/60 bg-card p-3 space-y-2 shadow-sm">
                            <div className="text-xs font-semibold text-muted-foreground flex items-center gap-2">
                                <TrendingUp size={14} className="text-primary" />
                                Curva de Aprendizaje
                            </div>
                            <div className="text-[11px] text-muted-foreground leading-relaxed">
                                Visualiza cómo evoluciona el <b className="text-foreground">F1-Score</b>, la <b className="text-foreground">Exactitud</b> y la <b className="text-foreground">Confianza Media</b> de la IA con cada entrenamiento. Más datos = mayor precisión.
                            </div>
                        </div>
                        <div className="rounded-lg border border-primary/20 bg-primary/5 p-3 text-[11px] text-muted-foreground leading-relaxed space-y-1">
                            <p className="font-semibold text-foreground">¿Cómo funciona?</p>
                            <p>Cada vez que subís un CSV o la IA se reentrena por feedback, se registra un punto en la curva.</p>
                            <p>La <span className="text-primary font-semibold">Confianza Media</span> mide qué tan segura está la IA al clasificar — a mayor volumen de datos variados, más alta.</p>
                        </div>
                    </motion.div>
                )}

                {/* INFERENCIA CIEGA */}
                {activeTab === "predict" && (
                    <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.12, ease: "easeOut" }} className="space-y-4">
                        <div className="rounded-lg border border-border/60 bg-card p-3 space-y-2 shadow-sm">
                            <div className="text-xs font-semibold text-muted-foreground flex items-center gap-2">
                                <Zap size={14} className="text-primary" />
                                Clasificación Masiva (CSV Ciego)
                            </div>
                            <div className="text-[11px] text-muted-foreground leading-relaxed">
                                Sube un archivo CSV con miles de tickets <b className="text-foreground">SIN</b> departamento. La IA predecirá todos en tiempo real.
                            </div>
                        </div>

                        <div
                            onDragOver={(e) => e.preventDefault()}
                            onDrop={handleDrop}
                            className="rounded-lg border-2 border-dashed border-border/60 bg-muted/20 hover:bg-muted/40 transition-colors duration-300 p-6 flex flex-col items-center justify-center text-center cursor-pointer h-[200px]"
                            onClick={handleBrowseClick}
                        >
                            <Zap size={36} className="text-primary mb-3" />
                            <div className="text-sm font-bold text-foreground break-all px-2">
                                {csvFile ? csvFile.name : "Selecciona o arrastra tu CSV Ciego"}
                            </div>
                            <div className="text-xs text-muted-foreground mt-1 px-2">
                                Columnas requeridas: "text" o "titulo" y "descripcion"
                            </div>
                        </div>

                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".csv"
                            className="hidden"
                            onChange={(e) => {
                                const f = e.target.files?.[0]
                                if (!f) return
                                handleFileSelect(f)
                            }}
                            disabled={isProcessing}
                        />

                        <Button
                            onClick={() => {
                                if (!csvFile) {
                                    toast.warning("Sube un archivo primero")
                                    return
                                }
                                onBlindUpload(csvFile)
                            }}
                            disabled={isProcessing || !csvFile}
                            className="w-full h-11 rounded-lg bg-primary hover:bg-primary/90 text-primary-foreground font-semibold shadow-md transition-all duration-150"
                        >
                            {isProcessing ? (
                                <><Loader2 size={16} className="animate-spin mr-2" /> Clasificando...</>
                            ) : (
                                <><Zap size={16} className="mr-2" /> Clasificar CSV</>
                            )}
                        </Button>
                    </motion.div>
                )}
            </div>
        </div>
    )
}