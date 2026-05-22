"use client"

import { Terminal, Maximize2, X } from "lucide-react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"

interface TerminalPanelProps {
    logs: string[]
    isProcessing: boolean
    terminalRef: React.RefObject<HTMLDivElement>
}

export function TerminalPanel({ logs, isProcessing, terminalRef }: TerminalPanelProps) {
    const [isExpanded, setIsExpanded] = useState(false)
    const expandedRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (terminalRef.current) {
            terminalRef.current.scrollTop = terminalRef.current.scrollHeight
        }
    }, [logs, terminalRef])

    useEffect(() => {
        if (isExpanded && expandedRef.current) {
            expandedRef.current.scrollTop = expandedRef.current.scrollHeight
        }
    }, [logs, isExpanded])

    const TerminalContent = ({ isModal = false }: { isModal?: boolean }) => (
        <>
            <div className="h-[40px] bg-muted/30 border-b border-border/60 px-4 flex items-center justify-between transition-colors duration-500">
                <div className="flex items-center gap-2 overflow-hidden">
                    <Terminal size={14} className="text-muted-foreground shrink-0" />
                    <span className="font-mono text-[11px] text-muted-foreground font-semibold truncate">
                        root@ai-engine: /var/log/uvicorn.log
                    </span>
                </div>

                {!isModal ? (
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 text-muted-foreground hover:bg-muted hover:text-foreground rounded-md transition-colors shrink-0"
                        onClick={(e) => { e.stopPropagation(); setIsExpanded(true) }}
                    >
                        <Maximize2 size={12} />
                    </Button>
                ) : (
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 text-muted-foreground hover:bg-red-500/10 hover:text-red-500 dark:hover:text-red-400 rounded-md transition-colors shrink-0"
                        onClick={() => setIsExpanded(false)}
                    >
                        <X size={14} />
                    </Button>
                )}
            </div>

            <div
                ref={isModal ? expandedRef : terminalRef}
                className={`flex-1 overflow-y-auto font-mono leading-relaxed whitespace-pre-wrap break-words bg-background transition-colors duration-500 ${isModal ? 'p-4 md:p-6 text-[11px] md:text-[13px]' : 'p-4 text-[11px]'}`}
            >
                {logs.map((log, i) => (
                    <div
                        key={i}
                        className={`${log.includes("[ERROR]") || log.includes("CRITICAL")
                            ? "text-red-600 dark:text-red-400"
                            : log.includes("[ÉXITO]") || log.includes("[SUCCESS]")
                                ? "text-green-600 dark:text-emerald-400"
                                : log.includes("[INFO]")
                                    ? "text-sky-600 dark:text-sky-400"
                                    : "text-foreground/80"
                            }`}
                    >
                        {log}
                    </div>
                ))}
                {isProcessing && <div className="text-green-600 dark:text-emerald-400 animate-soft-blink mt-1 block">█</div>}
            </div>
        </>
    )

    return (
        <>
            <div
                className="h-full bg-card border border-border/60 shadow-sm rounded-xl overflow-hidden flex flex-col cursor-pointer hover:border-primary/50 transition-colors duration-300"
                onClick={() => !isExpanded && setIsExpanded(true)}
            >
                <TerminalContent />
            </div>

            {isExpanded && (
                <div
                    className="fixed inset-0 z-[100] flex items-center justify-center bg-background/80 backdrop-blur-sm p-4 sm:p-6 animate-in fade-in duration-300"
                    onClick={() => setIsExpanded(false)}
                >
                    <div
                        className="w-[95vw] max-w-5xl h-[85vh] sm:h-[75vh] shadow-2xl animate-in zoom-in-95 duration-300 rounded-xl overflow-hidden border border-border/60 flex flex-col bg-card"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <TerminalContent isModal={true} />
                    </div>
                </div>
            )}
        </>
    )
}