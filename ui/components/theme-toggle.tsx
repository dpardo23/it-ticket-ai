"use client";

import * as React from "react";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";

export function ThemeToggle() {
    const { theme, setTheme } = useTheme();
    // 1. Estado para verificar si ya estamos en el cliente (navegador)
    const [mounted, setMounted] = React.useState(false);

    // 2. Efecto que se ejecuta solo una vez al cargar en el cliente
    React.useEffect(() => {
        setMounted(true);
    }, []);

    // 3. Renderizado de seguridad ("placeholder") mientras el servidor carga
    // Esto evita que React se confunda con las animaciones de Framer Motion
    if (!mounted) {
        return (
            <Button variant="ghost" size="icon" className="relative h-9 w-9 rounded-full opacity-0">
                <span className="sr-only">Cargando tema...</span>
            </Button>
        );
    }

    return (
        <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="relative h-9 w-9 rounded-full"
        >
            <motion.div
                initial={false}
                animate={{
                    rotate: theme === "dark" ? 0 : 180,
                    scale: theme === "dark" ? 1 : 0
                }}
                transition={{ duration: 0.3, ease: "easeInOut" }}
                className="absolute"
            >
                <Moon className="h-5 w-5 text-foreground" />
            </motion.div>
            <motion.div
                initial={false}
                animate={{
                    rotate: theme === "light" ? 0 : -180,
                    scale: theme === "light" ? 1 : 0
                }}
                transition={{ duration: 0.3, ease: "easeInOut" }}
                className="absolute"
            >
                <Sun className="h-5 w-5 text-foreground" />
            </motion.div>
            <span className="sr-only">Toggle theme</span>
        </Button>
    );
}