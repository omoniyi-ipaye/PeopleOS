'use client'

import React, { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain } from 'lucide-react'

export interface SplashScreenProps {
    finishLoading?: () => void
}

export const SplashScreen: React.FC<SplashScreenProps> = ({ finishLoading }) => {
    const [isMounted, setIsMounted] = useState(false)

    useEffect(() => {
        setIsMounted(true)
        const timeout = setTimeout(() => {
            if (finishLoading) finishLoading()
        }, 2500)
        return () => clearTimeout(timeout)
    }, [finishLoading])

    if (!isMounted) return null

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-950 overflow-hidden">
            {/* Background Animated Gradients */}
            <div className="absolute inset-0 overflow-hidden">
                <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-accent/20 rounded-full blur-[120px] animate-pulse" />
                <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-success/10 rounded-full blur-[120px] animate-pulse-slow" />
            </div>

            <div className="relative flex flex-col items-center">
                <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                    className="relative"
                >
                    <div className="absolute inset-0 bg-accent/30 blur-2xl rounded-full" />
                    <div className="relative bg-slate-900 border border-slate-800 p-6 rounded-3xl shadow-2xl">
                        <Brain className="w-16 h-16 text-accent animate-pulse" />
                    </div>
                </motion.div>

                <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.5, duration: 0.8 }}
                    className="mt-8 text-center"
                >
                    <h1 className="text-4xl font-bold tracking-tighter text-white sm:text-5xl">
                        People<span className="text-accent">OS</span>
                    </h1>
                    <div className="mt-4 flex items-center justify-center space-x-2">
                        <div className="h-1 w-1 bg-accent rounded-full animate-bounce [animation-delay:-0.3s]" />
                        <div className="h-1 w-1 bg-accent rounded-full animate-bounce [animation-delay:-0.15s]" />
                        <div className="h-1 w-1 bg-accent rounded-full animate-bounce" />
                    </div>
                    <p className="mt-6 text-sm font-medium text-slate-400 uppercase tracking-[0.3em]">
                        Strategic Intelligence
                    </p>
                </motion.div>
            </div>

            {/* Glassmorphism Bottom Card */}
            <motion.div
                initial={{ y: 100, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 1, duration: 1 }}
                className="absolute bottom-12 px-6 py-3 rounded-full border border-white/10 bg-white/5 backdrop-blur-md text-slate-400 text-xs font-medium tracking-wide"
            >
                Privacy-Preserving People Analytics Platform
            </motion.div>
        </div>
    )
}
