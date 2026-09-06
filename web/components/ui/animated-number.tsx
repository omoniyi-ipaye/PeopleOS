"use client"

import { useEffect, useRef } from "react"
import { useInView, useMotionValue, useSpring } from "framer-motion"

export function AnimatedNumber({
    value,
    direction = "up",
    delay = 0,
    className
}: {
    value: number
    direction?: "up" | "down"
    delay?: number
    className?: string
}) {
    const ref = useRef<HTMLSpanElement>(null)
    const motionValue = useMotionValue(direction === "down" ? value : 0)
    const springValue = useSpring(motionValue, {
        damping: 60,
        stiffness: 100,
    })
    const isInView = useInView(ref, { once: true, margin: "-100px" })

    useEffect(() => {
        if (isInView) {
            const timeout = window.setTimeout(() => {
                motionValue.set(direction === "down" ? 0 : value)
            }, delay * 1000)
            return () => window.clearTimeout(timeout)
        }
    }, [motionValue, isInView, delay, value, direction])

    useEffect(() => {
        return springValue.on("change", (latest) => {
            if (ref.current) {
                ref.current.textContent = Intl.NumberFormat("en-US").format(Math.round(latest))
            }
        })
    }, [springValue])

    return <span className={className} ref={ref} />
}
