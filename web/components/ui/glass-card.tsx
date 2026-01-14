import { cn } from "@/lib/utils"

interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
    children: React.ReactNode
    variant?: "default" | "hover" | "active"
    className?: string
}

export function GlassCard({
    children,
    variant = "default",
    className,
    ...props
}: GlassCardProps) {
    return (
        <div
            className={cn(
                "glass-card p-6",
                variant === "hover" && "hover:scale-[1.02] hover:bg-white/10 dark:hover:bg-white/5 cursor-pointer",
                variant === "active" && "border-accent/40 bg-accent/5",
                className
            )}
            {...props}
        >
            {/* Glossy overlay effect */}
            <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent opacity-0 pointer-events-none transition-opacity duration-300 group-hover:opacity-100" />

            <div className="relative z-10">
                {children}
            </div>
        </div>
    )
}
