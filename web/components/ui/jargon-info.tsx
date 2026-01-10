'use client'

import { Info } from 'lucide-react'
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from './tooltip'

interface JargonInfoProps {
    term: string
    description: string
}

export function JargonInfo({ term, description }: JargonInfoProps) {
    return (
        <TooltipProvider>
            <Tooltip>
                <TooltipTrigger asChild>
                    <button className="inline-flex items-center gap-1 text-text-muted hover:text-accent transition-colors">
                        <Info className="w-3.5 h-3.5" />
                    </button>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs bg-surface dark:bg-surface-dark border border-border dark:border-border-dark p-3 shadow-xl">
                    <div className="space-y-1">
                        <div className="text-xs font-bold text-text-primary dark:text-text-dark-primary">{term}</div>
                        <div className="text-[10px] text-text-secondary dark:text-text-dark-secondary leading-relaxed">
                            {description}
                        </div>
                    </div>
                </TooltipContent>
            </Tooltip>
        </TooltipProvider>
    )
}
