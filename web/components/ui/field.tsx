import { forwardRef, useId } from 'react'
import { cn } from '@/lib/utils'
import { designTokens } from '@/design-system/tokens'

interface FieldShellProps {
  label?: string
  description?: string
  error?: string
  required?: boolean
  inputId: string
  children: React.ReactNode
}

function FieldShell({ label, description, error, required, inputId, children }: FieldShellProps) {
  const descriptionId = description ? `${inputId}-description` : undefined
  const errorId = error ? `${inputId}-error` : undefined
  return (
    <div className="space-y-1.5">
      {label && <label htmlFor={inputId} className="block text-sm font-semibold text-slate-700 dark:text-slate-200">{label}{required && <span className="ml-1 text-red-600" aria-hidden="true">*</span>}</label>}
      {children}
      {description && !error && <p id={descriptionId} className="text-xs leading-5 text-slate-500 dark:text-slate-400">{description}</p>}
      {error && <p id={errorId} role="alert" className="text-xs leading-5 text-red-600 dark:text-red-300">{error}</p>}
    </div>
  )
}

const controlClass = cn(
  'w-full border border-slate-300 bg-white text-sm text-slate-900 placeholder:text-slate-400',
  'dark:border-white/15 dark:bg-slate-900 dark:text-slate-100 dark:placeholder:text-slate-500',
  'disabled:cursor-not-allowed disabled:bg-slate-100 disabled:opacity-60 dark:disabled:bg-white/5',
  designTokens.radius.control,
  designTokens.motion.interactive,
  designTokens.focus
)

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  description?: string
  error?: string
}

export const Input = forwardRef<HTMLInputElement, InputProps>(({ id, label, description, error, required, className, ...props }, ref) => {
  const generatedId = useId()
  const inputId = id ?? generatedId
  return (
    <FieldShell label={label} description={description} error={error} required={required} inputId={inputId}>
      <input ref={ref} id={inputId} required={required} aria-invalid={Boolean(error) || undefined} aria-describedby={error ? `${inputId}-error` : description ? `${inputId}-description` : undefined} className={cn(controlClass, 'h-10 px-3', error && 'border-red-500', className)} {...props} />
    </FieldShell>
  )
})
Input.displayName = 'Input'

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string
  description?: string
  error?: string
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(({ id, label, description, error, required, className, ...props }, ref) => {
  const generatedId = useId()
  const inputId = id ?? generatedId
  return (
    <FieldShell label={label} description={description} error={error} required={required} inputId={inputId}>
      <textarea ref={ref} id={inputId} required={required} aria-invalid={Boolean(error) || undefined} aria-describedby={error ? `${inputId}-error` : description ? `${inputId}-description` : undefined} className={cn(controlClass, 'min-h-24 resize-y px-3 py-2.5', error && 'border-red-500', className)} {...props} />
    </FieldShell>
  )
})
Textarea.displayName = 'Textarea'
