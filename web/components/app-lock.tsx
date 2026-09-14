'use client'

import { FormEvent, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { LockKeyhole, ShieldCheck, UnlockKeyhole } from 'lucide-react'
import { api } from '@/lib/api-client'
import { Button, Input, SectionHeader, StatusBadge, Surface } from '@/components/ui'

export interface AppLockStatus {
  enabled: boolean
  locked: boolean
}

const appLockQueryKey = ['app-lock', 'status'] as const

function errorMessage(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback
}

export function AppLockSetup({ context = 'setup' }: { context?: 'setup' | 'settings' }) {
  const queryClient = useQueryClient()
  const status = useQuery<AppLockStatus>({ queryKey: appLockQueryKey, queryFn: api.appLock.getStatus, retry: false })
  const [pin, setPin] = useState('')
  const [confirmation, setConfirmation] = useState('')
  const [currentPin, setCurrentPin] = useState('')
  const [nextPin, setNextPin] = useState('')
  const [nextConfirmation, setNextConfirmation] = useState('')
  const [validationError, setValidationError] = useState<string | null>(null)
  const setup = useMutation({
    mutationFn: () => api.appLock.setup(pin),
    onSuccess: next => {
      queryClient.setQueryData(appLockQueryKey, next)
      void queryClient.invalidateQueries({ queryKey: ['platform', 'status'] })
      setPin('')
      setConfirmation('')
      setValidationError(null)
    },
  })
  const lock = useMutation({
    mutationFn: api.appLock.lock,
    onSuccess: next => queryClient.setQueryData(appLockQueryKey, next),
  })
  const change = useMutation({
    mutationFn: () => api.appLock.change(currentPin, nextPin),
    onSuccess: next => {
      queryClient.setQueryData(appLockQueryKey, next)
      setCurrentPin('')
      setNextPin('')
      setNextConfirmation('')
      setValidationError(null)
    },
  })
  const disable = useMutation({
    mutationFn: () => api.appLock.disable(currentPin),
    onSuccess: next => {
      queryClient.setQueryData(appLockQueryKey, next)
      setCurrentPin('')
      setValidationError(null)
    },
  })
  const pinLengthHint = pin.length > 0 && pin.length < 6 ? `${pin.length}/6 digits entered` : 'Use exactly six digits.'
  const currentPinLengthHint = currentPin.length > 0 && currentPin.length < 6 ? `${currentPin.length}/6 digits entered` : undefined
  const confirmationError = confirmation.length === 6 && pin !== confirmation ? 'PIN entries do not match.' : undefined

  if (status.isLoading) return null
  if (status.isError) return <Surface tone="warning" padding="compact"><div role="alert" className="text-sm text-red-700 dark:text-red-300">PeopleOS could not check the local app-lock state.</div></Surface>

  if (status.data?.enabled) {
    return <Surface padding="lg" className="border-violet-200/70 dark:border-violet-500/20">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-3"><div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300"><LockKeyhole className="h-5 w-5" aria-hidden="true" /></div><div><div className="text-sm font-semibold text-slate-950 dark:text-white">App lock is ready</div><p className="mt-1 max-w-xl text-xs leading-5 text-slate-600 dark:text-slate-400">A six-digit owner PIN protects this local PeopleOS installation when you leave it unattended.</p></div></div>
        <Button variant="secondary" disabled={lock.isPending} isLoading={lock.isPending} onClick={() => lock.mutate()}><LockKeyhole className="h-4 w-4" />Lock app</Button>
      </div>
      {lock.isError && <p role="alert" className="mt-3 text-xs text-red-600 dark:text-red-300">{errorMessage(lock.error, 'PeopleOS could not be locked.')}</p>}
      {context === 'settings' && <div className="mt-5 border-t border-slate-200/80 pt-5 dark:border-white/10">
        <div className="text-sm font-semibold text-slate-950 dark:text-white">Manage owner lock</div>
        <p className="mt-1 text-xs leading-5 text-slate-600 dark:text-slate-400">Changing or removing the lock requires the current PIN.</p>
        <form onSubmit={event => {
          event.preventDefault()
          if (!/^\d{6}$/.test(currentPin) || !/^\d{6}$/.test(nextPin)) return setValidationError('Use exactly six digits for each PIN.')
          if (nextPin !== nextConfirmation) return setValidationError('The new PIN entries do not match.')
          setValidationError(null)
          change.mutate()
        }} className="mt-4 grid gap-4 sm:grid-cols-3" noValidate>
          <Input id="current-owner-pin" label="Current owner PIN" type="password" inputMode="numeric" autoComplete="current-password" maxLength={6} value={currentPin} onChange={event => setCurrentPin(event.target.value.replace(/\D/g, '').slice(0, 6))} error={validationError ?? undefined} helperText={currentPinLengthHint} />
          <Input id="new-owner-pin" label="New owner PIN" type="password" inputMode="numeric" autoComplete="new-password" maxLength={6} value={nextPin} onChange={event => setNextPin(event.target.value.replace(/\D/g, '').slice(0, 6))} />
          <Input id="new-owner-pin-confirmation" label="Confirm new PIN" type="password" inputMode="numeric" autoComplete="new-password" maxLength={6} value={nextConfirmation} onChange={event => setNextConfirmation(event.target.value.replace(/\D/g, '').slice(0, 6))} />
          <div className="flex flex-wrap items-center gap-3 sm:col-span-3"><Button type="submit" variant="secondary" disabled={change.isPending} isLoading={change.isPending}><LockKeyhole className="h-4 w-4" />Change PIN</Button><Button type="button" variant="ghost" disabled={disable.isPending} isLoading={disable.isPending} onClick={() => {
            if (!/^\d{6}$/.test(currentPin)) return setValidationError('Enter the current six-digit PIN to remove the lock.')
            setValidationError(null)
            disable.mutate()
          }}>Remove app lock</Button>{(change.isError || disable.isError) && <span role="alert" className="text-xs text-red-600 dark:text-red-300">{errorMessage(change.error ?? disable.error, 'The app-lock change could not be saved.')}</span>}</div>
        </form>
      </div>}
    </Surface>
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!/^\d{6}$/.test(pin)) return setValidationError('Use exactly six digits for the owner PIN.')
    if (pin !== confirmation) return setValidationError('The PIN entries do not match.')
    setValidationError(null)
    setup.mutate()
  }

  return <Surface padding="lg" className="border-slate-200 dark:border-white/10">
    <SectionHeader title={context === 'setup' ? 'Protect this installation' : 'App lock'} description="Optional local protection for the owner of this PeopleOS installation." />
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(280px,0.7fr)]">
      <form onSubmit={submit} className="grid gap-4 sm:grid-cols-2" noValidate>
        <Input id="owner-pin" label="Six-digit owner PIN" type="password" inputMode="numeric" autoComplete="new-password" maxLength={6} pattern="[0-9]{6}" value={pin} onChange={event => setPin(event.target.value.replace(/\D/g, '').slice(0, 6))} error={validationError ?? undefined} helperText={pinLengthHint} />
        <Input id="owner-pin-confirmation" label="Confirm PIN" type="password" inputMode="numeric" autoComplete="new-password" maxLength={6} pattern="[0-9]{6}" value={confirmation} onChange={event => setConfirmation(event.target.value.replace(/\D/g, '').slice(0, 6))} error={confirmationError} helperText="Re-enter the same six digits." />
        <div className="flex flex-wrap items-center gap-3 sm:col-span-2"><Button type="submit" disabled={setup.isPending || !/^\d{6}$/.test(pin) || pin !== confirmation} isLoading={setup.isPending}><LockKeyhole className="h-4 w-4" />Set owner lock</Button>{setup.isError && <span role="alert" className="text-xs text-red-600 dark:text-red-300">{errorMessage(setup.error, 'The owner PIN could not be saved.')}</span>}</div>
      </form>
      <div className="rounded-2xl border border-slate-200/80 bg-slate-50 p-4 dark:border-white/10 dark:bg-white/[0.03]"><div className="flex items-start gap-3"><ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-violet-600 dark:text-violet-300" /><p className="text-xs leading-5 text-slate-600 dark:text-slate-400">The PIN is stored as a one-way hash in local app configuration. It is a local unattended-session lock, not a replacement for your computer login or full-disk encryption.</p></div><StatusBadge tone="neutral" className="mt-3">Owner only · local</StatusBadge></div>
    </div>
  </Surface>
}

export function AppLockScreen() {
  const queryClient = useQueryClient()
  const [pin, setPin] = useState('')
  const unlock = useMutation({
    mutationFn: () => api.appLock.unlock(pin),
    onSuccess: next => {
      queryClient.setQueryData(appLockQueryKey, next)
      void queryClient.invalidateQueries()
      setPin('')
    },
  })

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (/^\d{6}$/.test(pin)) unlock.mutate()
  }

  return <div className="mx-auto flex min-h-[80vh] max-w-lg items-center justify-center p-4"><Surface padding="lg" className="w-full text-center shadow-lg shadow-slate-200/30 dark:shadow-none"><div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300"><LockKeyhole className="h-7 w-7" aria-hidden="true" /></div><div className="mt-5"><div className="text-xs font-bold uppercase tracking-[0.16em] text-violet-600 dark:text-violet-300">PeopleOS is locked</div><h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950 dark:text-white">Enter the owner PIN to continue</h1><p className="mx-auto mt-3 max-w-sm text-sm leading-6 text-slate-600 dark:text-slate-400">Your workforce data remains in this local installation. Unlock it when you are ready to return.</p></div><form onSubmit={submit} className="mx-auto mt-6 max-w-xs space-y-4" noValidate><Input id="unlock-owner-pin" label="Owner PIN" type="password" inputMode="numeric" autoComplete="current-password" maxLength={6} pattern="[0-9]{6}" value={pin} onChange={event => setPin(event.target.value.replace(/\D/g, '').slice(0, 6))} autoFocus error={unlock.isError ? errorMessage(unlock.error, 'That PIN did not unlock PeopleOS.') : undefined} /><Button type="submit" className="w-full" disabled={!/^\d{6}$/.test(pin) || unlock.isPending} isLoading={unlock.isPending}><UnlockKeyhole className="h-4 w-4" />Unlock PeopleOS</Button></form></Surface></div>
}
