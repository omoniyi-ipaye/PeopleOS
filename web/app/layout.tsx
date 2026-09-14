'use client'

import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from './providers'
import { AppShell } from '@/components/app-shell'
import { cn } from '@/lib/utils'

const inter = Inter({ subsets: ['latin'] })

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <title>PeopleOS — People Intelligence</title>
        <meta name="description" content="Governed, privacy-preserving people intelligence for workforce decisions" />
      </head>
      <body className={cn(inter.className, 'bg-slate-50 text-slate-950 antialiased dark:bg-slate-950 dark:text-white')}>
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  )
}
