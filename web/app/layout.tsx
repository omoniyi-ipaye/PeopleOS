'use client'

import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from './providers'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
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
          <div className="flex h-screen overflow-hidden">
            <Sidebar />
            <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
              <Header />
              <main className="flex-1 overflow-y-auto px-4 py-5 md:px-7 md:py-7">
                {children}
              </main>
            </div>
          </div>
        </Providers>
      </body>
    </html>
  )
}
