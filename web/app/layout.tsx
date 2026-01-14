'use client'

import { useState, useEffect } from 'react'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from './providers'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { cn } from '@/lib/utils'
import { SplashScreen } from '@/components/ui/splash-screen'
import { usePathname } from 'next/navigation'
import { AnimatePresence, motion } from 'framer-motion'

const inter = Inter({ subsets: ['latin'] })

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const [loading, setLoading] = useState(true)
  const pathname = usePathname()
  const isHome = pathname === '/'

  return (
    <html lang="en">
      <head>
        <title>PeopleOS - HR Intelligence Operating System</title>
        <meta name="description" content="Privacy-preserving People Analytics Platform" />
      </head>
      <body className={cn(inter.className, "bg-background dark:bg-background-dark text-text-primary dark:text-text-dark-primary antialiased")}>
        <Providers>
          <AnimatePresence mode="wait">
            {loading && isHome ? (
              <motion.div
                key="splash"
                initial={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.5 }}
              >
                <SplashScreen finishLoading={() => setLoading(false)} />
              </motion.div>
            ) : (
              <motion.div
                key="content"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5 }}
                className="flex h-screen overflow-hidden"
              >
                <Sidebar />
                <div className="flex-1 flex flex-col overflow-hidden">
                  <Header />
                  <main className="flex-1 overflow-auto p-6">
                    {children}
                  </main>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </Providers>
      </body>
    </html>
  )
}
