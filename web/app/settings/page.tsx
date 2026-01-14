'use client'

import { useQuery } from '@tanstack/react-query'
import { GlassCard } from '@/components/ui/glass-card'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api-client'
import { UploadStatus } from '@/types/api'
import {
  Server,
  Database,
  Brain,
  Cpu,
  Search,
  CheckCircle,
  XCircle,
  Info,
  Activity,
  Shield,
  Zap
} from 'lucide-react'
import { cn } from '@/lib/utils'

export default function SettingsPage() {
  const { data: status, isLoading: isStatusLoading } = useQuery<UploadStatus>({
    queryKey: ['api', 'status'],
    queryFn: () => api.getStatus() as Promise<UploadStatus>,
  })

  const { data: health, isLoading: isHealthLoading } = useQuery<{ status: string }>({
    queryKey: ['api', 'health'],
    queryFn: () => api.getHealth() as Promise<{ status: string }>,
  })

  const isLoading = isStatusLoading || isHealthLoading

  return (
    <div className="h-[calc(100vh-100px)] flex flex-col animate-in fade-in duration-700">
      <div className="flex-none mb-6">
        <h1 className="text-4xl font-display font-bold text-gradient bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-400">
          System Settings
        </h1>
        <p className="text-text-secondary dark:text-text-dark-secondary mt-2 text-lg font-light">
          Monitor system health, engine status, and configuration.
        </p>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 space-y-6">

        {/* System Health Hero */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <GlassCard className="md:col-span-2 relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-64 h-64 bg-accent/5 rounded-full blur-3xl -z-10 group-hover:bg-accent/10 transition-colors duration-500" />
            <div className="p-6">
              <div className="flex items-center gap-4 mb-6">
                <div className={cn(
                  "w-12 h-12 rounded-xl flex items-center justify-center shadow-lg",
                  health?.status === 'healthy' ? "bg-success/10 text-success" : "bg-danger/10 text-danger"
                )}>
                  <Activity className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-lg font-bold">System Status</h3>
                  <p className="text-sm text-text-secondary">Real-time operaional metrics</p>
                </div>
                <div className="ml-auto">
                  <Badge variant={health?.status === 'healthy' ? 'success' : 'danger'} className="animate-pulse">
                    {health?.status === 'healthy' ? 'Operational' : 'Issues Detected'}
                  </Badge>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="p-4 rounded-xl bg-surface-secondary/30 border border-white/5 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Server className="w-5 h-5 text-text-muted" />
                    <div>
                      <p className="font-semibold text-sm">API Backend</p>
                      <p className="text-xs text-text-secondary">FastAPI :8000</p>
                    </div>
                  </div>
                  <div className="h-2.5 w-2.5 rounded-full bg-success shadow-[0_0_8px_rgba(74,222,128,0.5)]" />
                </div>

                <div className="p-4 rounded-xl bg-surface-secondary/30 border border-white/5 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Database className="w-5 h-5 text-text-muted" />
                    <div>
                      <p className="font-semibold text-sm">Data Engine</p>
                      <p className="text-xs text-text-secondary">
                        {status?.data?.loaded ? `${status.data.row_count.toLocaleString()} rows` : 'Idle'}
                      </p>
                    </div>
                  </div>
                  <div className={cn(
                    "h-2.5 w-2.5 rounded-full shadow-lg transition-colors",
                    status?.data?.loaded ? "bg-success shadow-success/50" : "bg-warning shadow-warning/50"
                  )} />
                </div>
              </div>
            </div>
          </GlassCard>

          <GlassCard className="flex flex-col justify-center items-center text-center p-6 relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-transparent blur-xl" />
            <Shield className="w-12 h-12 text-purple-500 mb-4" />
            <h3 className="font-bold text-lg mb-1">Privacy First</h3>
            <p className="text-sm text-text-secondary mb-4">
              Data processed locally. <br /> No external uploads.
            </p>
            <Badge variant="outline" className="border-purple-500/30 text-purple-500 bg-purple-500/5">
              Secure Environment
            </Badge>
          </GlassCard>
        </div>

        {/* Analytics Engines Grid */}
        <GlassCard className="p-6">
          <div className="flex items-center gap-2 mb-6">
            <Cpu className="w-5 h-5 text-accent" />
            <h3 className="font-bold text-lg">Analytics Engines</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { name: 'Analytics Core', key: 'analytics', icon: Zap },
              { name: 'ML Predictions', key: 'ml', icon: Brain },
              { name: 'Compensation', key: 'compensation', icon: Database },
              { name: 'Succession', key: 'succession', icon: Database },
              { name: 'Team Dynamics', key: 'team_dynamics', icon: Database },
              { name: 'Fairness', key: 'fairness', icon: Database },
              { name: 'Semantic Search', key: 'vector_search', icon: Search },
              { name: 'AI Advisor', key: 'llm', icon: Brain },
            ].map((engine) => {
              const isActive = status?.engines?.[engine.key as keyof typeof status.engines]
              return (
                <div
                  key={engine.key}
                  className="flex items-center justify-between p-4 rounded-xl bg-surface-secondary/20 hover:bg-surface-secondary/40 border border-white/5 transition-all group"
                >
                  <div className="flex items-center gap-3">
                    <div className={cn(
                      "p-2 rounded-lg transition-colors",
                      isActive ? "bg-accent/10 text-accent" : "bg-gray-100 dark:bg-gray-800 text-gray-400"
                    )}>
                      <engine.icon className="w-4 h-4" />
                    </div>
                    <span className="text-sm font-bold">{engine.name}</span>
                  </div>
                  {isActive ? (
                    <CheckCircle className="w-4 h-4 text-success" />
                  ) : (
                    <XCircle className="w-4 h-4 text-text-muted opacity-50" />
                  )}
                </div>
              )
            })}
          </div>
        </GlassCard>

        {/* Features & Info */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {status?.features_enabled && (
            <GlassCard className="p-6">
              <h3 className="font-bold text-lg mb-4">Enabled Features</h3>
              <div className="flex flex-wrap gap-2">
                {Object.entries(status.features_enabled).map(([key, enabled]) => (
                  <Badge
                    key={key}
                    variant={enabled ? 'success' : 'outline'}
                    className={cn("uppercase tracking-wider text-xs", !enabled && "opacity-50")}
                  >
                    {key.replace(/_/g, ' ')}
                  </Badge>
                ))}
              </div>
            </GlassCard>
          )}

          <GlassCard className="p-6">
            <h3 className="font-bold text-lg mb-4 flex items-center gap-2">
              <Info className="w-4 h-4 text-accent" />
              About PeopleOS
            </h3>
            <div className="space-y-4 text-sm text-text-secondary">
              <p>
                Version 1.0.0 • Local-first HR Analytics Platform
              </p>
              <div className="flex flex-wrap gap-2 pt-2 border-t border-white/10">
                {['Next.js 14', 'FastAPI', 'PyTorch', 'FAISS', 'Tailwind'].map(tech => (
                  <Badge key={tech} variant="outline" className="bg-surface-secondary dark:bg-white/5 text-[10px]">
                    {tech}
                  </Badge>
                ))}
              </div>
            </div>
          </GlassCard>
        </div>
      </div>
    </div>
  )
}
