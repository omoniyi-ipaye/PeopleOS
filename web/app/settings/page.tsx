'use client'

import { useQuery } from '@tanstack/react-query'
import { Card } from '@/components/ui/card'
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
} from 'lucide-react'

export default function SettingsPage() {
  const { data: status, isLoading } = useQuery<UploadStatus>({
    queryKey: ['api', 'status'],
    queryFn: () => api.getStatus() as Promise<UploadStatus>,
  })

  const { data: health } = useQuery<{ status: string }>({
    queryKey: ['api', 'health'],
    queryFn: () => api.getHealth() as Promise<{ status: string }>,
  })

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Page Title */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">Settings</h1>
        <p className="text-text-secondary dark:text-text-dark-secondary mt-1">
          System status and configuration
        </p>
      </div>

      {/* System Status */}
      <Card title="System Status">
        <div className="flex items-center gap-3 mb-6">
          <div
            className={`w-3 h-3 rounded-full ${health?.status === 'healthy' ? 'bg-success shadow-[0_0_8px_rgba(74,222,128,0.5)]' : 'bg-danger shadow-[0_0_8px_rgba(248,113,113,0.5)]'
              }`}
          />
          <span className="font-semibold text-text-primary dark:text-text-dark-primary">
            {health?.status === 'healthy' ? 'All Systems Operational' : 'System Issue'}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* API Server */}
          <div className="p-4 rounded-lg bg-surface hover:bg-surface-hover dark:bg-surface-dark hover:dark:bg-surface-dark-hover border border-border dark:border-border-dark transition-colors shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Server className="w-4 h-4 text-text-muted dark:text-text-dark-muted" />
                <span className="font-semibold text-text-primary dark:text-text-dark-primary">API Server</span>
              </div>
              <Badge variant="success" size="sm">Running</Badge>
            </div>
            <p className="text-sm text-text-secondary dark:text-text-dark-secondary">
              FastAPI backend on port 8000
            </p>
          </div>

          {/* Data Status */}
          <div className="p-4 rounded-lg bg-surface hover:bg-surface-hover dark:bg-surface-dark hover:dark:bg-surface-dark-hover border border-border dark:border-border-dark transition-colors shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-text-muted dark:text-text-dark-muted" />
                <span className="font-semibold text-text-primary dark:text-text-dark-primary">Data</span>
              </div>
              <Badge
                variant={status?.data?.loaded ? 'success' : 'warning'}
                size="sm"
              >
                {status?.data?.loaded ? 'Loaded' : 'Not Loaded'}
              </Badge>
            </div>
            <p className="text-sm text-text-secondary dark:text-text-dark-secondary">
              {status?.data?.loaded
                ? `${status.data.row_count.toLocaleString()} rows`
                : 'Upload data to get started'}
            </p>
          </div>
        </div>
      </Card>

      {/* Engine Status */}
      <Card title="Analytics Engines" subtitle="Status of backend processing engines">
        {isLoading ? (
          <div className="animate-pulse space-y-3">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-12 bg-surface-dark-hover rounded-lg" />
            ))}
          </div>
        ) : (
          <div className="space-y-3">
            {[
              { name: 'Analytics Engine', key: 'analytics', icon: Cpu },
              { name: 'ML Predictions', key: 'ml', icon: Brain },
              { name: 'Compensation Analysis', key: 'compensation', icon: Database },
              { name: 'Succession Planning', key: 'succession', icon: Database },
              { name: 'Team Dynamics', key: 'team_dynamics', icon: Database },
              { name: 'Fairness Analysis', key: 'fairness', icon: Database },
              { name: 'Semantic Search', key: 'vector_search', icon: Search },
              { name: 'AI Advisor (LLM)', key: 'llm', icon: Brain },
            ].map((engine) => {
              const isActive = status?.engines?.[engine.key]
              return (
                <div
                  key={engine.key}
                  className="flex items-center justify-between p-3 rounded-lg bg-surface hover:bg-surface-hover dark:bg-surface-dark hover:dark:bg-surface-dark-hover border border-border dark:border-border-dark transition-colors shadow-sm"
                >
                  <div className="flex items-center gap-3">
                    <engine.icon className="w-4 h-4 text-text-muted dark:text-text-dark-muted" />
                    <span className="text-sm font-semibold text-text-primary dark:text-text-dark-primary">{engine.name}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {isActive ? (
                      <>
                        <CheckCircle className="w-4 h-4 text-success" />
                        <span className="text-sm font-medium text-success">Active</span>
                      </>
                    ) : (
                      <>
                        <XCircle className="w-4 h-4 text-text-muted dark:text-text-dark-muted" />
                        <span className="text-sm font-medium text-text-muted dark:text-text-dark-muted">Inactive</span>
                      </>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </Card>

      {/* Features Enabled */}
      {status?.features_enabled && (
        <Card title="Features Enabled" subtitle="Based on uploaded data columns">
          <div className="flex flex-wrap gap-2">
            {Object.entries(status.features_enabled).map(([key, enabled]) => (
              <Badge
                key={key}
                variant={enabled ? 'success' : 'default'}
                size="sm"
              >
                {key.replace(/_/g, ' ')}
              </Badge>
            ))}
          </div>
        </Card>
      )}

      {/* About */}
      <Card title="About PeopleOS">
        <div className="space-y-4">
          <div className="flex items-start gap-3">
            <Info className="w-5 h-5 text-accent mt-0.5" />
            <div>
              <h4 className="font-medium">Version 1.0.0</h4>
              <p className="text-sm text-text-secondary mt-1">
                PeopleOS is a comprehensive HR analytics platform with ML-powered
                insights, predictive analytics, and strategic workforce planning.
              </p>
            </div>
          </div>

          <div className="pt-4 border-t border-border dark:border-border-dark">
            <h4 className="text-sm font-medium text-text-secondary mb-3">
              Technology Stack
            </h4>
            <div className="flex flex-wrap gap-2">
              {[
                'Next.js 14',
                'FastAPI',
                'Python',
                'scikit-learn',
                'SHAP',
                'FAISS',
                'Tailwind CSS',
                'Recharts',
              ].map((tech) => (
                <Badge key={tech} variant="info" size="sm">
                  {tech}
                </Badge>
              ))}
            </div>
          </div>

          <div className="pt-4 border-t border-border dark:border-border-dark text-sm text-text-muted dark:text-text-dark-muted text-center italic">
            <p>Local-first analytics • Privacy by design • No data leaves your machine</p>
          </div>
        </div>
      </Card>
    </div>
  )
}
