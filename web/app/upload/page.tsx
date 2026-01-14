'use client'

import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { GlassCard } from '@/components/ui/glass-card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api-client'
import { UploadStatus, UploadResponse } from '@/types/api'
import { cn } from '@/lib/utils'
import {
  Upload,
  FileText,
  CheckCircle,
  AlertTriangle,
  Database,
  Trash2,
  Download,
  Sparkles,
  ArrowRight,
  Loader2
} from 'lucide-react'

export default function UploadPage() {
  const queryClient = useQueryClient()
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null)

  const { data: status } = useQuery<UploadStatus>({
    queryKey: ['upload', 'status'],
    queryFn: api.upload.getStatus as any,
  })

  const uploadMutation = useMutation({
    mutationFn: api.upload.uploadFile,
    onSuccess: (data: any) => {
      setUploadResult(data)
      queryClient.invalidateQueries()
    },
  })

  const loadSampleMutation = useMutation({
    mutationFn: api.upload.loadSample,
    onSuccess: (data: any) => {
      setUploadResult(data)
      queryClient.invalidateQueries()
    },
  })

  const resetMutation = useMutation({
    mutationFn: api.upload.reset,
    onSuccess: () => {
      setUploadResult(null)
      queryClient.invalidateQueries()
    },
  })

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        uploadMutation.mutate(acceptedFiles[0])
      }
    },
    [uploadMutation]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/json': ['.json'],
    },
    maxFiles: 1,
  })

  return (
    <div className="h-[calc(100vh-100px)] flex flex-col relative overflow-hidden animate-in fade-in duration-700">
      <div className="absolute top-[-20%] left-[-10%] w-[500px] h-[500px] bg-accent/5 rounded-full blur-3xl -z-10 animate-pulse-subtle" />

      <div className="flex-none mb-8 text-center pt-8">
        <h1 className="text-5xl font-display font-bold text-gradient bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-500 dark:from-white dark:to-gray-400 mb-4">
          Import Data
        </h1>
        <p className="text-xl text-text-secondary dark:text-text-dark-secondary font-light max-w-2xl mx-auto">
          Upload your HR dataset to unlock AI-powered insights. <br className="hidden md:block" /> We support CSV and JSON formats.
        </p>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar px-4 pb-12">
        <div className="max-w-4xl mx-auto space-y-8">

          {/* Active Data Status */}
          {status?.has_data && (
            <div className="animate-in fade-in slide-in-from-top-4">
              <GlassCard className="bg-success/5 border-success/20 p-4 flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="p-3 bg-success/10 rounded-xl text-success">
                    <Database className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="font-bold text-success flex items-center gap-2">
                      System Data Active <CheckCircle className="w-4 h-4" />
                    </h3>
                    <p className="text-sm text-text-secondary">
                      <span className="font-bold text-text-primary dark:text-white">{status.employee_count.toLocaleString()}</span> employees loaded
                    </p>
                  </div>
                </div>
                <Button
                  variant="danger"
                  size="sm"
                  onClick={() => resetMutation.mutate()}
                  isLoading={resetMutation.isPending}
                  className="rounded-xl"
                >
                  <Trash2 className="w-4 h-4 mr-2" /> Reset
                </Button>
              </GlassCard>
            </div>
          )}

          {/* Main Dropzone */}
          <div {...getRootProps()} className="group cursor-pointer">
            <input {...getInputProps()} />
            <GlassCard className={cn(
              "relative border-2 border-dashed transition-all duration-300 min-h-[300px] flex flex-col items-center justify-center text-center p-12 overflow-hidden",
              isDragActive ? "border-accent bg-accent/5 scale-[0.99]" : "border-white/10 hover:border-accent/40 hover:bg-surface-secondary/30"
            )}>
              <div className={cn(
                "w-20 h-20 rounded-3xl bg-surface-secondary/50 flex items-center justify-center mb-6 transition-all duration-500",
                isDragActive ? "bg-accent text-white rotate-12 scale-110" : "text-text-muted group-hover:text-accent group-hover:bg-accent/10"
              )}>
                {uploadMutation.isPending ? (
                  <Loader2 className="w-10 h-10 animate-spin" />
                ) : (
                  <Upload className="w-10 h-10" />
                )}
              </div>

              <h3 className="text-2xl font-bold mb-2 group-hover:text-accent transition-colors">
                {isDragActive ? 'Drop file to upload' : 'Drag & Drop your dataset'}
              </h3>
              <p className="text-text-secondary mb-8 max-w-sm mx-auto leading-relaxed">
                Support for standard HR CSV exports. <br /> Minimum 50 records required for ML analysis.
              </p>

              <div className="flex items-center gap-4">
                <Button className="rounded-xl pointer-events-none bg-white dark:bg-slate-800 text-black dark:text-white border border-border shadow-lg">
                  Browse Files
                </Button>
                {uploadMutation.isPending && (
                  <Badge variant="default" className="bg-accent/10 text-accent border-accent/20 animate-pulse">
                    Processing...
                  </Badge>
                )}
              </div>
            </GlassCard>
          </div>

          {/* Quick Actions */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Template Download */}
            <GlassCard
              className="p-6 cursor-pointer hover:border-accent/30 transition-all flex items-center justify-between group"
              onClick={() => api.upload.downloadTemplate()}
            >
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-xl bg-surface-secondary/50 text-text-muted group-hover:text-primary transition-colors">
                  <Download className="w-5 h-5" />
                </div>
                <div className="text-left">
                  <h4 className="font-bold">Download Template</h4>
                  <p className="text-sm text-text-secondary">See required format</p>
                </div>
              </div>
            </GlassCard>

            {/* Load Sample */}
            <GlassCard
              className={cn(
                "p-6 cursor-pointer hover:border-accent/30 transition-all flex items-center justify-between group",
                status?.has_data && "opacity-50 pointer-events-none"
              )}
              onClick={() => !status?.has_data && loadSampleMutation.mutate()}
            >
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-xl bg-purple-500/10 text-purple-500">
                  <Sparkles className="w-5 h-5" />
                </div>
                <div className="text-left">
                  <h4 className="font-bold group-hover:text-purple-400 transition-colors">Load Sample Data</h4>
                  <p className="text-sm text-text-secondary">Try with 1,000 records</p>
                </div>
              </div>
              {loadSampleMutation.isPending && <Loader2 className="w-5 h-5 animate-spin text-purple-500" />}
            </GlassCard>
          </div>

          {/* Result Message */}
          {uploadResult && (
            <div className="animate-in fade-in slide-in-from-bottom-4">
              <GlassCard className={cn("p-6 border-l-4", uploadResult.success ? "border-l-success" : "border-l-danger")}>
                <div className="flex items-start gap-4">
                  <div className={cn("p-2 rounded-lg", uploadResult.success ? "bg-success/10 text-success" : "bg-danger/10 text-danger")}>
                    {uploadResult.success ? <CheckCircle className="w-6 h-6" /> : <AlertTriangle className="w-6 h-6" />}
                  </div>
                  <div className="flex-1">
                    <h3 className="font-bold text-lg">{uploadResult.success ? 'Upload Complete' : 'Upload Failed'}</h3>
                    <p className="text-text-secondary mt-1">{uploadResult.message}</p>

                    {uploadResult.success && (
                      <div className="flex gap-2 mt-4">
                        {uploadResult.features_enabled?.predictive && (
                          <Badge variant="success" className="animate-in fade-in zoom-in">ML Ready</Badge>
                        )}
                        {uploadResult.features_enabled?.nlp && (
                          <Badge variant="success" className="animate-in fade-in zoom-in delay-100">NLP Ready</Badge>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </GlassCard>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
