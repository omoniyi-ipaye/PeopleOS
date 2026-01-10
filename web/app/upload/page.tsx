'use client'

import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api-client'
import { UploadStatus, UploadResponse } from '@/types/api'
import {
  Upload,
  FileText,
  CheckCircle,
  AlertTriangle,
  Database,
  Trash2,
  Download,
} from 'lucide-react'

export default function UploadPage() {
  const queryClient = useQueryClient()
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null)

  const { data: status, refetch: refetchStatus } = useQuery<UploadStatus>({
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
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Page Title */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary dark:text-text-dark-primary">Upload Data</h1>
        <p className="text-text-secondary dark:text-text-dark-secondary mt-1">
          Import your HR data to start analyzing
        </p>
      </div>

      {/* Current Status */}
      {status?.has_data && (
        <Card className="bg-success/10 border-success/20">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Database className="w-5 h-5 text-success" />
              <div>
                <h3 className="font-medium text-success">Data Loaded</h3>
                <p className="text-sm text-text-secondary dark:text-text-dark-secondary">
                  {status.employee_count.toLocaleString()} employees
                </p>
              </div>
            </div>
            <Button
              variant="danger"
              size="sm"
              onClick={() => resetMutation.mutate()}
              isLoading={resetMutation.isPending}
            >
              <Trash2 className="w-4 h-4 mr-2" />
              Reset Data
            </Button>
          </div>
        </Card>
      )}

      {/* Upload Area */}
      <Card padding="none">
        <div
          {...getRootProps()}
          className={`
            p-12 border-2 border-dashed rounded-xl cursor-pointer transition-colors
            ${isDragActive
              ? 'border-accent bg-accent/5'
              : 'border-border dark:border-border-dark hover:border-accent/50 hover:bg-surface-hover dark:hover:bg-surface-dark-hover bg-surface dark:bg-surface-dark'
            }
          `}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center text-center">
            <Upload
              className={`w-12 h-12 mb-4 ${isDragActive ? 'text-accent' : 'text-text-muted'
                }`}
            />
            <h3 className="text-lg font-medium mb-2 text-text-primary dark:text-text-dark-primary">
              {isDragActive ? 'Drop file here' : 'Drag & drop your file'}
            </h3>
            <p className="text-text-secondary dark:text-text-dark-secondary mb-4">
              or click to browse. Supports CSV and JSON files.
            </p>
            {uploadMutation.isPending && (
              <Badge variant="info">Uploading...</Badge>
            )}
          </div>
        </div>
      </Card>

      {/* Or Load Sample Data */}
      <div className="flex items-center gap-4">
        <div className="flex-1 h-px bg-border dark:bg-border-dark" />
        <span className="text-text-muted dark:text-text-dark-muted text-sm">or</span>
        <div className="flex-1 h-px bg-border dark:bg-border-dark" />
      </div>

      <Card>
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-medium text-text-primary dark:text-text-dark-primary">Load Sample Data</h3>
            <p className="text-sm text-text-secondary dark:text-text-dark-secondary mt-1">
              Try PeopleOS with 1,000+ sample employees
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              onClick={() => api.upload.downloadTemplate()}
            >
              <Download className="w-4 h-4 mr-2" />
              Download Template
            </Button>
            <Button
              onClick={() => loadSampleMutation.mutate()}
              isLoading={loadSampleMutation.isPending}
              disabled={status?.has_data}
            >
              <FileText className="w-4 h-4 mr-2" />
              Load Sample
            </Button>
          </div>
        </div>
      </Card>

      {/* Upload Result */}
      {uploadResult && (
        <Card
          className={
            uploadResult.success
              ? 'bg-success/10 border-success/20'
              : 'bg-danger/10 border-danger/20'
          }
        >
          <div className="flex items-start gap-3">
            {uploadResult.success ? (
              <CheckCircle className="w-5 h-5 text-success mt-0.5" />
            ) : (
              <AlertTriangle className="w-5 h-5 text-danger mt-0.5" />
            )}
            <div className="flex-1">
              <h3
                className={`font-medium ${uploadResult.success ? 'text-success' : 'text-danger'
                  }`}
              >
                {uploadResult.message}
              </h3>
              {uploadResult.success && (
                <div className="mt-3 space-y-2">
                  <div className="flex flex-wrap gap-2">
                    {uploadResult.features_enabled?.predictive && (
                      <Badge variant="success" size="sm">
                        ML Predictions
                      </Badge>
                    )}
                    {uploadResult.features_enabled?.nlp && (
                      <Badge variant="success" size="sm">
                        NLP Analysis
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm text-text-secondary dark:text-text-dark-secondary">
                    Columns: {uploadResult.columns?.slice(0, 5).join(', ')}
                    {(uploadResult.columns?.length ?? 0) > 5 &&
                      ` +${(uploadResult.columns?.length ?? 0) - 5} more`}
                  </p>
                </div>
              )}
            </div>
          </div>
        </Card>
      )}

      {/* Requirements */}
      <Card title="Data Requirements">
        <div className="space-y-4">
          <div>
            <h4 className="text-sm font-medium text-text-secondary dark:text-text-dark-secondary mb-2">
              Required Columns
            </h4>
            <div className="flex flex-wrap gap-2">
              {[
                'EmployeeID',
                'Dept',
                'Tenure',
                'Salary',
                'LastRating',
                'Age',
              ].map((col) => (
                <Badge key={col} variant="default" size="sm">
                  {col}
                </Badge>
              ))}
            </div>
          </div>

          <div>
            <h4 className="text-sm font-medium text-text-secondary dark:text-text-dark-secondary mb-2">
              Optional (Enable Features)
            </h4>
            <div className="flex flex-wrap gap-2">
              <Badge variant="info" size="sm">
                Attrition → ML Predictions
              </Badge>
              <Badge variant="info" size="sm">
                PerformanceText → NLP & Search
              </Badge>
              <Badge variant="info" size="sm">
                Gender → Pay Equity Analysis
              </Badge>
            </div>
          </div>

          <p className="text-sm text-text-muted dark:text-text-dark-muted">
            Column names are matched automatically. Minimum 50 rows required.
          </p>
        </div>
      </Card>
    </div>
  )
}
