import type { LLMStatus } from '@/types/api'

/**
 * API client for PeopleOS FastAPI backend
 */

// The development rewrite is convenient for short requests, but its proxy
// lifetime is shorter than a cold local-model inference. Talk directly to the
// loopback API during local development so the browser can observe the full
// bounded response. Packaged/production builds retain same-origin routing.
const API_BASE = process.env.NEXT_PUBLIC_API_URL || (process.env.NODE_ENV === 'development' ? 'http://127.0.0.1:8000' : '')
const DEFAULT_API_TIMEOUT_MS = 15_000
const LONG_RUNNING_API_TIMEOUT_MS = 180_000

async function fetchAPI<T>(endpoint: string, options?: RequestInit, timeoutMs = DEFAULT_API_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs)
  const externalSignal = options?.signal
  const abortFromExternal = () => controller.abort()
  if (externalSignal) {
    if (externalSignal.aborted) controller.abort()
    else externalSignal.addEventListener('abort', abortFromExternal, { once: true })
  }
  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
      signal: controller.signal,
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Request failed' }))
      throw new Error(error.detail || `API error: ${response.status}`)
    }
    return response.json()
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('PeopleOS did not respond in time. Check the local app and retry.')
    }
    throw error
  } finally {
    window.clearTimeout(timeout)
    externalSignal?.removeEventListener('abort', abortFromExternal)
  }
}

export const api = {
  // Local owner app lock
  appLock: {
    getStatus: () => fetchAPI<{ enabled: boolean; locked: boolean }>('/api/app-lock/status'),
    setup: (pin: string) => fetchAPI<{ enabled: boolean; locked: boolean }>('/api/app-lock/setup', { method: 'POST', body: JSON.stringify({ pin }) }),
    lock: () => fetchAPI<{ enabled: boolean; locked: boolean }>('/api/app-lock/lock', { method: 'POST' }),
    unlock: (pin: string) => fetchAPI<{ enabled: boolean; locked: boolean }>('/api/app-lock/unlock', { method: 'POST', body: JSON.stringify({ pin }) }),
    change: (currentPin: string, newPin: string) => fetchAPI<{ enabled: boolean; locked: boolean }>('/api/app-lock/change', { method: 'POST', body: JSON.stringify({ current_pin: currentPin, new_pin: newPin }) }),
    disable: (pin: string) => fetchAPI<{ enabled: boolean; locked: boolean }>('/api/app-lock/disable', { method: 'POST', body: JSON.stringify({ pin }) }),
  },

  // Upload endpoints
  upload: {
    uploadFile: async (file: File, pay?: { annual: boolean; currency: string }, mapping?: Record<string, string | null>) => {
      const formData = new FormData()
      formData.append('file', file)
      if (pay?.annual) formData.append('salary_basis', 'annual')
      if (pay?.currency.trim()) formData.append('salary_currency', pay.currency.trim().toUpperCase())
      if (mapping) formData.append('column_mapping', JSON.stringify(mapping))

      const response = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
        throw new Error(error.detail || 'Upload failed')
      }

      return response.json()
    },
    previewFile: async (file: File, options?: { annual?: boolean; currency?: string; mapping?: Record<string, string | null>; useLLM?: boolean }) => {
      const formData = new FormData()
      formData.append('file', file)
      if (options?.annual) formData.append('salary_basis', 'annual')
      if (options?.currency?.trim()) formData.append('salary_currency', options.currency.trim().toUpperCase())
      if (options?.mapping) formData.append('column_mapping', JSON.stringify(options.mapping))
      if (options?.useLLM) formData.append('use_llm', 'true')

      const response = await fetch(`${API_BASE}/api/upload/preview`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Import review failed' }))
        throw new Error(error.detail || 'Import review failed')
      }

      return response.json()
    },
    getStatus: () => fetchAPI('/api/upload/status'),
    loadSample: () => fetchAPI('/api/upload/load-sample', { method: 'POST' }),
    downloadTemplate: () => {
      const link = document.createElement('a')
      link.href = `${API_BASE}/api/upload/template`
      link.download = 'peopleos-template.csv'
      link.rel = 'noreferrer'
      document.body.appendChild(link)
      link.click()
      link.remove()
    },
    reset: () => fetchAPI('/api/upload/reset', { method: 'POST' }),
  },

  // Analytics endpoints
  analytics: {
    getSummary: () => fetchAPI('/api/analytics/summary'),
    getDepartments: () => fetchAPI('/api/analytics/departments'),
    getDistributions: () => fetchAPI('/api/analytics/distributions'),
    getCorrelations: (limit = 10) =>
      fetchAPI(`/api/analytics/correlations?limit=${limit}`),
    getHighRiskDepartments: () => fetchAPI('/api/analytics/high-risk-departments'),
    getClusters: (nClusters = 3) =>
      fetchAPI(`/api/analytics/clusters?n_clusters=${nClusters}`),
    getForecast: (metric = 'headcount', periods = 12) =>
      fetchAPI(`/api/analytics/forecast?metric=${metric}&periods=${periods}`),
    compareGroups: (groupBy: string, metric: string) =>
      fetchAPI(`/api/analytics/compare-groups?group_by=${encodeURIComponent(groupBy)}&metric=${encodeURIComponent(metric)}`),
    getClusterMembers: (clusterId: number, nClusters: number = 4) =>
      fetchAPI(`/api/analytics/cluster-members/${clusterId}?n_clusters=${nClusters}`),
  },

  // Predictions endpoints
  predictions: {
    getModelMetrics: () => fetchAPI('/api/predictions/model-metrics'),
    getFeatureImportance: (limit = 10) =>
      fetchAPI(`/api/predictions/feature-importance?limit=${limit}`),
    getRisk: (category?: string, limit = 100) => {
      const params = new URLSearchParams()
      if (category) params.set('risk_category', category)
      params.set('limit', limit.toString())
      return fetchAPI(`/api/predictions/risk?${params}`)
    },
    getEmployeeDetail: (employeeId: string) =>
      fetchAPI(`/api/predictions/employee/${employeeId}`),
    getHighRisk: (limit = 20) =>
      fetchAPI(`/api/predictions/high-risk-employees?limit=${limit}`),
  },

  // Compensation endpoints
  compensation: {
    getSummary: () => fetchAPI('/api/compensation/summary'),
    getEquity: () => fetchAPI('/api/compensation/equity'),
    getOutliers: () => fetchAPI('/api/compensation/outliers'),
    getCompaRatio: () => fetchAPI('/api/compensation/compa-ratio'),
    getGenderPayGap: () => fetchAPI('/api/compensation/gender-pay-gap'),
    getByTenure: () => fetchAPI('/api/compensation/by-tenure'),
    getAnalysis: () => fetchAPI('/api/compensation/analysis'),
  },

  // Succession endpoints
  succession: {
    getReadiness: () => fetchAPI('/api/succession/readiness'),
    getHighPotentials: () => fetchAPI('/api/succession/high-potentials'),
    getPipeline: () => fetchAPI('/api/succession/pipeline'),
    getBenchStrength: () => fetchAPI('/api/succession/bench-strength'),
    getGaps: () => fetchAPI('/api/succession/gaps'),
    getRecommendations: () => fetchAPI('/api/succession/recommendations'),
    get9Box: () => fetchAPI('/api/succession/9box'),
    get9BoxSummary: () => fetchAPI('/api/succession/9box/summary'),
  },

  // Team endpoints
  team: {
    getHealth: () => fetchAPI('/api/team/health'),
    getDiversity: () => fetchAPI('/api/team/diversity'),
    getAnalysis: () => fetchAPI('/api/team/analysis'),
    getFilters: () => fetchAPI('/api/team/filters'),
    getComprehensive: (filters?: {
      departments?: string[];
      locations?: string[];
      countries?: string[];
      genders?: string[];
      age_groups?: string[];
      job_levels?: number[];
      tenure_ranges?: string[];
      min_tenure?: number;
      max_tenure?: number;
    }) => {
      const params = new URLSearchParams();
      if (filters?.departments?.length) params.append('departments', filters.departments.join(','));
      if (filters?.locations?.length) params.append('locations', filters.locations.join(','));
      if (filters?.countries?.length) params.append('countries', filters.countries.join(','));
      if (filters?.genders?.length) params.append('genders', filters.genders.join(','));
      if (filters?.age_groups?.length) params.append('age_groups', filters.age_groups.join(','));
      if (filters?.job_levels?.length) params.append('job_levels', filters.job_levels.join(','));
      if (filters?.tenure_ranges?.length) params.append('tenure_ranges', filters.tenure_ranges.join(','));
      if (filters?.min_tenure !== undefined) params.append('min_tenure', String(filters.min_tenure));
      if (filters?.max_tenure !== undefined) params.append('max_tenure', String(filters.max_tenure));
      const queryString = params.toString();
      return fetchAPI(`/api/team/comprehensive${queryString ? `?${queryString}` : ''}`);
    },
  },

  // Fairness endpoints
  fairness: {
    getFourFifths: () => fetchAPI('/api/fairness/four-fifths'),
    getAnalysis: () => fetchAPI('/api/fairness/analysis'),
    getDemographicParity: () => fetchAPI('/api/fairness/demographic-parity'),
  },

  // Search endpoints
  search: {
    search: (query: string, topK = 10) =>
      fetchAPI(`/api/search?query=${encodeURIComponent(query)}&top_k=${topK}`, {
        method: 'POST',
      }),
    prepare: () => fetchAPI('/api/search/prepare', { method: 'POST' }),
    getStatus: () => fetchAPI('/api/search/status'),
  },

  // Advisor endpoints
  advisor: {
    getStatus: () => fetchAPI('/api/advisor/status'),
    getSummary: () => fetchAPI('/api/advisor/summary', undefined, LONG_RUNNING_API_TIMEOUT_MS),
    ask: (question: string) =>
      fetchAPI(`/api/advisor/ask?question=${encodeURIComponent(question)}`, {
        method: 'POST',
      }, LONG_RUNNING_API_TIMEOUT_MS),
  },

  // Governed People Intelligence endpoints
  intelligence: {
    investigate: (request: { question: string; dataset_version?: string; agentic?: boolean }, signal?: AbortSignal) =>
      fetchAPI('/api/intelligence/investigate', {
        method: 'POST',
        signal,
        body: JSON.stringify(request),
      }, LONG_RUNNING_API_TIMEOUT_MS),
  },

  // Optional local AI setup endpoints
  llm: {
    getStatus: () => fetchAPI<LLMStatus>('/api/llm/status'),
    configure: (request: { provider: 'none' | 'ollama'; enabled: boolean; model?: string }) =>
      fetchAPI<LLMStatus>('/api/llm/configure', {
        method: 'POST',
        body: JSON.stringify(request),
      }),
    setup: (model?: string) =>
      fetchAPI<LLMStatus>('/api/llm/setup', {
        method: 'POST',
        body: JSON.stringify(model ? { model } : {}),
      }),
    test: () => fetchAPI<{ passed: boolean; model: string; response: string; elapsed_ms: number }>('/api/llm/test', { method: 'POST' }, LONG_RUNNING_API_TIMEOUT_MS),
  },

  // NLP endpoints
  nlp: {
    getAnalysis: () => fetchAPI('/api/nlp/analysis', undefined, LONG_RUNNING_API_TIMEOUT_MS),
  },

  // Survival Analysis endpoints
  survival: {
    getAnalysis: () => fetchAPI('/api/survival/analysis'),
    getKaplanMeier: (segmentBy?: string) => {
      const params = segmentBy ? `?segment_by=${segmentBy}` : ''
      return fetchAPI(`/api/survival/kaplan-meier${params}`)
    },
    getCoxModel: () => fetchAPI('/api/survival/cox-model'),
    getHazardOverTime: () => fetchAPI('/api/survival/hazard-over-time'),
    getCohortInsights: (filters?: Record<string, string | number>) => {
      const params = new URLSearchParams()
      if (filters) {
        Object.entries(filters).forEach(([key, value]) => {
          params.set(key, String(value))
        })
      }
      return fetchAPI(`/api/survival/cohort-insights?${params}`)
    },
    getAtRisk: (limit = 20) => fetchAPI(`/api/survival/at-risk?limit=${limit}`),
    getEmployeeSurvival: (employeeId: string) =>
      fetchAPI(`/api/survival/employee/${employeeId}`),
  },

  // Quality of Hire endpoints
  qualityOfHire: {
    getAnalysis: () => fetchAPI('/api/quality-of-hire/analysis'),
    getSourceEffectiveness: () => fetchAPI('/api/quality-of-hire/source-effectiveness'),
    getCorrelations: (outcome = 'LastRating') =>
      fetchAPI(`/api/quality-of-hire/correlations?outcome=${outcome}`),
    getInsights: () => fetchAPI('/api/quality-of-hire/insights'),
    getCohortAnalysis: (cohortBy = 'HireSource', minTenureMonths = 6) =>
      fetchAPI(`/api/quality-of-hire/cohort-analysis?cohort_by=${cohortBy}&min_tenure_months=${minTenureMonths}`),
    getNewHireRisks: (months = 6) =>
      fetchAPI(`/api/quality-of-hire/new-hire-risks?months=${months}`),
    getBestPredictors: (limit = 5) =>
      fetchAPI(`/api/quality-of-hire/best-predictors?limit=${limit}`),
  },

  // Structural Analysis endpoints
  structural: {
    getAnalysis: () => fetchAPI('/api/structural/analysis'),
    getStagnation: (filters?: { dept?: string; category?: string; minTenure?: number; limit?: number }) => {
      const params = new URLSearchParams()
      if (filters?.dept) params.set('dept', filters.dept)
      if (filters?.category) params.set('category', filters.category)
      if (filters?.minTenure) params.set('min_tenure', String(filters.minTenure))
      if (filters?.limit) params.set('limit', String(filters.limit))
      return fetchAPI(`/api/structural/stagnation?${params}`)
    },
    getStagnationHotspots: () => fetchAPI('/api/structural/stagnation/hotspots'),
    getSpanOfControl: (filters?: { category?: string; dept?: string; minReports?: number }) => {
      const params = new URLSearchParams()
      if (filters?.category) params.set('category', filters.category)
      if (filters?.dept) params.set('dept', filters.dept)
      if (filters?.minReports) params.set('min_reports', String(filters.minReports))
      return fetchAPI(`/api/structural/span-of-control?${params}`)
    },
    getSpanAnalysis: () => fetchAPI('/api/structural/span-of-control/analysis'),
    getPromotionEquity: () => fetchAPI('/api/structural/promotion-equity'),
    getPromotionBottlenecks: () => fetchAPI('/api/structural/promotion-bottlenecks'),
    getEmployeeStagnation: (employeeId: string) =>
      fetchAPI(`/api/structural/employee/${employeeId}/stagnation`),
  },

  // Sentiment Analysis endpoints
  sentiment: {
    getAnalysis: () => fetchAPI('/api/sentiment/analysis'),
    getENPS: (groupBy?: string, dateFrom?: string, dateTo?: string) => {
      const params = new URLSearchParams()
      if (groupBy) params.set('group_by', groupBy)
      if (dateFrom) params.set('date_from', dateFrom)
      if (dateTo) params.set('date_to', dateTo)
      return fetchAPI(`/api/sentiment/enps?${params}`)
    },
    getENPSTrends: (period = 'month') =>
      fetchAPI(`/api/sentiment/enps/trends?period=${period}`),
    getENPSDrivers: () => fetchAPI('/api/sentiment/enps/drivers'),
    getOnboarding: (employeeId?: string) => {
      const params = employeeId ? `?employee_id=${employeeId}` : ''
      return fetchAPI(`/api/sentiment/onboarding${params}`)
    },
    getOnboardingHealth: () => fetchAPI('/api/sentiment/onboarding/health'),
    getEarlyWarnings: () => fetchAPI('/api/sentiment/early-warnings'),
    getTemplates: () => fetchAPI('/api/sentiment/templates'),
    uploadENPS: async (file: File) => {
      const formData = new FormData()
      formData.append('file', file)
      const response = await fetch(`${API_BASE}/api/sentiment/upload/enps`, {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
        throw new Error(error.detail || 'Upload failed')
      }
      return response.json()
    },
    uploadOnboarding: async (file: File) => {
      const formData = new FormData()
      formData.append('file', file)
      const response = await fetch(`${API_BASE}/api/sentiment/upload/onboarding`, {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
        throw new Error(error.detail || 'Upload failed')
      }
      return response.json()
    },
  },

  // Employee Experience endpoints
  experience: {
    getAnalysis: () => fetchAPI('/api/experience/analysis'),
    getIndex: (groupBy?: string) => {
      const params = groupBy ? `?group_by=${groupBy}` : ''
      return fetchAPI(`/api/experience/index${params}`)
    },
    getEmployeeExperience: (employeeId: string) =>
      fetchAPI(`/api/experience/index/employee/${employeeId}`),
    getSegments: () => fetchAPI('/api/experience/segments'),
    getDrivers: () => fetchAPI('/api/experience/drivers'),
    getAtRisk: (threshold?: number, limit = 20) => {
      const params = new URLSearchParams()
      if (threshold) params.set('threshold', String(threshold))
      params.set('limit', String(limit))
      return fetchAPI(`/api/experience/at-risk?${params}`)
    },
    getLifecycle: () => fetchAPI('/api/experience/lifecycle'),
    getManagerImpact: () => fetchAPI('/api/experience/manager-impact'),
    getSignals: () => fetchAPI('/api/experience/signals'),
    getTrends: (period = 'month') =>
      fetchAPI(`/api/experience/trends?period=${period}`),
  },

  // Scenario Planning endpoints
  scenario: {
    simulateCompensation: (request: {
      adjustment_type: 'percentage' | 'absolute' | 'market_adjustment'
      target: { scope: string; department?: string; employee_ids?: string[] }
      adjustment_value: number
      time_horizon_months?: number
    }) =>
      fetchAPI('/api/scenario/simulate/compensation', {
        method: 'POST',
        body: JSON.stringify(request),
      }),
    simulateHeadcount: (request: {
      change_type: 'reduction' | 'expansion'
      target: { scope: string; department?: string }
      change_count?: number
      change_percentage?: number
      selection_criteria?: 'performance' | 'tenure' | 'cost'
    }) =>
      fetchAPI('/api/scenario/simulate/headcount', {
        method: 'POST',
        body: JSON.stringify(request),
      }),
    simulateIntervention: (request: {
      intervention_type: 'retention_bonus' | 'career_path' | 'manager_change'
      target_employees: string | string[]
      intervention_params?: Record<string, number>
    }) =>
      fetchAPI('/api/scenario/simulate/intervention', {
        method: 'POST',
        body: JSON.stringify(request),
      }),
    getTemplates: () => fetchAPI('/api/scenario/templates'),
    compareScenarios: (scenarioIds: string[]) =>
      fetchAPI('/api/scenario/compare', {
        method: 'POST',
        body: JSON.stringify(scenarioIds),
      }),
    saveScenario: (request: { scenario_id: string; scenario_name: string }) =>
      fetchAPI('/api/scenario/save', {
        method: 'POST',
        body: JSON.stringify(request),
      }),
    drilldownScenarios: (request: { scenario_ids: string[]; question: string }) =>
      fetchAPI('/api/scenario/drilldown', {
        method: 'POST',
        body: JSON.stringify(request),
      }, LONG_RUNNING_API_TIMEOUT_MS),
    getScenario: (scenarioId: string) =>
      fetchAPI(`/api/scenario/${scenarioId}`),
    deleteScenario: (scenarioId: string) =>
      fetchAPI(`/api/scenario/${scenarioId}`, { method: 'DELETE' }),
    getRecentScenarios: (limit = 10) =>
      fetchAPI(`/api/scenario/history/recent?limit=${limit}`),
    analyzeSensitivity: (request: {
      scenario_type: 'compensation' | 'headcount' | 'intervention'
      base_request: Record<string, unknown>
      variable: string
      range_values: number[]
    }) =>
      fetchAPI('/api/scenario/sensitivity', {
        method: 'POST',
        body: JSON.stringify(request),
      }),
  },

  // Sessions endpoints
  sessions: {
    list: () => fetchAPI('/api/sessions'),
    save: (name: string) =>
      fetchAPI('/api/sessions', {
        method: 'POST',
        body: JSON.stringify({ name }),
      }),
    load: (filepath: string) =>
      fetchAPI(`/api/sessions/load?filepath=${encodeURIComponent(filepath)}`, {
        method: 'POST',
      }),
    delete: (filepath: string) =>
      fetchAPI(`/api/sessions?filepath=${encodeURIComponent(filepath)}`, {
        method: 'DELETE',
      }),
  },

  // Model Lab endpoints
  modelLab: {
    getValidation: (daysBack = 90) => fetchAPI(`/api/model-lab/validation?days_back=${daysBack}`),
    getSensitivity: () => fetchAPI('/api/model-lab/sensitivity'),
    getRefinementPlan: () => fetchAPI('/api/model-lab/refinement-plan'),
    optimize: () => fetchAPI('/api/model-lab/optimize', { method: 'POST' }),
  },

  // Geographic Distribution endpoints
  geo: {
    getDistribution: () => fetchAPI<Array<{ country: string; count: number; percentage: number }>>('/api/geo/distribution'),
    getSummary: () => fetchAPI<{ total_employees: number; countries_represented: number; remote_workers: number; remote_percentage: number }>('/api/geo/summary'),
  },

  // Causal Inference endpoints
  causal: {
    getImpact: (treatment: string, outcome = 'Attrition') =>
      fetchAPI(`/api/causal/impact?treatment=${encodeURIComponent(treatment)}&outcome=${encodeURIComponent(outcome)}`),
    getRecommendations: () => fetchAPI('/api/causal/recommendations'),
  },

  // Network Analysis endpoints
  network: {
    getInfluencers: (limit = 10) => fetchAPI(`/api/network/influencers?limit=${limit}`),
    getIsolated: (limit = 10) => fetchAPI(`/api/network/isolated?limit=${limit}`),
    getSummary: () => fetchAPI('/api/network/summary'),
  },

  // General status
  getStatus: () => fetchAPI('/api/status'),
  getHealth: () => fetchAPI('/api/health'),
}
