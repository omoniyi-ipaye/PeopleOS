'use client'

import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '@/lib/api-client'
import {
  AlertTriangle,
  ChevronRight,
  Globe,
  MapPin,
  RotateCcw,
  Search,
  Users,
  ZoomIn,
  ZoomOut,
} from 'lucide-react'

const GEOJSON_URL = 'https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson'
const VIEWBOX_WIDTH = 1000
const VIEWBOX_HEIGHT = 500

type Position = [number, number]
type LinearRing = Position[]
type PolygonCoordinates = LinearRing[]
type MultiPolygonCoordinates = PolygonCoordinates[]

type Geometry =
  | { type: 'Polygon'; coordinates: PolygonCoordinates }
  | { type: 'MultiPolygon'; coordinates: MultiPolygonCoordinates }

type GeoProperties = Record<string, string | number | null | undefined>

interface GeoFeature {
  type: 'Feature'
  id?: string | number
  properties: GeoProperties
  geometry: Geometry | null
}

interface GeoFeatureCollection {
  type: 'FeatureCollection'
  features: GeoFeature[]
}

interface GeoDistribution {
  country: string
  count: number
  percentage: number
}

interface TooltipContent {
  country: string
  count: number
  percentage: number
}

const COUNTRY_NAME_MAP: Record<string, string> = {
  'United States': 'United States of America',
  'United Kingdom': 'United Kingdom',
  'South Korea': 'South Korea',
  'Czech Republic': 'Czechia',
  'Czechia': 'Czechia',
  'UK': 'United Kingdom',
  'USA': 'United States of America',
  'US': 'United States of America',
}

function normalizeCountryName(value: string): string {
  const trimmed = value.trim()
  return (COUNTRY_NAME_MAP[trimmed] || trimmed).toLowerCase()
}

function getFeatureName(feature: GeoFeature): string {
  const properties = feature.properties
  const candidates = [
    properties.ADMIN,
    properties.name,
    properties.NAME,
    properties.NAME_EN,
    properties.SOVEREIGNT,
  ]
  const value = candidates.find((candidate) => typeof candidate === 'string' && candidate.length > 0)
  return typeof value === 'string' ? value : 'Unknown'
}

function project([longitude, latitude]: Position): Position {
  const x = ((longitude + 180) / 360) * VIEWBOX_WIDTH
  const y = ((90 - latitude) / 180) * VIEWBOX_HEIGHT
  return [x, y]
}

function ringToPath(ring: LinearRing): string {
  if (ring.length === 0) return ''
  return ring
    .map((point, index) => {
      const [x, y] = project(point)
      return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ') + ' Z'
}

function geometryToPath(geometry: Geometry | null): string {
  if (!geometry) return ''
  if (geometry.type === 'Polygon') {
    return geometry.coordinates.map(ringToPath).join(' ')
  }
  return geometry.coordinates
    .flatMap((polygon) => polygon.map(ringToPath))
    .join(' ')
}

function getCountryColor(count: number, maxCount: number): string {
  if (count <= 0 || maxCount <= 0) return '#e2e8f0'
  const ratio = Math.min(count / maxCount, 1)
  const intensity = 0.2 + ratio * 0.8
  const saturation = Math.round(40 + intensity * 50)
  const lightness = Math.round(70 - intensity * 40)
  return `hsl(142, ${saturation}%, ${lightness}%)`
}

function getFeatureKey(feature: GeoFeature, index: number): string {
  return String(feature.id ?? `${getFeatureName(feature)}-${index}`)
}

export function WorldMap() {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [distribution, setDistribution] = useState<GeoDistribution[]>([])
  const [features, setFeatures] = useState<GeoFeature[]>([])
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null)
  const [tooltipContent, setTooltipContent] = useState<TooltipContent | null>(null)
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 })
  const [searchQuery, setSearchQuery] = useState('')
  const [zoom, setZoom] = useState(1)

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const [distData, geoResponse] = await Promise.all([
          api.geo.getDistribution(),
          fetch(GEOJSON_URL, { cache: 'force-cache' }),
        ])

        if (!geoResponse.ok) {
          throw new Error(`Map geometry request failed with ${geoResponse.status}`)
        }

        const geoJson = (await geoResponse.json()) as GeoFeatureCollection
        if (!cancelled) {
          if (!Array.isArray(distData) || distData.length === 0) {
            setError('No geographic data available. Ensure your data has a Country column.')
          } else if (!Array.isArray(geoJson.features) || geoJson.features.length === 0) {
            setError('World map geometry is unavailable.')
          } else {
            setDistribution(distData as GeoDistribution[])
            setFeatures(geoJson.features)
          }
        }
      } catch (loadError) {
        console.error('WorldMap: failed to load geographic data', loadError)
        if (!cancelled) {
          setError('Failed to load geographic data.')
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [])

  const dataByCountry = useMemo(() => {
    const lookup = new Map<string, GeoDistribution>()
    for (const item of distribution) {
      lookup.set(normalizeCountryName(item.country), item)
      lookup.set(item.country.toLowerCase(), item)
    }
    return lookup
  }, [distribution])

  const maxCount = useMemo(
    () => Math.max(...distribution.map((item) => item.count), 1),
    [distribution],
  )

  const totalEmployees = useMemo(
    () => distribution.reduce((sum, item) => sum + item.count, 0),
    [distribution],
  )

  const filteredCountries = useMemo(() => {
    const sorted = [...distribution].sort((a, b) => b.count - a.count)
    if (!searchQuery.trim()) return sorted
    const query = searchQuery.trim().toLowerCase()
    return sorted.filter((item) => item.country.toLowerCase().includes(query))
  }, [distribution, searchQuery])

  const paths = useMemo(
    () => features.map((feature) => geometryToPath(feature.geometry)),
    [features],
  )

  const getDistributionForFeature = useCallback(
    (feature: GeoFeature): GeoDistribution | undefined => {
      const featureName = getFeatureName(feature)
      return dataByCountry.get(normalizeCountryName(featureName)) || dataByCountry.get(featureName.toLowerCase())
    },
    [dataByCountry],
  )

  const handleMouseMove = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    setTooltipPosition({ x: event.clientX, y: event.clientY })
  }, [])

  const handleFeatureEnter = useCallback(
    (feature: GeoFeature) => {
      const data = getDistributionForFeature(feature)
      setTooltipContent(
        data
          ? { country: data.country, count: data.count, percentage: data.percentage }
          : { country: getFeatureName(feature), count: 0, percentage: 0 },
      )
    },
    [getDistributionForFeature],
  )

  const handleFeatureClick = useCallback(
    (feature: GeoFeature) => {
      const data = getDistributionForFeature(feature)
      if (!data || data.count <= 0) return
      setSelectedCountry((current) => (current === data.country ? null : data.country))
    },
    [getDistributionForFeature],
  )

  const handleZoomIn = useCallback(() => setZoom((value) => Math.min(value * 1.35, 4)), [])
  const handleZoomOut = useCallback(() => setZoom((value) => Math.max(value / 1.35, 1)), [])
  const handleReset = useCallback(() => {
    setZoom(1)
    setSelectedCountry(null)
  }, [])

  if (loading) {
    return (
      <div className="h-[calc(100vh-180px)] flex items-center justify-center bg-slate-50 dark:bg-slate-900 rounded-xl">
        <div className="text-center">
          <Globe className="w-16 h-16 text-accent animate-pulse mx-auto mb-4" />
          <p className="text-text-secondary font-medium">Loading global workforce map...</p>
          <p className="text-text-muted text-sm mt-1">Fetching employee locations</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="h-[calc(100vh-180px)] flex items-center justify-center bg-slate-50 dark:bg-slate-900 rounded-xl">
        <div className="text-center max-w-md">
          <AlertTriangle className="w-12 h-12 text-warning mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">Unable to Load Map Data</h3>
          <p className="text-text-secondary text-sm mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-accent text-white rounded-lg text-sm hover:bg-accent/90"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div
      className="h-[calc(100vh-180px)] flex bg-slate-50 dark:bg-slate-900 rounded-xl overflow-hidden border border-border dark:border-border-dark"
      onMouseMove={handleMouseMove}
    >
      <div className="flex-1 relative min-w-0">
        <div className="absolute top-4 left-4 z-10 bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm px-4 py-3 rounded-xl shadow-lg border border-border dark:border-border-dark">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-accent/10 rounded-lg">
              <Globe className="w-5 h-5 text-accent" />
            </div>
            <div>
              <h2 className="text-lg font-bold">Global Workforce</h2>
              <p className="text-xs text-text-muted">
                {totalEmployees.toLocaleString()} employees across {distribution.length} countries
              </p>
            </div>
          </div>
        </div>

        <div className="absolute top-4 right-4 z-10 flex flex-col gap-2">
          <button
            onClick={handleZoomIn}
            disabled={zoom >= 4}
            className="p-2 bg-white/95 dark:bg-slate-800/95 rounded-lg shadow-lg border border-border hover:bg-slate-100 dark:hover:bg-slate-700 disabled:opacity-50"
            title="Zoom In"
          >
            <ZoomIn className="w-5 h-5" />
          </button>
          <button
            onClick={handleZoomOut}
            disabled={zoom <= 1}
            className="p-2 bg-white/95 dark:bg-slate-800/95 rounded-lg shadow-lg border border-border hover:bg-slate-100 dark:hover:bg-slate-700 disabled:opacity-50"
            title="Zoom Out"
          >
            <ZoomOut className="w-5 h-5" />
          </button>
          <button
            onClick={handleReset}
            className="p-2 bg-white/95 dark:bg-slate-800/95 rounded-lg shadow-lg border border-border hover:bg-slate-100 dark:hover:bg-slate-700"
            title="Reset View"
          >
            <RotateCcw className="w-5 h-5" />
          </button>
        </div>

        <div className="absolute bottom-4 left-4 z-10 bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm px-4 py-3 rounded-xl shadow-lg border border-border dark:border-border-dark">
          <p className="text-xs font-semibold text-text-muted mb-2">Employee Density</p>
          <div className="flex items-center gap-2 text-[11px] text-text-muted">
            <span>Low</span>
            {[0.15, 0.35, 0.55, 0.75, 1].map((ratio) => (
              <span
                key={ratio}
                className="w-6 h-3 rounded-sm"
                style={{ backgroundColor: getCountryColor(maxCount * ratio, maxCount) }}
              />
            ))}
            <span>High</span>
          </div>
        </div>

        <div className="w-full h-full overflow-hidden bg-slate-100 dark:bg-slate-950">
          <svg
            viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`}
            className="w-full h-full"
            role="img"
            aria-label="Global workforce distribution map"
          >
            <g
              transform={`translate(${VIEWBOX_WIDTH / 2} ${VIEWBOX_HEIGHT / 2}) scale(${zoom}) translate(${-VIEWBOX_WIDTH / 2} ${-VIEWBOX_HEIGHT / 2})`}
              className="transition-transform duration-200"
            >
              {features.map((feature, index) => {
                const data = getDistributionForFeature(feature)
                const isSelected = Boolean(data && selectedCountry === data.country)
                const count = data?.count ?? 0
                return (
                  <path
                    key={getFeatureKey(feature, index)}
                    d={paths[index]}
                    fill={getCountryColor(count, maxCount)}
                    stroke={isSelected ? '#0f172a' : '#ffffff'}
                    strokeWidth={isSelected ? 1.6 : 0.45}
                    vectorEffect="non-scaling-stroke"
                    className="cursor-pointer transition-opacity hover:opacity-80"
                    onMouseEnter={() => handleFeatureEnter(feature)}
                    onMouseLeave={() => setTooltipContent(null)}
                    onClick={() => handleFeatureClick(feature)}
                  />
                )
              })}
            </g>
          </svg>
        </div>
      </div>

      <aside className="w-80 shrink-0 bg-white dark:bg-slate-900 border-l border-border dark:border-border-dark flex flex-col">
        <div className="p-4 border-b border-border dark:border-border-dark">
          <div className="flex items-center gap-2 mb-3">
            <Users className="w-4 h-4 text-accent" />
            <h3 className="font-semibold">Countries</h3>
          </div>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
            <input
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="Search countries"
              className="w-full pl-9 pr-3 py-2 rounded-lg border border-border dark:border-border-dark bg-slate-50 dark:bg-slate-800 text-sm outline-none focus:ring-2 focus:ring-accent/30"
            />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {filteredCountries.map((item) => {
            const selected = selectedCountry === item.country
            return (
              <button
                key={item.country}
                onClick={() => setSelectedCountry(selected ? null : item.country)}
                className={`w-full text-left p-3 rounded-lg mb-1 transition-colors ${
                  selected ? 'bg-accent/10' : 'hover:bg-slate-100 dark:hover:bg-slate-800'
                }`}
              >
                <div className="flex items-center gap-3">
                  <MapPin className={`w-4 h-4 shrink-0 ${selected ? 'text-accent' : 'text-text-muted'}`} />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-medium text-sm truncate">{item.country}</span>
                      <span className="text-sm font-semibold">{item.count.toLocaleString()}</span>
                    </div>
                    <div className="mt-1.5 h-1.5 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.max(item.percentage, 1)}%`,
                          backgroundColor: getCountryColor(item.count, maxCount),
                        }}
                      />
                    </div>
                    <p className="text-[11px] text-text-muted mt-1">{item.percentage.toFixed(1)}% of workforce</p>
                  </div>
                  <ChevronRight className="w-4 h-4 text-text-muted shrink-0" />
                </div>
              </button>
            )
          })}
        </div>
      </aside>

      {tooltipContent && (
        <div
          className="fixed z-50 pointer-events-none bg-slate-950 text-white px-3 py-2 rounded-lg shadow-xl text-xs"
          style={{ left: tooltipPosition.x + 14, top: tooltipPosition.y + 14 }}
        >
          <p className="font-semibold">{tooltipContent.country}</p>
          <p className="text-slate-300">
            {tooltipContent.count.toLocaleString()} employees · {tooltipContent.percentage.toFixed(1)}%
          </p>
        </div>
      )}
    </div>
  )
}
