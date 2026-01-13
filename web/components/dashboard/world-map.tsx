'use client'

// Extend window for debug logging
declare global {
    interface Window {
        _geoLogged?: boolean
    }
}

import React, { useState, useEffect, useMemo, useCallback } from 'react'
import {
    ComposableMap,
    Geographies,
    Geography,
    ZoomableGroup
} from 'react-simple-maps'
import { api } from '@/lib/api-client'
import { cn } from '@/lib/utils'
import {
    ZoomIn,
    ZoomOut,
    RotateCcw,
    Globe,
    Users,
    MapPin,
    ChevronRight,
    AlertTriangle,
    Search,
} from 'lucide-react'

// World map topology - Natural Earth data via world-atlas
// Properties include 'name' for country names
const GEO_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-110m.json"

// Map from our data country names to TopoJSON names (Natural Earth naming)
// The world-atlas uses Natural Earth country names
const COUNTRY_NAME_MAP: Record<string, string> = {
    // Our data name -> TopoJSON/Natural Earth name
    'United States': 'United States of America',
    'United Kingdom': 'United Kingdom',
    'Singapore': 'Singapore',
    'Germany': 'Germany',
    'India': 'India',
    'Australia': 'Australia',
    'Canada': 'Canada',
    'Brazil': 'Brazil',
    'Japan': 'Japan',
    'France': 'France',
    'Netherlands': 'Netherlands',
    'China': 'China',
    'Mexico': 'Mexico',
    'Spain': 'Spain',
    'Italy': 'Italy',
    'South Korea': 'South Korea',
    'Ireland': 'Ireland',
    'Switzerland': 'Switzerland',
    'Sweden': 'Sweden',
    'Norway': 'Norway',
    'Denmark': 'Denmark',
    'Poland': 'Poland',
    'Belgium': 'Belgium',
    'Austria': 'Austria',
    'New Zealand': 'New Zealand',
    'South Africa': 'South Africa',
    'United Arab Emirates': 'United Arab Emirates',
    'Israel': 'Israel',
    'Philippines': 'Philippines',
    'Indonesia': 'Indonesia',
    'Thailand': 'Thailand',
    'Vietnam': 'Vietnam',
    'Malaysia': 'Malaysia',
    'Portugal': 'Portugal',
    'Czech Republic': 'Czechia',
    'Czechia': 'Czechia',
    'Romania': 'Romania',
    'Argentina': 'Argentina',
    'Chile': 'Chile',
    'Colombia': 'Colombia',
    'Peru': 'Peru',
    'Russia': 'Russia',
    'Ukraine': 'Ukraine',
    'Nigeria': 'Nigeria',
    'Kenya': 'Kenya',
    'Egypt': 'Egypt',
    'Morocco': 'Morocco',
    'Saudi Arabia': 'Saudi Arabia',
    'Qatar': 'Qatar',
    'Turkey': 'Turkey',
    'Greece': 'Greece',
    'Hungary': 'Hungary',
    'Finland': 'Finland',
    'Estonia': 'Estonia',
    'Latvia': 'Latvia',
    'Lithuania': 'Lithuania',
    'UK': 'United Kingdom',
    'USA': 'United States of America',
    'US': 'United States of America',
}

// Reverse mapping: TopoJSON name -> our data name
const TOPO_TO_DATA_NAME: Record<string, string> = {}
Object.entries(COUNTRY_NAME_MAP).forEach(([dataName, topoName]) => {
    if (!TOPO_TO_DATA_NAME[topoName]) {
        TOPO_TO_DATA_NAME[topoName] = dataName
    }
})

// ISO3 code mapping for fallback matching
const COUNTRY_TO_ISO3: Record<string, string> = {
    'United States': 'USA',
    'United Kingdom': 'GBR',
    'Singapore': 'SGP',
    'Germany': 'DEU',
    'India': 'IND',
    'Australia': 'AUS',
    'Canada': 'CAN',
    'Brazil': 'BRA',
    'Japan': 'JPN',
    'France': 'FRA',
    'Netherlands': 'NLD',
    'China': 'CHN',
    'Mexico': 'MEX',
    'Spain': 'ESP',
    'Italy': 'ITA',
    'South Korea': 'KOR',
    'Ireland': 'IRL',
    'Switzerland': 'CHE',
    'Sweden': 'SWE',
    'Norway': 'NOR',
}

const ISO3_TO_COUNTRY: Record<string, string> = Object.fromEntries(
    Object.entries(COUNTRY_TO_ISO3).map(([k, v]) => [v, k])
)

interface GeoDistribution {
    country: string
    count: number
    percentage: number
}

// Green gradient based on employee count
function getCountryColor(count: number, maxCount: number): string {
    if (count === 0 || maxCount === 0) return '#e2e8f0' // Gray for no employees

    // Calculate intensity (0.2 to 1 for visible gradient)
    const ratio = count / maxCount
    const intensity = 0.2 + ratio * 0.8

    // Green palette: HSL(142, saturation%, lightness%)
    // Higher count = higher saturation, lower lightness (darker green)
    const saturation = Math.round(40 + intensity * 50) // 40-90%
    const lightness = Math.round(70 - intensity * 40) // 70-30%

    return `hsl(142, ${saturation}%, ${lightness}%)`
}

export function WorldMap() {
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [distribution, setDistribution] = useState<GeoDistribution[]>([])
    const [hoveredCountry, setHoveredCountry] = useState<string | null>(null)
    const [selectedCountry, setSelectedCountry] = useState<string | null>(null)
    const [tooltipContent, setTooltipContent] = useState<{ country: string; count: number; percentage: number } | null>(null)
    const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 })
    const [searchQuery, setSearchQuery] = useState('')

    // Zoom state
    const [position, setPosition] = useState<{ coordinates: [number, number]; zoom: number }>({
        coordinates: [0, 20],
        zoom: 1
    })

    // Fetch geo data
    useEffect(() => {
        async function fetchData() {
            try {
                setLoading(true)
                setError(null)
                const distData = await api.geo.getDistribution()
                console.log('WorldMap: Fetched distribution data:', distData)
                if (Array.isArray(distData) && distData.length > 0) {
                    setDistribution(distData)
                } else {
                    setError('No geographic data available. Ensure your data has a Country column.')
                }
            } catch (err) {
                console.error('WorldMap: Failed to fetch geo data:', err)
                setError('Failed to load geographic data.')
            } finally {
                setLoading(false)
            }
        }
        fetchData()
    }, [])

    // Build lookup map by TopoJSON country name and ISO3 code
    const countryDataByTopoName = useMemo(() => {
        const nameMap: Record<string, GeoDistribution> = {}
        distribution.forEach(item => {
            // Get the TopoJSON name for this country
            const topoName = COUNTRY_NAME_MAP[item.country] || item.country
            nameMap[topoName] = item
            // Also map by lowercase for case-insensitive matching
            nameMap[topoName.toLowerCase()] = item
            // Also map by our original name
            nameMap[item.country] = item
            nameMap[item.country.toLowerCase()] = item
            // Also map by ISO3 code
            const iso3 = COUNTRY_TO_ISO3[item.country]
            if (iso3) {
                nameMap[iso3] = item
            }
        })
        console.log('WorldMap: Name mapping built with', distribution.length, 'countries')
        console.log('WorldMap: Distribution data:', distribution)
        console.log('WorldMap: Mapping keys:', Object.keys(nameMap))
        return nameMap
    }, [distribution])

    const maxCount = useMemo(() => {
        const max = Math.max(...distribution.map(d => d.count), 1)
        console.log('WorldMap: Max count is', max)
        return max
    }, [distribution])

    const totalEmployees = useMemo(() => {
        return distribution.reduce((sum, d) => sum + d.count, 0)
    }, [distribution])

    const filteredCountries = useMemo(() => {
        const sorted = [...distribution].sort((a, b) => b.count - a.count)
        if (!searchQuery) return sorted
        return sorted.filter(d =>
            d.country.toLowerCase().includes(searchQuery.toLowerCase())
        )
    }, [distribution, searchQuery])

    // Event handlers
    const handleMouseMove = useCallback((event: React.MouseEvent) => {
        setTooltipPosition({ x: event.clientX, y: event.clientY })
    }, [])

    const handleMouseEnter = useCallback((geo: any) => {
        // Get country name and ISO3 from TopoJSON properties
        const geoName = geo.properties?.name || geo.properties?.NAME || geo.properties?.ADMIN || 'Unknown'
        const iso3 = geo.properties?.ISO_A3 || geo.properties?.iso_a3 || geo.properties?.id || ''

        // Try to find data for this country - try name first, then ISO3
        const data = countryDataByTopoName[geoName]
            || countryDataByTopoName[geoName.toLowerCase()]
            || countryDataByTopoName[iso3]

        setHoveredCountry(geoName)
        if (data) {
            setTooltipContent({
                country: data.country,
                count: data.count,
                percentage: data.percentage
            })
        } else {
            // Try to get friendly name from mappings
            const friendlyName = TOPO_TO_DATA_NAME[geoName] || ISO3_TO_COUNTRY[iso3] || geoName
            setTooltipContent({
                country: friendlyName,
                count: 0,
                percentage: 0
            })
        }
    }, [countryDataByTopoName])

    const handleMouseLeave = useCallback(() => {
        setHoveredCountry(null)
        setTooltipContent(null)
    }, [])

    const handleCountryClick = useCallback((geo: any) => {
        const geoName = geo.properties?.name || geo.properties?.NAME || geo.properties?.ADMIN || ''
        const iso3 = geo.properties?.ISO_A3 || geo.properties?.iso_a3 || geo.properties?.id || ''
        const data = countryDataByTopoName[geoName]
            || countryDataByTopoName[geoName.toLowerCase()]
            || countryDataByTopoName[iso3]
        if (data && data.count > 0) {
            setSelectedCountry(prev => prev === data.country ? null : data.country)
        }
    }, [countryDataByTopoName])

    // Zoom controls
    const handleZoomIn = useCallback(() => {
        setPosition(pos => ({ ...pos, zoom: Math.min(pos.zoom * 1.5, 4) }))
    }, [])

    const handleZoomOut = useCallback(() => {
        setPosition(pos => ({ ...pos, zoom: Math.max(pos.zoom / 1.5, 1) }))
    }, [])

    const handleReset = useCallback(() => {
        setPosition({ coordinates: [0, 20], zoom: 1 })
        setSelectedCountry(null)
    }, [])

    const handleMoveEnd = useCallback((newPosition: { coordinates: [number, number]; zoom: number }) => {
        setPosition(newPosition)
    }, [])

    // Loading state
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

    // Error state
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
            {/* Map Container */}
            <div className="flex-1 relative">
                {/* Header Overlay */}
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

                {/* Zoom Controls */}
                <div className="absolute top-4 right-4 z-10 flex flex-col gap-2">
                    <button
                        onClick={handleZoomIn}
                        disabled={position.zoom >= 4}
                        className="p-2 bg-white/95 dark:bg-slate-800/95 rounded-lg shadow-lg border border-border hover:bg-slate-100 dark:hover:bg-slate-700 disabled:opacity-50"
                        title="Zoom In"
                    >
                        <ZoomIn className="w-5 h-5" />
                    </button>
                    <button
                        onClick={handleZoomOut}
                        disabled={position.zoom <= 1}
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

                {/* Legend */}
                <div className="absolute bottom-4 left-4 z-10 bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm px-4 py-3 rounded-xl shadow-lg border border-border">
                    <p className="text-xs font-semibold text-text-muted mb-2">Employee Density</p>
                    <div className="flex items-center gap-1">
                        <div className="w-8 h-3 rounded" style={{ background: '#e2e8f0' }} title="No employees" />
                        <div className="w-8 h-3 rounded" style={{ background: 'hsl(142, 50%, 65%)' }} title="Few" />
                        <div className="w-8 h-3 rounded" style={{ background: 'hsl(142, 65%, 50%)' }} title="Some" />
                        <div className="w-8 h-3 rounded" style={{ background: 'hsl(142, 80%, 40%)' }} title="Many" />
                        <div className="w-8 h-3 rounded" style={{ background: 'hsl(142, 90%, 30%)' }} title="Most" />
                    </div>
                    <div className="flex justify-between text-[10px] text-text-muted mt-1">
                        <span>None</span>
                        <span>Most</span>
                    </div>
                </div>

                {/* Instructions */}
                <div className="absolute bottom-4 right-4 z-10 bg-white/80 dark:bg-slate-800/80 px-3 py-2 rounded-lg text-xs text-text-muted">
                    Drag to pan • Scroll to zoom • Click for details
                </div>

                {/* The Map */}
                <ComposableMap
                    projectionConfig={{
                        rotate: [-10, 0, 0],
                        scale: 147
                    }}
                    style={{ width: '100%', height: '100%', background: '#ffffff' }}
                >
                    <ZoomableGroup
                        zoom={position.zoom}
                        center={position.coordinates}
                        onMoveEnd={handleMoveEnd}
                        minZoom={1}
                        maxZoom={4}
                    >
                        <Geographies geography={GEO_URL}>
                            {({ geographies }) => {
                                // Debug: Log geography data to help troubleshoot
                                if (geographies.length > 0 && !window._geoLogged) {
                                    window._geoLogged = true
                                    console.log('WorldMap: Total geographies:', geographies.length)
                                    console.log('WorldMap: Sample geography properties:', geographies[0].properties)

                                    // Check for matches with our data
                                    const matches: string[] = []
                                    geographies.forEach(g => {
                                        const name = g.properties?.name || ''
                                        if (countryDataByTopoName[name] || countryDataByTopoName[name.toLowerCase()]) {
                                            matches.push(`${name} -> ${countryDataByTopoName[name]?.count || countryDataByTopoName[name.toLowerCase()]?.count} employees`)
                                        }
                                    })
                                    console.log('WorldMap: Matched countries:', matches)
                                }

                                return geographies.map((geo) => {
                                    // Get country name and ISO3 from TopoJSON - try multiple property names
                                    const geoName = geo.properties?.name || geo.properties?.NAME || geo.properties?.ADMIN || ''
                                    const iso3 = geo.properties?.ISO_A3 || geo.properties?.iso_a3 || geo.properties?.id || ''

                                    // Look up our employee data - try name first, then ISO3
                                    const data = countryDataByTopoName[geoName]
                                        || countryDataByTopoName[geoName.toLowerCase()]
                                        || countryDataByTopoName[iso3]
                                    const count = data?.count || 0
                                    const isSelected = selectedCountry && data?.country === selectedCountry

                                    // Apply green gradient color based on employee count
                                    const fillColor = isSelected
                                        ? 'hsl(142, 90%, 25%)'
                                        : count > 0
                                            ? getCountryColor(count, maxCount)
                                            : '#e2e8f0'

                                    return (
                                        <Geography
                                            key={geo.rsmKey}
                                            geography={geo}
                                            onMouseEnter={() => handleMouseEnter(geo)}
                                            onMouseLeave={handleMouseLeave}
                                            onClick={() => handleCountryClick(geo)}
                                            style={{
                                                default: {
                                                    fill: fillColor,
                                                    stroke: isSelected ? '#166534' : '#94a3b8',
                                                    strokeWidth: isSelected ? 1.5 : 0.5,
                                                    outline: 'none',
                                                },
                                                hover: {
                                                    fill: count > 0 ? 'hsl(142, 80%, 35%)' : '#cbd5e1',
                                                    stroke: '#1e293b',
                                                    strokeWidth: 1,
                                                    outline: 'none',
                                                    cursor: count > 0 ? 'pointer' : 'default',
                                                },
                                                pressed: {
                                                    fill: 'hsl(142, 90%, 25%)',
                                                    stroke: '#166534',
                                                    strokeWidth: 1.5,
                                                    outline: 'none',
                                                },
                                            }}
                                        />
                                    )
                                })
                            }}
                        </Geographies>
                    </ZoomableGroup>
                </ComposableMap>

                {/* Tooltip */}
                {tooltipContent && (
                    <div
                        className="fixed z-50 px-4 py-3 bg-slate-900 text-white rounded-xl shadow-2xl pointer-events-none"
                        style={{
                            left: tooltipPosition.x + 10,
                            top: tooltipPosition.y - 60,
                        }}
                    >
                        <div className="flex items-center gap-2 mb-1">
                            <MapPin className="w-4 h-4 text-green-400" />
                            <span className="font-bold">{tooltipContent.country}</span>
                        </div>
                        {tooltipContent.count > 0 ? (
                            <div className="flex items-center gap-2">
                                <Users className="w-3 h-3 text-green-400" />
                                <span className="text-green-400 font-semibold">
                                    {tooltipContent.count.toLocaleString()} employees
                                </span>
                                <span className="text-slate-400 text-sm">
                                    ({tooltipContent.percentage}%)
                                </span>
                            </div>
                        ) : (
                            <p className="text-slate-400 text-sm">No employees</p>
                        )}
                    </div>
                )}
            </div>

            {/* Country List Sidebar */}
            <div className="w-72 bg-white dark:bg-slate-800 border-l border-border dark:border-border-dark flex flex-col">
                {/* Header */}
                <div className="p-4 border-b border-border dark:border-border-dark">
                    <h3 className="font-semibold mb-3">Employees by Country</h3>
                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
                        <input
                            type="text"
                            placeholder="Search countries..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full pl-9 pr-3 py-2 text-sm border border-border rounded-lg bg-surface focus:outline-none focus:ring-2 focus:ring-accent/50"
                        />
                    </div>
                </div>

                {/* Country List */}
                <div className="flex-1 overflow-y-auto">
                    {filteredCountries.length === 0 ? (
                        <div className="p-4 text-center text-text-muted text-sm">
                            No countries found
                        </div>
                    ) : (
                        <div className="divide-y divide-border dark:divide-border-dark">
                            {filteredCountries.map((item) => (
                                <button
                                    key={item.country}
                                    onClick={() => setSelectedCountry(
                                        selectedCountry === item.country ? null : item.country
                                    )}
                                    className={cn(
                                        "w-full px-4 py-3 flex items-center justify-between hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors text-left",
                                        selectedCountry === item.country && "bg-accent/5 border-l-2 border-accent"
                                    )}
                                >
                                    <div className="flex items-center gap-3">
                                        <div
                                            className="w-3 h-3 rounded-full flex-shrink-0"
                                            style={{ backgroundColor: getCountryColor(item.count, maxCount) }}
                                        />
                                        <span className="text-sm font-medium truncate">
                                            {item.country}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-2 flex-shrink-0">
                                        <span className="text-sm font-bold text-accent">
                                            {item.count.toLocaleString()}
                                        </span>
                                        <span className="text-xs text-text-muted">
                                            ({item.percentage}%)
                                        </span>
                                        <ChevronRight className={cn(
                                            "w-4 h-4 text-text-muted transition-transform",
                                            selectedCountry === item.country && "rotate-90"
                                        )} />
                                    </div>
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                {/* Footer Summary */}
                <div className="p-4 border-t border-border dark:border-border-dark bg-slate-50 dark:bg-slate-900/50">
                    <div className="flex justify-between items-center text-sm">
                        <span className="text-text-muted">Total Employees</span>
                        <span className="font-bold">{totalEmployees.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between items-center text-sm mt-1">
                        <span className="text-text-muted">Countries</span>
                        <span className="font-bold">{distribution.length}</span>
                    </div>
                </div>
            </div>
        </div>
    )
}
