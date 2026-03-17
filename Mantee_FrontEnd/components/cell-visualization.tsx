'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import {
  Play,
  Pause,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  MousePointer2,
  Pencil,
  X,
  Info,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface Cell {
  id: number
  x: number
  y: number
  cluster: string
  expression: number
  baseExpression: number
  velocity: { x: number; y: number }
  phase: number
  annotation?: string
}

interface Annotation {
  cellId: number
  text: string
  x: number
  y: number
}

const CLUSTER_COLORS: Record<string, string> = {
  A: '#22d3ee', // cyan
  B: '#a78bfa', // violet
  C: '#4ade80', // green
  D: '#fb923c', // orange
  E: '#f472b6', // pink
}

const TOTAL_CELLS = 293

function generateInitialCells(): Cell[] {
  const cells: Cell[] = []
  const clusters = ['A', 'B', 'C', 'D', 'E']
  const clusterCenters = [
    { x: 0.25, y: 0.3 },
    { x: 0.7, y: 0.25 },
    { x: 0.5, y: 0.55 },
    { x: 0.2, y: 0.75 },
    { x: 0.8, y: 0.7 },
  ]

  for (let i = 0; i < TOTAL_CELLS; i++) {
    const clusterIdx = Math.floor(Math.random() * clusters.length)
    const center = clusterCenters[clusterIdx]
    const spread = 0.12

    cells.push({
      id: i,
      x: center.x + (Math.random() - 0.5) * spread * 2,
      y: center.y + (Math.random() - 0.5) * spread * 2,
      cluster: clusters[clusterIdx],
      expression: Math.random(),
      baseExpression: 0.3 + Math.random() * 0.5,
      velocity: {
        x: (Math.random() - 0.5) * 0.0003,
        y: (Math.random() - 0.5) * 0.0003,
      },
      phase: Math.random() * Math.PI * 2,
    })
  }
  return cells
}

export function CellVisualization() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [cells, setCells] = useState<Cell[]>([])
  const [isPlaying, setIsPlaying] = useState(true)
  const [zoom, setZoom] = useState(1)
  const [selectedCells, setSelectedCells] = useState<Set<number>>(new Set())
  const [tool, setTool] = useState<'select' | 'annotate'>('select')
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [annotationInput, setAnnotationInput] = useState('')
  const [annotatingCell, setAnnotatingCell] = useState<Cell | null>(null)
  const [hoveredCell, setHoveredCell] = useState<Cell | null>(null)
  const [time, setTime] = useState(0)
  const animationRef = useRef<number>()

  const getCanvasCoords = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current
      if (!canvas) return null
      const rect = canvas.getBoundingClientRect()
      const x = (e.clientX - rect.left) / rect.width / zoom
      const y = (e.clientY - rect.top) / rect.height / zoom
      return { x, y }
    },
    [zoom]
  )

  const findCellAtPosition = useCallback(
    (x: number, y: number) => {
      const threshold = 0.025 / zoom
      return cells.find((cell) => {
        const dx = cell.x - x
        const dy = cell.y - y
        return Math.sqrt(dx * dx + dy * dy) < threshold
      })
    },
    [cells, zoom]
  )

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const coords = getCanvasCoords(e)
      if (!coords) return

      const clickedCell = findCellAtPosition(coords.x, coords.y)

      if (tool === 'select') {
        if (clickedCell) {
          if (e.shiftKey) {
            setSelectedCells((prev) => {
              const newSet = new Set(prev)
              if (newSet.has(clickedCell.id)) {
                newSet.delete(clickedCell.id)
              } else {
                newSet.add(clickedCell.id)
              }
              return newSet
            })
          } else {
            setSelectedCells(new Set([clickedCell.id]))
          }
        } else {
          setSelectedCells(new Set())
        }
      } else if (tool === 'annotate' && clickedCell) {
        setAnnotatingCell(clickedCell)
        setAnnotationInput(
          annotations.find((a) => a.cellId === clickedCell.id)?.text || ''
        )
      }
    },
    [tool, getCanvasCoords, findCellAtPosition, annotations]
  )

  const handleCanvasMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const coords = getCanvasCoords(e)
      if (!coords) return
      const cell = findCellAtPosition(coords.x, coords.y)
      setHoveredCell(cell || null)
    },
    [getCanvasCoords, findCellAtPosition]
  )

  const saveAnnotation = useCallback(() => {
    if (!annotatingCell) return

    setAnnotations((prev) => {
      const existing = prev.findIndex((a) => a.cellId === annotatingCell.id)
      if (annotationInput.trim() === '') {
        if (existing >= 0) {
          return prev.filter((a) => a.cellId !== annotatingCell.id)
        }
        return prev
      }

      const newAnnotation: Annotation = {
        cellId: annotatingCell.id,
        text: annotationInput.trim(),
        x: annotatingCell.x,
        y: annotatingCell.y,
      }

      if (existing >= 0) {
        const updated = [...prev]
        updated[existing] = newAnnotation
        return updated
      }
      return [...prev, newAnnotation]
    })

    setAnnotatingCell(null)
    setAnnotationInput('')
  }, [annotatingCell, annotationInput])

  useEffect(() => {
    setCells(generateInitialCells())
  }, [])

  useEffect(() => {
    if (!isPlaying) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      return
    }

    const animate = () => {
      setTime((t) => t + 0.02)

      setCells((prevCells) =>
        prevCells.map((cell) => {
          let newX = cell.x + cell.velocity.x
          let newY = cell.y + cell.velocity.y

          if (newX < 0.05 || newX > 0.95) cell.velocity.x *= -1
          if (newY < 0.05 || newY > 0.95) cell.velocity.y *= -1

          newX = Math.max(0.05, Math.min(0.95, newX))
          newY = Math.max(0.05, Math.min(0.95, newY))

          const expressionNoise = Math.sin(time + cell.phase) * 0.15
          const newExpression = Math.max(
            0.1,
            Math.min(1, cell.baseExpression + expressionNoise)
          )

          return {
            ...cell,
            x: newX,
            y: newY,
            expression: newExpression,
          }
        })
      )

      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isPlaying, time])

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const rect = container.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    const width = rect.width
    const height = rect.height

    ctx.fillStyle = '#0a0a14'
    ctx.fillRect(0, 0, width, height)

    ctx.save()
    ctx.scale(zoom, zoom)

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 10; i++) {
      const x = (i / 10) * (width / zoom)
      const y = (i / 10) * (height / zoom)
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height / zoom)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width / zoom, y)
      ctx.stroke()
    }

    // Draw cells
    cells.forEach((cell) => {
      const x = cell.x * (width / zoom)
      const y = cell.y * (height / zoom)
      const baseRadius = 6
      const radius = baseRadius * (0.6 + cell.expression * 0.6)
      const color = CLUSTER_COLORS[cell.cluster]

      const isSelected = selectedCells.has(cell.id)
      const isHovered = hoveredCell?.id === cell.id
      const hasAnnotation = annotations.some((a) => a.cellId === cell.id)

      // Glow effect
      const glowIntensity = cell.expression * 0.4
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius * 3)
      gradient.addColorStop(0, color + Math.floor(glowIntensity * 255).toString(16).padStart(2, '0'))
      gradient.addColorStop(1, 'transparent')
      ctx.fillStyle = gradient
      ctx.beginPath()
      ctx.arc(x, y, radius * 3, 0, Math.PI * 2)
      ctx.fill()

      // Cell body
      ctx.fillStyle = color
      ctx.globalAlpha = 0.3 + cell.expression * 0.5
      ctx.beginPath()
      ctx.arc(x, y, radius, 0, Math.PI * 2)
      ctx.fill()
      ctx.globalAlpha = 1

      // Cell border
      ctx.strokeStyle = color
      ctx.lineWidth = isSelected ? 2.5 : isHovered ? 2 : 1
      ctx.beginPath()
      ctx.arc(x, y, radius, 0, Math.PI * 2)
      ctx.stroke()

      // Selection ring
      if (isSelected) {
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 1.5
        ctx.setLineDash([3, 3])
        ctx.beginPath()
        ctx.arc(x, y, radius + 5, 0, Math.PI * 2)
        ctx.stroke()
        ctx.setLineDash([])
      }

      // Annotation marker
      if (hasAnnotation) {
        ctx.fillStyle = '#fbbf24'
        ctx.beginPath()
        ctx.arc(x + radius, y - radius, 4, 0, Math.PI * 2)
        ctx.fill()
      }
    })

    ctx.restore()

    // Draw annotation labels
    annotations.forEach((annotation) => {
      const cell = cells.find((c) => c.id === annotation.cellId)
      if (!cell) return
      const x = cell.x * width * zoom
      const y = cell.y * height * zoom

      ctx.fillStyle = 'rgba(30, 30, 40, 0.9)'
      ctx.strokeStyle = '#fbbf24'
      ctx.lineWidth = 1

      const textWidth = ctx.measureText(annotation.text).width
      const padding = 6
      const boxWidth = textWidth + padding * 2
      const boxHeight = 20
      const boxX = x + 15
      const boxY = y - 10

      ctx.beginPath()
      ctx.roundRect(boxX, boxY, boxWidth, boxHeight, 4)
      ctx.fill()
      ctx.stroke()

      ctx.fillStyle = '#fbbf24'
      ctx.font = '11px system-ui'
      ctx.fillText(annotation.text, boxX + padding, boxY + 14)
    })
  }, [cells, zoom, selectedCells, hoveredCell, annotations])

  const resetView = () => {
    setCells(generateInitialCells())
    setZoom(1)
    setSelectedCells(new Set())
    setAnnotations([])
  }

  const stats = {
    total: TOTAL_CELLS,
    active: cells.filter((c) => c.expression > 0.6).length,
    avgExpression: cells.reduce((sum, c) => sum + c.expression, 0) / cells.length,
    clusterCounts: Object.fromEntries(
      Object.keys(CLUSTER_COLORS).map((k) => [k, cells.filter((c) => c.cluster === k).length])
    ),
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border bg-card p-3">
        <div>
          <h1 className="text-lg font-semibold text-foreground">
            Transcriptome Cell Visualization
          </h1>
          <p className="text-xs text-muted-foreground">
            {TOTAL_CELLS} cells across 5 clusters - Real-time expression monitoring
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div
            className={cn(
              'flex items-center gap-1 rounded-full px-2 py-1 text-xs',
              isPlaying ? 'bg-chart-1/20 text-chart-1' : 'bg-muted text-muted-foreground'
            )}
          >
            <span className={cn('h-1.5 w-1.5 rounded-full', isPlaying ? 'animate-pulse bg-chart-1' : 'bg-muted-foreground')} />
            {isPlaying ? 'Live' : 'Paused'}
          </div>
        </div>
      </div>

      {/* Toolbar */}
      <div className="flex items-center gap-4 border-b border-border bg-card/50 px-3 py-2">
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() => setIsPlaying(!isPlaying)}
          >
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </Button>
          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={resetView}>
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>

        <div className="h-4 w-px bg-border" />

        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() => setZoom((z) => Math.min(3, z + 0.25))}
          >
            <ZoomIn className="h-4 w-4" />
          </Button>
          <span className="w-12 text-center text-xs text-muted-foreground">
            {Math.round(zoom * 100)}%
          </span>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
          >
            <ZoomOut className="h-4 w-4" />
          </Button>
        </div>

        <div className="h-4 w-px bg-border" />

        <div className="flex items-center gap-1">
          <Button
            variant={tool === 'select' ? 'secondary' : 'ghost'}
            size="sm"
            className="h-8 gap-1 text-xs"
            onClick={() => setTool('select')}
          >
            <MousePointer2 className="h-3 w-3" />
            Select
          </Button>
          <Button
            variant={tool === 'annotate' ? 'secondary' : 'ghost'}
            size="sm"
            className="h-8 gap-1 text-xs"
            onClick={() => setTool('annotate')}
          >
            <Pencil className="h-3 w-3" />
            Annotate
          </Button>
        </div>

        {selectedCells.size > 0 && (
          <>
            <div className="h-4 w-px bg-border" />
            <span className="text-xs text-muted-foreground">
              {selectedCells.size} cell{selectedCells.size > 1 ? 's' : ''} selected
            </span>
          </>
        )}
      </div>

      {/* Main content */}
      <div className="relative flex-1 overflow-hidden">
        <div ref={containerRef} className="h-full w-full">
          <canvas
            ref={canvasRef}
            className="h-full w-full cursor-crosshair"
            onClick={handleCanvasClick}
            onMouseMove={handleCanvasMouseMove}
          />
        </div>

        {/* Hover tooltip */}
        {hoveredCell && (
          <div className="pointer-events-none absolute left-4 top-4 rounded-lg border border-border bg-card/95 p-3 shadow-lg backdrop-blur-sm">
            <div className="flex items-center gap-2">
              <div
                className="h-3 w-3 rounded-full"
                style={{ backgroundColor: CLUSTER_COLORS[hoveredCell.cluster] }}
              />
              <span className="text-sm font-medium text-foreground">
                Cell #{hoveredCell.id}
              </span>
            </div>
            <div className="mt-2 space-y-1 text-xs text-muted-foreground">
              <p>Cluster: {hoveredCell.cluster}</p>
              <p>Expression: {(hoveredCell.expression * 100).toFixed(1)}%</p>
              <p>
                Position: ({hoveredCell.x.toFixed(3)}, {hoveredCell.y.toFixed(3)})
              </p>
            </div>
          </div>
        )}

        {/* Stats panel */}
        <div className="absolute bottom-4 left-4 rounded-lg border border-border bg-card/95 p-3 shadow-lg backdrop-blur-sm">
          <div className="flex items-center gap-1 text-xs font-medium text-foreground">
            <Info className="h-3 w-3" />
            Statistics
          </div>
          <div className="mt-2 grid grid-cols-3 gap-3 text-xs">
            <div>
              <p className="text-muted-foreground">Total</p>
              <p className="font-mono text-foreground">{stats.total}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Active</p>
              <p className="font-mono text-chart-1">{stats.active}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Avg Expr</p>
              <p className="font-mono text-foreground">{(stats.avgExpression * 100).toFixed(1)}%</p>
            </div>
          </div>
        </div>

        {/* Cluster legend */}
        <div className="absolute bottom-4 right-4 rounded-lg border border-border bg-card/95 p-3 shadow-lg backdrop-blur-sm">
          <p className="text-xs font-medium text-foreground">Clusters</p>
          <div className="mt-2 space-y-1">
            {Object.entries(CLUSTER_COLORS).map(([cluster, color]) => (
              <div key={cluster} className="flex items-center gap-2">
                <div className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
                <span className="text-xs text-muted-foreground">
                  {cluster}: {stats.clusterCounts[cluster]}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Annotation modal */}
        {annotatingCell && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm">
            <div className="w-80 rounded-lg border border-border bg-card p-4 shadow-xl">
              <div className="flex items-center justify-between">
                <h3 className="font-medium text-foreground">Annotate Cell #{annotatingCell.id}</h3>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6"
                  onClick={() => setAnnotatingCell(null)}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <input
                type="text"
                value={annotationInput}
                onChange={(e) => setAnnotationInput(e.target.value)}
                placeholder="Enter annotation..."
                className="mt-3 w-full rounded-md border border-border bg-input px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === 'Enter') saveAnnotation()
                  if (e.key === 'Escape') setAnnotatingCell(null)
                }}
              />
              <div className="mt-3 flex justify-end gap-2">
                <Button variant="ghost" size="sm" onClick={() => setAnnotatingCell(null)}>
                  Cancel
                </Button>
                <Button size="sm" onClick={saveAnnotation}>
                  Save
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
