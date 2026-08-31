import { useCallback, useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { FolderOpen, RefreshCw, X } from 'lucide-react'
import { backendFetch } from '../lib/backend'
import { pathToFileUrl } from '../lib/file-url'

/** Recorded by the backend when the file was generated. Absent for hand-dropped files. */
export interface OutputGenerationParams {
  mode: string
  prompt: string
  model: string
  model_label?: string | null
  duration?: number | null
  resolution: string
  aspect_ratio: string
  fps: number
  audio: boolean
  camera_motion: string
  seed?: number | null
}

export interface OutputItem {
  path: string
  name: string
  size_bytes: number
  /** POSIX seconds. */
  modified_at: number
  generation_params?: OutputGenerationParams | null
}

interface OutputsListResponse {
  outputs: OutputItem[]
  total_count: number
  has_more: boolean
  next_offset: number | null
  outputs_dir: string
}

const PAGE_SIZE = 24

const IMAGE_SUFFIXES = ['.png', '.jpg', '.jpeg']

function isImage(name: string): boolean {
  const lower = name.toLowerCase()
  return IMAGE_SUFFIXES.some((suffix) => lower.endsWith(suffix))
}

function formatSize(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  return `${Math.max(1, Math.round(bytes / 1024))} KB`
}

function formatWhen(posixSeconds: number): string {
  return new Date(posixSeconds * 1000).toLocaleString()
}

export interface OutputsProjectOption {
  id: string
  name: string
}

export interface OutputsGalleryModalProps {
  onClose: () => void
  /** Projects the file can be imported into. Empty disables importing. */
  projects: OutputsProjectOption[]
  /** Copies the file into the project and returns once it is registered. */
  onImport: (item: OutputItem, projectId: string) => Promise<void>
}

/**
 * Browses the backend's outputs folder.
 *
 * Renders reach that folder however they were started — including from outside this
 * window, which the project list in localStorage never sees. This is the only view
 * that shows those.
 */
export function OutputsGalleryModal({ onClose, projects, onImport }: OutputsGalleryModalProps) {
  const [targetProjectId, setTargetProjectId] = useState(projects[0]?.id ?? '')
  const [importingPath, setImportingPath] = useState<string | null>(null)
  const [importError, setImportError] = useState<string | null>(null)
  const [items, setItems] = useState<OutputItem[]>([])
  const [outputsDir, setOutputsDir] = useState<string>('')
  const [totalCount, setTotalCount] = useState(0)
  const [nextOffset, setNextOffset] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async (offset: number) => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await backendFetch(`/api/outputs?limit=${PAGE_SIZE}&offset=${offset}`)
      if (!response.ok) throw new Error(`Backend returned ${response.status}`)
      const body: OutputsListResponse = await response.json()
      setItems((previous) => (offset === 0 ? body.outputs : [...previous, ...body.outputs]))
      setOutputsDir(body.outputs_dir)
      setTotalCount(body.total_count)
      setNextOffset(body.next_offset)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    void load(0)
  }, [load])

  const importItem = useCallback(
    async (item: OutputItem) => {
      if (!targetProjectId) return
      setImportingPath(item.path)
      setImportError(null)
      try {
        await onImport(item, targetProjectId)
        onClose()
      } catch (cause) {
        setImportError(cause instanceof Error ? cause.message : String(cause))
      } finally {
        setImportingPath(null)
      }
    },
    [onClose, onImport, targetProjectId],
  )

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6"
      onClick={onClose}
    >
      <div
        className="flex h-[80dvh] w-full max-w-5xl flex-col rounded-2xl bg-surface shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="flex items-center justify-between border-b border-border px-5 py-4">
          <div className="min-w-0">
            <h2 className="text-base font-semibold">Generated files</h2>
            <p className="truncate text-xs text-muted-foreground" title={outputsDir}>
              {totalCount} file{totalCount === 1 ? '' : 's'}
              {outputsDir ? ` in ${outputsDir}` : ''}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {projects.length > 0 ? (
              <label className="flex items-center gap-2 text-xs text-muted-foreground">
                Add to
                <select
                  className="rounded-lg border border-border bg-transparent px-2 py-1 text-xs"
                  value={targetProjectId}
                  onChange={(event) => setTargetProjectId(event.target.value)}
                >
                  {projects.map((project) => (
                    <option key={project.id} value={project.id}>
                      {project.name}
                    </option>
                  ))}
                </select>
              </label>
            ) : (
              <span className="text-xs text-muted-foreground">Create a project to import</span>
            )}
            <button
              type="button"
              className="rounded-lg p-2 hover:bg-muted disabled:opacity-50"
              onClick={() => void load(0)}
              disabled={isLoading}
              title="Refresh"
            >
              <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
            <button type="button" className="rounded-lg p-2 hover:bg-muted" onClick={onClose} title="Close">
              <X className="h-4 w-4" />
            </button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-5">
          {importingPath && (
            <p className="mb-3 rounded-lg bg-muted p-3 text-sm">Importing into the project…</p>
          )}

          {importError && (
            <p className="mb-3 rounded-lg bg-red-500/10 p-3 text-sm text-red-400">
              Could not import: {importError}
            </p>
          )}

          {error && (
            <p className="rounded-lg bg-red-500/10 p-3 text-sm text-red-400">
              Could not read the outputs folder: {error}
            </p>
          )}

          {!error && items.length === 0 && !isLoading && (
            <div className="flex h-full flex-col items-center justify-center gap-2 text-muted-foreground">
              <FolderOpen className="h-8 w-8" />
              <p className="text-sm">Nothing generated yet.</p>
            </div>
          )}

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            {items.map((item) => (
              <button
                key={item.path}
                type="button"
                className="group overflow-hidden rounded-xl border border-border text-left transition hover:border-primary"
                onClick={() => void importItem(item)}
                title={item.path}
                disabled={!targetProjectId || importingPath !== null}
              >
                <div className="aspect-video bg-black">
                  {isImage(item.name) ? (
                    <img
                      src={pathToFileUrl(item.path)}
                      alt={item.name}
                      loading="lazy"
                      className="h-full w-full object-contain"
                    />
                  ) : (
                    // No poster frame exists on disk, so let the video element render its
                    // own first frame rather than extracting thumbnails for every file.
                    <video
                      src={pathToFileUrl(item.path)}
                      preload="metadata"
                      muted
                      className="h-full w-full object-contain"
                      onMouseEnter={(event) => void event.currentTarget.play().catch(() => {})}
                      onMouseLeave={(event) => {
                        event.currentTarget.pause()
                        event.currentTarget.currentTime = 0
                      }}
                    />
                  )}
                </div>
                <div className="px-2 py-1.5">
                  <p className="truncate text-xs font-medium">{item.name}</p>
                  <p className="text-[11px] text-muted-foreground">
                    {formatSize(item.size_bytes)} · {formatWhen(item.modified_at)}
                  </p>
                  {item.generation_params && (
                    <p className="truncate text-[11px] text-muted-foreground">
                      {item.generation_params.resolution} · {item.generation_params.aspect_ratio} ·{' '}
                      {item.generation_params.fps}fps
                    </p>
                  )}
                </div>
              </button>
            ))}
          </div>

          {nextOffset !== null && (
            <div className="mt-4 flex justify-center">
              <button
                type="button"
                className="rounded-lg border border-border px-4 py-2 text-sm hover:bg-muted disabled:opacity-50"
                onClick={() => void load(nextOffset)}
                disabled={isLoading}
              >
                {isLoading ? 'Loading…' : `Load more (${totalCount - items.length} left)`}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>,
    document.body,
  )
}
