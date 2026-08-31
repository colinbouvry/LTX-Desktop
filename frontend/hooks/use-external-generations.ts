import { useEffect, useRef } from 'react'
import { useProjects } from '../contexts/ProjectContext'
import { backendFetch } from '../lib/backend'
import { addVisualAssetToProject } from '../lib/asset-copy'
import { logger } from '../lib/logger'
import type { GenerationParams } from '../types/project-model'

/** Recorded by the backend beside each generated file. */
interface OutputGenerationParams {
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

interface OutputItem {
  path: string
  name: string
  size_bytes: number
  modified_at: number
  generation_params?: OutputGenerationParams | null
}

const POLL_INTERVAL_MS = 5_000
const WATERMARK_KEY_PREFIX = 'ltx-external-generations-seen-'

function watermarkKey(projectId: string): string {
  return `${WATERMARK_KEY_PREFIX}${projectId}`
}

function readWatermark(projectId: string): number | null {
  try {
    const stored = localStorage.getItem(watermarkKey(projectId))
    if (!stored) return null
    const parsed = Number(stored)
    return Number.isFinite(parsed) ? parsed : null
  } catch {
    return null
  }
}

function writeWatermark(projectId: string, value: number): void {
  try {
    localStorage.setItem(watermarkKey(projectId), String(value))
  } catch {
    // Storage full or blocked: the next poll re-derives a watermark. Losing it only
    // risks re-offering a file, never losing one.
  }
}

/**
 * Adds generations produced outside this window — by the MCP server, or a script —
 * to the open project.
 *
 * They reach the shared outputs folder but never the project list, which lives in this
 * renderer's localStorage. Polling the backend is the only way in: the writer has no
 * access to that storage.
 *
 * Only files carrying backend-recorded provenance are taken. A file dropped into the
 * folder by hand has none, and inventing settings for it would put a fabricated
 * generation in the project's history.
 */
export function useExternalGenerations(projectId: string | null): void {
  const { addAsset, getProject } = useProjects()
  // Guards against a slow import overlapping the next tick, which would import twice.
  const isImportingRef = useRef(false)

  useEffect(() => {
    if (!projectId) return

    let cancelled = false

    const poll = async () => {
      if (cancelled || isImportingRef.current) return
      try {
        const response = await backendFetch('/api/outputs?limit=50')
        if (!response.ok || cancelled) return
        const body: { outputs: OutputItem[] } = await response.json()
        const candidates = body.outputs.filter(item => item.generation_params)

        const watermark = readWatermark(projectId)
        if (watermark === null) {
          // First sight of this project: adopt the current newest file as the starting
          // point. Without this, opening a project would sweep in the whole history.
          const newest = candidates.reduce((max, item) => Math.max(max, item.modified_at), 0)
          writeWatermark(projectId, newest)
          return
        }

        const fresh = candidates
          .filter(item => item.modified_at > watermark)
          .sort((a, b) => a.modified_at - b.modified_at)
        if (fresh.length === 0) return

        isImportingRef.current = true
        try {
          for (const item of fresh) {
            if (cancelled) break
            // Re-read each time: the project changed under us if a previous iteration
            // added an asset, and a stale copy would drop it.
            const project = getProject(projectId)
            if (!project) break
            const alreadyPresent = project.assets.some(asset => asset.path.endsWith(item.name))
            if (!alreadyPresent) {
              const copied = await addVisualAssetToProject(item.path, projectId, 'video')
              if (!copied) {
                logger.error(`Could not copy external generation into the project: ${item.path}`)
                continue
              }
              const recorded = item.generation_params!
              addAsset(projectId, {
                type: 'video',
                path: copied.path,
                bigThumbnailPath: copied.bigThumbnailPath,
                smallThumbnailPath: copied.smallThumbnailPath,
                width: copied.width,
                height: copied.height,
                prompt: recorded.prompt,
                resolution: recorded.resolution,
                duration: recorded.duration ?? undefined,
                generationParams: {
                  mode: recorded.mode as GenerationParams['mode'],
                  prompt: recorded.prompt,
                  model: recorded.model,
                  modelLabel: recorded.model_label ?? undefined,
                  duration: recorded.duration ?? null,
                  resolution: recorded.resolution,
                  fps: recorded.fps,
                  audio: recorded.audio,
                  cameraMotion: recorded.camera_motion,
                },
              })
              logger.info(`Imported external generation ${item.name} into the open project`)
            }
            // Advance per file, so a failure part-way does not replay what already landed.
            writeWatermark(projectId, item.modified_at)
          }
        } finally {
          isImportingRef.current = false
        }
      } catch (error) {
        // The backend may still be starting, or have been stopped. Stay quiet and retry.
        logger.error(`External generation poll failed: ${error}`)
      }
    }

    void poll()
    const timer = window.setInterval(() => void poll(), POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [addAsset, getProject, projectId])
}
