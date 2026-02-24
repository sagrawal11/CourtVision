"use client"

/**
 * Court Setup Page — /matches/[id]/court-setup
 *
 * Step 2 of the upload flow. After the user uploads a video, a lightweight
 * Batch job extracts frame 1000 and runs CourtDetector to generate 14
 * AI-suggested keypoints. This page:
 *  1. Polls /api/videos/{id}/status until court_setup_status='ready'
 *  2. Renders the video frame with 14 draggable numbered dots + connecting lines
 *  3. On "Confirm Court", saves the keypoints and triggers full analysis
 *
 * Keypoint index reference: see docs/video_pipeline.md
 */

import { useEffect, useRef, useState, useCallback } from "react"
import { useParams, useRouter } from "next/navigation"
import { createClient } from "@/lib/supabase/client"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
const POLL_INTERVAL_MS = 3000

// Court line connections: pairs of keypoint indices to draw as lines
// Matches the standard tennis court geometry (see docs/video_pipeline.md)
const COURT_LINES: [number, number][] = [
    [0, 1],   // Far baseline
    [2, 3],   // Near baseline
    [0, 2],   // Left doubles sideline
    [1, 3],   // Right doubles sideline
    [4, 5],   // Left singles sideline
    [6, 7],   // Right singles sideline
    [4, 6],   // Far baseline singles
    [5, 7],   // Near baseline singles
    [8, 9],   // Far service line
    [10, 11], // Near service line
    [8, 10],  // Left service box vertical
    [9, 11],  // Right service box vertical
    [12, 13], // Center service line
]

interface Keypoint {
    x: number
    y: number
}

type Keypoints = (Keypoint | null)[]

export default function CourtSetupPage() {
    const params = useParams()
    const router = useRouter()
    const matchId = params.id as string
    const supabase = createClient()

    const canvasRef = useRef<HTMLCanvasElement>(null)
    const imageRef = useRef<HTMLImageElement | null>(null)

    const [status, setStatus] = useState<"waiting" | "ready" | "saving" | "done">("waiting")
    const [frameUrl, setFrameUrl] = useState<string | null>(null)
    const [keypoints, setKeypoints] = useState<Keypoints>(Array(14).fill(null))
    const [dragIndex, setDragIndex] = useState<number | null>(null)
    const [error, setError] = useState<string | null>(null)

    // Debug video state
    const [debugVideoStatus, setDebugVideoStatus] = useState<"idle" | "generating" | "ready">("idle")
    const [debugVideoUrl, setDebugVideoUrl] = useState<string | null>(null)
    const debugPollRef = useRef<NodeJS.Timeout | null>(null)

    // ── Polling: wait for AI keypoints to be ready ────────────────────────────
    useEffect(() => {
        let timer: NodeJS.Timeout
        const poll = async () => {
            try {
                const { data: { session } } = await supabase.auth.getSession()
                if (!session) return

                const statusRes = await fetch(`${API_URL}/api/videos/${matchId}/status`, {
                    headers: { Authorization: `Bearer ${session.access_token}` },
                })
                if (!statusRes.ok) return
                const { court_setup_status } = await statusRes.json()

                if (court_setup_status === "ready" || court_setup_status === "confirmed") {
                    // Fetch the actual keypoints from court_configs
                    const { data: config } = await supabase
                        .from("court_configs")
                        .select("*")
                        .eq("match_id", matchId)
                        .single()

                    if (config) {
                        const kps: Keypoints = Array.from({ length: 14 }, (_, i) => {
                            const x = config[`kp${i}_x`]
                            const y = config[`kp${i}_y`]
                            return x != null && y != null ? { x, y } : null
                        })
                        setKeypoints(kps)
                    }

                    // Get the frame image URL from the match record
                    const { data: match } = await supabase
                        .from("matches")
                        .select("s3_temp_key, video_filename")
                        .eq("id", matchId)
                        .single()

                    if (match?.s3_temp_key) {
                        // Request a short-lived presigned GET URL from backend
                        const imgRes = await fetch(`${API_URL}/api/videos/${matchId}/frame-url`, {
                            headers: { Authorization: `Bearer ${session.access_token}` },
                        })
                        if (imgRes.ok) {
                            const { url } = await imgRes.json()
                            setFrameUrl(url)
                        }
                    }

                    setStatus("ready")
                    return // Stop polling
                }

                timer = setTimeout(poll, POLL_INTERVAL_MS)
            } catch (e) {
                console.error("Polling error:", e)
                timer = setTimeout(poll, POLL_INTERVAL_MS)
            }
        }
        poll()
        return () => clearTimeout(timer)
    }, [matchId, supabase])

    // ── Canvas rendering ───────────────────────────────────────────────────────
    const draw = useCallback(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext("2d")
        if (!ctx) return

        ctx.clearRect(0, 0, canvas.width, canvas.height)

        // Draw video frame as background
        if (imageRef.current?.complete) {
            ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height)
        }

        // Draw court lines
        ctx.strokeStyle = "rgba(80, 200, 120, 0.8)"
        ctx.lineWidth = 2
        ctx.setLineDash([])
        for (const [a, b] of COURT_LINES) {
            const kpA = keypoints[a]
            const kpB = keypoints[b]
            if (kpA && kpB) {
                ctx.beginPath()
                ctx.moveTo(kpA.x, kpA.y)
                ctx.lineTo(kpB.x, kpB.y)
                ctx.stroke()
            }
        }

        // Draw keypoint dots
        keypoints.forEach((kp, i) => {
            if (!kp) return
            // Outer white ring
            ctx.beginPath()
            ctx.arc(kp.x, kp.y, 14, 0, Math.PI * 2)
            ctx.fillStyle = "rgba(255,255,255,0.15)"
            ctx.fill()
            ctx.strokeStyle = "white"
            ctx.lineWidth = 2
            ctx.stroke()
            // Coloured fill
            ctx.beginPath()
            ctx.arc(kp.x, kp.y, 10, 0, Math.PI * 2)
            ctx.fillStyle = dragIndex === i ? "#FFD700" : "#50C878"
            ctx.fill()
            // Index label
            ctx.fillStyle = "white"
            ctx.font = "bold 11px Inter, sans-serif"
            ctx.textAlign = "center"
            ctx.textBaseline = "middle"
            ctx.fillText(String(i), kp.x, kp.y)
        })
    }, [keypoints, dragIndex])

    useEffect(() => {
        draw()
    }, [draw])

    // Load frame image once URL is available
    useEffect(() => {
        if (!frameUrl) return
        const img = new Image()
        img.crossOrigin = "anonymous"
        img.onload = () => {
            imageRef.current = img
            const canvas = canvasRef.current
            if (canvas) {
                canvas.width = img.naturalWidth
                canvas.height = img.naturalHeight
            }
            draw()
        }
        img.src = frameUrl
    }, [frameUrl, draw])

    // ── Mouse/touch drag handlers ──────────────────────────────────────────────
    const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement>): { x: number; y: number } => {
        const canvas = canvasRef.current!
        const rect = canvas.getBoundingClientRect()
        const scaleX = canvas.width / rect.width
        const scaleY = canvas.height / rect.height
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        }
    }

    const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
        const { x, y } = getCanvasCoords(e)
        const hitIndex = keypoints.findIndex((kp) => {
            if (!kp) return false
            return Math.hypot(kp.x - x, kp.y - y) < 18
        })
        if (hitIndex !== -1) setDragIndex(hitIndex)
    }

    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (dragIndex === null) return
        const { x, y } = getCanvasCoords(e)
        setKeypoints((prev) => {
            const next = [...prev]
            next[dragIndex] = { x, y }
            return next
        })
    }

    const handleMouseUp = () => setDragIndex(null)

    // ── Generate debug video ───────────────────────────────────────────────────
    const handleGenerateDebugVideo = async () => {
        setDebugVideoStatus("generating")
        setDebugVideoUrl(null)
        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) throw new Error("Not authenticated")

            const res = await fetch(`${API_URL}/api/videos/${matchId}/generate-debug-video`, {
                method: "POST",
                headers: { Authorization: `Bearer ${session.access_token}` },
            })
            if (!res.ok) {
                const err = await res.json()
                throw new Error(err.detail || "Failed to start debug video generation")
            }

            // Start polling for completion
            const pollDebugVideo = async () => {
                try {
                    const { data: { session: s } } = await supabase.auth.getSession()
                    if (!s) return
                    const urlRes = await fetch(`${API_URL}/api/videos/${matchId}/debug-video-url`, {
                        headers: { Authorization: `Bearer ${s.access_token}` },
                    })
                    if (urlRes.status === 200) {
                        const { url } = await urlRes.json()
                        setDebugVideoUrl(url)
                        setDebugVideoStatus("ready")
                        return  // stop polling
                    }
                    // 202 = still generating, keep polling
                    debugPollRef.current = setTimeout(pollDebugVideo, 5000)
                } catch {
                    debugPollRef.current = setTimeout(pollDebugVideo, 5000)
                }
            }
            pollDebugVideo()
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : "Failed to generate debug video")
            setDebugVideoStatus("idle")
        }
    }

    // Cleanup poll on unmount
    useEffect(() => () => { if (debugPollRef.current) clearTimeout(debugPollRef.current) }, [])

    // ── Confirm keypoints ──────────────────────────────────────────────────────
    const handleConfirm = async () => {
        setStatus("saving")
        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) throw new Error("Not authenticated")

            const payload: Record<string, number | null | boolean> = { ai_suggested: false }
            keypoints.forEach((kp, i) => {
                payload[`kp${i}_x`] = kp?.x ?? null
                payload[`kp${i}_y`] = kp?.y ?? null
            })

            const res = await fetch(`${API_URL}/api/videos/${matchId}/court-keypoints`, {
                method: "PUT",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${session.access_token}`,
                },
                body: JSON.stringify(payload),
            })
            if (!res.ok) throw new Error("Failed to save keypoints")

            setStatus("done")
            router.push(`/matches/${matchId}`)
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : "Save failed")
            setStatus("ready")
        }
    }

    // ── Render ─────────────────────────────────────────────────────────────────
    return (
        <div className="min-h-screen bg-[#0d0d0d] text-white flex flex-col">
            {/* Header */}
            <div className="border-b border-[#232323] px-6 py-4 flex items-center justify-between">
                <div>
                    <h1 className="text-xl font-semibold">Set Court Keypoints</h1>
                    <p className="text-sm text-gray-400 mt-0.5">
                        {status === "waiting"
                            ? "Analysing your video — this takes ~30 seconds…"
                            : "Drag each numbered dot to the correct court line intersection, then confirm."}
                    </p>
                </div>
                {/* Action buttons */}
                <div className="flex items-center gap-3">
                    {status === "ready" && (
                        <button
                            onClick={handleConfirm}
                            className="px-5 py-2 rounded-lg bg-[#50C878] hover:bg-[#45b069] text-black font-semibold text-sm transition-colors"
                        >
                            Confirm Court
                        </button>
                    )}
                    {status === "saving" && (
                        <span className="text-sm text-gray-400">Saving…</span>
                    )}
                    {status === "done" && (
                        <span className="text-sm text-[#50C878]">✓ Saved! Starting analysis…</span>
                    )}
                </div>
            </div>

            {/* Legend */}
            {status === "ready" && (
                <div className="px-6 py-2 flex gap-6 text-xs text-gray-500 border-b border-[#1a1a1a]">
                    <span className="flex items-center gap-1.5">
                        <span className="w-3 h-3 rounded-full bg-[#50C878] inline-block" /> Keypoint (drag to adjust)
                    </span>
                    <span className="flex items-center gap-1.5">
                        <span className="w-4 h-0.5 bg-[#50C878]/70 inline-block" /> Court line
                    </span>
                </div>
            )}

            {/* Main canvas area */}
            <div className="flex-1 flex items-center justify-center p-4 overflow-hidden">
                {status === "waiting" ? (
                    <div className="flex flex-col items-center gap-4 text-center">
                        <div className="w-12 h-12 rounded-full border-4 border-[#50C878] border-t-transparent animate-spin" />
                        <p className="text-gray-400 text-sm">AI is analysing frame 1000 of your video</p>
                    </div>
                ) : (
                    <div className="relative w-full max-w-5xl">
                        <canvas
                            ref={canvasRef}
                            className="w-full rounded-xl border border-[#333] cursor-crosshair select-none"
                            style={{ maxHeight: "70vh", objectFit: "contain" }}
                            onMouseDown={handleMouseDown}
                            onMouseMove={handleMouseMove}
                            onMouseUp={handleMouseUp}
                            onMouseLeave={handleMouseUp}
                        />
                        {!frameUrl && (
                            <div className="absolute inset-0 flex items-center justify-center bg-[#111] rounded-xl">
                                <p className="text-gray-500 text-sm">Loading frame…</p>
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Error */}
            {error && (
                <div className="mx-6 mb-4 p-3 rounded-lg bg-red-900/20 border border-red-800 text-red-300 text-sm">
                    {error}
                </div>
            )}

            {/* Debug video panel — appears after court is confirmed */}
            {(status === "done" || status === "saving" || debugVideoStatus !== "idle") && (
                <div className="mx-6 mb-6 p-4 rounded-xl border border-[#2a2a2a] bg-[#111]">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-semibold text-white">CV Verification Debug Video</p>
                            <p className="text-xs text-gray-500 mt-0.5">
                                Renders the first 60s of your video with court lines, zone overlays, ball tracking, and player boxes.
                                Requires the backend to be running locally.
                            </p>
                        </div>
                        <div className="flex items-center gap-3 ml-4">
                            {debugVideoStatus === "idle" && (
                                <button
                                    onClick={handleGenerateDebugVideo}
                                    className="px-4 py-2 rounded-lg border border-[#50C878] text-[#50C878] hover:bg-[#50C878]/10 text-sm font-medium transition-colors whitespace-nowrap"
                                >
                                    Generate Debug Video
                                </button>
                            )}
                            {debugVideoStatus === "generating" && (
                                <div className="flex items-center gap-2 text-sm text-gray-400">
                                    <div className="w-4 h-4 rounded-full border-2 border-[#50C878] border-t-transparent animate-spin" />
                                    Rendering… (this takes 2–5 min)
                                </div>
                            )}
                            {debugVideoStatus === "ready" && debugVideoUrl && (
                                <a
                                    href={debugVideoUrl}
                                    download="court_debug.mp4"
                                    className="px-4 py-2 rounded-lg bg-[#50C878] hover:bg-[#45b069] text-black text-sm font-semibold transition-colors whitespace-nowrap"
                                >
                                    ↓ Download Debug Video
                                </a>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
