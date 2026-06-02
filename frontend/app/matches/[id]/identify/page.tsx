"use client"

import React, { useEffect, useState, useRef } from "react"
import { useParams, useRouter } from "next/navigation"
import { MainLayout } from "@/components/layout/main-layout"
import { createClient } from "@/lib/supabase/client"

interface DetectedFrame {
  frame_index: number
  image_base64: string
  boxes: [number, number, number, number][]
  width: number
  height: number
}

interface Selection {
  frameIndex: number
  boxIdx: number
  clickXPct: number
  clickYPct: number
  boxes: [number, number, number, number][]
  frameWidth: number
  frameHeight: number
}

export default function IdentifyPage() {
  const params = useParams()
  const router = useRouter()
  const matchId = params.id as string
  const supabase = createClient()
  const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

  const [frames, setFrames] = useState<DetectedFrame[]>([])
  const [status, setStatus] = useState<"loading" | "polling" | "ready" | "error">("loading")
  const [selection, setSelection] = useState<Selection | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const tokenRef = useRef<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const tryFetchFrames = async (token: string): Promise<boolean> => {
    try {
      const res = await fetch(`${API}/api/videos/${matchId}/player-selection-data`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (res.ok) {
        const data = await res.json()
        const f: DetectedFrame[] = (data.frames ?? []).filter((fr: DetectedFrame) => fr.boxes.length > 0)
        if (f.length > 0) {
          setFrames(f)
          setStatus("ready")
          return true
        }
      }
    } catch {}
    return false
  }

  useEffect(() => {
    let stopped = false
    const init = async () => {
      const { data: { session } } = await supabase.auth.getSession()
      if (!session) { setStatus("error"); return }
      tokenRef.current = session.access_token

      const ready = await tryFetchFrames(session.access_token)
      if (ready || stopped) return

      setStatus("polling")
      pollRef.current = setInterval(async () => {
        if (stopped || !tokenRef.current) return
        const ok = await tryFetchFrames(tokenRef.current)
        if (ok) clearInterval(pollRef.current!)
      }, 3000)
    }
    init()
    return () => {
      stopped = true
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [matchId]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleFrameClick = (frame: DetectedFrame, e: React.MouseEvent<HTMLDivElement>) => {
    if (frame.boxes.length === 0) return
    const rect = e.currentTarget.getBoundingClientRect()
    const relX = (e.clientX - rect.left) / rect.width
    const relY = (e.clientY - rect.top) / rect.height
    const px = relX * frame.width
    const py = relY * frame.height

    let nearestIdx = 0
    let minDist = Infinity
    frame.boxes.forEach((box, i) => {
      const cx = (box[0] + box[2]) / 2
      const cy = (box[1] + box[3]) / 2
      const d = (cx - px) ** 2 + (cy - py) ** 2
      if (d < minDist) { minDist = d; nearestIdx = i }
    })

    setSelection({
      frameIndex: frame.frame_index,
      boxIdx: nearestIdx,
      clickXPct: relX * 100,
      clickYPct: relY * 100,
      boxes: frame.boxes,
      frameWidth: frame.width,
      frameHeight: frame.height,
    })
  }

  const handleSubmit = async () => {
    if (!selection || submitting) return
    setSubmitting(true)
    setError(null)

    const { data: { session } } = await supabase.auth.getSession()
    if (!session) { setSubmitting(false); return }

    try {
      const res = await fetch(`${API}/api/videos/identify-player`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${session.access_token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          match_id: matchId,
          frame_data: { frame_index: selection.frameIndex },
          selected_player_coords: { x: selection.clickXPct, y: selection.clickYPct },
          boxes: selection.boxes,
          frame_width: selection.frameWidth,
          frame_height: selection.frameHeight,
        }),
      })
      if (!res.ok) throw new Error("Failed")
      router.push(`/matches/${matchId}/court-setup`)
    } catch {
      setError("Failed to save. Please try again.")
      setSubmitting(false)
    }
  }

  return (
    <MainLayout>
      <div className="mx-auto px-4 py-8 max-w-6xl">
        <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6">
          <h1 className="text-2xl font-bold text-white mb-1">Identify Your Player</h1>
          <p className="text-gray-400 text-sm mb-6">
            Click on yourself in any frame below. The green boxes show detected players.
          </p>

          {status === "loading" && (
            <p className="text-gray-400 py-12 text-center">Loading frames...</p>
          )}

          {status === "polling" && (
            <div className="py-12 text-center">
              <div className="inline-block w-6 h-6 border-2 border-[#50C878] border-t-transparent rounded-full animate-spin mb-3" />
              <p className="text-gray-400">Generating player selection frames...</p>
              <p className="text-gray-600 text-xs mt-1">This takes about 30 seconds</p>
            </div>
          )}

          {status === "error" && (
            <p className="text-red-400 py-12 text-center">Failed to load frames. Please refresh.</p>
          )}

          {status === "ready" && (
            <>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3 mb-6">
                {frames.map((frame) => {
                  const isActiveFrame = selection?.frameIndex === frame.frame_index
                  return (
                    <div key={frame.frame_index} className="flex flex-col gap-1.5">
                      <div
                        className={`relative cursor-crosshair rounded overflow-hidden border-2 transition-colors ${
                          isActiveFrame ? "border-[#50C878]" : "border-[#333]"
                        }`}
                        style={{ aspectRatio: `${frame.width} / ${frame.height}` }}
                        onClick={(e) => handleFrameClick(frame, e)}
                      >
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={frame.image_base64}
                          alt={`Frame ${frame.frame_index}`}
                          className="w-full h-full object-cover pointer-events-none select-none"
                          draggable={false}
                        />
                        {/* Box overlays */}
                        {frame.boxes.map((box, bi) => {
                          const [x1, y1, x2, y2] = box
                          const left = (x1 / frame.width) * 100
                          const top = (y1 / frame.height) * 100
                          const width = ((x2 - x1) / frame.width) * 100
                          const height = ((y2 - y1) / frame.height) * 100
                          const isSelected = isActiveFrame && selection?.boxIdx === bi
                          return (
                            <div
                              key={bi}
                              className="absolute border-2 pointer-events-none transition-colors"
                              style={{
                                left: `${left}%`,
                                top: `${top}%`,
                                width: `${width}%`,
                                height: `${height}%`,
                                borderColor: isSelected ? "#50C878" : "transparent",
                                boxShadow: isSelected ? "0 0 0 2px #50C87880" : "none",
                              }}
                            />
                          )
                        })}
                      </div>
                      <p className="text-xs text-center text-gray-500">
                        {isActiveFrame ? (
                          <span className="text-[#50C878]">Selected</span>
                        ) : (
                          "Click to select"
                        )}
                      </p>
                    </div>
                  )
                })}
              </div>

              {error && (
                <div className="bg-red-900/20 border border-red-800 rounded p-3 mb-4">
                  <p className="text-sm text-red-400">{error}</p>
                </div>
              )}

              <div className="flex items-center justify-between">
                <p className="text-xs text-gray-500">
                  {selection
                    ? `Player selected in frame ${selection.frameIndex}`
                    : "No player selected yet"}
                </p>
                <button
                  onClick={handleSubmit}
                  disabled={!selection || submitting}
                  className="px-6 py-2 rounded-lg bg-[#50C878] text-black text-sm font-semibold hover:bg-[#45b069] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  {submitting ? "Saving..." : "Continue →"}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </MainLayout>
  )
}
