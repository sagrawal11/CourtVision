"use client"

/**
 * Upload Modal — handles match video uploads and instant frontend court setup.
 * 
 * Flow:
 *  1. User selects a video file & enters metadata.
 *  2. Invisible <video> element instantly seeks to ~33s and extracts a frame to a canvas.
 *  3. Modal flips to "Step 2", showing the frame with 14 draggable court keypoints.
 *  4. User adjusts keypoints manually.
 *  5. Click "Upload & Confirm" -> 
 *       a) POST /api/videos/prepare-upload (gets signed URL)
 *       b) PUT video file to Supabase Storage
 *       c) POST /api/videos/{id}/confirm-upload (sends keypoints & completes flow)
 */

import type React from "react"
import { useState, useEffect, useRef } from "react"
import { useRouter } from "next/navigation"
import { X, UploadCloud, CheckCircle2, Info } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useAuth } from "@/hooks/useAuth"
import { useProfile } from "@/hooks/useProfile"
import { useTeams } from "@/hooks/useTeams"
import { useActivation } from "@/hooks/useActivation"
import { createClient } from "@/lib/supabase/client"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
const ACCEPTED_VIDEO_TYPES = "video/mp4,video/quicktime,video/x-msvideo,video/webm"
const MAX_FILE_SIZE_MB = 50  // Supabase Storage free tier global limit

// Fallback coordinates for a 1280x720 video (will be scaled internally by the canvas)
const DEFAULT_KEYPOINTS = [
  { x: 260, y: 720 }, { x: 420, y: 720 }, { x: 640, y: 720 },  // Bottom baseline
  { x: 860, y: 720 }, { x: 1020, y: 720 },
  { x: 400, y: 450 }, { x: 480, y: 450 }, { x: 640, y: 450 },  // Service line
  { x: 800, y: 450 }, { x: 880, y: 450 },
  { x: 520, y: 300 }, { x: 640, y: 300 }, { x: 760, y: 300 },  // Net line
  { x: 640, y: 585 } // Center tee
]

// Connectivity graph for drawing court lines
const EDGES = [
  [0, 4], [5, 9], [10, 12], // horizontals: bl, sl, net
  [0, 10], [1, 11], [3, 11], [4, 12], // verticals: left alley, left singles, right singles, right alley
  [2, 7], [7, 13], [13, 2] // center line
]

interface UploadModalProps {
  isOpen: boolean
  onClose: () => void
}

export function UploadModal({ isOpen, onClose }: UploadModalProps) {
  const router = useRouter()
  const { getUser } = useAuth()
  const { profile } = useProfile()
  const { teams } = useTeams()
  const { isActivated } = useActivation()
  const supabase = createClient()

  // -- Step 1 Refs
  const fileInputRef = useRef<HTMLInputElement>(null)

  // -- Step 2 Refs
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [frameExtracted, setFrameExtracted] = useState(false)
  const [keypoints, setKeypoints] = useState(DEFAULT_KEYPOINTS.map(kp => ({ ...kp })))
  const [draggingIdx, setDraggingIdx] = useState<number | null>(null)
  const [videoAspect, setVideoAspect] = useState(16 / 9)

  // -- State
  const [step, setStep] = useState<1 | 2>(1)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [playerName, setPlayerName] = useState("")
  const [selectedPlayerId, setSelectedPlayerId] = useState<string>("")
  const [matchDate, setMatchDate] = useState("")
  const [opponent, setOpponent] = useState("")
  const [notes, setNotes] = useState("")
  const [teamMembers, setTeamMembers] = useState<any[]>([])

  const [uploadPhase, setUploadPhase] = useState<"idle" | "requesting" | "uploading" | "confirming" | "done">("idle")
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const isCoach = profile?.role === "coach"
  const isLoading = uploadPhase !== "idle" && uploadPhase !== "done"

  // Block upload if unactivated coach
  useEffect(() => {
    if (isCoach && !isActivated && isOpen) onClose()
  }, [isCoach, isActivated, isOpen, onClose])

  // Reset form on open
  useEffect(() => {
    if (isOpen) {
      setStep(1)
      setSelectedFile(null)
      setFrameExtracted(false)
      setKeypoints(DEFAULT_KEYPOINTS.map(kp => ({ ...kp })))
      setPlayerName("")
      setSelectedPlayerId("")
      setMatchDate("")
      setOpponent("")
      setNotes("")
      setError(null)
      setUploadPhase("idle")
      setUploadProgress(0)
    }
  }, [isOpen])

  // Fetch team members
  useEffect(() => {
    const fetchTeamMembers = async () => {
      if (!isCoach || teams.length === 0 || !isOpen) { setTeamMembers([]); return }
      const allMembers: any[] = []
      for (const team of teams) {
        const { data: { session } } = await supabase.auth.getSession()
        if (!session) continue
        try {
          const res = await fetch(`${API_URL}/api/teams/${team.id}/members`, {
            headers: { Authorization: `Bearer ${session.access_token}` },
          })
          if (res.ok) {
            const data = await res.json()
            const members = (data.members || []).filter((m: any) => m.users?.role === "player")
            allMembers.push(...members.map((m: any) => ({ id: m.users?.id, name: m.users?.name || m.users?.email || "Unknown" })))
          }
        } catch (err) { console.error("Error fetching team:", err) }
      }
      setTeamMembers(Array.from(new Map(allMembers.map(m => [m.id, m])).values()))
    }
    if (isOpen && isCoach && teams.length > 0) fetchTeamMembers()
  }, [isCoach, teams, isOpen, supabase])

  // ── Frame Extraction Logic ───────────────────────────────────────────────

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setError(`File exceeds ${MAX_FILE_SIZE_MB} MB limit (Supabase Free Tier)`)
      return
    }
    setSelectedFile(file)
    setError(null)

    // Create local object URL for the hidden video element
    const objectUrl = URL.createObjectURL(file)
    if (videoRef.current) {
      videoRef.current.src = objectUrl
      videoRef.current.load()
      // We want to jump to ~33 seconds in to bypass warmups
      videoRef.current.currentTime = 33
    }
  }

  // Triggered when the hidden <video> successfully seeks to the requested time
  const handleVideoSeeked = () => {
    const video = videoRef.current
    if (!video) return

    setVideoAspect(video.videoWidth / video.videoHeight)

    // Initial draw to the hidden step 2 canvas mapping
    requestAnimationFrame(() => drawCanvas())
    setFrameExtracted(true)
  }

  // ── Canvas Interaction Logic ──────────────────────────────────────────────

  const getCanvasCoords = (e: React.MouseEvent | React.TouchEvent | MouseEvent | TouchEvent) => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }

    const rect = canvas.getBoundingClientRect()
    // calculate scaling factor between internal 1280x720 space and display size
    const scaleX = 1280 / rect.width
    const scaleY = 720 / rect.height

    let clientX, clientY
    if ("touches" in e) {
      clientX = e.touches[0].clientX
      clientY = e.touches[0].clientY
    } else {
      clientX = (e as React.MouseEvent).clientX
      clientY = (e as React.MouseEvent).clientY
    }

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    }
  }

  const drawCanvas = () => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw the video frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    // Dark overlay for contrast
    ctx.fillStyle = "rgba(0,0,0,0.1)"
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw lines
    ctx.strokeStyle = "rgba(80, 200, 120, 0.7)"
    ctx.lineWidth = 3
    for (const [i, j] of EDGES) {
      const p1 = keypoints[i]
      const p2 = keypoints[j]
      ctx.beginPath()
      ctx.moveTo(p1.x, p1.y)
      ctx.lineTo(p2.x, p2.y)
      ctx.stroke()
    }

    // Draw points
    keypoints.forEach((kp, i) => {
      ctx.beginPath()
      ctx.arc(kp.x, kp.y, draggingIdx === i ? 8 : 6, 0, 2 * Math.PI)
      ctx.fillStyle = draggingIdx === i ? "#fff" : "#50C878"
      ctx.fill()
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 2
      ctx.stroke()

      // Label
      ctx.fillStyle = "#fff"
      ctx.font = "12px sans-serif"
      ctx.fillText(i.toString(), kp.x + 10, kp.y - 10)
    })
  }

  // Redraw when keypoints or drag state change
  useEffect(() => {
    if (step === 2 && frameExtracted) {
      drawCanvas()
    }
  }, [keypoints, draggingIdx, step, frameExtracted])

  const handlePointerDown = (e: React.MouseEvent | React.TouchEvent) => {
    if (isLoading) return
    const { x, y } = getCanvasCoords(e)

    // Find closest keypoint within an interaction radius (scaled to 1280x720 space)
    const HIT_RADIUS = 30
    let closestIdx = -1
    let minDist = Infinity

    keypoints.forEach((kp, i) => {
      const dist = Math.hypot(kp.x - x, kp.y - y)
      if (dist < HIT_RADIUS && dist < minDist) {
        minDist = dist
        closestIdx = i
      }
    })

    if (closestIdx !== -1) {
      setDraggingIdx(closestIdx)
    }
  }

  const handlePointerMove = (e: React.MouseEvent | React.TouchEvent) => {
    if (draggingIdx === null || isLoading) return
    const { x, y } = getCanvasCoords(e)

    setKeypoints(prev => {
      const next = [...prev]
      next[draggingIdx] = { x, y }
      return next
    })
  }

  const handlePointerUp = () => {
    setDraggingIdx(null)
  }

  // ── Network Submission ──────────────────────────────────────────────────

  const handleNextStep = (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedFile) { setError("Please select a video file"); return }
    if (!frameExtracted) { setError("Still processing the video. Please wait a second."); return }
    setError(null)
    setStep(2)
  }

  const handleFinalSubmit = async () => {
    setError(null)
    if (!selectedFile) return

    const user = await getUser()
    const { data: { session } } = await supabase.auth.getSession()
    if (!user || !session) { setError("Please sign in first"); return }

    const authHeaders = { Authorization: `Bearer ${session.access_token}`, "Content-Type": "application/json" }
    const matchUserId = isCoach && selectedPlayerId ? selectedPlayerId : undefined
    const finalPlayerName = !isCoach ? (profile?.name || undefined) : (playerName || undefined)

    try {
      // ── Step 1: Create match record + get signed storage upload URL ─────────
      setUploadPhase("requesting")
      const prepRes = await fetch(`${API_URL}/api/videos/prepare-upload`, {
        method: "POST",
        headers: authHeaders,
        body: JSON.stringify({
          filename: selectedFile.name,
          player_name: finalPlayerName,
          player_user_id: matchUserId,
          match_date: matchDate || undefined,
          opponent: opponent || undefined,
          notes: notes || undefined,
        }),
      })
      if (!prepRes.ok) throw new Error((await prepRes.json()).detail || "Failed to prepare upload")
      const { match_id, storage_path, upload_url } = await prepRes.json()

      // ── Step 2: Upload file directly to Supabase Storage via signed URL ─────
      setUploadPhase("uploading")
      await uploadWithProgress(upload_url, selectedFile, (pct) => setUploadProgress(pct))

      // ── Step 3: Notify backend & send the manually confirmed keypoints ──────────
      setUploadPhase("confirming")

      const keypointsPayload: Record<string, number | null | boolean> = { ai_suggested: false }
      keypoints.forEach((kp, i) => {
        keypointsPayload[`kp${i}_x`] = kp.x
        keypointsPayload[`kp${i}_y`] = kp.y
      })

      const confirmRes = await fetch(`${API_URL}/api/videos/${match_id}/confirm-upload`, {
        method: "POST",
        headers: authHeaders,
        body: JSON.stringify({
          storage_path,
          keypoints: keypointsPayload
        }),
      })
      if (!confirmRes.ok) throw new Error("Failed to confirm upload and save court")

      setUploadPhase("done")
      setTimeout(() => { onClose(); router.push(`/matches/${match_id}`) }, 800)

    } catch (err: unknown) {
      console.error("Upload error:", err)
      setError(err instanceof Error ? err.message : "Upload failed")
      setUploadPhase("idle")
    }
  }

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div
        className={`bg-[#1a1a1a] transition-all duration-300 rounded-2xl p-6 border border-[#333333] shadow-2xl flex flex-col ${step === 2 ? 'max-w-4xl w-full max-h-[90vh]' : 'max-w-md w-full'}`}
      >
        <div className="flex justify-between items-center mb-4 shrink-0">
          <h2 className="text-xl font-semibold text-white">
            {step === 1 ? "Upload Match Video" : "Step 2: Confirm Court Dimensions"}
          </h2>
          {!isLoading && (
            <Button variant="ghost" size="icon" onClick={onClose} className="text-gray-400 hover:text-white hover:bg-[#262626]">
              <X className="h-5 w-5" />
            </Button>
          )}
        </div>

        {/* Hidden internal video player for taking frame screenshots */}
        <video
          ref={videoRef}
          className="hidden"
          crossOrigin="anonymous"
          onSeeked={handleVideoSeeked}
          muted playsInline
        />

        {/* ── STEP 1: METADATA & FILE ────────────────────────────────────── */}
        {step === 1 && (
          <form onSubmit={handleNextStep} className="space-y-4">
            <div>
              <Label className="text-gray-400 text-sm font-medium">Match Video</Label>
              <div
                onClick={() => fileInputRef.current?.click()}
                className={`mt-1 flex flex-col items-center justify-center gap-2 p-6 rounded-xl border-2 border-dashed cursor-pointer transition-colors
                  ${selectedFile ? "border-[#50C878] bg-[#50C878]/5" : "border-[#333333] hover:border-[#50C878]/50 bg-black/30"}`}
              >
                {selectedFile ? (
                  <>
                    <CheckCircle2 className="h-8 w-8 text-[#50C878]" />
                    <p className="text-sm text-white font-medium truncate max-w-full text-center">{selectedFile.name}</p>
                    <p className="text-xs text-gray-500">{(selectedFile.size / 1024 / 1024).toFixed(1)} MB</p>
                  </>
                ) : (
                  <>
                    <UploadCloud className="h-8 w-8 text-gray-500" />
                    <p className="text-sm text-gray-400">Click to choose a video file</p>
                    <p className="text-xs text-gray-600">MP4, MOV, AVI, WebM — up to {MAX_FILE_SIZE_MB} MB</p>
                  </>
                )}
              </div>
              <input ref={fileInputRef} type="file" accept={ACCEPTED_VIDEO_TYPES} onChange={handleFileChange} className="hidden" />
            </div>

            {isCoach && teamMembers.length > 0 && (
              <div>
                <Label className="text-gray-400 text-sm font-medium">Select Player</Label>
                <Select value={selectedPlayerId} onValueChange={setSelectedPlayerId}>
                  <SelectTrigger className="mt-1 bg-black/50 border-[#333333] text-white">
                    <SelectValue placeholder="Select a player" />
                  </SelectTrigger>
                  <SelectContent className="bg-[#1a1a1a] border-[#333333]">
                    {teamMembers.map((m) => (
                      <SelectItem key={m.id} value={m.id} className="text-white hover:bg-[#262626]">{m.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}

            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label className="text-gray-400 text-sm font-medium">Match Date</Label>
                <Input type="date" value={matchDate} onChange={(e) => setMatchDate(e.target.value)}
                  className="mt-1 bg-black/50 border-[#333333] text-white" />
              </div>
              <div>
                <Label className="text-gray-400 text-sm font-medium">Opponent</Label>
                <Input type="text" value={opponent} onChange={(e) => setOpponent(e.target.value)}
                  placeholder="Name / school" className="mt-1 bg-black/50 border-[#333333] text-white placeholder-gray-500" />
              </div>
            </div>

            <div>
              <Label className="text-gray-400 text-sm font-medium">Notes (Optional)</Label>
              <Textarea value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Any notes…"
                rows={2} className="mt-1 bg-black/50 border-[#333333] text-white placeholder-gray-500 resize-none" />
            </div>

            {error && (
              <div className="bg-red-900/20 border border-red-800 rounded-lg p-3">
                <p className="text-sm text-red-300">{error}</p>
              </div>
            )}

            <div className="flex gap-2 justify-end">
              <Button type="button" variant="outline" onClick={onClose}
                className="border-[#333333] text-gray-300 hover:border-[#50C878] hover:text-white bg-transparent">
                Cancel
              </Button>
              <Button type="submit" disabled={!selectedFile || !frameExtracted}
                className="bg-[#50C878] hover:bg-[#45b069] text-black font-semibold">
                {frameExtracted ? "Next: Setup Court" : "Extracting Video..."}
              </Button>
            </div>
          </form>
        )}

        {/* ── STEP 2: COURT EDITOR ────────────────────────────────────── */}
        {step === 2 && (
          <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
            <div className="mb-4 text-sm text-gray-400 flex items-start gap-2 shrink-0">
              <Info className="h-4 w-4 mt-0.5 text-[#50C878] shrink-0" />
              <p>
                Drag the green dots to perfectly align with the 14 intersections of your tennis court.
                This teaches the AI the dimensions of your specific camera angle.
              </p>
            </div>

            <div className="relative flex-1 bg-black rounded-lg border border-[#333] overflow-hidden flex items-center justify-center isolate">
              <canvas
                ref={canvasRef}
                width={1280}
                height={720}
                className="w-full h-full object-contain cursor-crosshair touch-none select-none z-10"
                style={{ aspectRatio: videoAspect }}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerLeave={handlePointerUp}
                onPointerCancel={handlePointerUp}
              />

              {(uploadPhase === "uploading" || uploadPhase === "confirming") && (
                <div className="absolute inset-0 z-20 bg-black/70 flex flex-col items-center justify-center p-8 backdrop-blur-sm">
                  <div className="w-full max-w-sm">
                    <h3 className="text-lg font-semibold text-white mb-2 text-center text-shadow">
                      {uploadPhase === "uploading" ? "Uploading video..." : "Finalising setup..."}
                    </h3>
                    <div className="flex justify-between text-xs text-gray-300 mb-2">
                      <span>{uploadProgress}%</span>
                    </div>
                    <div className="w-full bg-[#222] rounded-full h-2 overflow-hidden shadow-inner">
                      <div className="bg-[#50C878] h-2 rounded-full transition-all duration-300" style={{ width: `${uploadProgress}%` }} />
                    </div>
                  </div>
                </div>
              )}

              {uploadPhase === "done" && (
                <div className="absolute inset-0 z-20 bg-black/80 flex flex-col items-center justify-center backdrop-blur-sm">
                  <CheckCircle2 className="h-16 w-16 text-[#50C878] shadow-lg animate-bounce" />
                  <h3 className="text-xl font-bold text-white mt-4 text-shadow text-center">Court Set!</h3>
                  <p className="text-gray-300 mt-2 text-center max-w-xs">Analysis has started. Redirecting you to the match page...</p>
                </div>
              )}
            </div>

            {error && (
              <div className="mt-4 bg-red-900/20 border border-red-800 rounded-lg p-3 shrink-0">
                <p className="text-sm text-red-300">{error}</p>
              </div>
            )}

            <div className="flex gap-2 justify-end mt-4 shrink-0">
              <Button type="button" variant="outline" onClick={() => setStep(1)} disabled={isLoading}
                className="border-[#333333] text-gray-300 hover:border-[#50C878] hover:text-white bg-transparent">
                Back
              </Button>
              <Button type="button" onClick={handleFinalSubmit} disabled={isLoading}
                className="bg-[#50C878] hover:bg-[#45b069] text-black font-semibold">
                Upload & Confirm Court
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Upload a file to a presigned URL via XHR so we can track progress.
 * Supabase Storage signed upload URLs accept a standard PUT request.
 */
async function uploadWithProgress(
  signedUrl: string,
  file: File,
  onProgress: (pct: number) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open("PUT", signedUrl, true)
    xhr.setRequestHeader("Content-Type", file.type || "video/*")
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100))
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve()
      else reject(new Error(`Upload failed (status ${xhr.status}): ${xhr.responseText}`))
    }
    xhr.onerror = () => reject(new Error("Network error during upload"))
    xhr.send(file)
  })
}
