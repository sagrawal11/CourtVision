"use client"

/**
 * Upload Modal — handles match video uploads via Supabase Storage.
 * 
 * Flow:
 *  1. User selects a video file
 *  2. POST /api/videos/prepare-upload → backend creates match record, returns signed upload URL
 *  3. Browser uploads the file directly to Supabase Storage (no backend bandwidth used)
 *  4. POST /api/videos/{id}/confirm-upload → backend verifies + triggers local court_setup_job
 *  5. Navigate to /matches/{id}/court-setup for the interactive court editor
 */

import type React from "react"
import { useState, useEffect, useRef } from "react"
import { useRouter } from "next/navigation"
import { X, UploadCloud, CheckCircle2 } from "lucide-react"
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
const MAX_FILE_SIZE_GB = 1  // Supabase Storage free tier: 1 GB total

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
  const fileInputRef = useRef<HTMLInputElement>(null)

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
      setSelectedFile(null)
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

  // Fetch team members (coaches only)
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
        } catch (err) { console.error("Error fetching team members:", err) }
      }
      setTeamMembers(Array.from(new Map(allMembers.map(m => [m.id, m])).values()))
    }
    if (isOpen && isCoach && teams.length > 0) fetchTeamMembers()
  }, [isCoach, teams, isOpen, supabase])

  if (!isOpen) return null
  if (isCoach && !isActivated) return null

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (file.size > MAX_FILE_SIZE_GB * 1024 ** 3) {
      setError(`File exceeds ${MAX_FILE_SIZE_GB} GB limit`)
      return
    }
    setSelectedFile(file)
    setError(null)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    if (!selectedFile) { setError("Please select a video file"); return }

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

      // ── Step 3: Notify backend — triggers local court_setup_job ────────────
      setUploadPhase("confirming")
      const confirmRes = await fetch(`${API_URL}/api/videos/${match_id}/confirm-upload`, {
        method: "POST",
        headers: authHeaders,
        body: JSON.stringify({ storage_path }),
      })
      if (!confirmRes.ok) throw new Error("Upload confirmed but court setup failed to start")

      setUploadPhase("done")
      setTimeout(() => { onClose(); router.push(`/matches/${match_id}/court-setup`) }, 800)

    } catch (err: unknown) {
      console.error("Upload error:", err)
      setError(err instanceof Error ? err.message : "Upload failed")
      setUploadPhase("idle")
    }
  }

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-[#1a1a1a] max-w-md w-full rounded-2xl p-6 border border-[#333333] shadow-2xl">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold text-white">Upload Match Video</h2>
          <Button variant="ghost" size="icon" onClick={onClose} className="text-gray-400 hover:text-white hover:bg-[#262626]">
            <X className="h-5 w-5" />
          </Button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* File picker */}
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
                  <p className="text-xs text-gray-600">MP4, MOV, AVI, WebM — up to {MAX_FILE_SIZE_GB} GB</p>
                </>
              )}
            </div>
            <input ref={fileInputRef} type="file" accept={ACCEPTED_VIDEO_TYPES} onChange={handleFileChange} className="hidden" />
          </div>

          {/* Upload progress */}
          {uploadPhase === "uploading" && (
            <div>
              <div className="flex justify-between text-xs text-gray-500 mb-1">
                <span>Uploading video…</span><span>{uploadProgress}%</span>
              </div>
              <div className="w-full bg-[#333] rounded-full h-1.5">
                <div className="bg-[#50C878] h-1.5 rounded-full transition-all duration-200" style={{ width: `${uploadProgress}%` }} />
              </div>
            </div>
          )}
          {uploadPhase === "requesting" && <p className="text-xs text-gray-500 text-center">Preparing upload…</p>}
          {uploadPhase === "confirming" && <p className="text-xs text-gray-500 text-center">Finalising… almost there!</p>}
          {uploadPhase === "done" && <p className="text-xs text-[#50C878] text-center">✓ Heading to court setup…</p>}

          {/* Coach: player selector */}
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

          {/* Match metadata */}
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
            <Button type="button" variant="outline" onClick={onClose} disabled={isLoading}
              className="border-[#333333] text-gray-300 hover:border-[#50C878] hover:text-white bg-transparent">
              Cancel
            </Button>
            <Button type="submit" disabled={isLoading || !selectedFile}
              className="bg-[#50C878] hover:bg-[#45b069] text-black font-semibold">
              {isLoading ? "Uploading…" : "Upload"}
            </Button>
          </div>
        </form>
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
      else reject(new Error(`Upload failed with status ${xhr.status}`))
    }
    xhr.onerror = () => reject(new Error("Network error during upload"))
    xhr.send(file)
  })
}
