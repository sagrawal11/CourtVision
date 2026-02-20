"use client"

import type React from "react"

import { useState } from "react"
import { useParams, useRouter } from "next/navigation"
import { MainLayout } from "@/components/layout/main-layout"
import { Button } from "@/components/ui/button"

export default function IdentifyPage() {
  const params = useParams()
  const router = useRouter()
  const matchId = params.id as string

  const [frames] = useState<string[]>([
    "/tennis-match-frame-1.jpg",
    "/tennis-match-frame-2.jpg",
    "/tennis-match-frame-3.jpg",
    "/tennis-match-frame-4.jpg",
    "/tennis-match-frame-5.jpg",
  ])
  const [selectedCoords, setSelectedCoords] = useState<Array<{ x: number; y: number } | null>>([
    null,
    null,
    null,
    null,
    null,
  ])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFrameClick = (frameIndex: number, event: React.MouseEvent<HTMLImageElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * 100
    const y = ((event.clientY - rect.top) / rect.height) * 100

    const newCoords = [...selectedCoords]
    newCoords[frameIndex] = { x, y }
    setSelectedCoords(newCoords)
  }

  const handleSubmit = async () => {
    const hasSelection = selectedCoords.some((c) => c !== null)
    if (!hasSelection) {
      setError("Please select yourself in at least one frame")
      return
    }

    setSubmitting(true)
    setError(null)

    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1000))
      router.push(`/matches/${matchId}`)
    } catch {
      setError("Failed to submit. Please try again.")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <MainLayout>
      <div className="mx-auto px-4 py-8 max-w-7xl">
        <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl">
          <h1 className="text-2xl font-bold text-white mb-2">Identify Yourself</h1>
          <p className="text-gray-400 mb-6">
            Click on yourself in each frame to help us track your performance throughout the match.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-6">
            {frames.map((frame: string, index: number) => (
              <div key={index} className="relative">
                <div className="aspect-video bg-[#262626] rounded border-2 border-dashed border-[#333333] overflow-hidden">
                  <img
                    src={frame || "/placeholder.svg"}
                    alt={`Frame ${index + 1}`}
                    className="w-full h-full object-cover cursor-crosshair"
                    onClick={(e) => handleFrameClick(index, e)}
                  />
                  {selectedCoords[index] && (
                    <div
                      className="absolute w-4 h-4 bg-blue-500 rounded-full border-2 border-white shadow-lg transform -translate-x-1/2 -translate-y-1/2 pointer-events-none"
                      style={{
                        left: `${selectedCoords[index]!.x}%`,
                        top: `${selectedCoords[index]!.y}%`,
                      }}
                    />
                  )}
                </div>
                <p className="text-xs text-gray-500 text-center mt-2">
                  {selectedCoords[index] ? "✓ Selected" : "Click to select"}
                </p>
              </div>
            ))}
          </div>

          {error && (
            <div className="bg-red-900/20 border-2 border-red-800 rounded-lg p-3 mb-4">
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}

          <div className="flex justify-end">
            <Button
              onClick={handleSubmit}
              disabled={submitting}
              className="bg-[#50C878] hover:bg-[#45b069] text-black font-semibold"
            >
              {submitting ? "Submitting..." : "Submit & Start Processing"}
            </Button>
          </div>
        </div>
      </div>
    </MainLayout>
  )
}
