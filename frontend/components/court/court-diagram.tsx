"use client"

type CourtShot = {
  id: string
  start_pos: { x: number; y: number }
  end_pos: { x: number; y: number }
  result: "winner" | "error" | "in_play"
  video_timestamp: number | null
}

interface CourtDiagramProps {
  shots?: CourtShot[]
  onShotClick?: (shot: CourtShot) => void
}

const courtWidth = 800
const courtHeight = 400

export function CourtDiagram({ shots = [], onShotClick }: CourtDiagramProps) {
  const getColor = (result: string) => {
    switch (result) {
      case "winner":
        return "#22c55e"
      case "error":
        return "#ef4444"
      default:
        return "#3b82f6"
    }
  }

  return (
    <div className="w-full bg-[#1a1a1a] rounded-lg border border-[#333333] p-4 overflow-x-auto">
      <svg width={courtWidth} height={courtHeight} viewBox={`0 0 ${courtWidth} ${courtHeight}`} className="mx-auto">
        {/* Court outline */}
        <rect
          x="20"
          y="20"
          width={courtWidth - 40}
          height={courtHeight - 40}
          fill="#4ade80"
          stroke="#22c55e"
          strokeWidth="2"
        />

        {/* Net line */}
        <line x1={courtWidth / 2} y1="20" x2={courtWidth / 2} y2={courtHeight - 20} stroke="white" strokeWidth="3" />

        {/* Service boxes - Left side */}
        <rect
          x="20"
          y="20"
          width={(courtWidth - 40) / 4}
          height={(courtHeight - 40) / 2}
          fill="#86efac"
          stroke="#22c55e"
          strokeWidth="1"
        />
        <rect
          x="20"
          y={(courtHeight - 40) / 2 + 20}
          width={(courtWidth - 40) / 4}
          height={(courtHeight - 40) / 2}
          fill="#86efac"
          stroke="#22c55e"
          strokeWidth="1"
        />

        {/* Service boxes - Right side */}
        <rect
          x={courtWidth - 20 - (courtWidth - 40) / 4}
          y="20"
          width={(courtWidth - 40) / 4}
          height={(courtHeight - 40) / 2}
          fill="#86efac"
          stroke="#22c55e"
          strokeWidth="1"
        />
        <rect
          x={courtWidth - 20 - (courtWidth - 40) / 4}
          y={(courtHeight - 40) / 2 + 20}
          width={(courtWidth - 40) / 4}
          height={(courtHeight - 40) / 2}
          fill="#86efac"
          stroke="#22c55e"
          strokeWidth="1"
        />

        {/* Center service line */}
        <line x1="20" y1={courtHeight / 2} x2={courtWidth - 20} y2={courtHeight / 2} stroke="white" strokeWidth="1" />

        {/* Shot lines */}
        {shots.map((shot) => (
          <line
            key={shot.id}
            x1={20 + shot.start_pos.x * (courtWidth - 40)}
            y1={20 + shot.start_pos.y * (courtHeight - 40)}
            x2={20 + shot.end_pos.x * (courtWidth - 40)}
            y2={20 + shot.end_pos.y * (courtHeight - 40)}
            stroke={getColor(shot.result)}
            strokeWidth="2"
            strokeDasharray={shot.result === "error" ? "5,5" : undefined}
            className="cursor-pointer hover:opacity-70 transition-opacity"
            onClick={() => onShotClick?.(shot)}
          />
        ))}
      </svg>

      {/* Legend */}
      <div className="flex gap-4 justify-center mt-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-0.5 bg-green-500" />
          <span className="text-gray-400">Winners</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-0.5 border-t-2 border-dashed border-red-500" />
          <span className="text-gray-400">Errors</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-0.5 bg-blue-500" />
          <span className="text-gray-400">In Play</span>
        </div>
      </div>
    </div>
  )
}
