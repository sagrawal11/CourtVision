export interface User {
  id: string
  email: string
  name: string | null
  role: "coach" | "player"
  team_id: string | null
  created_at: string
  updated_at: string
}

export interface Team {
  id: string
  name: string
  code: string
  created_at: string
  updated_at: string
}

export interface MatchStats {
  total_points: number
  poi_points_won: number
  opp_points_won: number
  poi_shots: number
  poi_winners: number
  poi_errors: number
  poi_in_play: number
  poi_winner_pct: number
  poi_serves_total: number
  poi_first_serves_in: number
  poi_first_serves_attempted: number
  poi_second_serves_attempted: number
  poi_second_serves_in: number
  poi_double_faults: number
  poi_serve_1_pct: number
  poi_first_serve_in_pct: number
  poi_second_serve_in_pct: number
  poi_aces: number
  serve_zones: {
    deuce_t: number
    deuce_body: number
    deuce_wide: number
    ad_t: number
    ad_body: number
    ad_wide: number
  }
  rally_lengths: number[]
  avg_rally_length: number
  poi_serve_speed_avg: number
  poi_forehand_speed_avg: number
  poi_backhand_speed_avg: number
  poi_forehands: number
  poi_backhands: number
}

// Lightweight shot shape stored in matches.analysis_shots JSONB by the pipeline
export interface AnalysisShot {
  frame: number
  point_idx: number | null
  x: number | null
  y: number | null
  player: string | null
  speed_kmh: number | null
  shot_type: string | null
}

export interface Match {
  id: string
  user_id: string
  player_name: string | null
  playsight_link: string | null
  video_url: string | null
  s3_temp_key: string | null
  poi_start_side: "near" | "far" | null
  status: "pending" | "generating_frames" | "player_selection" | "court_setup" | "processing" | "completed" | "failed"
  court_setup_status: string | null
  match_date: string | null
  opponent: string | null
  notes: string | null
  processed_at: string | null
  analyzed_at: string | null
  analysis_error: string | null
  stats: MatchStats | null
  analysis_shots: AnalysisShot[] | null
  created_at: string
  updated_at: string
}

export interface Shot {
  id: string
  match_id: string
  frame: number | null
  point_idx: number | null
  shot_type: string | null
  start_pos: { x: number; y: number } | null
  end_pos: { x: number; y: number } | null
  timestamp: number
  video_timestamp: number | null
  result: "winner" | "error" | "in_play"
  speed_kmh: number | null
  player: string | null
  is_winner: boolean
  is_error: boolean
  created_at: string
}

export interface PointRecord {
  id: string
  match_id: string
  point_idx: number
  start_frame: number | null
  end_frame: number | null
  start_timestamp_s: number | null
  end_timestamp_s: number | null
  serve_player: "near" | "far" | null
  rally_length: number
  manual_outcome: "winner" | "forced_error" | "unforced_error" | "ace" | "double_fault" | null
  reviewed_at: string | null
  created_at: string
}
