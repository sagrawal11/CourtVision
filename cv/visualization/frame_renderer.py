"""
Visualization utilities for hero video generation.
Creates clean, professional visualizations combining player tracking and ball detection.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any


class FrameRenderer:
    """Creates polished visualizations for promotional videos."""
    
    def __init__(
        self,
        player_color: Tuple[int, int, int] = (120, 200, 80),  # BGR emerald green
        ball_color: Tuple[int, int, int] = (120, 200, 80),  # BGR emerald green
        court_color: Tuple[int, int, int] = (120, 200, 80),  # BGR emerald green
        trail_length: int = 30,
        skeleton_line_width: int = 2,
        keypoint_radius: int = 4,
        ball_radius: int = 8,
    ):
        """
        Initialize visualizer.
        
        Args:
            player_color: BGR color for player skeletons
            ball_color: BGR color for ball and trajectory
            court_color: BGR color for court lines
            trail_length: Maximum number of points in trajectory trail
            skeleton_line_width: Width of skeleton connection lines
            keypoint_radius: Radius of keypoint circles
            ball_radius: Radius of ball circle
        """
        self.player_color = player_color
        self.ball_color = ball_color
        self.court_color = court_color
        self.trail_length = trail_length
        self.skeleton_line_width = skeleton_line_width
        self.keypoint_radius = keypoint_radius
        self.ball_radius = ball_radius
        
        # Define skeleton connections (simplified MHR70 keypoints)
        # Format: (start_idx, end_idx)
        self.skeleton_connections = [
            # Head to shoulders
            (0, 1), (0, 2),
            # Torso
            (1, 2), (1, 3), (2, 4),
            # Arms
            (1, 5), (5, 7), (7, 9),  # Left arm
            (2, 6), (6, 8), (8, 10),  # Right arm
            # Legs
            (3, 11), (11, 13), (13, 15),  # Left leg
            (4, 12), (12, 14), (14, 16),  # Right leg
        ]
    
    def create_frame(
        self,
        frame: np.ndarray,
        player_outputs: List[Dict[str, Any]],
        ball_detection: Optional[Tuple[Tuple[int, int], float, np.ndarray]],
        ball_trajectory: List[Tuple[int, int]],
        skeleton_visualizer=None,
        court_keypoints: Optional[List[Tuple]] = None,
    ) -> np.ndarray:
        """
        Create a single annotated frame.
        
        Args:
            frame: Original frame (BGR) - used only for dimensions
            player_outputs: List of SAM-3d-body outputs for each player
            ball_detection: Ball detection result (center, confidence, mask) or None
            ball_trajectory: List of recent ball positions
            skeleton_visualizer: SkeletonVisualizer instance for keypoints-only mode
            court_keypoints: List of court keypoints (x, y) tuples or None
            
        Returns:
            Annotated frame (BGR) with black background and overlays only
        """
        # Create black frame instead of copying original
        vis_frame = np.zeros_like(frame)
        print(f"[DEBUG create_frame] Created black frame, shape: {vis_frame.shape}, dtype: {vis_frame.dtype}")
        
        # Draw court lines first (behind everything)
        if court_keypoints:
            print(f"[DEBUG create_frame] Drawing court with {len(court_keypoints)} keypoints")
            vis_frame = self._draw_court(vis_frame, court_keypoints)
            print(f"[DEBUG create_frame] After court: non-zero pixels = {np.count_nonzero(vis_frame)}")
        else:
            print(f"[DEBUG create_frame] No court keypoints provided")
        
        # Draw ball trajectory (behind players)
        if len(ball_trajectory) > 1:
            print(f"[DEBUG create_frame] Drawing ball trajectory with {len(ball_trajectory)} points")
            vis_frame = self._draw_trajectory(vis_frame, ball_trajectory)
            print(f"[DEBUG create_frame] After trajectory: non-zero pixels = {np.count_nonzero(vis_frame)}")
        else:
            print(f"[DEBUG create_frame] No ball trajectory (length: {len(ball_trajectory)})")
        
        # Draw players
        print(f"[DEBUG create_frame] Drawing {len(player_outputs)} player(s)")
        for i, player_output in enumerate(player_outputs):
            print(f"[DEBUG create_frame] Drawing player {i+1}/{len(player_outputs)}")
            vis_frame = self._draw_player(
                vis_frame,
                player_output,
                skeleton_visualizer
            )
            print(f"[DEBUG create_frame] After player {i+1}: non-zero pixels = {np.count_nonzero(vis_frame)}")
        
        # Draw ball (on top)
        if ball_detection:
            print(f"[DEBUG create_frame] Drawing ball detection")
            vis_frame = self._draw_ball(vis_frame, ball_detection)
            print(f"[DEBUG create_frame] After ball: non-zero pixels = {np.count_nonzero(vis_frame)}")
        else:
            print(f"[DEBUG create_frame] No ball detection provided")
        
        print(f"[DEBUG create_frame] Final frame - non-zero pixels: {np.count_nonzero(vis_frame)} / {vis_frame.size}")
        print(f"[DEBUG create_frame] Final frame pixel range: [{vis_frame.min()}, {vis_frame.max()}]")
        return vis_frame
    
    def _draw_player(
        self,
        frame: np.ndarray,
        player_output: Dict[str, Any],
        skeleton_visualizer=None,
    ) -> np.ndarray:
        """Draw player skeleton on frame."""
        vis_frame = frame.copy()
        
        # Get keypoints
        keypoints_2d = player_output.get("pred_keypoints_2d", None)
        if keypoints_2d is None:
            return vis_frame
        
        # Use skeleton visualizer if available (keypoints-only mode)
        # Note: skeleton_visualizer colors are set during initialization
        if skeleton_visualizer is not None:
            # Add visibility column
            keypoints_2d_with_vis = np.concatenate(
                [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))],
                axis=-1
            )
            # Draw skeleton (colors are pre-configured in visualizer)
            # Convert BGR to RGB for skeleton visualizer
            vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
            vis_frame_rgb = skeleton_visualizer.draw_skeleton(
                vis_frame_rgb,
                keypoints_2d_with_vis,
                kpt_thr=0.3
            )
            # Convert back to BGR
            vis_frame = cv2.cvtColor(vis_frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            # Manual drawing (fallback)
            vis_frame = self._draw_skeleton_manual(
                vis_frame,
                keypoints_2d
            )
        
        return vis_frame
    
    def _draw_skeleton_manual(
        self,
        frame: np.ndarray,
        keypoints_2d: np.ndarray,
    ) -> np.ndarray:
        """Manually draw skeleton connections and keypoints."""
        vis_frame = frame.copy()
        
        # Draw connections
        for start_idx, end_idx in self.skeleton_connections:
            if (start_idx < len(keypoints_2d) and 
                end_idx < len(keypoints_2d)):
                start_kp = keypoints_2d[start_idx]
                end_kp = keypoints_2d[end_idx]
                
                # Check visibility (assuming 3rd dimension is visibility)
                if len(start_kp) > 2 and start_kp[2] > 0:
                    if len(end_kp) > 2 and end_kp[2] > 0:
                        cv2.line(
                            vis_frame,
                            (int(start_kp[0]), int(start_kp[1])),
                            (int(end_kp[0]), int(end_kp[1])),
                            self.player_color,
                            self.skeleton_line_width,
                            cv2.LINE_AA
                        )
        
        # Draw keypoints
        for kp in keypoints_2d:
            if len(kp) > 2 and kp[2] > 0:  # Visibility check
                cv2.circle(
                    vis_frame,
                    (int(kp[0]), int(kp[1])),
                    self.keypoint_radius,
                    self.player_color,
                    -1,
                    cv2.LINE_AA
                )
                # White outline for visibility
                cv2.circle(
                    vis_frame,
                    (int(kp[0]), int(kp[1])),
                    self.keypoint_radius,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
        
        return vis_frame
    
    def _draw_ball(
        self,
        frame: np.ndarray,
        ball_detection: Tuple[Tuple[int, int], float, np.ndarray],
    ) -> np.ndarray:
        """Draw ball detection on frame."""
        vis_frame = frame.copy()
        center, confidence, mask = ball_detection
        
        # Draw subtle mask overlay (optional, can be disabled for cleaner look)
        # mask_colored = np.zeros_like(frame)
        # mask_colored[:, :, 0] = mask * self.ball_color[0]  # B
        # mask_colored[:, :, 1] = mask * self.ball_color[1]  # G
        # mask_colored[:, :, 2] = mask * self.ball_color[2]  # R
        # vis_frame = cv2.addWeighted(vis_frame, 0.9, mask_colored, 0.1, 0)
        
        # Draw ball center
        cx, cy = center
        cv2.circle(
            vis_frame,
            (cx, cy),
            self.ball_radius,
            self.ball_color,
            -1,
            cv2.LINE_AA
        )
        # White outline
        cv2.circle(
            vis_frame,
            (cx, cy),
            self.ball_radius,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        return vis_frame
    
    def _draw_trajectory(
        self,
        frame: np.ndarray,
        trajectory: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Draw ball trajectory trail with fading effect."""
        vis_frame = frame.copy()
        
        if len(trajectory) < 2:
            return vis_frame
        
        # Draw trajectory with fading opacity
        num_points = len(trajectory)
        for i in range(len(trajectory) - 1):
            pt1 = trajectory[i]
            pt2 = trajectory[i + 1]
            
            # Calculate opacity based on position in trail (newer = more opaque)
            alpha = (i + 1) / num_points
            color = tuple(int(c * alpha) for c in self.ball_color)
            
            # Thinner lines for older points
            thickness = max(1, int(2 * alpha))
            
            cv2.line(
                vis_frame,
                pt1,
                pt2,
                color,
                thickness,
                cv2.LINE_AA
            )
        
        return vis_frame
    
    def _draw_court(
        self,
        frame: np.ndarray,
        court_keypoints: List[Tuple],
        line_color: Tuple[int, int, int] = (100, 100, 100),  # Gray
        line_thickness: int = 1,
    ) -> np.ndarray:
        """Draw court lines from keypoints."""
        vis_frame = frame.copy()
        
        if len(court_keypoints) < 14:
            return vis_frame
        
        # Use instance court color if not specified
        if line_color is None:
            line_color = self.court_color
        
        # Define court line connections (14 keypoints)
        # Based on standard tennis court keypoint order
        court_lines = [
            # Baselines
            (0, 1),  # Top baseline
            (2, 3),  # Bottom baseline
            # Service boxes
            (4, 6),  # Top service line left
            (6, 1),  # Top service line right
            (5, 7),  # Bottom service line left
            (7, 3),  # Bottom service line right
            # Service box vertical lines
            (4, 8),   # Top left service box
            (8, 10),  # Left service box vertical
            (6, 9),   # Top right service box
            (9, 11),  # Right service box vertical
            # Net (center line)
            (8, 12),  # Net left
            (12, 9),  # Net center to right
            # Side lines (optional, if keypoints are valid)
            (4, 5),   # Left side line
            (6, 7),   # Right side line
        ]
        
        # Draw court lines
        for start_idx, end_idx in court_lines:
            if (start_idx < len(court_keypoints) and 
                end_idx < len(court_keypoints)):
                start_pt = court_keypoints[start_idx]
                end_pt = court_keypoints[end_idx]
                
                # Check if points are valid (not None)
                if (start_pt and end_pt and 
                    start_pt[0] is not None and start_pt[1] is not None and
                    end_pt[0] is not None and end_pt[1] is not None):
                    cv2.line(
                        vis_frame,
                        (int(start_pt[0]), int(start_pt[1])),
                        (int(end_pt[0]), int(end_pt[1])),
                        line_color,
                        line_thickness,
                        cv2.LINE_AA
                    )
        
        return vis_frame
