#!/usr/bin/env python3
"""
Hero Video Generator - Combines SAM-3d-body player tracking with TrackNet ball detection
to create promotional videos for the CourtVision hero section.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import json
import subprocess
import tempfile
import shutil

import cv2
import numpy as np
import torch
import gc
from tqdm import tqdm
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Add SAM-3d-body to path
SAM3D_BODY_PATH = PROJECT_ROOT / "SAM-3d-body" / "sam-3d-body"
if SAM3D_BODY_PATH.exists():
    sys.path.insert(0, str(SAM3D_BODY_PATH))

# Import SAM-3d-body utilities
try:
    from notebook.utils import setup_sam_3d_body, load_sam_3d_body_hf
    from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator
    from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
    from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
except ImportError as e:
    print(f"Error importing SAM-3d-body: {e}")
    print("Make sure SAM-3d-body is set up correctly.")
    sys.exit(1)

# Import YOLO human detector
try:
    from yolo_human_detector import YOLOHumanDetector
    YOLO_DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: YOLO human detector not available: {e}")
    YOLO_DETECTOR_AVAILABLE = False

# Import TrackNet ball detector
try:
    from ensemble_ball_detector import EnsembleBallDetector
    ENSEMBLE_DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TrackNet detector not available: {e}")
    ENSEMBLE_DETECTOR_AVAILABLE = False

# Import visualizer
from visualizer import HeroVideoVisualizer

# Import custom mesh visualizer for emerald green mesh (optional, for full mesh mode)
try:
    from mesh_visualizer import visualize_sample_together_emerald
    MESH_VISUALIZER_AVAILABLE = True
except ImportError:
    MESH_VISUALIZER_AVAILABLE = False
    # Fallback to original if custom not available
    try:
        from tools.vis_utils import visualize_sample_together
        visualize_sample_together_emerald = visualize_sample_together
        MESH_VISUALIZER_AVAILABLE = True
    except ImportError:
        MESH_VISUALIZER_AVAILABLE = False

# Try to import court detection (optional)
COURT_DETECTION_AVAILABLE = False
CourtDetector = None
try:
    # Try importing from old codebase
    sys.path.insert(0, str(PROJECT_ROOT / "old" / "scripts" / "legacy" / "demos"))
    from court_demo import CourtDetector
    COURT_DETECTION_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    try:
        # Try importing from core module
        sys.path.insert(0, str(PROJECT_ROOT / "old" / "src" / "core"))
        from tennis_CV import CourtDetector
        COURT_DETECTION_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        # Court detection not available - that's okay, it's optional
        CourtDetector = None


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)  # BGR format


def process_video(
    input_path: Path,
    output_path: Path,
    ball_prompt: str = "tennis ball",
    frame_skip: int = 30,  # Default to every 30th frame for speed (full mesh is VERY slow!)
    fps: float = 30.0,
    player_color: str = "#50C878",
    ball_color: str = "#50C878",
    trail_length: int = 30,
    keypoints_only: bool = False,  # Default to full mesh (user preference)
    device: Optional[str] = None,
    enable_court_detection: bool = False,  # Disable by default for speed
    court_model_path: Optional[Path] = None,
    process_resolution: Optional[int] = 720,  # Default to 720px width for speed (full mesh is VERY slow!)
):
    """
    Process video with both player tracking and ball detection.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        ball_prompt: Unused, kept for backwards compatibility
        frame_skip: Process every Nth frame
        fps: Output video FPS
        player_color: Hex color for player skeletons
        ball_color: Hex color for ball and trajectory
        trail_length: Number of frames in ball trajectory trail
        keypoints_only: Use fast keypoints-only mode
        device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"Error: Input video not found at {input_path}")
        return
    
    # Auto-detect device
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved
            
            print(f"Using CUDA")
            print(f"   GPU: {gpu_name}")
            print(f"   GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {free:.2f} GB free, {total:.2f} GB total")
            
            # A100-specific optimizations
            if "A100" in gpu_name:
                print(f"   ✅ A100 detected! You have plenty of memory ({free:.2f} GB free).")
                print(f"   💡 Consider using higher quality settings:")
                print(f"      - Higher process_resolution (e.g., 1080 or 1280 instead of 720)")
                print(f"      - Enable ensemble ball detection for better accuracy")
                print(f"      - Process every frame (frame_skip=1) for smoother output")
            elif "L4" in gpu_name:
                print(f"   ✅ L4 detected! Good memory headroom ({free:.2f} GB free).")
            elif "T4" in gpu_name:
                print(f"   ⚠️ T4 detected. Limited memory ({free:.2f} GB free).")
                if free < 10.0:
                    print(f"   💡 Consider restarting runtime or using keypoints_only mode to save memory.")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple Silicon MPS")
        else:
            device = "cpu"
            print("Using CPU")
    else:
        print(f"Using device: {device}")
    
    # Initialize models
    print("\n" + "="*60)
    print("Initializing Models")
    print("="*60)
    
    # Setup SAM-3d-body with YOLO human detector
    print("\n1. Setting up SAM-3d-body...")
    
    # Try to use YOLO human detector if available
    human_detector = None
    if YOLO_DETECTOR_AVAILABLE:
        yolo_model_path = PROJECT_ROOT / "old" / "models" / "player" / "playersnball5.pt"
        if yolo_model_path.exists():
            try:
                human_detector = YOLOHumanDetector(model_path=yolo_model_path, device=device)
                print("✓ YOLO human detector loaded (playersnball5.pt)")
            except Exception as e:
                print(f"⚠ Could not load YOLO detector: {e}")
                human_detector = None
    
    # Load SAM-3d-body model
    try:
        model, model_cfg = load_sam_3d_body_hf("facebook/sam-3d-body-dinov3", device=device)
        
        # Load FOV estimator
        try:
            from tools.build_fov_estimator import FOVEstimator
            fov_estimator = FOVEstimator(name="moge2", device=device)
        except Exception:
            fov_estimator = None
        
        # Create estimator with YOLO detector
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=human_detector,
            human_segmentor=None,  # Not needed for keypoints-only or full mesh
            fov_estimator=fov_estimator,
        )
        
        if human_detector:
            print("✓ SAM-3d-body ready (with YOLO human detector)")
        else:
            print("✓ SAM-3d-body ready (without detector, will process full image)")
        
        # Clear GPU cache after SAM-3d-body is loaded (model is now stored in estimator)
        if device == "cuda" and torch.cuda.is_available():
            print("   Clearing GPU cache before loading SAM3...")
            # Model is stored in estimator, so we can delete the direct references
            del model, model_cfg
            if fov_estimator is not None:
                del fov_estimator
            torch.cuda.empty_cache()
            gc.collect()
            print("   ✓ GPU cache cleared")
    except Exception as e:
        print(f"⚠ Error setting up SAM-3d-body: {e}")
        # Fallback to standard setup
        try:
            estimator = setup_sam_3d_body(
                hf_repo_id="facebook/sam-3d-body-dinov3",
                device=device,
                detector_name=None
            )
            print("✓ SAM-3d-body ready (fallback setup)")
            
            # Also clear cache for fallback path
            if device == "cuda" and torch.cuda.is_available():
                print("   Clearing GPU cache before loading SAM3...")
                torch.cuda.empty_cache()
                gc.collect()
                print("   ✓ GPU cache cleared")
        except Exception as e2:
            print(f"✗ Failed to setup SAM-3d-body: {e2}")
            return
    
    # Setup ball detector
    print("\n2. Setting up ball detector...")
    
    if ENSEMBLE_DETECTOR_AVAILABLE:
        ball_detector = EnsembleBallDetector(device=device)
        print("✓ TrackNet ball detector ready")
    else:
        print("Error: TrackNet ball detector not available")
        return
    
    # Initialize court detector (optional)
    court_detector = None
    if enable_court_detection and COURT_DETECTION_AVAILABLE and CourtDetector:
        print("\n3. Setting up court detector...")
        print(f"   PROJECT_ROOT: {PROJECT_ROOT}")
        print(f"   PROJECT_ROOT exists: {PROJECT_ROOT.exists()}")
        
        # Try to find court model
        if court_model_path is None:
            # Try default locations - prioritize models/court/model_tennis_court_det.pt
            possible_paths = [
                PROJECT_ROOT / "models" / "court" / "model_tennis_court_det.pt",  # Primary location
                PROJECT_ROOT / "old" / "models" / "court" / "model_tennis_court_det.pt",  # Legacy location
            ]
            
            print(f"   Searching for court model 'model_tennis_court_det.pt' in:")
            # First try exact paths
            for path in possible_paths:
                exists = path.exists()
                print(f"     {'✓' if exists else '✗'} {path}")
                if exists:
                    court_model_path = path
                    print(f"   ✅ Found court model at: {court_model_path}")
                    break
            
            # If not found, try to find any .pt file in models/court/
            if court_model_path is None:
                court_dir = PROJECT_ROOT / "models" / "court"
                print(f"   Checking for any .pt files in: {court_dir}")
                if court_dir.exists():
                    pt_files = list(court_dir.glob("*.pt"))
                    print(f"   Found {len(pt_files)} .pt file(s) in court directory:")
                    for pt_file in pt_files:
                        print(f"     - {pt_file.name}")
                    if pt_files:
                        court_model_path = pt_files[0]  # Use first .pt file found
                        print(f"   ✅ Using: {court_model_path.name}")
                else:
                    print(f"   ✗ Court directory does not exist: {court_dir}")
        
        if court_model_path and court_model_path.exists():
            try:
                # Create a minimal config for court detector
                court_config = {
                    'input_width': 640,
                    'input_height': 360,
                    'low_threshold': 170,
                    'min_radius': 10,
                    'max_radius': 25,
                    'use_refine_kps': True,
                    'use_homography': True
                }
                court_detector = CourtDetector(
                    model_path=str(court_model_path),
                    config=court_config
                )
                if court_detector.model is not None:
                    print("✓ Court detector ready")
                else:
                    print("⚠ Court detector model failed to load (will skip court detection)")
                    court_detector = None
            except Exception as e:
                print(f"⚠ Court detector initialization failed: {e}")
                print("  Continuing without court detection...")
                court_detector = None
        else:
            print("⚠ Court detection model not found (will skip court detection)")
            print(f"  Searched for 'model_tennis_court_det.pt' in:")
            for path in possible_paths:
                exists = "✓" if path.exists() else "✗"
                print(f"    {exists} {path}")
            # Also check if court directory exists
            court_dir = PROJECT_ROOT / "models" / "court"
            if court_dir.exists():
                print(f"  Court directory exists: {court_dir}")
                all_files = list(court_dir.glob('*'))
                print(f"  Files in court directory ({len(all_files)} total):")
                for f in all_files:
                    print(f"    - {f.name} ({'file' if f.is_file() else 'dir'})")
            else:
                print(f"  ✗ Court directory does not exist: {court_dir}")
                print(f"  Make sure the model is at: {PROJECT_ROOT / 'models' / 'court' / 'model_tennis_court_det.pt'}")
    else:
        if not enable_court_detection:
            print("\n3. Court detection disabled by user")
        else:
            print("\n3. Court detection not available")
    
    # Initialize visualizer
    print("\n4. Setting up visualizer...")
    emerald_green_bgr = hex_to_bgr("#50C878")
    visualizer = HeroVideoVisualizer(
        player_color=hex_to_bgr(player_color),
        ball_color=hex_to_bgr(ball_color),
        court_color=emerald_green_bgr,  # Always use emerald green for court
        trail_length=trail_length
    )
    print("✓ Visualizer ready")
    
    # Open video
    print(f"\n5. Opening video: {input_path}")
    cap = cv2.VideoCapture(str(input_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Downscale if requested for speed
    scale_factor = 1.0
    if process_resolution and width > process_resolution:
        scale_factor = process_resolution / width
        process_width = process_resolution
        process_height = int(height * scale_factor)
        print(f"   ⚡ Processing at reduced resolution: {process_width}x{process_height} (scale: {scale_factor:.2f})")
    else:
        process_width = width
        process_height = height
    
    frames_to_process = (total_frames + frame_skip - 1) // frame_skip
    # Estimate time: keypoints-only ~0.5s/frame, full mesh ~60-300s/frame (VERY slow!)
    # Full mesh rendering is extremely slow, especially at higher resolutions
    # Based on actual observation: ~260s/frame at 1280px, so adjust estimates
    if keypoints_only:
        time_per_frame = 0.5
    else:
        # Full mesh: MUCH slower, especially at higher resolution
        # Actual observed: ~260s/frame at 1280px, so:
        # At 720px: ~120-180s/frame, at 960px: ~180-240s/frame, at 1280px: ~240-300s/frame, at full res: ~300-360s/frame
        if scale_factor < 0.5:  # Very downscaled (720px or less)
            time_per_frame = 120.0  # ~2 minutes per frame
        elif scale_factor < 0.6:  # Moderately downscaled (960px)
            time_per_frame = 180.0  # ~3 minutes per frame
        elif scale_factor < 0.8:  # Slightly downscaled (1280px)
            time_per_frame = 260.0  # ~4.3 minutes per frame (observed)
        else:
            time_per_frame = 300.0  # ~5 minutes per frame at full resolution
    
    if ENSEMBLE_DETECTOR_AVAILABLE:
        time_per_frame += 0.5  # TrackNet adds minimal time per frame
    
    if enable_court_detection:
        time_per_frame += 10.0  # Court detection adds time (but we skip it most frames)
    
    estimated_time_minutes = (frames_to_process * time_per_frame) / 60
    estimated_time_hours = estimated_time_minutes / 60
    
    print(f"   Original resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"   Processing resolution: {process_width}x{process_height}")
    print(f"   Original FPS: {original_fps:.2f}")
    print(f"   Total frames: {total_frames}")
    print(f"   Processing every {frame_skip} frame(s) = {frames_to_process} frames")
    if estimated_time_hours >= 1.0:
        print(f"   Estimated time: ~{estimated_time_hours:.1f} hours ({estimated_time_minutes:.0f} minutes)")
    else:
        print(f"   Estimated time: ~{estimated_time_minutes:.1f} minutes")
    
    if not keypoints_only:
        print(f"   ✓ Using full 3D mesh rendering (slower but highest quality)")
    if frame_skip == 1:
        print(f"   ⚠ WARNING: Processing every frame. Use --frame-skip 5 or higher for much faster processing")
    if scale_factor >= 1.0 and not keypoints_only:
        print(f"   ⚠ TIP: Use --process-resolution 1280 to speed up mesh rendering significantly")
    
    # Setup output video writer (use original resolution for output)
    output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try multiple codecs - in Colab, some codecs report success but don't actually work
    # Try H.264 variants first (most compatible), then fallback options
    fourcc_options = ['H264', 'avc1', 'XVID', 'mp4v', 'MJPG']
    fourcc = None
    out = None
    use_ffmpeg_fallback = False
    temp_frame_dir = None
    # Initialize output_fps (used in both VideoWriter and ffmpeg fallback)
    output_fps = original_fps  # Keep original FPS, we'll write frames multiple times
    
    for codec in fourcc_options:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (output_width, output_height))
            # Test if writer is actually working by checking isOpened() AND trying to write a test frame
            if out.isOpened():
                # Try writing a test frame to verify the codec actually works
                # Use contiguous array to match what we'll write during processing
                test_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                test_frame = np.ascontiguousarray(test_frame)
                test_write = out.write(test_frame)
                if test_write:
                    print(f"✓ VideoWriter initialized successfully with codec '{codec}' (test write passed)")
                    # Note: We wrote one black test frame, which is fine - it's negligible
                    break
                else:
                    print(f"⚠ Codec '{codec}' initialized but test write failed, trying next...")
                    out.release()
                    out = None
            else:
                out.release()
                out = None
                print(f"⚠ Codec '{codec}' failed to open, trying next...")
        except Exception as e:
            print(f"⚠ Codec '{codec}' error: {e}")
            if out:
                out.release()
            out = None
            continue
    
    # If all OpenCV codecs failed, use ffmpeg fallback (common in Colab)
    if out is None or not out.isOpened():
        print(f"⚠ All OpenCV codecs failed, using ffmpeg fallback...")
        print(f"   This is common in Colab environments")
        
        # Create temporary directory for frames
        temp_frame_dir = Path(tempfile.mkdtemp(prefix="hero_video_frames_"))
        print(f"   Temporary frame directory: {temp_frame_dir}")
        use_ffmpeg_fallback = True
        out = None  # We'll write frames as images instead
        
        # Verify ffmpeg is available
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode != 0:
                print(f"❌ ERROR: ffmpeg is not available!")
                print(f"   Please install ffmpeg: apt-get install -y ffmpeg")
                cap.release()
                return
            print(f"✓ ffmpeg is available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"❌ ERROR: ffmpeg is not installed!")
            print(f"   Please install ffmpeg: apt-get install -y ffmpeg")
            cap.release()
            return
    
    # Print GPU memory status if using CUDA
    if device == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        gpu_name = torch.cuda.get_device_name(0)
        
        print(f"\n📊 GPU Memory Status:")
        print(f"   GPU: {gpu_name}")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved: {reserved:.2f} GB")
        print(f"   Free: {free:.2f} GB")
        print(f"   Total: {total:.2f} GB")
        
        # Adjust warning threshold based on GPU
        if "A100" in gpu_name:
            if free < 5.0:
                print(f"   ⚠️ WARNING: Low free GPU memory ({free:.2f} GB) for A100!")
                print(f"   Detections may fail. Consider restarting the Colab runtime.")
            else:
                print(f"   ✅ Excellent memory headroom for A100!")
        elif "L4" in gpu_name:
            if free < 3.0:
                print(f"   ⚠️ WARNING: Low free GPU memory ({free:.2f} GB) for L4!")
                print(f"   Detections may fail. Consider restarting the Colab runtime.")
            else:
                print(f"   ✅ Good memory headroom for L4!")
        else:
            if free < 2.0:
                print(f"   ⚠️ WARNING: Very little free GPU memory ({free:.2f} GB)!")
                print(f"   Detections may fail. Consider restarting the Colab runtime.")
    
    # Initialize skeleton visualizer for keypoints
    if keypoints_only:
        # Convert emerald green from BGR to RGB for skeleton visualizer
        emerald_green_rgb = tuple(reversed(hex_to_bgr("#50C878")))  # Always use emerald green
        skeleton_visualizer = SkeletonVisualizer(
            line_width=2,
            radius=4,
            kpt_color=emerald_green_rgb,
            link_color=emerald_green_rgb
        )
        skeleton_visualizer.set_pose_meta(mhr70_pose_info)
    else:
        skeleton_visualizer = None
    
    # Ball trajectory tracking
    ball_trajectory: List[Tuple[int, int]] = []
    
    # Process frames
    print("\n6. Processing frames...")
    print("="*60)
    
    frame_count = 0
    processed_count = 0
    
    # Calculate the last frame number we should process
    last_frame_to_process = (frames_to_process - 1) * frame_skip
    
    with tqdm(total=frames_to_process, desc="Processing") as pbar:
        while cap.isOpened():
            # Safety check: stop if we've read past the last frame we need to process
            if frame_count > last_frame_to_process:
                print(f"\n✓ Reached last frame to process (frame {last_frame_to_process}), stopping")
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Safety check: if we've processed more frames than expected, break
            if processed_count >= frames_to_process:
                print(f"\n✓ Processed expected {frames_to_process} frames, stopping")
                break
            
            # Skip frames if needed
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue  # Don't update progress bar for skipped frames
            
            print(f"\n{'='*80}")
            print(f"[DEBUG] ========== PROCESSING FRAME {frame_count} ==========")
            print(f"[DEBUG Frame {frame_count}] Frame shape: {frame.shape}, dtype: {frame.dtype}")
            print(f"[DEBUG Frame {frame_count}] Frame pixel range: [{frame.min()}, {frame.max()}]")
            
            # Downscale frame if requested
            if scale_factor < 1.0:
                print(f"[DEBUG Frame {frame_count}] Downscaling frame from {frame.shape[:2]} to ({process_height}, {process_width})")
                frame = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_AREA)
                print(f"[DEBUG Frame {frame_count}] After downscale: {frame.shape}")
            
            # Clear GPU cache before processing to prevent memory accumulation
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Process with SAM-3d-body
            print(f"\n[DEBUG Frame {frame_count}] Starting SAM-3d-body processing...")
            try:
                # Convert BGR to RGB for SAM-3d-body
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                print(f"[DEBUG Frame {frame_count}] Frame converted to RGB, shape: {frame_rgb.shape}")
                
                # Run inference with lower thresholds for better multi-person detection
                print(f"[DEBUG Frame {frame_count}] Calling estimator.process_one_image()...")
                # Check if detector is being used
                if hasattr(estimator, 'detector') and estimator.detector is not None:
                    print(f"[DEBUG Frame {frame_count}] Using human detector: {type(estimator.detector).__name__}")
                else:
                    print(f"[DEBUG Frame {frame_count}] ⚠️ No human detector - will process full image as single person")
                
                outputs = estimator.process_one_image(
                    frame_rgb,
                    inference_type="full" if not keypoints_only else "keypoints_only",
                    bbox_thr=0.15,  # Even lower threshold to detect more people (was 0.25, default is 0.5)
                    nms_thr=0.5,    # Higher NMS to keep separate people (was 0.4, default is 0.3)
                )
                print(f"[DEBUG Frame {frame_count}] SAM-3d-body returned {len(outputs)} output(s)")
                
                # Clear GPU cache after processing
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if len(outputs) > 0:
                    print(f"[DEBUG Frame {frame_count}] First output keys: {list(outputs[0].keys())}")
                    if "pred_keypoints_2d" in outputs[0]:
                        kp_shape = outputs[0]["pred_keypoints_2d"].shape
                        print(f"[DEBUG Frame {frame_count}] Keypoints shape: {kp_shape}")
                        print(f"[DEBUG Frame {frame_count}] Sample keypoint: {outputs[0]['pred_keypoints_2d'][0] if kp_shape[0] > 0 else 'N/A'}")
                else:
                    print(f"[DEBUG Frame {frame_count}] ⚠️ NO PLAYER DETECTIONS!")
            except Exception as e:
                print(f"[DEBUG Frame {frame_count}] ❌ SAM-3d-body processing failed: {e}")
                import traceback
                traceback.print_exc()
                outputs = []
            
            # Process with ball detector
            print(f"[DEBUG Frame {frame_count}] Starting ball detection...")
            ball_detection = None
            try:
                if ENSEMBLE_DETECTOR_AVAILABLE:
                    print(f"[DEBUG Frame {frame_count}] Using TrackNet ball detector")
                    ball_detection = ball_detector.detect_ball(frame, text_prompt=ball_prompt)
                else:
                    print(f"[DEBUG Frame {frame_count}] ⚠️ TrackNet Ball detector not available!")
                
                if ball_detection:
                    center, conf, mask = ball_detection
                    print(f"[DEBUG Frame {frame_count}] ✅ Ball detected at {center} (confidence: {conf:.2f})")
                    if mask is not None:
                        print(f"[DEBUG Frame {frame_count}] Ball mask shape: {mask.shape}, non-zero pixels: {np.count_nonzero(mask)}")
                else:
                    print(f"[DEBUG Frame {frame_count}] ⚠️ No ball detected")
            except Exception as e:
                print(f"[DEBUG Frame {frame_count}] ❌ Ball detection error: {e}")
                import traceback
                traceback.print_exc()
                pass
            
            # Process with court detector (optional, skip every few frames for speed)
            print(f"[DEBUG Frame {frame_count}] Checking court detection...")
            court_keypoints = None
            if enable_court_detection and court_detector and court_detector.model is not None:
                print(f"[DEBUG Frame {frame_count}] Court detector available, enable_court_detection={enable_court_detection}")
                # Only run court detection every 10th processed frame (saves significant time)
                if processed_count % 10 == 0:
                    print(f"[DEBUG Frame {frame_count}] Running court detection (every 10th frame)...")
                    try:
                        court_keypoints = court_detector.detect_court_in_frame(frame)
                        print(f"[DEBUG Frame {frame_count}] Court detector returned: {type(court_keypoints)}")
                        # Filter out None points
                        if court_keypoints:
                            court_keypoints = [
                                (kp[0], kp[1]) if kp and kp[0] is not None and kp[1] is not None else None
                                for kp in court_keypoints
                            ]
                            valid_points = sum(1 for kp in court_keypoints if kp is not None)
                            print(f"[DEBUG Frame {frame_count}] ✅ Court detected: {valid_points} valid keypoints out of {len(court_keypoints)}")
                        else:
                            print(f"[DEBUG Frame {frame_count}] ⚠️ Court detector returned None/empty")
                    except Exception as e:
                        print(f"[DEBUG Frame {frame_count}] ❌ Court detection error: {e}")
                        court_keypoints = None
                else:
                    print(f"[DEBUG Frame {frame_count}] Skipping court detection (not every 10th frame)")
                    # Reuse previous court detection (courts don't move much)
                    pass  # Will use None, which is fine
            else:
                print(f"[DEBUG Frame {frame_count}] Court detection disabled or unavailable")
            
            # Update ball trajectory
            print(f"[DEBUG Frame {frame_count}] Updating ball trajectory...")
            if ball_detection:
                center, confidence, mask = ball_detection
                ball_trajectory.append(center)
                print(f"[DEBUG Frame {frame_count}] Added ball to trajectory. Trajectory length: {len(ball_trajectory)}")
                # Keep only recent trajectory points
                if len(ball_trajectory) > trail_length:
                    ball_trajectory.pop(0)
                    print(f"[DEBUG Frame {frame_count}] Trimmed trajectory to {trail_length} points")
            else:
                # Keep trajectory even if ball not detected (fade out)
                if len(ball_trajectory) > 0:
                    ball_trajectory.pop(0)
                    print(f"[DEBUG Frame {frame_count}] No ball, fading trajectory. Remaining: {len(ball_trajectory)}")
            print(f"[DEBUG Frame {frame_count}] Final trajectory length: {len(ball_trajectory)}")
            
            # Create visualization
            print(f"[DEBUG Frame {frame_count}] Creating visualization...")
            print(f"[DEBUG Frame {frame_count}]   - keypoints_only: {keypoints_only}")
            print(f"[DEBUG Frame {frame_count}]   - MESH_VISUALIZER_AVAILABLE: {MESH_VISUALIZER_AVAILABLE}")
            print(f"[DEBUG Frame {frame_count}]   - len(outputs): {len(outputs)}")
            print(f"[DEBUG Frame {frame_count}]   - ball_detection: {ball_detection is not None}")
            print(f"[DEBUG Frame {frame_count}]   - len(ball_trajectory): {len(ball_trajectory)}")
            print(f"[DEBUG Frame {frame_count}]   - court_keypoints: {court_keypoints is not None}")
            
            # Always use create_frame which creates black background and draws all overlays
            # This works for both mesh and keypoints-only modes, and handles empty outputs
            print(f"[DEBUG Frame {frame_count}] Calling visualizer.create_frame()...")
            vis_frame = visualizer.create_frame(
                frame=frame,
                player_outputs=outputs,
                ball_detection=ball_detection,
                ball_trajectory=ball_trajectory,
                skeleton_visualizer=skeleton_visualizer if keypoints_only else None,
                court_keypoints=court_keypoints
            )
            print(f"[DEBUG Frame {frame_count}] create_frame() returned frame shape: {vis_frame.shape}, dtype: {vis_frame.dtype}")
            print(f"[DEBUG Frame {frame_count}] Frame pixel range: [{vis_frame.min()}, {vis_frame.max()}]")
            print(f"[DEBUG Frame {frame_count}] Non-zero pixels: {np.count_nonzero(vis_frame)} / {vis_frame.size}")
            
            # If we have mesh mode and outputs, overlay the 3D mesh on top of the skeleton frame
            if not keypoints_only and MESH_VISUALIZER_AVAILABLE and len(outputs) > 0:
                print(f"[DEBUG Frame {frame_count}] Attempting mesh rendering...")
                try:
                    # Get faces for mesh rendering from estimator (not from mhr70 module)
                    if not hasattr(estimator, 'faces') or estimator.faces is None:
                        raise AttributeError("estimator.faces is not available")
                    faces = estimator.faces
                    print(f"[DEBUG Frame {frame_count}] Faces loaded from estimator, shape: {faces.shape}")
                    # Convert current vis_frame (which has ball/court/skeleton) to RGB for mesh visualizer
                    vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                    print(f"[DEBUG Frame {frame_count}] Converted to RGB, calling visualize_sample_together_emerald()...")
                    # Render mesh with emerald green (this will composite mesh on top of existing overlays)
                    mesh_frame_rgb = visualize_sample_together_emerald(vis_frame_rgb, outputs, faces)
                    print(f"[DEBUG Frame {frame_count}] Mesh visualization returned, shape: {mesh_frame_rgb.shape}, dtype: {mesh_frame_rgb.dtype}")
                    print(f"[DEBUG Frame {frame_count}] Mesh frame pixel range: [{mesh_frame_rgb.min()}, {mesh_frame_rgb.max()}]")
                    print(f"[DEBUG Frame {frame_count}] Mesh frame non-zero pixels: {np.count_nonzero(mesh_frame_rgb)} / {mesh_frame_rgb.size}")
                    # Convert back to BGR
                    vis_frame = cv2.cvtColor(mesh_frame_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    print(f"[DEBUG Frame {frame_count}] Converted back to BGR, final shape: {vis_frame.shape}")
                except Exception as e:
                    # If mesh rendering fails, keep the skeleton frame we already have
                    print(f"[DEBUG Frame {frame_count}] ❌ Mesh rendering failed (using skeleton): {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[DEBUG Frame {frame_count}] Skipping mesh rendering (keypoints_only={keypoints_only}, MESH_AVAILABLE={MESH_VISUALIZER_AVAILABLE}, outputs={len(outputs)})")
            
            # Upscale back to original resolution if we downscaled
            if scale_factor < 1.0:
                print(f"[DEBUG Frame {frame_count}] Upscaling from {vis_frame.shape[:2]} to ({output_height}, {output_width})")
                vis_frame = cv2.resize(vis_frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
                print(f"[DEBUG Frame {frame_count}] Upscaled frame shape: {vis_frame.shape}")
            
            print(f"[DEBUG Frame {frame_count}] Final frame before writing - shape: {vis_frame.shape}, dtype: {vis_frame.dtype}")
            print(f"[DEBUG Frame {frame_count}] Final frame pixel range: [{vis_frame.min()}, {vis_frame.max()}]")
            print(f"[DEBUG Frame {frame_count}] Final frame non-zero pixels: {np.count_nonzero(vis_frame)} / {vis_frame.size}")
            
            # Write frame(s) - either via VideoWriter or save as images for ffmpeg
            if use_ffmpeg_fallback:
                # Save frame as image (write multiple times if frame_skip > 1)
                print(f"[DEBUG Frame {frame_count}] Saving frame as image(s) for ffmpeg (frame_skip={frame_skip})...")
                for i in range(frame_skip):
                    frame_filename = temp_frame_dir / f"frame_{processed_count * frame_skip + i:08d}.png"
                    # Ensure frame is contiguous
                    if not vis_frame.flags['C_CONTIGUOUS']:
                        vis_frame = np.ascontiguousarray(vis_frame)
                    cv2.imwrite(str(frame_filename), vis_frame)
                print(f"[DEBUG Frame {frame_count}] ✅ Frame saved as image(s)")
            else:
                # Check VideoWriter status before writing
                if not out.isOpened():
                    print(f"[DEBUG Frame {frame_count}] ❌ CRITICAL: VideoWriter is not open! Cannot write frames.")
                    break
                
                # Write frame multiple times to maintain original playback speed
                # If we process every Nth frame, write each frame N times
                print(f"[DEBUG Frame {frame_count}] Writing frame {frame_skip} time(s)...")
                write_success = True
                for i in range(frame_skip):
                    # Ensure frame is contiguous in memory (some codecs require this)
                    if not vis_frame.flags['C_CONTIGUOUS']:
                        vis_frame = np.ascontiguousarray(vis_frame)
                    
                    success = out.write(vis_frame)
                    if not success:
                        print(f"[DEBUG Frame {frame_count}] ❌ ERROR: VideoWriter.write() returned False on iteration {i}")
                        write_success = False
                        # Check if writer is still open
                        if not out.isOpened():
                            print(f"[DEBUG Frame {frame_count}] ❌ VideoWriter is no longer open!")
                            break
                if write_success:
                    print(f"[DEBUG Frame {frame_count}] ✅ Frame written successfully")
                else:
                    print(f"[DEBUG Frame {frame_count}] ⚠️ Frame write had issues but continuing...")
            
            processed_count += 1
            frame_count += 1
            pbar.update(1)
    
    # Cleanup
    cap.release()
    
    # If using ffmpeg fallback, combine frames into video
    if use_ffmpeg_fallback:
        print("\n" + "="*60)
        print("Combining frames with ffmpeg...")
        print("="*60)
        
        # Count actual frames saved
        frame_files = sorted(temp_frame_dir.glob("frame_*.png"))
        num_frames = len(frame_files)
        print(f"Found {num_frames} frame images to combine")
        
        if num_frames == 0:
            print("❌ ERROR: No frames were saved!")
            if temp_frame_dir and temp_frame_dir.exists():
                shutil.rmtree(temp_frame_dir)
            return
        
        # Use ffmpeg to combine frames into video
        # Pattern: frame_%08d.png (8-digit zero-padded frame numbers)
        frame_pattern = str(temp_frame_dir / "frame_%08d.png")
        
        # Build ffmpeg command
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-framerate', str(output_fps),  # Input frame rate
            '-i', frame_pattern,  # Input pattern
            '-c:v', 'libx264',  # H.264 codec
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            '-crf', '18',  # High quality (lower = better quality, 18-23 is good)
            '-preset', 'medium',  # Encoding speed vs compression
            str(output_path)
        ]
        
        print(f"Running ffmpeg command:")
        print(f"  {' '.join(ffmpeg_cmd)}")
        
        try:
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ ffmpeg completed successfully")
            if result.stderr:
                # ffmpeg outputs progress to stderr
                print(f"ffmpeg output: {result.stderr[-500:]}")  # Last 500 chars
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: ffmpeg failed!")
            print(f"   Return code: {e.returncode}")
            print(f"   stderr: {e.stderr}")
            print(f"   stdout: {e.stdout}")
            # Clean up temp directory
            if temp_frame_dir and temp_frame_dir.exists():
                shutil.rmtree(temp_frame_dir)
            return
        
        # Clean up temporary frame directory
        print(f"Cleaning up temporary frame directory...")
        if temp_frame_dir and temp_frame_dir.exists():
            shutil.rmtree(temp_frame_dir)
            print(f"✓ Temporary files cleaned up")
    else:
        # Normal VideoWriter cleanup
        if out:
            out.release()
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"Processed {processed_count} frames (every {frame_skip} frame(s))")
    print(f"Output saved to: {output_path}")
    print(f"Output resolution: {output_width}x{output_height}")
    print(f"Output FPS: {output_fps:.2f} (original: {original_fps:.2f}, frames written: {processed_count * frame_skip})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate hero video with player and ball tracking"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output video file"
    )
    parser.add_argument(
        "--ball-prompt",
        type=str,
        default="tennis ball",
        help="Text prompt for ball detection (unused with TrackNet, kept for compatibility)"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        help="Process every Nth frame (default: 5 = every 5th frame for speed)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video FPS (default: 30.0)"
    )
    parser.add_argument(
        "--player-color",
        type=str,
        default="#50C878",
        help="Hex color for player skeletons (default: #50C878 - emerald green)"
    )
    parser.add_argument(
        "--ball-color",
        type=str,
        default="#50C878",
        help="Hex color for ball and trajectory (default: #50C878 - emerald green)"
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=30,
        help="Number of frames in ball trajectory trail (default: 30)"
    )
    parser.add_argument(
        "--keypoints-only",
        action="store_true",
        help="Use fast keypoints-only mode (no 3D mesh rendering) - 10x+ faster but no mesh"
    )
    parser.add_argument(
        "--process-resolution",
        type=int,
        default=720,
        help="Downscale video to this width for processing (default: 720 for full mesh - it's VERY slow!). Higher = slower but better quality. Use 0 for original resolution."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--no-court-detection",
        action="store_true",
        help="Disable court detection (skip if model/dependencies unavailable)"
    )
    parser.add_argument(
        "--court-model",
        type=str,
        default=None,
        help="Path to court detection model file (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    process_video(
        input_path=Path(args.input),
        output_path=Path(args.output),
        ball_prompt=args.ball_prompt,
        frame_skip=args.frame_skip,
        fps=args.fps,
        player_color=args.player_color,
        ball_color=args.ball_color,
        trail_length=args.trail_length,
        keypoints_only=args.keypoints_only,
        device=None if args.device == "auto" else args.device,
        enable_court_detection=not args.no_court_detection,
        court_model_path=Path(args.court_model) if args.court_model else None,
        process_resolution=args.process_resolution if args.process_resolution > 0 else None,
    )


if __name__ == "__main__":
    main()
