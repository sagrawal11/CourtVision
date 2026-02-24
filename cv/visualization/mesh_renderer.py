"""
Custom mesh visualization with emerald green color for hero videos.
Based on SAM-3d-body visualization utilities.
"""

import numpy as np
import cv2
try:
    from sam_3d_body.visualization.renderer import Renderer
    from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
    from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info

    # Emerald green in RGB (normalized 0-1)
    # #50C878 = RGB(80, 200, 120) = normalized (0.314, 0.784, 0.471)
    EMERALD_GREEN = (80 / 255.0, 200 / 255.0, 120 / 255.0)  # #50C878

    visualizer = SkeletonVisualizer(line_width=2, radius=5)
    visualizer.set_pose_meta(mhr70_pose_info)
    SAM_READY = True
except ImportError:
    SAM_READY = False


def render_mesh(img_cv2, outputs, faces):
    """
    Render mesh on top of existing image (which should have ball/court already drawn).
    Based on visualize_sample_together but with custom emerald green color.
    Composites the mesh on top of the input image.
    if not SAM_READY:
        print("Warning: SAM-3d-body not ready, returning original image.")
        return img_cv2

    """
    print(f"[DEBUG mesh_visualizer] visualize_sample_together_emerald called")
    print(f"[DEBUG mesh_visualizer] Input image shape: {img_cv2.shape}, dtype: {img_cv2.dtype}")
    print(f"[DEBUG mesh_visualizer] Input image pixel range: [{img_cv2.min()}, {img_cv2.max()}]")
    print(f"[DEBUG mesh_visualizer] Input image non-zero pixels: {np.count_nonzero(img_cv2)} / {img_cv2.size}")
    print(f"[DEBUG mesh_visualizer] Number of outputs: {len(outputs)}")
    
    if len(outputs) == 0:
        # Return input image unchanged if no detections
        print(f"[DEBUG mesh_visualizer] No outputs, returning input unchanged")
        return img_cv2
    
    # Use the input image as base (it already has ball/court drawn)
    # Create separate frames for keypoints and mesh rendering
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()
    print(f"[DEBUG mesh_visualizer] Created copies for keypoints and mesh")

    # First, sort by depth, furthest to closest
    all_depths = np.stack([tmp['pred_cam_t'] for tmp in outputs], axis=0)[:, 2]
    outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

    # Then, draw all keypoints with emerald green
    # Update visualizer colors to emerald green
    # Convert normalized color to 0-255 range and create RGB tuple
    emerald_green_rgb = tuple(int(c * 255) for c in reversed(EMERALD_GREEN))  # RGB in 0-255 range: (120, 200, 80)
    # Create list of 70 colors (one per keypoint) - all the same emerald green
    num_keypoints = outputs_sorted[0]["pred_keypoints_2d"].shape[0]  # Should be 70 for MHR70
    # CRITICAL: visualizer expects kpt_color to be a list/array with length matching num_keypoints (70)
    # Each element should be a tuple (R, G, B) or a color value
    # Create list of 70 identical RGB tuples
    kpt_color_list = [emerald_green_rgb for _ in range(num_keypoints)]
    visualizer.kpt_color = kpt_color_list
    print(f"[DEBUG mesh_visualizer] Set kpt_color: type={type(visualizer.kpt_color)}, len={len(visualizer.kpt_color)}, first_element={visualizer.kpt_color[0] if len(visualizer.kpt_color) > 0 else 'N/A'}")
    # For link_color, create a list matching the number of skeleton links
    if visualizer.skeleton is not None:
        num_skeleton_links = len(visualizer.skeleton)
        link_color_list = [emerald_green_rgb for _ in range(num_skeleton_links)]
        visualizer.link_color = link_color_list
        print(f"[DEBUG mesh_visualizer] Set link_color: type={type(visualizer.link_color)}, len={len(visualizer.link_color)}")
    else:
        # If no skeleton, link_color can be a single color (string or tuple) - but this shouldn't happen if set_pose_meta was called
        visualizer.link_color = emerald_green_rgb
    
    for pid, person_output in enumerate(outputs_sorted):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_keypoints = visualizer.draw_skeleton(img_keypoints, keypoints_2d)

    # Combine all meshes together (original approach works better for alignment)
    # But use proper camera center based on image dimensions
    all_pred_vertices = []
    all_faces = []
    for pid, person_output in enumerate(outputs_sorted):
        all_pred_vertices.append(person_output["pred_vertices"] + person_output["pred_cam_t"])
        all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    # Calculate camera translation from all vertices (better than using just last two)
    # Use the center of all vertices for better alignment
    fake_pred_cam_t = np.mean(all_pred_vertices, axis=0)
    all_pred_vertices = all_pred_vertices - fake_pred_cam_t
    
    # Use average focal length for rendering (more stable than using one person's)
    avg_focal_length = np.mean([out["focal_length"] for out in outputs_sorted])
    print(f"[DEBUG mesh_visualizer] Rendering {len(outputs_sorted)} person(s) with avg focal_length={avg_focal_length:.2f}")
    print(f"[DEBUG mesh_visualizer] Camera translation: {fake_pred_cam_t}")
    print(f"[DEBUG mesh_visualizer] Image dimensions: {img_mesh.shape[1]}x{img_mesh.shape[0]}")
    
    # Render with proper camera center (image center)
    renderer = Renderer(focal_length=avg_focal_length, faces=all_faces)
    camera_center = [img_mesh.shape[1] / 2.0, img_mesh.shape[0] / 2.0]
    print(f"[DEBUG mesh_visualizer] Camera center: {camera_center}")
    
    img_mesh = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            img_mesh,
            mesh_base_color=EMERALD_GREEN,
            scene_bg_color=(0, 0, 0),
            camera_center=camera_center,  # Explicitly set camera center
        )
        * 255
    ).astype(np.uint8)
    print(f"[DEBUG mesh_visualizer] Renderer returned, shape: {img_mesh.shape}, dtype: {img_mesh.dtype}")
    print(f"[DEBUG mesh_visualizer] Rendered mesh pixel range: [{img_mesh.min()}, {img_mesh.max()}]")
    print(f"[DEBUG mesh_visualizer] Rendered mesh non-zero pixels: {np.count_nonzero(img_mesh)} / {img_mesh.size}")
    
    # Combine keypoints with mesh
    # Keypoints were drawn on img_keypoints (black background with green keypoints)
    # Mesh was rendered on img_mesh (black background with green mesh)
    # We need to combine them - use maximum to overlay keypoints on mesh
    # Both are on black backgrounds, so max will show both overlays
    # Note: img_keypoints is RGB, img_mesh is RGB, both are uint8
    print(f"[DEBUG mesh_visualizer] Combining keypoints and mesh using np.maximum")
    print(f"[DEBUG mesh_visualizer] Keypoints frame non-zero pixels: {np.count_nonzero(img_keypoints)}")
    print(f"[DEBUG mesh_visualizer] Mesh frame non-zero pixels: {np.count_nonzero(img_mesh)}")
    combined = np.maximum(img_mesh, img_keypoints)
    print(f"[DEBUG mesh_visualizer] Combined frame non-zero pixels: {np.count_nonzero(combined)} / {combined.size}")
    print(f"[DEBUG mesh_visualizer] Combined frame pixel range: [{combined.min()}, {combined.max()}]")
    
    return combined
