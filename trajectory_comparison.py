#!/usr/bin/env python3
"""
Clean trajectory comparison figure for paper
"""
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['grid.alpha'] = 0.15

def load_trajectories():
    """Load GPS and KISS-ICP trajectories"""
    # Load GPS
    gps_data = np.loadtxt("/home/taewook/ISIS/gps_trajectory.txt", comments='#')
    gps_utm = gps_data[:, :2]
    
    # Load KISS-ICP
    kissicp_poses = []
    with open("/home/taewook/ISIS/kiss_icp_poses.txt", 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 12:
                pose = np.array(vals).reshape(3, 4)
                kissicp_poses.append([pose[0, 3], pose[1, 3]])
    
    kissicp_traj = np.array(kissicp_poses)
    
    return gps_utm, kissicp_traj

def trim_and_align(gps_traj, kissicp_traj):
    """Trim GPS and align trajectories"""
    # Calculate cumulative distances
    gps_dists = np.insert(np.cumsum(np.linalg.norm(np.diff(gps_traj, axis=0), axis=1)), 0, 0)
    kissicp_dists = np.insert(np.cumsum(np.linalg.norm(np.diff(kissicp_traj, axis=0), axis=1)), 0, 0)
    
    # Trim GPS from beginning
    length_to_trim = gps_dists[-1] - kissicp_dists[-1]
    start_idx = np.searchsorted(gps_dists, length_to_trim)
    gps_trimmed = gps_traj[start_idx:]
    
    # Convert to relative coordinates
    gps_rel = gps_trimmed - gps_trimmed[0]
    
    # Align KISS-ICP
    from scipy.optimize import minimize
    
    def transform_2d(points, theta, tx, ty):
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        return (points @ R.T) + np.array([tx, ty])
    
    def objective(params):
        theta, tx, ty = params
        kissicp_transformed = transform_2d(kissicp_traj, theta, tx, ty)
        
        # Match endpoints
        endpoint_error = np.linalg.norm(kissicp_transformed[-1] - gps_rel[-1])
        startpoint_error = np.linalg.norm(kissicp_transformed[0] - gps_rel[0])
        
        # Sample middle points
        n_samples = min(20, len(kissicp_traj))
        indices = np.linspace(0, len(kissicp_traj)-1, n_samples).astype(int)
        gps_indices = np.linspace(0, len(gps_rel)-1, n_samples).astype(int)
        
        middle_errors = []
        for i, j in zip(indices, gps_indices):
            middle_errors.append(np.linalg.norm(kissicp_transformed[i] - gps_rel[j]))
        
        return endpoint_error * 10 + startpoint_error * 10 + np.mean(middle_errors)
    
    # Find best alignment
    best_params = None
    best_error = float('inf')
    
    for init_theta in np.linspace(0, 2*np.pi, 8):
        init_tx = gps_rel[-1, 0] - kissicp_traj[-1, 0]
        init_ty = gps_rel[-1, 1] - kissicp_traj[-1, 1]
        
        result = minimize(objective, [init_theta, init_tx, init_ty],
                         method='L-BFGS-B',
                         bounds=[(-2*np.pi, 2*np.pi), (-2000, 2000), (-2000, 2000)])
        if result.fun < best_error:
            best_error = result.fun
            best_params = result.x
    
    theta, tx, ty = best_params
    kissicp_aligned = transform_2d(kissicp_traj, theta, tx, ty)
    
    return gps_rel, kissicp_aligned

def create_clean_figure(gps_rel, kissicp_aligned):
    """Create clean trajectory comparison figure"""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    # Plot trajectories
    ax.plot(gps_rel[:, 0], gps_rel[:, 1], 
           color='#1565C0', linewidth=2.0, alpha=0.9, label='GPS Reference')
    ax.plot(kissicp_aligned[:, 0], kissicp_aligned[:, 1], 
           color='#C62828', linewidth=1.5, alpha=0.8, linestyle='--', label='SLAM Trajectory')
    
    # Mark start and end with black dots
    ax.scatter(gps_rel[0, 0], gps_rel[0, 1], 
              s=100, c='black', marker='o', zorder=5)
    ax.scatter(gps_rel[-1, 0], gps_rel[-1, 1], 
              s=100, c='black', marker='o', zorder=5)
    
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.grid(True, alpha=0.15)
    ax.set_aspect('equal')
    
    # Legend without start/end labels
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    
    # Clean axis styling
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.2)
    
    ax.tick_params(axis='both', which='major', labelsize=11,
                  length=5, width=1, color='#333333',
                  direction='in', top=True, right=True)
    
    plt.tight_layout()
    
    output_path = '/home/taewook/ISIS/trajectory_comparison_clean.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path

def main():
    # Load trajectories
    gps_traj, kissicp_traj = load_trajectories()
    
    # Trim and align
    gps_rel, kissicp_aligned = trim_and_align(gps_traj, kissicp_traj)
    
    # Create figure
    output = create_clean_figure(gps_rel, kissicp_aligned)
    
    print(f"✅ Figure saved: {output}")

if __name__ == "__main__":
    main()
