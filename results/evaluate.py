import os
import argparse
import json
import yaml
import numpy as np
import csv
from datetime import datetime
from tqdm import tqdm

def judge_success(data, collision_th, suc_dis, shortest_traj):
    # Calculate length of the shortest trajectory (Ground Truth)
    shortest_dis = 0
    for i in range(len(shortest_traj)-1):
        shortest_dis += np.linalg.norm(shortest_traj[i][:2] - shortest_traj[i+1][:2])
 
    goal = shortest_traj[-1][:2]
    d_0 = np.linalg.norm(data[0][:2] - goal)
    collision_num = 0
    arrive = False
    cul_dis = 0   
    
    # Iterate through the actual trajectory
    for i, d in enumerate(data):
        d_t = np.linalg.norm(d[:2] - goal)
        if i > 0:
            cul_dis += np.linalg.norm(d[:2] - data[i-1][:2])
        
        # Collision handling: stop if too many collisions (if using strict threshold)
        if d[4] == 1:
            collision_num += 1
        if collision_num >= collision_th:
            break
            
        # Success handling: stop if close enough to goal
        if np.linalg.norm(d[:2] - goal) < suc_dis:
            arrive = True
            break
            
    # Final distance to goal (if not arrived, it's the distance at breakdown point)
    final_dist = np.linalg.norm(data[min(i, len(data)-1)][:2] - goal)
    
    return arrive, collision_num, shortest_dis, cul_dis, d_0, final_dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="vgc", help="Name of the experiment folder in results/")
    parser.add_argument("--config", default="experiments/alpha_0p7_depth_route_full.yaml", help="Evaluation config containing data_root")
    parser.add_argument("--require-full-benchmark", action="store_true", help="Fail unless all fixed 500 episodes and references are valid")
    args = parser.parse_args()
    with open(args.config) as f:
        data_root = yaml.safe_load(f)["data_root"]

    result_dir = os.path.join("results/exps", args.exp_name)
    traj_dir = os.path.join(result_dir, "trajectory")
    
    if not os.path.exists(traj_dir):
        print(f"Error: Directory {traj_dir} not found.")
        raise SystemExit(1)

    # Configuration for evaluation
    # Report SR/SPL for every requested collision-budget and goal-distance pair.
    collision_th_list = [10, 30, 50]  # Maximum allowed collision steps
    suc_dis_list = [0.25, 0.30, 0.35] # Maximum final distance to the goal (m)
    
    # Store metrics for the "Standard" configuration to log to CSV
    # Standard CSV row: collision_th=10, suc_dis=0.30.
    std_metrics = None
    
    # Pre-load trajectory files
    traj_files = [f for f in os.listdir(traj_dir) if f.endswith(".txt")]
    if not traj_files:
        print("No trajectory files found.")
        raise SystemExit(1)

    if args.require_full_benchmark:
        with open("experiments/pointgoal_v3_results.json") as stream:
            tasks = json.load(stream)["tasks"]
        expected = {task["filename"] for task in tasks}
        if set(traj_files) != expected:
            raise ValueError(f"Benchmark mismatch: {len(expected - set(traj_files))} missing, {len(set(traj_files) - expected)} extra episodes")
        for task in tasks:
            rollout = np.loadtxt(os.path.join(traj_dir, task["filename"]), ndmin=2)
            ref = os.path.join(data_root, "scenes_117", "test", task["scene"], f"traj_{task['id']}", f"traj_{task['id']}.npy")
            reference = np.load(ref)
            if rollout.size == 0 or rollout.shape[1] < 5 or not np.isfinite(rollout).all() or len(reference) < 2 or not np.isfinite(reference).all():
                raise ValueError("Invalid benchmark trajectory: " + task["filename"])

    print(f"\nEvaluating experiment: {args.exp_name}")
    print(f"Found {len(traj_files)} trajectories.")

    # Loop through all threshold combinations
    for c_th in collision_th_list:
        for s_dis in suc_dis_list:
            print('-------------------------------------')
            print(f'collision count th: {c_th}, arrive th: {s_dis}')

            arrives = []
            collision_nums = []
            SPL = []
            SoftSPL = []
            dists_to_goal = []

            for f in tqdm(traj_files, leave=False):
                try:
                    data = np.loadtxt(os.path.join(traj_dir, f))
                    if data.ndim == 1: data = data.reshape(1, -1)
                except:
                    continue

                # Parse filename to find ground truth path
                # Format: scene_floor_traj_id.txt (e.g. Eudora_0_traj_1.txt)
                parts = f.replace(".txt", "").split("_")
                if len(parts) < 4: continue
                
                scene_floor = f"{parts[0]}_{parts[1]}"
                traj_name = f"{parts[2]}_{parts[3]}"
                
                # Ground truth path
                shortest_traj_path = os.path.join(data_root, "scenes_117", "test", scene_floor, traj_name, traj_name + ".npy")
                if not os.path.exists(shortest_traj_path):
                    continue
                
                shortest_traj = np.load(shortest_traj_path)

                arrive, collision_num, shortest_dis, cul_dis, d_0, d_t = judge_success(
                    data, c_th, s_dis, shortest_traj
                )

                arrives.append(arrive)
                collision_nums.append(collision_num)
                
                # SPL = Success * (shortest / max(actual, shortest))
                spl_val = arrive * shortest_dis / max(cul_dis, shortest_dis) if max(cul_dis, shortest_dis) > 0 else 0
                SPL.append(spl_val)
                
                # SoftSPL = (1 - final_dist / init_dist) * ...
                soft_spl_val = (1 - min(d_t / d_0, 1.0)) * shortest_dis / max(cul_dis, shortest_dis) if max(cul_dis, shortest_dis) > 0 and d_0 > 0 else 0
                SoftSPL.append(max(0, soft_spl_val))
                
                dists_to_goal.append(d_t)

            if not arrives: continue

            mean_sr = np.mean(arrives)
            mean_spl = np.mean(SPL)
            mean_softspl = np.mean(SoftSPL)
            mean_col = np.mean(collision_nums)
            mean_dist = np.mean(dists_to_goal)
            # col_rate: percentage of trajs that had at least 1 collision
            # Note: collision_nums is the COUNT of collision steps. 
            # To be consistent with standard metrics, usually we report Collision Rate (>0).
            # But the original script reported "collision mean nums". I will report both.
            
            print(f"{args.exp_name}:-----SR: {mean_sr:.4f} SPL: {mean_spl:.4f} SoftSPL: {mean_softspl:.4f}")
            print(f"collision mean nums: {mean_col:.2f}")

            # Capture metrics for CSV logging (standard config: c_th=10, s_dis=0.30).
            if c_th == 10 and s_dis == 0.3:
                # Calculate Collision Rate (binary) for CSV
                col_rate = np.mean([1 if c > 0 else 0 for c in collision_nums])
                std_metrics = {
                    "SR": mean_sr,
                    "SPL": mean_spl,
                    "Collision": col_rate, # Rate, not mean num
                    "Avg_Dist": mean_dist,
                    "Count": len(arrives)
                }

    # --- Log to CSV ---
    if std_metrics:
        summary_file = "evaluation_summary.csv"
        file_exists = os.path.isfile(summary_file)
        with open(summary_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Time", "Exp_Name", "SR", "SPL", "Collision_Rate", "Avg_Dist", "Count"])
            
            writer.writerow([
                datetime.now().strftime("%m-%d %H:%M"),
                args.exp_name,
                f"{std_metrics['SR']:.4f}",
                f"{std_metrics['SPL']:.4f}",
                f"{std_metrics['Collision']:.4f}",
                f"{std_metrics['Avg_Dist']:.4f}",
                std_metrics['Count']
            ])
        print(f"\n[Standard Metrics (c_th=10, s_dis=0.30)] appended to {summary_file}")

if __name__ == "__main__":
    main()
