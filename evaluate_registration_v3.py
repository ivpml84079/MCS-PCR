import numpy as np
import open3d as o3d
import yaml
import pandas as pd
from math import acos, degrees


def rotation_error(R_gt, R_est):
    """Compute rotation error in degrees."""
    R_diff = R_gt.T @ R_est
    trace_val = np.trace(R_diff)
    trace_val = np.clip((trace_val - 1) / 2, -1.0, 1.0)
    return degrees(acos(trace_val))


def translation_error(t_gt, t_est):
    """Compute translation error."""
    return np.linalg.norm(t_gt - t_est)


def compute_rmse_and_mee(source_points, T_gt, T_est):
    """Compute RMSE and MeE after transforming source using GT and Est."""
    src_gt = (T_gt @ source_points.T).T[:, :3]
    src_est = (T_est @ source_points.T).T[:, :3]

    errors = np.linalg.norm(src_gt - src_est, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))
    mee = np.median(errors)
    return rmse, mee


def load_transforms(filename, n):
    """Load n transformation matrices from txt file."""
    mats = []
    lines = open(filename).read().strip().split("\n")
    lines_per_mat = 4
    for i in range(n):
        block = lines[i * lines_per_mat:(i + 1) * lines_per_mat]
        mat = np.array([[float(v) for v in row.split()] for row in block])
        mats.append(mat)
    return mats


def load_time_table(time_file):
    """
    Load time statistics file and return dict:
    {(source, target): time_ms}
    """
    df = pd.read_csv(time_file)
    time_map = {}
    df.columns = [c.strip() for c in df.columns]

    for _, row in df.iterrows():
        src = row["Source"].strip()
        tgt = row["Target"].strip()
        time_ms = float(row["Time(ms)"])
        time_map[(src, tgt)] = time_ms

    return time_map


def evaluate(yaml_file, est_transform_file, rot_th, trans_th,
             excel_out="evaluation.xlsx", time_file=None):

    # Load YAML
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    root = config["root"]
    gt_dir = root + config["groundtruth"]
    raw_dir = root + config["raw_data"]
    pairs = config["pairs"]

    # Load estimated transforms
    est_mats = load_transforms(est_transform_file, len(pairs))

    # Load time table (optional)
    time_map = {}
    if time_file is not None:
        time_map = load_time_table(time_file)

    results = []

    for idx, pair in enumerate(pairs):
        src_name = pair["source"]
        tgt_name = pair["target"]

        print(f"Processing pair {idx}: {src_name} -> {tgt_name}")

        # Load GT transform
        try:
            T_gt = np.loadtxt(gt_dir + pair["transformation_file"])
        except:
            T_gt = np.identity(4)
        T_est = est_mats[idx]

        R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]
        R_est, t_est = T_est[:3, :3], T_est[:3, 3]

        r_err = rotation_error(R_gt, R_est)
        t_err = translation_error(t_gt, t_est)

        # Load source point cloud
        pcd_src = o3d.io.read_point_cloud(raw_dir + src_name)
        pts = np.asarray(pcd_src.points)
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])

        rmse, mee = compute_rmse_and_mee(pts_h, T_gt, T_est)

        success = (r_err < rot_th) and (t_err < trans_th)

        # Fetch time (ms)
        time_ms = time_map.get((src_name, tgt_name), np.nan)

        success = success and (time_ms > 0)

        results.append({
            "pair": f"{src_name}->{tgt_name}",
            "rotation_error": r_err,
            "translation_error": t_err,
            "RMSE": rmse,
            "MeE": mee,
            "time_ms": time_ms,
            "success": success
        })

    df = pd.DataFrame(results)

    # Summary
    success_rate = df["success"].mean()
    df_success = df[df["success"]]

    summary = {
        "success_rate": success_rate,
        "avg_rotation_error": df_success["rotation_error"].mean(),
        "avg_translation_error": df_success["translation_error"].mean(),
        "avg_RMSE": df_success["RMSE"].mean(),
        "avg_MeE": df_success["MeE"].mean(),
        "avg_time_ms_all": df["time_ms"].mean(),
        "avg_time_ms_success": df_success["time_ms"].mean()
    }

    # Save Excel
    with pd.ExcelWriter(excel_out) as writer:
        df.to_excel(writer, sheet_name="pairs", index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)

    print(f"Saved evaluation to {excel_out}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--est", required=True)
    parser.add_argument("--reg", default=None,
                        help="CSV file containing time(ms) per pair")
    parser.add_argument("--rot_th", type=float, default=5)
    parser.add_argument("--trans_th", type=float, default=1)
    parser.add_argument("--out", default="evaluation.xlsx")

    args = parser.parse_args()

    evaluate(
        args.yaml,
        args.est,
        args.rot_th,
        args.trans_th,
        excel_out=args.out,
        time_file=args.reg
    )
