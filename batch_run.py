import os
import subprocess
import sys

# ================= 設定區域 =================

# 1. C++ 執行檔路徑
EXE_PATH = r"C:\Timothy\Code\Research\TopologicalConsistencyRegis\x64\Release\TopologicalConsistencyRegis.exe"

# 2. Python 腳本名稱
EVAL_SCRIPT = "evaluate_registration_v3.py"  # 每次迴圈內執行
ANALYSIS_SCRIPT = "analyze_results.py"       # 全部跑完後執行

# 3. 參數列表 (ARGS_LIST)
ARGS_LIST = [
    "Apartment",
    "Boardroom",
    "RESSO_6i",
    "RESSO_7a",
    "Park",
    "Campus",
    # "",
    # 在此加入更多參數...
]

RES_LIST = [0.025, 0.025, 0.05, 0.1, 0.1, 0.1]

TRANS_LIST = [1, 1, 1, 2, 2, 2]

# ===========================================

def get_configured_env():
    """設定 PCL 與相關依賴的環境變數"""
    env = os.environ.copy()
    pcl_root = env.get("PCL_ROOT", "")
    
    if not pcl_root:
        print("警告: 系統環境變數中未找到 PCL_ROOT。")

    extra_paths = [
        os.path.join(pcl_root, "bin"),
        os.path.join(pcl_root, r"3rdParty\FLANN\bin"),
        os.path.join(pcl_root, r"3rdParty\VTK\bin"),
        os.path.join(pcl_root, r"3rdParty\Qhull\bin"),
        os.path.join(pcl_root, r"3rdParty\OpenNI2\Tools"),
        r"C:\Libraries\yaml-cpp-0.8.0\build\Release"
    ]
    
    # 將新路徑加入 PATH 開頭
    env["PATH"] = ";".join(extra_paths) + ";" + env["PATH"]
    return env

def main():
    custom_env = get_configured_env()
    
    print(f"--- 開始批次處理，共 {len(ARGS_LIST)} 組任務 ---")

    max_lines = "20"
    angle = "3.0"
    fac_epsilon = "2.0"
    fac_tau = input("fac_tau: ")

    for index, arg in enumerate(ARGS_LIST):
        print(f"\n[任務 {index+1}/{len(ARGS_LIST)}] 正在處理參數: {arg}")

        
        
        try:
            # --- 步驟 1: 執行 C++ 主程式 ---
            exe_args = f"{arg} {RES_LIST[index]} {max_lines} {angle} {fac_epsilon} {fac_tau}"
            print(f"  -> 執行 C++: {os.path.basename(EXE_PATH)} {exe_args}")
            subprocess.run([EXE_PATH, arg, str(RES_LIST[index]), max_lines, angle, fac_epsilon, fac_tau], env=custom_env, check=True)
            
            # --- 步驟 2: 執行 Python 評估腳本 (帶入相同參數) ---
            eval_args = f"--yaml .\\configs\\{arg}.yaml --est .\\reg_results\\{arg}\\est_transforms.txt --reg .\\reg_results\\{arg}\\registration_results.txt --trans_th {TRANS_LIST[index]} --out .\\reg_results\\{arg}\\evaluation.xlsx"
            print(f"  -> 執行評估: {EVAL_SCRIPT} {eval_args}")
            # 使用 sys.executable 確保用同一個 python 解譯器執行
            subprocess.run([sys.executable, EVAL_SCRIPT, 
                f"--yaml", f".\\configs\\{arg}.yaml",
                f"--est", f".\\reg_results\\{max_lines}_{angle}_{fac_epsilon}_{fac_tau}\\{arg}\\est_transforms.txt",
                f"--reg", f".\\reg_results\\{max_lines}_{angle}_{fac_epsilon}_{fac_tau}\\{arg}\\registration_results.txt",
                f"--trans_th", f"{TRANS_LIST[index]}",
                f"--out", f".\\reg_results\\{max_lines}_{angle}_{fac_epsilon}_{fac_tau}\\{arg}\\evaluation.xlsx"
            ], check=True)

        except subprocess.CalledProcessError as e:
            # 如果 C++ 失敗，或者 Python評估腳本失敗，會跳到這裡
            print(f"  [錯誤] 在處理 {arg} 時發生錯誤 (Exit code: {e.returncode})")
            print("  -> 跳過此任務，繼續下一個...")
            continue # 繼續下一個迴圈，不中斷整個批次
            
        except FileNotFoundError as e:
            print(f"  [錯誤] 找不到檔案: {e.filename}")
            break
            
        except Exception as e:
            print(f"  [未預期錯誤] {e}")

    print("\n--- 所有迴圈任務結束，開始執行最終分析 ---")

    # --- 步驟 3: 執行最終結果分析 ---
    if os.path.exists(ANALYSIS_SCRIPT):
        try:
            subprocess.run([sys.executable, ANALYSIS_SCRIPT, f".\\reg_results\\{max_lines}_{angle}_{fac_epsilon}_{fac_tau}"], check=True)
            print("最終分析完成。")
        except subprocess.CalledProcessError:
            print("最終分析腳本執行失敗。")
    else:
        print(f"找不到分析腳本: {ANALYSIS_SCRIPT}")

if __name__ == "__main__":
    main()