import os
import pandas as pd
import numpy as np
import sys

def calculate_global_stats(root_dir):
    """
    1. 遍歷 root_dir 下的「第一層」子資料夾，讀取 evaluation.xlsx。
    2. 統計全域成功率 (Success Rate)。
    3. 針對「成功 (Success) 的 Pair」計算誤差平均值 (Error Metrics)。
    """
    all_pairs_data = []
    folder_count = 0

    print(f"開始搜尋目錄 (僅限第一層子資料夾): {root_dir} ...")

    if not os.path.exists(root_dir):
        print(f"錯誤: 找不到路徑 {root_dir}")
        return

    # 1. 取得 root 下的所有項目
    for item in os.listdir(root_dir):
        sub_dir_path = os.path.join(root_dir, item)

        # 確保是資料夾
        if os.path.isdir(sub_dir_path):
            target_file = os.path.join(sub_dir_path, "evaluation.xlsx")

            if os.path.exists(target_file):
                try:
                    # 讀取 Excel
                    df = pd.read_excel(target_file, sheet_name='pairs')
                    
                    if 'success' in df.columns:
                        # 統一將 success 欄位轉為 boolean，避免字串 'True'/'False' 混用造成問題
                        # 如果是字串，轉小寫後比對 'true'；如果是布林，直接保持
                        if df['success'].dtype == 'object':
                            df['success'] = df['success'].astype(str).str.lower() == 'true'
                        else:
                            df['success'] = df['success'].astype(bool)

                        all_pairs_data.append(df)
                        folder_count += 1
                        print(f"[讀取成功] {item} (Pairs: {len(df)})")
                    else:
                        print(f"[格式警告] {item} 缺少 'success' 欄位，已跳過。")
                        
                except Exception as e:
                    print(f"[讀取錯誤] 無法讀取 {item}: {e}")

    if not all_pairs_data:
        print("\n找不到任何有效的 evaluation.xlsx 檔案。")
        return

    # 2. 合併所有數據 (Raw Data)
    full_df = pd.concat(all_pairs_data, ignore_index=True)
    
    # 3. 計算成功率 (使用全部資料)
    total_count = len(full_df)
    success_count = full_df['success'].sum() # True 為 1, False 為 0
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0

    # 4. 篩選出成功的資料 (用於計算誤差平均)
    success_df = full_df[full_df['success'] == True]

    # 定義要計算的數值欄位
    numeric_metrics = ['rotation_error', 'translation_error', 'RMSE', 'MeE', 'time_ms']

    print("\n" + "="*50)
    print("全域統計結果 (Global Statistics)")
    print("="*50)
    print(f"處理資料夾數 : {folder_count}")
    print(f"總 Pair 數量 : {total_count}")
    print(f"成功 Pair 數量 : {success_count}")
    print(f"成功率 (Success Rate): {success_rate:.2f}%")
    print("-" * 50)
    print("以下數據僅統計「成功 (Success)」的案例：")
    print("-" * 50)

    if len(success_df) == 0:
        print("沒有任何成功的 Pair，無法計算誤差平均值。")
    else:
        for metric in numeric_metrics:
            if metric in success_df.columns:
                avg_val = success_df[metric].mean()
                print(f"平均 {metric:<20}: {avg_val:.6f}")
            else:
                print(f"警告: 找不到欄位 {metric}")
    
    print("="*50)

if __name__ == "__main__":
    # --- 設定 ---
    # 請將此處改為你要搜尋的根目錄路徑
    target_root_directory = sys.argv[1] if len(sys.argv) > 1 else r".\reg_results"
    
    # 執行函數
    calculate_global_stats(target_root_directory)