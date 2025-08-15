#!/usr/bin/env python3
"""
測試基本串流demo的腳本
"""

import subprocess
import sys
import os

def test_basic_streaming():
    """測試基本串流處理"""
    print("=== 測試基本串流Demo ===")
    
    # 檢查視頻文件是否存在
    video_path = "./data/video.mp4"
    if not os.path.exists(video_path):
        print(f"警告: 視頻文件不存在 {video_path}")
        print("請確保data/video.mp4文件存在，或使用--video_path參數指定其他視頻文件")
        return
    
    try:
        # 運行基本串流demo（處理前10秒）
        cmd = [sys.executable, "basic.py", "--video_path", video_path, "--num_workers", "2"]
        print(f"執行命令: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd()
        )
        
        # 等待一段時間然後終止（避免處理整個視頻）
        try:
            stdout, stderr = process.communicate(timeout=30)
            print("標準輸出:")
            print(stdout)
            if stderr:
                print("錯誤輸出:")
                print(stderr)
        except subprocess.TimeoutExpired:
            print("測試運行30秒後自動停止")
            process.terminate()
            stdout, stderr = process.communicate()
            print("部分輸出:")
            print(stdout[-1000:] if stdout else "無輸出")
        
    except Exception as e:
        print(f"測試失敗: {e}")

if __name__ == "__main__":
    test_basic_streaming()
