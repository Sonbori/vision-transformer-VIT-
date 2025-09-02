#!/usr/bin/env python3
"""
노트북 일괄 실행 스크립트 (헤드리스 모드)
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import logging
from datetime import datetime

# 로그 설정
os.makedirs('experiments/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'experiments/logs/notebook_execution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def setup_environment():
    """환경 설정"""
    # matplotlib 백엔드 설정
    import matplotlib
    matplotlib.use('Agg')
    
    # 랜덤 시드 고정
    import torch
    import numpy as np
    torch.manual_seed(42)
    np.random.seed(42)
    
    # CPU 전용 모드
    device = 'cpu'
    logging.info(f"Environment setup completed. Device: {device}")
    return device

def execute_notebook(notebook_path, timeout=1200):
    """개별 노트북 실행"""
    logging.info(f"Starting execution of {notebook_path}")
    start_time = time.time()
    
    try:
        # jupyter nbconvert로 실행
        cmd = [
            'jupyter', 'nbconvert', 
            '--to', 'notebook',
            '--execute',
            '--inplace',
            '--ExecutePreprocessor.timeout=1200',
            '--ExecutePreprocessor.kernel_name=python3',
            str(notebook_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            logging.info(f"✅ {notebook_path} completed successfully in {execution_time:.1f}s")
            return True, execution_time, ""
        else:
            logging.error(f"❌ {notebook_path} failed: {result.stderr}")
            return False, execution_time, result.stderr
            
    except subprocess.TimeoutExpired:
        logging.error(f"⏰ {notebook_path} timed out after {timeout}s")
        return False, timeout, "Timeout"
    except Exception as e:
        logging.error(f"💥 {notebook_path} crashed: {str(e)}")
        return False, time.time() - start_time, str(e)

def check_outputs(notebook_name):
    """산출물 존재 여부 확인"""
    expected_outputs = {
        '01_infer_vit': ['reports/figures/attention_overlay_final.png', 'reports/tables/preds_topk_final.csv'],
        '02_rectangular_inputs': ['reports/figures/pareto_accuracy_latency_final.png', 'reports/tables/rect_results_final.csv'],
        '03_patchify_linear_vs_conv': ['reports/figures/throughput_bar_final.png', 'reports/tables/match_report_final.json'],
        '04_positional_embedding': ['reports/figures/pe_heatmap_final.png', 'reports/tables/pos_embedding_analysis_final.csv'],
        '05_train_cifar100_deit': ['reports/figures/train_curves_final.png', 'reports/tables/cifar100_results_final.csv']
    }
    
    outputs = expected_outputs.get(notebook_name, [])
    existing = []
    missing = []
    
    for output in outputs:
        if os.path.exists(output):
            size = os.path.getsize(output)
            existing.append(f"{output} ({size} bytes)")
        else:
            missing.append(output)
    
    return existing, missing

def main():
    """메인 실행"""
    setup_environment()
    
    notebooks = [
        'notebooks/01_infer_vit.ipynb',
        'notebooks/02_rectangular_inputs.ipynb', 
        'notebooks/03_patchify_linear_vs_conv.ipynb',
        'notebooks/04_positional_embedding.ipynb',
        'notebooks/05_train_cifar100_deit.ipynb'
    ]
    
    results = {}
    total_start = time.time()
    
    for notebook in notebooks:
        notebook_name = Path(notebook).stem
        
        # 실행
        success, exec_time, error = execute_notebook(notebook)
        
        # 산출물 확인
        existing, missing = check_outputs(notebook_name)
        
        results[notebook_name] = {
            'success': success,
            'execution_time': exec_time,
            'error': error,
            'outputs_created': existing,
            'outputs_missing': missing
        }
        
        # 실시간 보고
        print(f"\n{'='*60}")
        print(f"📓 {notebook_name}")
        print(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"Time: {exec_time:.1f}s")
        if existing:
            print(f"Created: {len(existing)} files")
            for file in existing:
                print(f"  ✅ {file}")
        if missing:
            print(f"Missing: {len(missing)} files") 
            for file in missing:
                print(f"  ❌ {file}")
        if error:
            print(f"Error: {error[:100]}...")
    
    total_time = time.time() - total_start
    
    # 최종 요약
    print(f"\n{'='*60}")
    print(f"📋 EXECUTION SUMMARY ({total_time:.1f}s total)")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results.values() if r['success'])
    total_count = len(results)
    
    print(f"Overall: {success_count}/{total_count} notebooks successful")
    
    for name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {name}: {result['execution_time']:.1f}s, {len(result['outputs_created'])} files")
    
    return results

if __name__ == '__main__':
    main()
