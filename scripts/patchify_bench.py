#!/usr/bin/env python3
"""
패치 임베딩 등가성 검증 및 성능 벤치마크 스크립트
Usage: python scripts/patchify_bench.py --output reports/tables/patchify_results_v2.csv
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import json
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class PatchEmbedLinear(nn.Module):
    """Flatten → Linear 방식의 패치 임베딩"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_chans * patch_size * patch_size
        
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 패치로 분할
        x = x.unfold(2, self.patch_size, self.patch_size) \
             .unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.flatten(2)
        
        # Linear projection
        x = self.proj(x)
        return x


class PatchEmbedConv(nn.Module):
    """Conv2d를 사용한 패치 임베딩"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Convolution projection
        x = self.proj(x)
        # Flatten and transpose
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchifyBenchmark:
    """패치 임베딩 벤치마크 클래스"""
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.tolerance = 1e-6
        
        print(f"✅ 벤치마크 초기화: {self.device}")
    
    def linear_to_conv_weights(self, linear_weight: torch.Tensor, 
                              patch_size: int, in_chans: int) -> torch.Tensor:
        """Linear 가중치를 Conv2d 가중치로 변환"""
        embed_dim, patch_dim = linear_weight.shape
        conv_weight = linear_weight.view(embed_dim, in_chans, patch_size, patch_size)
        return conv_weight
    
    def make_equivalent(self, linear_model: PatchEmbedLinear, 
                       conv_model: PatchEmbedConv) -> None:
        """두 모델을 동일한 가중치로 설정"""
        with torch.no_grad():
            linear_weight = linear_model.proj.weight.data
            linear_bias = linear_model.proj.bias.data
            
            conv_weight = self.linear_to_conv_weights(
                linear_weight, linear_model.patch_size, 3
            )
            
            conv_model.proj.weight.data = conv_weight
            conv_model.proj.bias.data = linear_bias.clone()
    
    def verify_equivalence(self, linear_model: PatchEmbedLinear,
                          conv_model: PatchEmbedConv,
                          num_tests: int = 10) -> Dict:
        """등가성 검증"""
        batch_sizes = [1, 4, 8, 16]
        all_results = []
        
        for test_case in range(num_tests):
            for batch_size in batch_sizes:
                # 무작위 입력 생성
                test_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
                
                # 모델 실행
                with torch.no_grad():
                    linear_output = linear_model(test_input)
                    conv_output = conv_model(test_input)
                
                # 차이 계산
                output_diff = torch.norm(linear_output - conv_output).item()
                max_abs_diff = torch.max(torch.abs(linear_output - conv_output)).item()
                relative_diff = output_diff / torch.norm(linear_output).item()
                
                all_results.append({
                    'test_case': test_case,
                    'batch_size': batch_size,
                    'l2_diff': output_diff,
                    'max_abs_diff': max_abs_diff,
                    'relative_diff': relative_diff,
                    'passed': output_diff < self.tolerance
                })
        
        results_df = pd.DataFrame(all_results)
        passed_tests = results_df['passed'].sum()
        total_tests = len(results_df)
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests,
            'mean_l2_diff': float(results_df['l2_diff'].mean()),
            'max_l2_diff': float(results_df['l2_diff'].max()),
            'verification_passed': passed_tests == total_tests
        }
    
    def benchmark_performance(self, model: nn.Module, batch_sizes: List[int],
                             num_trials: int = 1000) -> List[Dict]:
        """성능 벤치마크"""
        results = []
        
        # GPU 동기화 함수
        if self.device.type == 'cuda':
            sync_fn = torch.cuda.synchronize
        else:
            sync_fn = lambda: None
        
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
            model.eval()
            
            # 워밍업
            with torch.no_grad():
                for _ in range(100):
                    _ = model(test_input)
                    sync_fn()
            
            # 성능 측정
            times = []
            with torch.no_grad():
                for _ in range(num_trials):
                    start_time = time.perf_counter()
                    _ = model(test_input)
                    sync_fn()
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
            
            times = np.array(times) * 1000  # ms 변환
            
            results.append({
                'batch_size': batch_size,
                'mean_ms': float(np.mean(times)),
                'std_ms': float(np.std(times)),
                'min_ms': float(np.min(times)),
                'max_ms': float(np.max(times)),
                'throughput_fps': float(1000 / np.mean(times) * batch_size)
            })
        
        return results
    
    def run_full_benchmark(self, batch_sizes: List[int] = [1, 4, 8, 16, 32]) -> Dict:
        """전체 벤치마크 실행"""
        print("🔧 모델 생성 중...")
        
        # 모델 생성
        linear_model = PatchEmbedLinear().to(self.device)
        conv_model = PatchEmbedConv().to(self.device)
        
        # 가중치 동일화
        self.make_equivalent(linear_model, conv_model)
        
        print("🧪 등가성 검증 중...")
        equivalence_result = self.verify_equivalence(linear_model, conv_model)
        
        print("⚡ 성능 벤치마크 실행 중...")
        linear_perf = self.benchmark_performance(linear_model, batch_sizes)
        conv_perf = self.benchmark_performance(conv_model, batch_sizes)
        
        # 성능 비교
        performance_comparison = []
        for l_perf, c_perf in zip(linear_perf, conv_perf):
            speedup = l_perf['mean_ms'] / c_perf['mean_ms']
            performance_comparison.append({
                'batch_size': l_perf['batch_size'],
                'linear_ms': l_perf['mean_ms'],
                'conv_ms': c_perf['mean_ms'],
                'speedup': speedup,
                'conv_faster': speedup > 1
            })
        
        return {
            'equivalence': equivalence_result,
            'linear_performance': linear_perf,
            'conv_performance': conv_perf,
            'performance_comparison': performance_comparison,
            'overall_speedup': np.mean([comp['speedup'] for comp in performance_comparison])
        }


def main():
    parser = argparse.ArgumentParser(description='패치 임베딩 벤치마크')
    parser.add_argument('--output', type=str, required=True,
                       help='결과 저장 경로 (JSON 파일)')
    parser.add_argument('--device', type=str, default='auto',
                       help='사용할 디바이스 (auto, cuda, cpu)')
    parser.add_argument('--batch-sizes', type=str, default='1,4,8,16,32',
                       help='테스트할 배치 크기 (쉼표로 구분)')
    parser.add_argument('--trials', type=int, default=1000,
                       help='성능 측정 반복 횟수')
    
    args = parser.parse_args()
    
    # 배치 크기 파싱
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')]
    
    print(f"📐 테스트 배치 크기: {batch_sizes}")
    print(f"🔢 측정 반복: {args.trials}회")
    
    # 벤치마크 실행
    benchmark = PatchifyBenchmark(device=args.device)
    results = benchmark.run_full_benchmark(batch_sizes)
    
    # 결과 저장
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 결과 저장: {args.output}")
    
    # 요약 출력
    print(f"\n📊 벤치마크 결과:")
    print(f"   등가성 검증: {'✅ 통과' if results['equivalence']['verification_passed'] else '❌ 실패'}")
    print(f"   평균 L2 차이: {results['equivalence']['mean_l2_diff']:.2e}")
    print(f"   전체 평균 속도 향상: {results['overall_speedup']:.2f}x")
    
    print(f"\n🏆 배치별 성능:")
    for comp in results['performance_comparison']:
        print(f"   배치 {comp['batch_size']}: Conv2d가 {comp['speedup']:.2f}x {'빠름' if comp['conv_faster'] else '느림'}")


if __name__ == '__main__':
    main()
