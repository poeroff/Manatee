#!/usr/bin/env python3
"""
Manatee VAE 추론 스크립트
사용법:
    # 모드 1 - predict: 유전자 발현 데이터(X) 입력 → 재구성(X_rec) + 잠재벡터(Z) 출력
    python inference.py --mode predict \
        --data GSE72857/processed/data_x.csv.gz \
        --output_dir ./output

    # 모드 2 - generate: TF 잠재벡터(Z) 입력 → 유전자 발현(X_rec) 출력
    python inference.py --mode generate \
        --data GSE72857/processed/data_z.csv.gz \
        --output_dir ./output

공통 옵션:
    --model     모델 경로 (기본값: GSE72857/model/GSE72857_fold_4.pt)
    --genes     유전자 목록 파일 (기본값: GSE72857/processed/genes.txt)
    --tfs       TF 목록 파일 (기본값: GSE72857/processed/tfs.txt)
    --depth     인코더/디코더 레이어 수 (기본값: 3)
    --no_umap   UMAP 시각화 생략
"""

import sys
import os
import argparse
import numpy as np
import torch

# src 폴더를 경로에 추가 (VAE 클래스 임포트용)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from vae import VAE


def load_model(model_path, genes_path, tfs_path, depth=3):
    """VAE 모델 로드"""
    with open(genes_path) as f:
        genes = f.read().splitlines()
    with open(tfs_path) as f:
        tfs = f.read().splitlines()

    mask = np.isin(np.array(genes), np.array(tfs))
    print(f"  유전자 수: {len(genes)}, TF 수: {len(tfs)}, 마스크 True 수: {mask.sum()}")

    model = VAE(mask_tf=mask, depth=depth)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  모델 로드 완료: {model_path}")
    return model, genes, tfs


def run_predict(model, data, output_dir, job_name, use_umap=True, random_seed=1):
    """
    predict 모드: X (유전자 발현) → X_rec (재구성) + Z (잠재벡터)
    data shape: [n_cells, n_genes=4342]
    """
    torch.manual_seed(random_seed)
    with torch.no_grad():
        x_rec, z, mu, logvar = model.forward(torch.FloatTensor(data))
    x_rec = x_rec.numpy()
    z = z.numpy()

    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, f'{job_name}_x_rec.csv.gz'), x_rec, delimiter=' ')
    np.savetxt(os.path.join(output_dir, f'{job_name}_z.csv.gz'), z, delimiter=' ')
    np.savetxt(os.path.join(output_dir, f'{job_name}_mu.csv.gz'), mu.numpy(), delimiter=' ')
    print(f"  저장: {job_name}_x_rec.csv.gz  (shape: {x_rec.shape})")
    print(f"  저장: {job_name}_z.csv.gz       (shape: {z.shape})")

    if use_umap:
        import umap
        np.random.seed(random_seed)
        reducer = umap.UMAP(random_state=random_seed, min_dist=0.5, n_neighbors=15)
        umap_x = reducer.fit_transform(data)
        umap_xrec = reducer.fit_transform(x_rec)
        np.savetxt(os.path.join(output_dir, f'{job_name}_x_umap.csv.gz'), umap_x, delimiter=' ')
        np.savetxt(os.path.join(output_dir, f'{job_name}_x_rec_umap.csv.gz'), umap_xrec, delimiter=' ')
        print(f"  저장: {job_name}_x_umap.csv.gz, {job_name}_x_rec_umap.csv.gz")

    return x_rec, z


def run_generate(model, data, output_dir, job_name, use_umap=True, random_seed=1):
    """
    generate 모드: Z (TF 잠재벡터) → X_rec (유전자 발현 예측)
    data shape: [n_cells, n_tfs=296]
    """
    with torch.no_grad():
        x_rec = model.decode(torch.FloatTensor(data))
    x_rec = x_rec.numpy()

    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, f'{job_name}_x_rec.csv.gz'), x_rec, delimiter=' ')
    print(f"  저장: {job_name}_x_rec.csv.gz  (shape: {x_rec.shape})")

    if use_umap:
        import umap
        np.random.seed(random_seed)
        reducer = umap.UMAP(random_state=random_seed, min_dist=0.5, n_neighbors=15)
        umap_xrec = reducer.fit_transform(x_rec)
        np.savetxt(os.path.join(output_dir, f'{job_name}_x_rec_umap.csv.gz'), umap_xrec, delimiter=' ')
        print(f"  저장: {job_name}_x_rec_umap.csv.gz")

    return x_rec


def main():
    base = os.path.dirname(__file__)

    parser = argparse.ArgumentParser(description='Manatee VAE 추론')
    parser.add_argument('--mode', choices=['predict', 'generate'], required=True,
                        help='predict: X→X_rec+Z, generate: Z→X_rec')
    parser.add_argument('--data', required=True,
                        help='입력 데이터 경로 (.csv.gz, 공백 구분)')
    parser.add_argument('--model', default=os.path.join(base, 'GSE72857/model/GSE72857_fold_4.pt'),
                        help='모델 체크포인트 경로')
    parser.add_argument('--genes', default=os.path.join(base, 'GSE72857/processed/genes.txt'),
                        help='유전자 목록 파일')
    parser.add_argument('--tfs', default=os.path.join(base, 'GSE72857/processed/tfs.txt'),
                        help='TF 목록 파일')
    parser.add_argument('--output_dir', default='./output',
                        help='결과 저장 폴더')
    parser.add_argument('--job', default='result',
                        help='출력 파일명 prefix')
    parser.add_argument('--depth', type=int, default=3,
                        help='인코더/디코더 레이어 수')
    parser.add_argument('--seed', type=int, default=1,
                        help='랜덤 시드')
    parser.add_argument('--no_umap', action='store_true',
                        help='UMAP 시각화 생략 (umap-learn 미설치 시 사용)')
    args = parser.parse_args()

    print("=" * 50)
    print("Manatee VAE 추론")
    print("=" * 50)
    print(f"  모드      : {args.mode}")
    print(f"  입력 데이터: {args.data}")
    print(f"  모델      : {args.model}")
    print(f"  출력 폴더  : {args.output_dir}")
    print()

    # 1. 모델 로드
    print("[1/3] 모델 로딩 중...")
    model, genes, tfs = load_model(args.model, args.genes, args.tfs, depth=args.depth)

    # 2. 데이터 로드
    print("[2/3] 데이터 로딩 중...")
    data = np.genfromtxt(args.data, delimiter=' ')
    print(f"  데이터 shape: {data.shape}")

    # 3. 추론
    print("[3/3] 추론 실행 중...")
    use_umap = not args.no_umap
    if args.mode == 'predict':
        x_rec, z = run_predict(model, data, args.output_dir, args.job,
                               use_umap=use_umap, random_seed=args.seed)
        print(f"\n완료! 재구성 발현 행렬: {x_rec.shape}, 잠재벡터: {z.shape}")
    else:
        x_rec = run_generate(model, data, args.output_dir, args.job,
                             use_umap=use_umap, random_seed=args.seed)
        print(f"\n완료! 예측 발현 행렬: {x_rec.shape}")


if __name__ == '__main__':
    main()
