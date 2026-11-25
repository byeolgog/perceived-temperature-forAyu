"""옷차림 모델 학습.

사용법:
  python train.py                         # 합성 데이터로 학습 (바로 시연 가능)
  python train.py --aws data/aws_2025.csv # 기상청 AWS 2025년 시간자료로 학습
  python train.py --model xgb             # XGBoost 사용
학생 체감 제보(data/feedback.csv)가 있으면 자동으로 함께 학습합니다.
"""
import argparse

import config
import model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aws", help="기상청 AWS/ASOS 시간자료 CSV 경로")
    p.add_argument("--model", choices=["rf", "xgb"], default="rf")
    args = p.parse_args()

    if args.aws:
        weather = model.load_aws_csv(args.aws)
        print(f"AWS 자료 {len(weather):,}시간 불러옴")
    else:
        weather = model.synthetic_weather()
        print(f"합성 날씨 {len(weather):,}건 생성")

    rep = model.train(weather, args.model)
    print(f"모델: {args.model} | 학습 {rep['n_train']:,}건 (학생 제보 {rep['n_feedback']}건 포함)")
    print(f"정확도(단계 정확히 일치): {rep['accuracy']:.1%}")
    print(f"±1단계 이내 일치:        {rep['within_1']:.1%}")
    print(f"판단 기준 7/10 (70%) → {'통과' if rep['accuracy'] >= 0.7 else '미달'}")
    print(f"저장: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
