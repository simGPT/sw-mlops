# SW 기업 연계 프로젝트(오브젠)

개발 기간: 2026년 4월 15일 → 2026년 6월 16일

## 📌 Overview

쿠버네티스 기반 MLOps를 활용한 이커머스 고객 이탈 예측 및 마케팅 자동화 서비스

## 📝 Summary

**AI 서비스의 안정적인 운영을 위해서는 모델의 학습, 배포, 모니터링, 재학습까지 전 생애주기를 체계적으로 관리하는 MLOps 체계**가 필요합니다. 이를 기반으로 이커머스 고객의 이탈 가능성을 예측하고, 마케팅 효과가 기대되는 고객을 선별하여 마케팅을 제공하는 프로젝트입니다.

<img width="978" height="494" alt="image" src="https://github.com/user-attachments/assets/0108ffef-97b6-4d3a-b37b-abf2ee211449" />
<img width="610" height="816" alt="image" src="https://github.com/user-attachments/assets/b34a6758-b367-4f61-86ad-e558dd746a92" />

## ✨ Key Features

1. 고객 이탈 예측 모델
    - Kaggle E-commerce Customer Churn Dataset 기반 학습 데이터 구축
    - 로지스틱 회귀를 이용한 이탈 예측 모델 개발
    - Accuracy, F1 Score 등을 활용한 성능 평가
2. 마케팅 효과 예측 모델
    - Treatment 그룹과 Control 그룹의 반응 차이를 학습하여 마케팅 효과 예측
    - 마케팅 효과 유효 고객을 선별하여 마케팅 자동화 수행
3. 모델 버전 관리
    - MLflow를 활용한 학습 파라미터 및 성능 지표 자동 기록
    - 모델 등록 기반 버전 관리 및 운영 모델 관리
    - PostgreSQL 기반 메타데이터 저장, AWS S3 기반 모델 아티팩트 관리
4. 자동 배포
    - 모델 재학습 후 성능 기준을 충족한 모델에 대해 자동 배포 수행
    - GitHub Actions 기반 CI/CD 파이프라인 구축
5. 모델 모니터링 및 알림 시스템
    - Prometheus를 활용한 실시간 운영 지표 수집
    - 이탈 예측 비율, 모델 신뢰도, 추론 지연시간, 피처 드리프트 등 모니터링
    - Grafana 대시보드 시각화 및 Alertmanager 기반 관리자 이메일 알림 제공
6. Kubernetes 기반 MLOps 인프라
    - AWS EC2 환경에 k3s 클러스터 구축
    - MLflow, Model API, Monitoring, Shopping 서비스의 네임스페이스 분리 운영
    - Traefik Ingress Controller 및 cert-manager를 활용한 HTTPS 기반 서비스 제공
    - Docker 및 GitHub Actions를 활용한 자동 배포 환경 구축
7. 쇼핑몰 연계 AI 서비스
    - Spring Boot 기반 쇼핑몰 백엔드 API 구축
    - React 기반 사용자 웹 서비스 제공
    - 고객 이탈 예측 결과와 마케팅 효과 예측 결과를 활용한 마케팅 자동화 연계
    - A/B 테스트 결과를 수집하여 마케팅 효과 예측 모델 재학습 데이터로 사용할 수 있도록 csv 파일 추출 api 구현
