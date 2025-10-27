# テスト構造ドキュメント

## 概要

このドキュメントでは、プロジェクトのテスト構造と各ディレクトリの役割について説明します。

## ディレクトリ構造

```
tests/
├── unit/                    # 単体テスト
│   ├── algorithms/         # アルゴリズム関連テスト
│   ├── analysis/           # 分析関連テスト
│   ├── cache/              # キャッシュ関連テスト
│   ├── config/             # 設定関連テスト
│   ├── data/               # データ関連テスト
│   ├── environment/        # 環境関連テスト
│   ├── evaluation/         # 評価関連テスト
│   ├── features/           # 特徴量関連テスト
│   ├── inference/          # 推論関連テスト
│   ├── metrics/            # メトリクス関連テスト
│   ├── models/             # モデル関連テスト
│   ├── risk/               # リスク関連テスト
│   ├── scripts/            # スクリプト関連テスト
│   ├── trading/            # 取引関連テスト
│   ├── training/           # トレーニング関連テスト
│   └── utils/              # ユーティリティ関連テスト
├── integration/            # 統合テスト
│   ├── test_*.py          # エンドツーエンド統合テスト
│   └── smoke_tests.py     # スモークテスト
├── benchmark/              # ベンチマークテスト
│   ├── v68/               # バージョン68ベンチマーク
│   ├── final/             # 最終ベンチマーク
│   ├── final_v2/          # 最終ベンチマークv2
│   └── ...                # その他のバージョン
├── multimodal/            # マルチモーダル関連テスト
├── training/              # トレーニング統合テスト
├── scripts/               # テスト実行スクリプト
├── legacy_tests/          # レガシーテスト（移行予定）
├── conftest.py            # pytest設定ファイル
└── __init__.py           # Pythonパッケージ初期化
```

## 各ディレクトリの役割

### unit/
単体テストを格納します。各サブディレクトリは機能別に分類されています。

- **algorithms/**: SAC, PPOなどのアルゴリズムの実装テスト
- **analysis/**: データ分析、統計解析のテスト
- **cache/**: キャッシュシステムのテスト
- **config/**: 設定管理、設定ファイルのテスト
- **data/**: データ処理、パイプラインのテスト
- **environment/**: 取引環境、Gym環境のテスト
- **evaluation/**: モデル評価、バックテストのテスト
- **features/**: 特徴量エンジニアリングのテスト
- **inference/**: モデル推論、予測のテスト
- **metrics/**: 評価メトリクス、KPIのテスト
- **models/**: モデル構造、保存/読み込みのテスト
- **risk/**: リスク管理、ポジションサイジングのテスト
- **scripts/**: ユーティリティスクリプトのテスト
- **trading/**: 取引ロジック、注文執行のテスト
- **training/**: トレーニングループ、コールバックのテスト
- **utils/**: 汎用ユーティリティ関数のテスト

### integration/
統合テストを格納します。複数のコンポーネントが連携して動作するテストです。

- **test_*.py**: エンドツーエンドの統合テスト
- **smoke_tests.py**: 基本機能の健全性を確認するスモークテスト

### benchmark/
ベンチマークテストを格納します。パフォーマンス比較や回帰テストに使用します。

各サブディレクトリはバージョンやテストスイート別に分類されています。

### multimodal/
マルチモーダル（複数データソース）関連のテストを格納します。

### training/
トレーニング関連の統合テストを格納します。

### scripts/
テスト実行用のスクリプトやユーティリティを格納します。

### legacy_tests/
古いテストコードを一時的に格納します。将来的に統合または削除されます。

## テスト実行方法

### 全テスト実行
```bash
pytest
```

### 単体テストのみ実行
```bash
pytest tests/unit/
```

### 統合テストのみ実行
```bash
pytest tests/integration/
```

### 特定のモジュールテスト実行
```bash
pytest tests/unit/risk/
pytest tests/unit/trading/
```

### ベンチマークテスト実行
```bash
pytest tests/benchmark/
```

## テスト作成ガイドライン

### ファイル命名規則
- テストファイル: `test_*.py`
- テストクラス: `Test*`
- テストメソッド: `test_*`

### テスト構造
```python
import pytest
from ztb.module import ClassToTest

class TestClassToTest:
    def setup_method(self):
        """テスト前準備"""
        pass

    def test_feature(self):
        """機能テスト"""
        pass

    def test_edge_case(self):
        """境界条件テスト"""
        pass
```

### テストデータの扱い
- テストデータは `tests/data/` またはフィクスチャを使用
- 大きなデータファイルはGitignore対象
- モックを使用した外部依存の分離

## 注意事項

- 実行時生成物（ログ、キャッシュ、アーティファクト）は `results/` ディレクトリに格納
- テスト専用の一時ファイルは適切にクリーンアップ
- CI/CDでは全テストスイートを実行
- 新機能追加時は対応するテストも追加
