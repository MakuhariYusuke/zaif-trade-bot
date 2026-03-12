"""
データバリデーション機能の実装

金融時系列データに対する包括的なバリデーションを提供：
- データ整合性チェック（型チェック, 範囲チェック, 一貫性チェック）
- スキーマ検証（必須フィールド, データ型, 制約条件）
- 品質メトリクス計算（完全性, 正確性, 適時性）
- 異常検知（統計的特性, 分布変化, データ欠損パターン）
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

import numpy as np
import pandas as pd
from scipy import stats

from ztb.types.common import MetricsDict, ObjectMap

logger = logging.getLogger(__name__)

StringList = list[str]
SchemaRule = dict[str, object]
SchemaDefinition = dict[str, SchemaRule]

class RuleEvaluationResult(TypedDict):
    """Result payload for individual validation rules."""

    valid: bool
    level: str
    messages: StringList

class RuleIssueResult(TypedDict):
    """Container for collected errors/warnings."""

    errors: StringList
    warnings: StringList

class SchemaValidationResult(TypedDict):
    """Schema validation summary."""

    errors: StringList
    warnings: StringList
    metrics: MetricsDict

class AnomalyDetectionResult(TypedDict):
    """Anomaly detection summary."""

    warnings: StringList
    details: ObjectMap

class IntegrityCheckResult(TypedDict):
    """Data integrity check summary."""

    errors: StringList
    warnings: StringList
    metrics: MetricsDict
    details: ObjectMap

class AdditionalValidationRule(TypedDict, total=False):
    """User-provided validation rule schema."""

    type: str
    columns: list[str]
    params: ObjectMap

ValidationRule = Callable[..., RuleEvaluationResult]

@dataclass
class ValidationResult:
    """バリデーション結果を格納するデータクラス。"""

    is_valid: bool
    errors: StringList
    warnings: StringList
    metrics: MetricsDict
    details: ObjectMap

@dataclass
class DataQualityMetrics:
    """データ品質メトリクスを格納するデータクラス。"""

    completeness: float  # 完全性 (0-1)
    accuracy: float  # 正確性 (0-1)
    timeliness: float  # 適時性 (0-1)
    consistency: float  # 一貫性 (0-1)
    validity: float  # 有効性 (0-1)
    uniqueness: float  # 独自性 (0-1)

class DataValidator:
    """
    金融時系列データに対する包括的なバリデーションを行うクラス。

    データの品質を多角的に評価し、問題点を特定。
    """

    def __init__(self):
        """DataValidatorを初期化。"""
        self.validation_rules: dict[str, ValidationRule] = {}
        self._register_default_rules()

    def _register_default_rules(self):
        """デフォルトのバリデーションルールを登録。"""
        self.validation_rules = {
            "not_null": self._validate_not_null,
            "data_type": self._validate_data_type,
            "range": self._validate_range,
            "pattern": self._validate_pattern,
            "uniqueness": self._validate_uniqueness,
            "consistency": self._validate_consistency,
            "temporal_order": self._validate_temporal_order,
            "statistical_properties": self._validate_statistical_properties,
        }

    def validate_data(
        self,
        data: pd.DataFrame,
        schema: SchemaDefinition,
        additional_rules: list[AdditionalValidationRule] | None = None,
    ) -> ValidationResult:
        """
        データに対して包括的なバリデーションを実行。

        Args:
            data: バリデーション対象のデータ
            schema: データスキーマ定義
            additional_rules: 追加のバリデーションルール

        Returns:
            バリデーション結果

        Example:
            >>> validator = DataValidator()
            >>> schema = {
            ...     "price": {"type": "float", "range": [0, float('inf')], "not_null": True},
            ...     "volume": {"type": "int", "range": [0, 1000000], "not_null": True},
            ...     "timestamp": {"type": "datetime", "not_null": True}
            ... }
            >>> result = validator.validate_data(data, schema)
        """
        errors: StringList = []
        warnings: StringList = []
        metrics: MetricsDict = {}
        details: ObjectMap = {}

        # スキーマベースのバリデーション
        schema_result = self._validate_schema(data, schema)
        errors.extend(schema_result["errors"])
        warnings.extend(schema_result["warnings"])
        metrics.update(schema_result["metrics"])

        # 追加ルールの適用
        if additional_rules:
            for rule in additional_rules:
                rule_result = self._apply_validation_rule(data, rule)
                errors.extend(rule_result["errors"])
                warnings.extend(rule_result["warnings"])

        # データ品質メトリクスの計算
        quality_metrics = self._calculate_quality_metrics(data)
        metrics.update(
            {
                "completeness": quality_metrics.completeness,
                "accuracy": quality_metrics.accuracy,
                "timeliness": quality_metrics.timeliness,
                "consistency": quality_metrics.consistency,
                "validity": quality_metrics.validity,
                "uniqueness": quality_metrics.uniqueness,
            }
        )

        # 異常検知
        anomaly_result = self._detect_anomalies(data)
        warnings.extend(anomaly_result["warnings"])
        details.update(anomaly_result["details"])

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            details=details,
        )

    def _validate_schema(
        self, data: pd.DataFrame, schema: SchemaDefinition
    ) -> SchemaValidationResult:
        """
        スキーマベースのバリデーションを実行。

        Args:
            data: 対象データ
            schema: スキーマ定義

        Returns:
            バリデーション結果
        """
        errors: StringList = []
        warnings: StringList = []
        metrics: MetricsDict = {}

        for column, rules in schema.items():
            if column not in data.columns:
                if rules.get("required", False):
                    errors.append(f"Required column '{column}' is missing")
                continue

            column_data = data[column]

            # 各ルールの適用
            for rule_name, rule_value in rules.items():
                if rule_name in self.validation_rules:
                    try:
                        rule_result = self.validation_rules[rule_name](
                            column_data, rule_value
                        )
                        if not rule_result["valid"]:
                            self._append_rule_messages(
                                rule_result, errors, warnings
                            )
                    except Exception as e:
                        errors.append(
                            f"Validation failed for {column}.{rule_name}: {e}"
                        )

            # 列レベルのメトリクス計算
            row_count = float(len(column_data))
            metrics[f"{column}_completeness"] = 1 - self._safe_ratio(
                float(column_data.isnull().sum()),
                row_count,
            )
            non_null_count = float(len(column_data.dropna()))
            metrics[f"{column}_uniqueness"] = self._safe_ratio(
                float(column_data.nunique()),
                non_null_count,
                default=1.0,
            )

        return {"errors": errors, "warnings": warnings, "metrics": metrics}

    def _apply_validation_rule(
        self, data: pd.DataFrame, rule: AdditionalValidationRule
    ) -> RuleIssueResult:
        """
        カスタムバリデーションルールを適用。

        Args:
            data: 対象データ
            rule: ルール定義

        Returns:
            バリデーション結果
        """
        errors: StringList = []
        warnings: StringList = []

        rule_type = str(rule.get("type", ""))
        raw_columns = rule.get("columns", [])
        columns = (
            [str(col) for col in raw_columns]
            if isinstance(raw_columns, list)
            else []
        )
        raw_params = rule.get("params", {})
        params = raw_params if isinstance(raw_params, dict) else {}

        if rule_type not in self.validation_rules:
            errors.append(f"Unknown validation rule type: {rule_type}")
            return {"errors": errors, "warnings": warnings}

        for column in columns:
            if column not in data.columns:
                errors.append(f"Column '{column}' not found for rule {rule_type}")
                continue

            try:
                result = self.validation_rules[rule_type](data[column], **params)
                if not result["valid"]:
                    self._append_rule_messages(
                        result, errors, warnings, prefix=f"{column}: "
                    )
            except Exception as e:
                errors.append(f"Rule application failed for {column}.{rule_type}: {e}")

        return {"errors": errors, "warnings": warnings}

    def _validate_not_null(
        self, data: pd.Series, required: bool = True
    ) -> RuleEvaluationResult:
        """Null値チェック。"""
        null_count = data.isnull().sum()
        is_valid = not required or null_count == 0

        messages = []
        if null_count > 0:
            messages.append(f"Found {null_count} null values")

        return {
            "valid": is_valid,
            "level": "error" if required else "warning",
            "messages": messages,
        }

    def _validate_data_type(
        self, data: pd.Series, expected_type: str
    ) -> RuleEvaluationResult:
        """データ型チェック。"""
        try:
            if expected_type == "int":
                pd.to_numeric(data, downcast="integer")
            elif expected_type == "float":
                pd.to_numeric(data, downcast="float")
            elif expected_type == "datetime":
                pd.to_datetime(data)
            elif expected_type == "string":
                data.astype(str)
            else:
                return {
                    "valid": False,
                    "level": "error",
                    "messages": [f"Unsupported data type: {expected_type}"],
                }

            return {"valid": True, "level": "info", "messages": []}
        except Exception as e:
            return {
                "valid": False,
                "level": "error",
                "messages": [f"Data type conversion failed: {e}"],
            }

    def _validate_range(
        self,
        data: pd.Series,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> RuleEvaluationResult:
        """値範囲チェック。"""
        valid_data = data.dropna()

        messages = []
        is_valid = True

        if min_val is not None:
            below_min = (valid_data < min_val).sum()
            if below_min > 0:
                messages.append(f"{below_min} values below minimum {min_val}")
                is_valid = False

        if max_val is not None:
            above_max = (valid_data > max_val).sum()
            if above_max > 0:
                messages.append(f"{above_max} values above maximum {max_val}")
                is_valid = False

        return {"valid": is_valid, "level": "error", "messages": messages}

    def _validate_pattern(self, data: pd.Series, pattern: str) -> RuleEvaluationResult:
        """パターン一致チェック。"""

        valid_data = data.dropna().astype(str)
        matches = valid_data.str.match(pattern)

        non_matches = (~matches).sum()
        is_valid = non_matches == 0

        messages = []
        if non_matches > 0:
            messages.append(f"{non_matches} values do not match pattern {pattern}")

        return {"valid": is_valid, "level": "error", "messages": messages}

    def _validate_uniqueness(
        self, data: pd.Series, should_be_unique: bool = True
    ) -> RuleEvaluationResult:
        """独自性チェック。"""
        unique_count = data.nunique()
        total_count = len(data.dropna())

        is_unique = unique_count == total_count
        is_valid = is_unique if should_be_unique else True

        messages = []
        if should_be_unique and not is_unique:
            duplicates = total_count - unique_count
            messages.append(f"Found {duplicates} duplicate values")

        return {"valid": is_valid, "level": "warning", "messages": messages}

    def _validate_consistency(
        self, data: pd.Series, related_column: str | None = None
    ) -> RuleEvaluationResult:
        """一貫性チェック（関連列との整合性）。"""
        # この実装は具体的なビジネスロジックによる
        # 例: 価格と出来高の相関関係チェック
        messages = []
        is_valid = True

        # 基本的な統計的一貫性チェック
        if len(data.dropna()) > 10:
            # 極端な値の割合チェック
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            extreme_values = ((data < q1 - 3 * iqr) | (data > q3 + 3 * iqr)).sum()
            extreme_ratio = extreme_values / len(data.dropna())

            if extreme_ratio > 0.1:  # 10%以上が極端な値
                messages.append(
                    f"High proportion of extreme values: {extreme_ratio:.2%}"
                )
                is_valid = False

        return {"valid": is_valid, "level": "warning", "messages": messages}

    def _validate_temporal_order(
        self, data: pd.Series, is_datetime: bool = True
    ) -> RuleEvaluationResult:
        """時系列順序チェック。"""
        messages = []
        is_valid = True

        if is_datetime:
            try:
                datetime_data = pd.to_datetime(data.dropna())
                is_monotonic = datetime_data.is_monotonic_increasing
                if not is_monotonic:
                    messages.append("Datetime values are not in chronological order")
                    is_valid = False
            except Exception as e:
                messages.append(f"Datetime validation failed: {e}")
                is_valid = False

        return {"valid": is_valid, "level": "error", "messages": messages}

    def _validate_statistical_properties(
        self,
        data: pd.Series,
        expected_mean: float | None = None,
        expected_std: float | None = None,
        tolerance: float = 0.1,
    ) -> RuleEvaluationResult:
        """統計的特性チェック。"""
        messages = []
        is_valid = True

        valid_data = data.dropna()
        if len(valid_data) < 10:
            return {
                "valid": True,
                "level": "warning",
                "messages": ["Insufficient data for statistical validation"],
            }

        actual_mean = valid_data.mean()
        actual_std = valid_data.std()

        if expected_mean is not None:
            mean_diff = self._safe_ratio(
                abs(actual_mean - expected_mean),
                abs(expected_mean),
            )
            if mean_diff > tolerance:
                messages.append(
                    f"Mean deviation: expected {expected_mean}, got {actual_mean:.2f}"
                )
                is_valid = False

        if expected_std is not None:
            std_diff = self._safe_ratio(
                abs(actual_std - expected_std),
                abs(expected_std),
            )
            if std_diff > tolerance:
                messages.append(
                    f"Std deviation: expected {expected_std}, got {actual_std:.2f}"
                )
                is_valid = False

        return {"valid": is_valid, "level": "warning", "messages": messages}

    def _calculate_quality_metrics(self, data: pd.DataFrame) -> DataQualityMetrics:
        """
        データ品質メトリクスを計算。

        Args:
            data: 対象データ

        Returns:
            品質メトリクス
        """
        # 完全性: 非Null値の割合
        completeness = 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))

        # 正確性: 基本的な範囲チェック（仮定）
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        accuracy_scores = []

        for col in numeric_cols:
            valid_data = data[col].dropna()
            if len(valid_data) > 0:
                # 負の値や極端な値がないかをチェック
                negative_ratio = (valid_data < 0).mean()
                extreme_ratio = (
                    (valid_data < valid_data.quantile(0.01))
                    | (valid_data > valid_data.quantile(0.99))
                ).mean()
                accuracy_scores.append(1 - (negative_ratio + extreme_ratio) / 2)

        accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.5

        # 適時性: タイムスタンプの新しさ（仮定）
        timeliness = 0.8  # デフォルト値

        # 一貫性: 列間の相関関係
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            consistency = corr_matrix.abs().mean().mean()
        else:
            consistency = 0.5

        # 有効性: データ型の適切性
        validity_scores = []
        for col in data.columns:
            try:
                if data[col].dtype in ["int64", "float64", "object", "datetime64[ns]"]:
                    validity_scores.append(1.0)
                else:
                    validity_scores.append(0.5)
            except Exception:
                validity_scores.append(0.0)

        validity = np.mean(validity_scores) if validity_scores else 0.5

        # 独自性: 重複行の少なさ
        duplicate_ratio = data.duplicated().mean()
        uniqueness = 1 - duplicate_ratio

        return DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            timeliness=timeliness,
            consistency=consistency,
            validity=validity,
            uniqueness=uniqueness,
        )

    def _detect_anomalies(self, data: pd.DataFrame) -> AnomalyDetectionResult:
        """
        データの異常を検知。

        Args:
            data: 対象データ

        Returns:
            異常検知結果
        """
        warnings: StringList = []
        details: ObjectMap = {}

        # 欠損パターンの分析
        missing_pattern = data.isnull().sum(axis=1)
        if (missing_pattern > missing_pattern.mean() + 2 * missing_pattern.std()).any():
            warnings.append("Detected unusual missing data patterns")

        # 統計的異常の検知
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            values = data[col].dropna()
            if len(values) > 50:
                # 分布の変化検知（簡易版）
                first_half = values[: len(values) // 2]
                second_half = values[len(values) // 2 :]

                if len(first_half) > 10 and len(second_half) > 10:
                    try:
                        stat, p_value = stats.ks_2samp(first_half, second_half)
                        if p_value < 0.05:
                            warnings.append(
                                f"Distribution change detected in {col} (p={p_value:.3f})"
                            )
                            details[f"{col}_distribution_shift"] = {
                                "statistic": stat,
                                "p_value": p_value,
                            }
                    except Exception:
                        continue

        return {"warnings": warnings, "details": details}

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safely compute ratio without zero-division."""
        if abs(denominator) < 1e-12:
            return default
        return numerator / denominator

    @staticmethod
    def _append_rule_messages(
        result: RuleEvaluationResult,
        errors: StringList,
        warnings: StringList,
        prefix: str = "",
    ) -> None:
        """Route rule messages to error/warning buckets."""
        formatted_messages = [f"{prefix}{msg}" for msg in result["messages"]]
        if result["level"] == "error":
            errors.extend(formatted_messages)
            return
        warnings.extend(formatted_messages)

class DataIntegrityChecker:
    """
    データ整合性をチェックするクラス。
    """

    def __init__(self):
        """DataIntegrityCheckerを初期化。"""
        pass

    def check_integrity(self, data: pd.DataFrame) -> ValidationResult:
        """
        データの整合性を包括的にチェック。

        Args:
            data: チェック対象のデータ

        Returns:
            整合性チェック結果
        """
        errors = []
        warnings = []
        metrics = {}
        details = {}

        # 基本的な整合性チェック
        integrity_checks = [
            self._check_data_types,
            self._check_index_integrity,
            self._check_column_consistency,
            self._check_value_ranges,
            self._check_temporal_consistency,
            self._check_business_rules,
        ]

        for check_func in integrity_checks:
            try:
                result = check_func(data)
                errors.extend(result.get("errors", []))
                warnings.extend(result.get("warnings", []))
                metrics.update(result.get("metrics", {}))
                details.update(result.get("details", {}))
            except Exception as e:
                errors.append(f"Integrity check failed: {e}")

        # 全体的な品質スコア計算
        quality_score = self._calculate_integrity_score(errors, warnings, metrics)
        metrics["overall_integrity_score"] = quality_score

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            details=details,
        )

    def _check_data_types(self, data: pd.DataFrame) -> IntegrityCheckResult:
        """データ型の整合性をチェック。"""
        errors: StringList = []
        warnings: StringList = []
        metrics: MetricsDict = {}

        for col in data.columns:
            previous_error_count = len(errors)
            dtype = data[col].dtype

            # 数値列のチェック
            if col.lower() in ["price", "volume", "amount", "quantity"]:
                if not np.issubdtype(dtype, np.number):
                    errors.append(f"Column '{col}' should be numeric but is {dtype}")
                else:
                    # 負の値チェック
                    if (data[col] < 0).any():
                        warnings.append(f"Column '{col}' contains negative values")

            # タイムスタンプ列のチェック
            elif col.lower() in ["timestamp", "datetime", "time", "date"]:
                try:
                    pd.to_datetime(data[col])
                except Exception:
                    errors.append(f"Column '{col}' is not a valid datetime column")

            had_column_error = len(errors) > previous_error_count
            metrics[f"{col}_dtype_consistency"] = 0.0 if had_column_error else 1.0

        return {"errors": errors, "warnings": warnings, "metrics": metrics, "details": {}}

    def _check_index_integrity(self, data: pd.DataFrame) -> IntegrityCheckResult:
        """インデックスの整合性をチェック。"""
        errors: StringList = []
        warnings: StringList = []
        metrics: MetricsDict = {}

        # 重複インデックスチェック
        if data.index.duplicated().any():
            duplicate_count = data.index.duplicated().sum()
            errors.append(f"Found {duplicate_count} duplicate index values")

        # インデックスの順序チェック
        if not data.index.is_monotonic_increasing:
            warnings.append("Index is not in ascending order")

        # 欠損インデックスチェック
        if data.index.isnull().any():
            null_count = data.index.isnull().sum()
            errors.append(f"Found {null_count} null index values")

        metrics["index_integrity"] = 1.0 if len(errors) == 0 else 0.0

        return {"errors": errors, "warnings": warnings, "metrics": metrics, "details": {}}

    def _check_column_consistency(self, data: pd.DataFrame) -> IntegrityCheckResult:
        """列間の整合性をチェック。"""
        errors: StringList = []
        warnings: StringList = []
        metrics: MetricsDict = {}

        # 価格と出来高の相関関係チェック（取引データの場合）
        price_cols = [col for col in data.columns if "price" in col.lower()]
        volume_cols = [col for col in data.columns if "volume" in col.lower()]

        if price_cols and volume_cols:
            for price_col in price_cols:
                for volume_col in volume_cols:
                    if data[price_col].corr(data[volume_col]) < -0.1:  # 負の相関は異常
                        warnings.append(
                            f"Unusual correlation between {price_col} and {volume_col}"
                        )

        # 列名の整合性チェック
        invalid_chars = [" ", "-", "/", "\\"]
        for col in data.columns:
            for char in invalid_chars:
                if char in col:
                    warnings.append(
                        f"Column name '{col}' contains invalid character '{char}'"
                    )

        metrics["column_consistency"] = 1.0 if len(errors) == 0 else 0.5

        return {"errors": errors, "warnings": warnings, "metrics": metrics, "details": {}}

    def _check_value_ranges(self, data: pd.DataFrame) -> IntegrityCheckResult:
        """値範囲の妥当性をチェック。"""
        errors: StringList = []
        warnings: StringList = []
        metrics: MetricsDict = {}

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            values = data[col].dropna()

            if len(values) == 0:
                continue

            # 基本的な範囲チェック
            if "price" in col.lower():
                if (values <= 0).any():
                    errors.append(f"Column '{col}' contains non-positive prices")
                elif (values > values.quantile(0.99) * 10).any():
                    warnings.append(f"Column '{col}' contains extremely high values")

            elif "volume" in col.lower():
                if (values < 0).any():
                    errors.append(f"Column '{col}' contains negative volumes")

            # ゼロ分散チェック
            if values.std() == 0:
                warnings.append(f"Column '{col}' has zero variance")

            # 極端な値の割合
            extreme_ratio = (
                (values < values.quantile(0.01)) | (values > values.quantile(0.99))
            ).mean()
            if extreme_ratio > 0.1:
                warnings.append(
                    f"Column '{col}' has {extreme_ratio:.1%} extreme values"
                )

        metrics["value_range_validity"] = 1.0 if len(errors) == 0 else 0.0

        return {"errors": errors, "warnings": warnings, "metrics": metrics, "details": {}}

    def _check_temporal_consistency(self, data: pd.DataFrame) -> IntegrityCheckResult:
        """時系列データの整合性をチェック。"""
        errors: StringList = []
        warnings: StringList = []
        metrics: MetricsDict = {}

        # タイムスタンプ列の検出
        datetime_cols = []
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                datetime_cols.append(col)
                continue

            # Heuristic: if column name suggests timestamp, coerce and check success rate
            if any(k in col.lower() for k in ("timestamp", "ts", "date")):
                coerced = pd.to_datetime(data[col], errors="coerce")
                if coerced.notna().sum() >= max(1, len(coerced) * 0.7):
                    datetime_cols.append(col)
                    continue

        if datetime_cols:
            for col in datetime_cols:
                datetime_values = pd.to_datetime(data[col].dropna())

                # 時系列順序チェック
                if not datetime_values.is_monotonic_increasing:
                    errors.append(
                        f"Column '{col}' timestamps are not in chronological order"
                    )

                # 未来の日付チェック
                future_dates = (datetime_values > datetime.now()).sum()
                if future_dates > 0:
                    warnings.append(
                        f"Column '{col}' contains {future_dates} future dates"
                    )

                # 時間間隔のチェック
                if len(datetime_values) > 1:
                    time_diffs = datetime_values.diff().dropna()
                    median_diff = time_diffs.median()

                    # 不規則な間隔の検出
                    irregular_intervals = (time_diffs < median_diff * 0.1).sum()
                    if irregular_intervals > len(time_diffs) * 0.1:
                        warnings.append(f"Column '{col}' has irregular time intervals")

        metrics["temporal_consistency"] = 1.0 if len(errors) == 0 else 0.0

        return {"errors": errors, "warnings": warnings, "metrics": metrics, "details": {}}

    def _check_business_rules(self, data: pd.DataFrame) -> IntegrityCheckResult:
        """ビジネスルールベースの整合性チェック。"""
        errors: StringList = []
        warnings: StringList = []
        metrics: MetricsDict = {}

        # 取引データのビジネスルール（例）
        if "price" in data.columns and "volume" in data.columns:
            # 高価格・低出来高の異常パターン
            price_q95 = data["price"].quantile(0.95)
            volume_q5 = data["volume"].quantile(0.05)

            suspicious_trades = (
                (data["price"] > price_q95) & (data["volume"] < volume_q5)
            ).sum()
            if suspicious_trades > 0:
                warnings.append(
                    f"Found {suspicious_trades} suspicious high-price low-volume trades"
                )

        # OHLCデータの整合性チェック
        ohlc_cols = ["open", "high", "low", "close"]
        available_ohlc = [col for col in ohlc_cols if col in data.columns]

        if len(available_ohlc) >= 4:
            # OHLCの論理関係チェック
            invalid_ohlc = (
                (data["high"] < data["low"])
                | (data["open"] > data["high"])
                | (data["open"] < data["low"])
                | (data["close"] > data["high"])
                | (data["close"] < data["low"])
            ).sum()

            if invalid_ohlc > 0:
                errors.append(f"Found {invalid_ohlc} invalid OHLC relationships")

        metrics["business_rule_compliance"] = 1.0 if len(errors) == 0 else 0.0

        return {"errors": errors, "warnings": warnings, "metrics": metrics, "details": {}}

    def _calculate_integrity_score(
        self, errors: StringList, warnings: StringList, metrics: MetricsDict
    ) -> float:
        """
        全体的な整合性スコアを計算。

        Args:
            errors: エラーリスト
            warnings: 警告リスト
            metrics: メトリクス辞書

        Returns:
            整合性スコア (0-1)
        """
        # エラーの重み
        error_penalty = len(errors) * 0.5

        # 警告の重み
        warning_penalty = len(warnings) * 0.1

        # メトリクスの平均
        if metrics:
            metric_score = np.mean(list(metrics.values()))
        else:
            metric_score = 0.5

        # 最終スコア計算
        integrity_score = max(0, min(1, metric_score - error_penalty - warning_penalty))

        return integrity_score
