"""
SAC v395f 詳細分析と次の改善案
"""

# ログから分かること
results_v395f = {
    "critic_loss": {
        "initial": 9.78e7,
        "final": 6.36e7,
        "min": 3.11e6,
        "max": 3.58e8,
        "trend": "不安定（爆発的な変動）"
    },
    "actor_loss": {
        "initial": -7.7e4,
        "final": -4.05e4,
        "trend": "負の値で安定（異常）"
    },
    "ent_coef": {
        "initial": 1.09,
        "final": 3.58,
        "trend": "上昇傾向（まだ高い）"
    }
}

## 🔍 Actor Lossが負の理由

"""
SACのActor Loss:
actor_loss = alpha * log_prob - Q_value

負の値 = Q_valueが非常に大きい
→ 報酬（reward）が大きすぎる
→ Q値の推定が爆発
"""

## 💡 解決策: さらに報酬を小さく

# 現在の設定
current_reward_scale = 1000.0  # 0.1%利益 → 報酬 1.0
current_clip = (-10.0, 10.0)   # ±1%で飽和

# 問題: 報酬が-10 ~ +10 は、まだ大きすぎる
# SACは off-policy なので、Q値が積算される
# reward_scale を 100.0 に下げる → 報酬 -1.0 ~ +1.0

# 推奨設定 v395g
recommended_v395g = {
    "reward_scale": 100.0,      # 1/10に縮小
    "reward_clip_min": -1.0,    # 1/10に縮小
    "reward_clip_max": 1.0,     # 1/10に縮小
}

"""
新しい報酬解釈:
- 0.1%利益 → 報酬 0.1
- 1.0%利益 → 報酬 1.0（最大）
- 0.1%損失 → 報酬 -0.1
- 1.0%損失 → 報酬 -1.0（最小）

これにより:
- Q値が [-10, 10] 程度に収まる（γ=0.99で10ステップ先まで考慮）
- Critic Lossが < 100 に
- Actor Lossが正の小さい値に
- ent_coefが 0.5-1.5 で安定
"""

## 📋 v395g 設定案

v395g_config = {
    "model_name": "sac_v395g_micro_reward",
    "notes": "報酬スケールを1/10に縮小（-1.0 ~ +1.0）",
    "sac_hyperparameters": {
        "learning_rate": 0.0003,
        "buffer_size": 20000,
        "learning_starts": 500,
        "batch_size": 128,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "target_update_interval": 1,
        "target_entropy": "auto"  # -1.0 for dim=1
    },
    "environment": {
        "reward_settings": {
            "use_simple_reward": True,
            "reward_scale": 100.0,        # ← 1000.0から変更
            "reward_clip_min": -1.0,      # ← -10.0から変更
            "reward_clip_max": 1.0,       # ← 10.0から変更
            "enable_inactivity_penalty": False
        }
    }
}

print("=" * 80)
print("📊 SAC v395f Analysis")
print("=" * 80)
print("\n🔍 Key Findings:")
print(f"  • Critic Loss: {results_v395f['critic_loss']['min']:.2e} ~ {results_v395f['critic_loss']['max']:.2e}")
print(f"  • Actor Loss: {results_v395f['actor_loss']['final']:.2e} (NEGATIVE! 異常)")
print(f"  • ent_coef: {results_v395f['ent_coef']['final']:.2f} (まだ高い)")

print("\n💡 Root Cause:")
print("  • 報酬範囲 [-10, 10] が大きすぎる")
print("  • Q値が爆発（Actor Lossが負）")
print("  • γ=0.99 で10ステップ先まで考慮 → Q値が積算")

print("\n🎯 Solution: v395g")
print("  • reward_scale: 1000.0 → 100.0 (1/10)")
print("  • reward_clip: [-10, 10] → [-1, 1] (1/10)")
print("  • 期待される報酬: -1.0 ~ +1.0")
print("  • 期待されるQ値: -10 ~ +10")
print("  • 期待されるCritic Loss: < 100")

print("\n📊 Expected Improvements:")
print("  ✅ Critic Loss: < 100 (現在: 1e6-1e8)")
print("  ✅ Actor Loss: 0.1 ~ 10.0 (現在: -4e4)")
print("  ✅ ent_coef: 0.5 ~ 1.5 (現在: 3.58)")
print("=" * 80)
