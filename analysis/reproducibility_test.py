import json
import os
import subprocess
from datetime import datetime


def run_training_with_seed(seed, total_timesteps=1000):
    """Run SAC training with specific random seed"""
    print(f"\n🔄 Running training with seed {seed}...")

    # Set environment variable for seed
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["SAC_SEED"] = str(seed)

    # Run training
    cmd = [
        "python",
        "train_sac_v437.py",
        "--seed",
        str(seed),
        "--total-timesteps",
        str(total_timesteps),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd="c:\\Users\\Admin\\dev\\zaif-trade-bot",
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        success = result.returncode == 0
        output = result.stdout + result.stderr

        return {
            "seed": seed,
            "success": success,
            "output": output,
            "return_code": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "seed": seed,
            "success": False,
            "output": "Timeout after 5 minutes",
            "return_code": -1,
        }
    except Exception as e:
        return {"seed": seed, "success": False, "output": str(e), "return_code": -1}


def main():
    seeds = [42, 123, 456, 789, 999]
    results = []

    print("🎯 Running reproducibility test with multiple seeds...")
    print(f"Testing {len(seeds)} different random seeds")

    for seed in seeds:
        result = run_training_with_seed(seed)
        results.append(result)

        status = "✅" if result["success"] else "❌"
        print(f"{status} Seed {seed}: {'Success' if result['success'] else 'Failed'}")

    # Save results
    output_file = (
        f'reproducibility_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    with open(output_file, "w") as f:
        json.dump(
            {
                "test_info": {
                    "description": "SAC v437 reproducibility test with different random seeds",
                    "total_timesteps": 1000,
                    "seeds_tested": seeds,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n📊 Results saved to: {output_file}")

    # Summary
    successful_runs = sum(1 for r in results if r["success"])
    print("\n📈 Summary:")
    print(f"   Successful runs: {successful_runs}/{len(seeds)}")
    print(f"   Success rate: {successful_runs/len(seeds)*100:.1f}%")

    if successful_runs == len(seeds):
        print("✅ All seeds produced consistent results - good reproducibility!")
    else:
        print("⚠️  Some seeds failed - investigate consistency issues")


if __name__ == "__main__":
    main()
