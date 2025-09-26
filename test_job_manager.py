#!/usr/bin/env python3
"""
Test script for JobManager with mock training function
"""
import sys
import time
import random
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ztb.experiments.job_manager import JobManager

def mock_train_function(job_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mock training function that simulates ML training
    """
    job_id = job_config.get("job_id", "unknown")
    print(f"Starting mock training for job {job_id}")

    # Simulate training time (1-5 seconds)
    training_time = random.uniform(1, 5)
    time.sleep(training_time)

    # Simulate success/failure (90% success rate)
    if random.random() < 0.9:
        result = {
            "job_id": job_id,
            "status": "completed",
            "metrics": {
                "accuracy": random.uniform(0.5, 0.95),
                "loss": random.uniform(0.1, 1.0),
                "training_time": training_time
            },
            "timestamp": time.time()
        }
    else:
        raise Exception(f"Mock training failed for job {job_id}")

    print(f"Completed mock training for job {job_id}")
    return result

def main():
    """Test JobManager with mock training"""
    print("Testing JobManager with mock training function...")

    # Create job manager with small timeout for testing
    manager = JobManager(base_dir="tmp-test-job-manager", timeout_hours=1)  # 1 hour, but we'll interrupt

    # Temporarily modify for small test
    manager.num_repeats = 1  # Only 1 repeat instead of 10
    manager.job_size = 10000  # 10k steps per job instead of 100k

    # For testing, run sequentially to avoid pickle issues
    jobs = manager.split_jobs()
    test_jobs = jobs[:10]  # Test with 10 jobs

    print(f"Testing with {len(test_jobs)} jobs sequentially")

    completed_jobs = []
    failed_jobs = []

    for job in test_jobs:
        try:
            result = manager.execute_job(job, mock_train_function)
            if result["status"] == "completed":
                completed_jobs.append(result)
            else:
                failed_jobs.append(result)
        except Exception as e:
            print(f"Job {job['job_id']} failed: {e}")
            failed_jobs.append({
                "job_id": job["job_id"],
                "status": "failed",
                "error": str(e)
            })

    # Aggregate results manually for testing
    results = {
        "total_jobs": len(test_jobs),
        "completed_jobs": len(completed_jobs),
        "failed_jobs": len(failed_jobs),
        "success_rate": len(completed_jobs) / len(test_jobs) if test_jobs else 0,
        "job_results": completed_jobs
    }

    # Create job configurations (small test: 10 jobs instead of 100k)
    jobs = manager.split_jobs()
    # For testing, use only first 10 jobs
    test_jobs = jobs[:10]
    print(f"Testing with {len(test_jobs)} jobs out of {len(jobs)} total")

    # Run jobs manually for testing
    completed_jobs = []
    failed_jobs = []

    for job in test_jobs:
        try:
            result = manager.execute_job(job, mock_train_function)
            if result["status"] == "completed":
                completed_jobs.append(result)
            else:
                failed_jobs.append(result)
        except Exception as e:
            print(f"Job {job['job_id']} failed: {e}")
            failed_jobs.append({
                "job_id": job["job_id"],
                "status": "failed",
                "error": str(e)
            })

    # Aggregate results
    print("Test completed!")
    print(f"Total jobs: {results['total_jobs']}")
    print(f"Completed: {results['completed_jobs']}")
    print(f"Failed: {results['failed_jobs']}")

    if len(completed_jobs) > 0:
        print("Sample metrics from first completed job:")
        sample_job = completed_jobs[0]
        print(f"  Accuracy: {sample_job['result']['metrics']['accuracy']:.3f}")
        print(f"  Loss: {sample_job['result']['metrics']['loss']:.3f}")
        print(f"  Training time: {sample_job['result']['metrics']['training_time']:.2f}s")

if __name__ == "__main__":
    main()