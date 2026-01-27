"""
Performance Profiling Utilities for SynDX Pipeline

Measures execution time, memory usage, and scalability metrics
for replacing simulated data in Figures 6 and 8.

Author: Chatchai Tritham
Date: 2026-01-25
"""

import time
import psutil
import functools
from typing import Dict, Callable, Any, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SynDXProfiler:
    """
    Performance profiling for SynDX pipeline operations.

    Captures:
    - Execution time per function/phase
    - Memory consumption (start/end/delta)
    - Phase-level aggregated metrics
    - Scalability data for parameter sweeps

    Usage:
        profiler = SynDXProfiler(output_dir='outputs/profiling')

        @profiler.profile_time('phase1_sampling')
        def my_function():
            # ... code ...
            pass

        profiler.save_metrics('profiling_results.json')
    """

    def __init__(
            self,
            output_dir: str = 'outputs/profiling',
            enabled: bool = True):
        """
        Initialize profiler.

        Args:
            output_dir: Directory for saving profiling results
            enabled: If False, profiling is disabled (no overhead)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

        # Metrics storage
        self.metrics = {}  # Function-level metrics
        self.phase_metrics = {}  # Phase-level aggregated metrics

        # Process handle for memory monitoring
        self.process = psutil.Process()

        logger.info(
            f"Initialized SynDXProfiler (enabled={enabled}, output_dir={output_dir})")

    def profile_time(self, name: str):
        """
        Decorator for timing function execution with memory tracking.

        Args:
            name: Descriptive name for the profiled function

        Returns:
            Decorator function

        Example:
            @profiler.profile_time('vae_training')
            def train_vae(data):
                # ... training code ...
                pass
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                if not self.enabled:
                    # Profiling disabled - no overhead
                    return func(*args, **kwargs)

                # Capture start metrics
                start_time = time.perf_counter()
                start_memory = self.process.memory_info().rss / 1024**2  # MB

                try:
                    # Execute function
                    result = func(*args, **kwargs)

                    # Capture end metrics
                    end_time = time.perf_counter()
                    end_memory = self.process.memory_info().rss / 1024**2

                    # Store metrics
                    self.metrics[name] = {
                        'execution_time_sec': end_time - start_time,
                        'memory_start_mb': start_memory,
                        'memory_end_mb': end_memory,
                        'memory_delta_mb': end_memory - start_memory,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'status': 'success'
                    }

                    logger.debug(
                        f"Profiled '{name}': {end_time - start_time:.3f}s, "
                        f"Δmem={end_memory - start_memory:.2f}MB"
                    )

                    return result

                except Exception as e:
                    # Record failure
                    end_time = time.perf_counter()
                    self.metrics[name] = {
                        'execution_time_sec': end_time - start_time,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'status': 'failed',
                        'error': str(e)
                    }
                    raise

            return wrapper
        return decorator

    def start_phase(self, phase_name: str) -> float:
        """
        Mark the start of a phase for manual timing.

        Args:
            phase_name: Name of the phase

        Returns:
            Start timestamp (for use with end_phase)

        Example:
            start = profiler.start_phase('Phase 1')
            # ... phase code ...
            profiler.end_phase('Phase 1', start)
        """
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss / 1024**2

        if phase_name not in self.phase_metrics:
            self.phase_metrics[phase_name] = {
                'start_time': start_time,
                'start_memory_mb': start_memory,
                'subphases': {}
            }

        return start_time

    def end_phase(self, phase_name: str, start_time: float):
        """
        Mark the end of a phase and record metrics.

        Args:
            phase_name: Name of the phase
            start_time: Start timestamp from start_phase()
        """
        if not self.enabled:
            return

        end_time = time.perf_counter()
        end_memory = self.process.memory_info().rss / 1024**2

        if phase_name in self.phase_metrics:
            self.phase_metrics[phase_name].update({
                'total_time_sec': end_time - start_time,
                'end_memory_mb': end_memory,
                # Simplified (could use max tracking)
                'peak_memory_mb': end_memory,
                'memory_delta_mb': end_memory - self.phase_metrics[phase_name]['start_memory_mb']
            })

            logger.info(
                f"Phase '{phase_name}' completed: {
                    end_time -
                    start_time:.2f}s, " f"peak_mem={
                    end_memory:.1f}MB")

    def record_subphase(self, phase_name: str, subphase_name: str,
                        time_sec: float, memory_mb: Optional[float] = None):
        """
        Record metrics for a subphase within a phase.

        Args:
            phase_name: Parent phase name
            subphase_name: Subphase name
            time_sec: Execution time in seconds
            memory_mb: Optional memory consumption
        """
        if not self.enabled:
            return

        if phase_name not in self.phase_metrics:
            self.phase_metrics[phase_name] = {'subphases': {}}

        self.phase_metrics[phase_name]['subphases'][subphase_name] = {
            'time_sec': time_sec,
            'memory_mb': memory_mb
        }

    def get_phase_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for phase-level metrics.

        Returns:
            Dictionary with aggregated phase metrics

        Example output:
            {
                'phases': ['Phase 1', 'Phase 2', ...],
                'times': [12.3, 45.6, ...],
                'memory': [120.5, 340.2, ...]
            }
        """
        summary = {
            'phases': [],
            'times': [],
            'memory': []
        }

        for phase_name, metrics in self.phase_metrics.items():
            summary['phases'].append(phase_name)
            summary['times'].append(metrics.get('total_time_sec', 0))
            summary['memory'].append(metrics.get('peak_memory_mb', 0))

        return summary

    def save_metrics(self, filename: str = 'profiling_results.json') -> Path:
        """
        Save profiling results to JSON file.

        Args:
            filename: Output filename

        Returns:
            Path to saved file
        """
        if not self.enabled:
            logger.warning("Profiling disabled - no metrics to save")
            return None

        output_path = self.output_dir / filename

        all_metrics = {
            'function_metrics': self.metrics,
            'phase_metrics': self.phase_metrics,
            'summary': self.get_phase_summary()
        }

        with open(output_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        logger.info(f"Profiling metrics saved to {output_path}")
        return output_path

    def load_metrics(self, filename: str = 'profiling_results.json') -> Dict:
        """
        Load previously saved profiling metrics.

        Args:
            filename: Input filename

        Returns:
            Dictionary of metrics
        """
        filepath = self.output_dir / filename

        if not filepath.exists():
            logger.error(f"Metrics file not found: {filepath}")
            return {}

        with open(filepath, 'r') as f:
            metrics = json.load(f)

        logger.info(f"Loaded metrics from {filepath}")
        return metrics

    def generate_report(
            self,
            output_file: str = 'profiling_report.txt') -> str:
        """
        Generate human-readable profiling report.

        Args:
            output_file: Output filename for report

        Returns:
            Report content as string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SynDX PROFILING REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Function-level metrics
        report_lines.append("FUNCTION-LEVEL METRICS")
        report_lines.append("-" * 80)

        if self.metrics:
            for func_name, metrics in sorted(self.metrics.items()):
                time_sec = metrics.get('execution_time_sec', 0)
                mem_delta = metrics.get('memory_delta_mb', 0)
                status = metrics.get('status', 'unknown')

                report_lines.append(
                    f"{func_name:40s} {time_sec:8.3f}s  Δmem={mem_delta:+8.2f}MB  [{status}]"
                )
        else:
            report_lines.append("(No function metrics recorded)")

        report_lines.append("")

        # Phase-level metrics
        report_lines.append("PHASE-LEVEL METRICS")
        report_lines.append("-" * 80)

        if self.phase_metrics:
            for phase_name, metrics in sorted(self.phase_metrics.items()):
                total_time = metrics.get('total_time_sec', 0)
                peak_mem = metrics.get('peak_memory_mb', 0)

                report_lines.append(f"\n{phase_name}:")
                report_lines.append(f"  Total time: {total_time:.2f}s")
                report_lines.append(f"  Peak memory: {peak_mem:.1f}MB")

                if 'subphases' in metrics and metrics['subphases']:
                    report_lines.append("  Subphases:")
                    for subphase, sub_metrics in metrics['subphases'].items():
                        sub_time = sub_metrics.get('time_sec', 0)
                        report_lines.append(
                            f"    - {subphase:30s} {sub_time:.3f}s")
        else:
            report_lines.append("(No phase metrics recorded)")

        report_lines.append("")
        report_lines.append("=" * 80)

        report_content = '\n'.join(report_lines)

        # Save to file
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            f.write(report_content)

        logger.info(f"Profiling report saved to {output_path}")

        return report_content

    def reset(self):
        """Reset all profiling metrics."""
        self.metrics.clear()
        self.phase_metrics.clear()
        logger.info("Profiling metrics reset")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto-save metrics."""
        if self.enabled:
            self.save_metrics()


# Convenience function for one-off timing
def profile_function(func: Callable, *args, **kwargs) -> tuple:
    """
    Profile a single function call without decorator.

    Args:
        func: Function to profile
        *args, **kwargs: Arguments to pass to function

    Returns:
        Tuple of (result, execution_time_sec, memory_delta_mb)

    Example:
        result, time_sec, mem_mb = profile_function(my_func, arg1, arg2)
        print(f"Execution took {time_sec:.2f}s and used {mem_mb:.1f}MB")
    """
    process = psutil.Process()

    start_time = time.perf_counter()
    start_memory = process.memory_info().rss / 1024**2

    result = func(*args, **kwargs)

    end_time = time.perf_counter()
    end_memory = process.memory_info().rss / 1024**2

    execution_time = end_time - start_time
    memory_delta = end_memory - start_memory

    return result, execution_time, memory_delta


# Main demonstration
if __name__ == '__main__':
    import numpy as np

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("SynDXProfiler - Demo Mode")

    # Create profiler
    profiler = SynDXProfiler(output_dir='outputs/profiling_demo', enabled=True)

    # Example 1: Decorator-based profiling
    @profiler.profile_time('matrix_multiplication')
    def heavy_computation():
        """Simulate heavy computation"""
        size = 2000
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        C = np.dot(A, B)
        return C

    logger.info("Running heavy computation...")
    result = heavy_computation()

    # Example 2: Manual phase tracking
    phase_start = profiler.start_phase('Data Processing Phase')

    @profiler.profile_time('data_generation')
    def generate_data():
        return np.random.rand(10000, 100)

    @profiler.profile_time('data_normalization')
    def normalize_data(data):
        return (data - data.mean(axis=0)) / data.std(axis=0)

    data = generate_data()
    normalized = normalize_data(data)

    profiler.end_phase('Data Processing Phase', phase_start)

    # Save metrics
    profiler.save_metrics('demo_profiling.json')

    # Generate report
    report = profiler.generate_report('demo_report.txt')
    print("\n" + report)

    logger.info("Demo complete!")
