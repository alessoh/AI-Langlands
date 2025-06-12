#!/usr/bin/env python3
"""
Main program for discovering murmuration patterns in elliptic curves.

This demonstrates how AI can reveal hidden mathematical structures
that support the Langlands program.

Usage:
    python murmuration_discovery.py
    python murmuration_discovery.py --num-curves 50000 --max-prime 200
"""

import numpy as np
import click
import os
import sys
from typing import Dict

# Import our modules
from elliptic_curves import (
    generate_curve_family, 
    generate_primes, 
    compute_frobenius_traces,
    compute_average_traces,
    detect_murmuration_period
)
from ml_patterns import discover_hidden_structure
from visualizations import (
    plot_murmuration_patterns,
    plot_pca_analysis,
    plot_ml_results,
    plot_comprehensive_analysis,
    create_summary_figure
)
from utils import (
    save_curves_data,
    load_curves_data,
    save_patterns,
    generate_report,
    print_summary
)

def print_banner():
    """Print welcome banner."""
    banner = r"""
    ╔═══════════════════════════════════════════════════════════╗
    ║                 AI LANGLANDS PROGRAM                      ║
    ║          Discovering Mathematical Murmurations             ║
    ╚═══════════════════════════════════════════════════════════╝
    
    Following in the footsteps of Pozdnyakov's discovery...
    """
    print(banner)

@click.command()
@click.option('--num-curves', default=10000, help='Number of elliptic curves to generate')
@click.option('--max-prime', default=100, help='Maximum prime for computing traces')
@click.option('--seed', default=42, help='Random seed for reproducibility')
@click.option('--use-cache', is_flag=True, help='Use cached data if available')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def main(num_curves: int, max_prime: int, seed: int, use_cache: bool, debug: bool):
    """
    Main discovery pipeline.
    
    This recreates the essence of how AI discovered murmurations:
    1. Generate elliptic curves with known ranks
    2. Compute their Frobenius traces
    3. Apply machine learning to find patterns
    4. Visualize the hidden murmurations
    5. Connect to the Langlands program
    """
    print_banner()
    
    # Set random seeds
    np.random.seed(seed)
    
    # Step 1: Generate or load elliptic curves
    print("\n" + "="*50)
    print("STEP 1: ELLIPTIC CURVE GENERATION")
    print("="*50)
    
    data = None
    if use_cache:
        data = load_curves_data()
    
    if data is None:
        # Generate new data
        curves = generate_curve_family(num_curves, seed=seed)
        primes = generate_primes(max_prime)
        
        print(f"\nGenerated {len(curves)} elliptic curves")
        print(f"Using {len(primes)} primes up to {max(primes)}")
        
        # Compute Frobenius traces
        traces = compute_frobenius_traces(curves, primes)
        
        # Save for future use
        save_curves_data(curves, traces, primes)
    else:
        curves = data['curves']
        traces = data['traces']
        primes = data['primes']
        print(f"\nLoaded {len(curves)} curves from cache")
    
    # Extract ranks
    ranks = [curve.rank for curve in curves]
    
    # Step 2: Discover murmurations
    print("\n" + "="*50)
    print("STEP 2: DISCOVERING MURMURATION PATTERNS")
    print("="*50)
    
    average_traces = compute_average_traces(curves, traces, primes)
    
    print("\nComputing oscillation periods...")
    periods = {}
    for rank, avg_trace in average_traces.items():
        period = detect_murmuration_period(avg_trace, primes)
        periods[rank] = period
        print(f"  Rank {rank}: period ≈ {period:.1f}")
    
    # Step 3: Machine learning analysis
    print("\n" + "="*50)
    print("STEP 3: MACHINE LEARNING ANALYSIS")
    print("="*50)
    
    patterns = discover_hidden_structure(traces, ranks)
    patterns['primes'] = primes  # Add for later use
    
    # Step 4: Create visualizations
    print("\n" + "="*50)
    print("STEP 4: CREATING VISUALIZATIONS")
    print("="*50)
    
    # Murmuration patterns - the key discovery!
    plot_murmuration_patterns(average_traces, primes)
    
    # PCA analysis
    from ml_patterns import MurmurationAnalyzer
    analyzer = MurmurationAnalyzer(traces, ranks)
    X_pca = analyzer.apply_pca(n_components=2)
    plot_pca_analysis(X_pca, ranks)
    
    # ML results
    if 'ml_results' in patterns:
        plot_ml_results(patterns['ml_results'])
    
    # Comprehensive analysis
    plot_comprehensive_analysis(average_traces, primes, periods)
    
    # Summary figure
    create_summary_figure(patterns)
    
    # Step 5: Generate report
    print("\n" + "="*50)
    print("STEP 5: GENERATING REPORT")
    print("="*50)
    
    report = generate_report(patterns, average_traces, 
                           len(curves), len(primes))
    
    # Save patterns for future analysis
    save_patterns(patterns)
    
    # Print summary
    print_summary(patterns)
    
    # Final message
    print("\n" + "="*50)
    print("DISCOVERY COMPLETE!")
    print("="*50)
    print("\nThe murmurations have been revealed!")
    print("Check the 'output' directory for visualizations and report.")
    print("\nThese patterns existed for centuries, waiting to be discovered.")
    print("AI has given us new eyes to see the mathematical universe.")
    
    if debug:
        print("\n[DEBUG] Pattern details:")
        for key, value in patterns.items():
            if key != 'ml_results':  # Skip large objects
                print(f"  {key}: {value}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)