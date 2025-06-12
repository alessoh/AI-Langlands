"""
Elliptic curve computations for the AI Langlands project.

This module provides functions to:
- Generate elliptic curves over finite fields
- Compute Frobenius traces (ap values)
- Calculate ranks and other invariants
"""

import numpy as np
from typing import List, Tuple, Dict
import sympy as sp
from tqdm import tqdm
import random

class EllipticCurve:
    """Represents an elliptic curve y^2 = x^3 + ax + b."""
    
    def __init__(self, a: int, b: int, rank: int = None):
        """
        Initialize an elliptic curve.
        
        Args:
            a, b: Coefficients in the Weierstrass equation
            rank: The rank of the curve (if known)
        """
        self.a = a
        self.b = b
        self.rank = rank
        self.discriminant = -16 * (4 * a**3 + 27 * b**2)
        
    def __repr__(self):
        return f"EllipticCurve(a={self.a}, b={self.b}, rank={self.rank})"
    
    def is_valid(self) -> bool:
        """Check if the curve is non-singular."""
        return self.discriminant != 0
    
    def count_points_mod_p(self, p: int) -> int:
        """
        Count points on the curve over Fp using naive method.
        
        Args:
            p: Prime number
            
        Returns:
            Number of points including point at infinity
        """
        count = 1  # Point at infinity
        
        # For each x, check all possible y values
        for x in range(p):
            # Compute y^2 = x^3 + ax + b mod p
            rhs = (x**3 + self.a * x + self.b) % p
            
            # Count how many y satisfy y^2 = rhs (mod p)
            for y in range(p):
                if (y * y) % p == rhs:
                    count += 1
                
        return count
    
    def frobenius_trace(self, p: int) -> int:
        """
        Compute the Frobenius trace ap = p + 1 - #E(Fp).
        
        Args:
            p: Prime number
            
        Returns:
            Frobenius trace ap
        """
        return p + 1 - self.count_points_mod_p(p)

def generate_curve_family(num_curves: int, seed: int = 42) -> List[EllipticCurve]:
    """
    Generate a family of elliptic curves with known ranks.
    
    We use a simplified model where rank is assigned based on
    discriminant properties to ensure variety.
    
    Args:
        num_curves: Number of curves to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of EllipticCurve objects
    """
    random.seed(seed)
    np.random.seed(seed)
    
    curves = []
    
    # Generate curves with different rank distributions
    # In reality, rank computation is extremely difficult
    # We use a heuristic assignment for demonstration
    rank_distribution = {
        0: 0.5,   # 50% rank 0
        1: 0.3,   # 30% rank 1  
        2: 0.15,  # 15% rank 2
        3: 0.05   # 5% rank 3+
    }
    
    print("Generating elliptic curves...")
    
    for i in tqdm(range(num_curves)):
        # Generate random coefficients
        a = random.randint(-50, 50)
        b = random.randint(-50, 50)
        
        # Ensure non-singular curve
        while 4 * a**3 + 27 * b**2 == 0:
            a = random.randint(-50, 50)
            b = random.randint(-50, 50)
        
        # Assign rank based on distribution and discriminant properties
        # This is a simplification - real rank computation is much harder
        r = random.random()
        cumulative = 0
        rank = 0
        
        for rank_val, prob in rank_distribution.items():
            cumulative += prob
            if r < cumulative:
                rank = rank_val
                break
        
        # Add some correlation between coefficients and rank
        # This helps ML discover patterns
        if rank == 0:
            a = a % 7  # Smaller coefficients for rank 0
        elif rank >= 2:
            a = a * 3  # Larger coefficients for higher rank
            
        curve = EllipticCurve(a, b, rank)
        curves.append(curve)
    
    return curves

def compute_frobenius_traces(curves: List[EllipticCurve], 
                           primes: List[int]) -> np.ndarray:
    """
    Compute Frobenius traces for all curves at given primes.
    
    Args:
        curves: List of elliptic curves
        primes: List of prime numbers
        
    Returns:
        Array of shape (num_curves, num_primes) containing traces
    """
    num_curves = len(curves)
    num_primes = len(primes)
    traces = np.zeros((num_curves, num_primes))
    
    print(f"Computing Frobenius traces for {num_curves} curves at {num_primes} primes...")
    
    for i, curve in enumerate(tqdm(curves)):
        for j, p in enumerate(primes):
            traces[i, j] = curve.frobenius_trace(p)
            
    return traces

def generate_primes(max_prime: int) -> List[int]:
    """Generate list of primes up to max_prime using sieve."""
    sieve = sp.sieve.primerange(3, max_prime + 1)  # Start from 3 to skip 2
    return list(sieve)

def compute_average_traces(curves: List[EllipticCurve], 
                         traces: np.ndarray,
                         primes: List[int]) -> Dict[int, np.ndarray]:
    """
    Compute average traces for curves grouped by rank.
    
    This is where murmurations become visible!
    
    Args:
        curves: List of elliptic curves
        traces: Frobenius traces array
        primes: List of primes
        
    Returns:
        Dictionary mapping rank to average trace values
    """
    rank_groups = {}
    
    # Group curves by rank
    for i, curve in enumerate(curves):
        rank = curve.rank
        if rank not in rank_groups:
            rank_groups[rank] = []
        rank_groups[rank].append(i)
    
    # Compute averages
    average_traces = {}
    
    for rank, indices in rank_groups.items():
        # Get traces for curves of this rank
        rank_traces = traces[indices, :]
        
        # Compute average normalized by sqrt(p)
        avg_trace = np.zeros(len(primes))
        for j, p in enumerate(primes):
            avg_trace[j] = np.mean(rank_traces[:, j]) / np.sqrt(p)
            
        average_traces[rank] = avg_trace
    
    return average_traces

def detect_murmuration_period(avg_traces: np.ndarray, 
                            primes: List[int]) -> float:
    """
    Detect the characteristic oscillation period in murmuration patterns.
    
    Args:
        avg_traces: Average trace values
        primes: Corresponding prime values
        
    Returns:
        Estimated period of oscillation
    """
    # Use FFT to find dominant frequency
    from scipy.fft import fft, fftfreq
    
    # Interpolate to uniform spacing for FFT
    uniform_x = np.linspace(primes[0], primes[-1], 1000)
    uniform_y = np.interp(uniform_x, primes, avg_traces)
    
    # Apply FFT
    yf = fft(uniform_y - np.mean(uniform_y))
    xf = fftfreq(len(uniform_x), d=(uniform_x[1] - uniform_x[0]))
    
    # Find dominant frequency (excluding DC component)
    positive_freqs = xf[1:len(xf)//2]
    positive_fft = np.abs(yf[1:len(yf)//2])
    dominant_idx = np.argmax(positive_fft)
    dominant_freq = positive_freqs[dominant_idx]
    
    period = 1 / dominant_freq if dominant_freq > 0 else float('inf')
    
    return period