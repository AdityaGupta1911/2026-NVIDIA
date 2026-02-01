"""
Comprehensive test suite for LABS Quantum-Enhanced Optimization.
Run with: python -m unittest tests.py
"""

import unittest
import numpy as np
from typing import List, Tuple

# Import functions from main code (adjust import as needed)
try:
    from __main__ import (
        get_interactions,
        compute_labs_energy,
        compute_hamiltonian_from_labs,
        compute_hamiltonian_from_spins,
        compute_merit_factor
    )
except ImportError:
    # Define locally if import fails
    def get_interactions(N: int) -> Tuple[List[List[int]], List[List[int]]]:
        G2 = []
        G4 = []
        for k in range(1, N):
            limit = N - k
            for i in range(limit):
                for j in range(i + 1, limit):
                    if i + k == j:
                        G2.append([i, j + k])
                    else:
                        indices = [i, i + k, j, j + k]
                        if len(set(indices)) == 4:
                            G4.append(indices)
        return G2, G4
    
    def compute_labs_energy(s: np.ndarray, N: int) -> np.float64:
        s_arr = np.array(s, dtype=np.int64)
        E = np.float64(0.0)
        for k in range(1, N):
            C_k = np.sum(s_arr[:N-k] * s_arr[k:])
            E += C_k * C_k
        return E
    
    def compute_hamiltonian_from_labs(E_labs: np.float64, N: int) -> np.float64:
        return np.float64(-2.0) * E_labs
    
    def compute_hamiltonian_from_spins(s: np.ndarray, G2: List[List[int]], 
                                      G4: List[List[int]], N: int) -> np.float64:
        s_arr = np.array(s, dtype=np.int64)
        H = np.float64(-N * (N - 1))
        for i, j in G2:
            H += np.float64(-4.0) * s_arr[i] * s_arr[j]
        for i, j, k, l in G4:
            H += np.float64(-4.0) * s_arr[i] * s_arr[j] * s_arr[k] * s_arr[l]
        return H
    
    def compute_merit_factor(s: np.ndarray, N: int) -> np.float64:
        E = compute_labs_energy(s, N)
        if E <= 0:
            return np.float64(0.0)
        return np.float64(N * N) / (np.float64(2.0) * E)


class TestLABSPhysics(unittest.TestCase):
    """Test the core physics of the LABS problem."""
    
    def test_N4_sequence_energy(self):
        """Test that N=4, s=[1,1,1,-1] gives E=2.0"""
        N = 4
        s = np.array([1, 1, 1, -1], dtype=np.int64)
        
        # Manual calculation:
        # k=1: C1 = 1*1 + 1*1 + 1*(-1) = 1 + 1 - 1 = 1
        # k=2: C2 = 1*1 + 1*(-1) = 1 - 1 = 0  
        # k=3: C3 = 1*(-1) = -1
        # E = 1² + 0² + (-1)² = 1 + 0 + 1 = 2
        
        E = compute_labs_energy(s, N)
        self.assertAlmostEqual(E, 2.0, places=10,
                              msg=f"Expected E=2.0, got {E:.6f}")
        
        # Also test Hamiltonian
        G2, G4 = get_interactions(N)
        H_from_spins = compute_hamiltonian_from_spins(s, G2, G4, N)
        H_from_formula = compute_hamiltonian_from_labs(E, N)
        
        self.assertAlmostEqual(H_from_spins, H_from_formula, places=10,
                              msg="H from spins ≠ H from formula")
        self.assertAlmostEqual(H_from_spins, -4.0, places=10,
                              msg=f"Expected H=-4.0, got {H_from_spins:.6f}")
    
    def test_H_minus_2E_N4(self):
        """Test H = -2E for multiple N=4 sequences."""
        N = 4
        test_sequences = [
            [1, 1, 1, 1],
            [1, 1, 1, -1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [-1, -1, -1, -1]
        ]
        
        G2, G4 = get_interactions(N)
        
        for seq in test_sequences:
            s = np.array(seq, dtype=np.int64)
            E = compute_labs_energy(s, N)
            H_spins = compute_hamiltonian_from_spins(s, G2, G4, N)
            H_formula = compute_hamiltonian_from_labs(E, N)
            
            self.assertAlmostEqual(H_spins, H_formula, places=10,
                                  msg=f"H ≠ -2E for sequence {seq}")
            self.assertAlmostEqual(H_spins, -2.0 * E, places=10,
                                  msg=f"H ≠ -2E for sequence {seq}")
    
    def test_H_minus_2E_N10(self):
        """Test H = -2E for N=10 random sequences."""
        N = 10
        G2, G4 = get_interactions(N)
        
        for _ in range(5):  # Test 5 random sequences
            s = np.random.choice([-1, 1], N).astype(np.int64)
            E = compute_labs_energy(s, N)
            H_spins = compute_hamiltonian_from_spins(s, G2, G4, N)
            H_formula = compute_hamiltonian_from_labs(E, N)
            
            self.assertAlmostEqual(H_spins, H_formula, places=10,
                                  msg=f"H ≠ -2E for random N=10 sequence")
            self.assertAlmostEqual(H_spins, -2.0 * E, places=10,
                                  msg=f"H ≠ -2E for random N=10 sequence")
    
    def test_N20_term_counts(self):
        """Test that N=20 produces exactly 90 G2 and 1050 G4 terms."""
        N = 20
        G2, G4 = get_interactions(N)
        
        self.assertEqual(len(G2), 90,
                        f"Expected 90 G2 terms, got {len(G2)}")
        self.assertEqual(len(G4), 1050,
                        f"Expected 1050 G4 terms, got {len(G4)}")
        
        # Additional validation: all indices should be within bounds
        all_terms = G2 + G4
        max_index = max([max(term) for term in all_terms] + [0])
        self.assertLess(max_index, N,
                       f"Index {max_index} out of bounds for N={N}")
    
    def test_small_N_term_counts(self):
        """Test term counts for small N values."""
        test_cases = [
            (4, (3, 0)),   # N=4: 3 G2, 0 G4
            (5, (6, 0)),   # N=5: 6 G2, 0 G4  
            (6, (10, 1)),  # N=6: 10 G2, 1 G4
            (7, (15, 4)),  # N=7: 15 G2, 4 G4
            (8, (21, 10)), # N=8: 21 G2, 10 G4
        ]
        
        for N, (expected_G2, expected_G4) in test_cases:
            G2, G4 = get_interactions(N)
            self.assertEqual(len(G2), expected_G2,
                           f"N={N}: Expected {expected_G2} G2, got {len(G2)}")
            self.assertEqual(len(G4), expected_G4,
                           f"N={N}: Expected {expected_G4} G4, got {len(G4)}")
    
    def test_energy_non_negative(self):
        """Test that E = Σ C_k² is always non-negative."""
        for N in [4, 8, 12, 16]:
            for _ in range(3):
                s = np.random.choice([-1, 1], N).astype(np.int64)
                E = compute_labs_energy(s, N)
                self.assertGreaterEqual(E, 0.0,
                                       f"E should be ≥ 0, got {E:.6f} for N={N}")
    
    def test_merit_factor_calculation(self):
        """Test merit factor calculation."""
        # Known test case: N=4, s=[1,1,1,-1], E=2.0
        N = 4
        s = np.array([1, 1, 1, -1], dtype=np.int64)
        mf = compute_merit_factor(s, N)
        expected_mf = (N * N) / (2.0 * 2.0)  # N²/(2E) = 16/(2*2) = 4.0
        self.assertAlmostEqual(mf, expected_mf, places=10,
                              msg=f"Expected MF={expected_mf}, got {mf:.6f}")
        
        # Test all ones sequence
        s_ones = np.ones(N, dtype=np.int64)
        mf_ones = compute_merit_factor(s_ones, N)
        # For all ones: C_k = N-k, so E = Σ (N-k)²
        E_ones = sum([(N - k) ** 2 for k in range(1, N)])
        expected_mf_ones = (N * N) / (2.0 * E_ones)
        self.assertAlmostEqual(mf_ones, expected_mf_ones, places=10,
                              msg="MF incorrect for all ones")
    
    def test_constant_sequence_energy(self):
        """Test energy for constant sequences."""
        for N in [4, 6, 8]:
            # All +1
            s_plus = np.ones(N, dtype=np.int64)
            E_plus = compute_labs_energy(s_plus, N)
            
            # All -1 (should give same E)
            s_minus = -np.ones(N, dtype=np.int64)
            E_minus = compute_labs_energy(s_minus, N)
            
            self.assertAlmostEqual(E_plus, E_minus, places=10,
                                  msg=f"E(+++) ≠ E(---) for N={N}")
            
            # Verify formula: E = Σ (N-k)²
            expected_E = sum([(N - k) ** 2 for k in range(1, N)])
            self.assertAlmostEqual(E_plus, expected_E, places=10,
                                  msg=f"E incorrect for constant sequence N={N}")
    
    def test_alternating_sequence_energy(self):
        """Test energy for alternating sequences."""
        for N in [4, 6, 8, 10]:
            # Alternating +1, -1, +1, -1, ...
            s = np.ones(N, dtype=np.int64)
            s[1::2] = -1
            
            E = compute_labs_energy(s, N)
            
            # For alternating pattern:
            # C_k = 0 for odd k, C_k = N-k for even k
            expected_E = sum([(N - k) ** 2 for k in range(1, N) if k % 2 == 0])
            
            self.assertAlmostEqual(E, expected_E, places=10,
                                  msg=f"E incorrect for alternating sequence N={N}")
    
    def test_G2_G4_indices_valid(self):
        """Test that all G2 and G4 indices are valid and distinct."""
        for N in [10, 15, 20]:
            G2, G4 = get_interactions(N)
            
            # Check G2 indices
            for i, j in G2:
                self.assertLess(i, N, f"G2 index {i} ≥ N={N}")
                self.assertLess(j, N, f"G2 index {j} ≥ N={N}")
                self.assertNotEqual(i, j, f"G2 indices equal: ({i},{j})")
            
            # Check G4 indices
            for i, j, k, l in G4:
                indices = [i, j, k, l]
                self.assertTrue(all(idx < N for idx in indices),
                               f"G4 index out of bounds for N={N}")
                self.assertEqual(len(set(indices)), 4,
                                f"G4 indices not distinct: {indices}")
    
    def test_energy_symmetry(self):
        """Test energy symmetry: flipping all spins gives same E."""
        for N in [5, 10, 15]:
            s1 = np.random.choice([-1, 1], N).astype(np.int64)
            s2 = -s1  # Flip all spins
            
            E1 = compute_labs_energy(s1, N)
            E2 = compute_labs_energy(s2, N)
            
            self.assertAlmostEqual(E1, E2, places=10,
                                  msg=f"E not symmetric under global spin flip for N={N}")
    
    def test_energy_shift_invariance(self):
        """Test energy invariance under cyclic shifts."""
        N = 8
        s = np.random.choice([-1, 1], N).astype(np.int64)
        E_original = compute_labs_energy(s, N)
        
        # Test a few cyclic shifts
        for shift in [1, 2, 3]:
            s_shifted = np.roll(s, shift)
            E_shifted = compute_labs_energy(s_shifted, N)
            self.assertAlmostEqual(E_original, E_shifted, places=10,
                                  msg=f"E not invariant under shift={shift}")


class TestHamiltonianConsistency(unittest.TestCase):
    """Test Hamiltonian energy calculations."""
    
    def test_hamiltonian_formula_consistency(self):
        """Test H = -2E formula for various N."""
        for N in [4, 8, 12, 16, 20]:
            G2, G4 = get_interactions(N)
            
            for _ in range(3):  # Multiple random tests
                s = np.random.choice([-1, 1], N).astype(np.int64)
                
                E = compute_labs_energy(s, N)
                H_spins = compute_hamiltonian_from_spins(s, G2, G4, N)
                H_formula = compute_hamiltonian_from_labs(E, N)
                
                self.assertAlmostEqual(H_spins, H_formula, places=10,
                                      msg=f"H ≠ -2E for N={N}, E={E:.2f}")
                self.assertAlmostEqual(H_spins, -2.0 * E, places=10,
                                      msg=f"H ≠ -2E for N={N}")
    
    def test_hamiltonian_negative_definite(self):
        """Test that H is always negative (since we're minimizing)."""
        for N in [10, 15, 20]:
            G2, G4 = get_interactions(N)
            
            for _ in range(5):
                s = np.random.choice([-1, 1], N).astype(np.int64)
                H = compute_hamiltonian_from_spins(s, G2, G4, N)
                
                self.assertLess(H, 0.0,
                               f"H should be negative, got {H:.2f} for N={N}")
    
    def test_constant_term_correctness(self):
        """Test that the constant term -N(N-1) is correct."""
        for N in [4, 8, 12]:
            # For the all-ones sequence, we can compute H explicitly
            s = np.ones(N, dtype=np.int64)
            G2, G4 = get_interactions(N)
            
            H = compute_hamiltonian_from_spins(s, G2, G4, N)
            
            # Manually compute expected H
            # For all ones: s_i * s_j = 1 for any i,j
            # So each G2 term contributes -4 * 1 = -4
            # Each G4 term contributes -4 * 1 = -4
            # Total from terms: -4 * (len(G2) + len(G4))
            # Plus constant: -N(N-1)
            total_terms = len(G2) + len(G4)
            expected_H = -4.0 * total_terms - N * (N - 1)
            
            self.assertAlmostEqual(H, expected_H, places=10,
                                  msg=f"Constant term incorrect for N={N}")


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
