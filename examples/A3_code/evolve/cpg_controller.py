import numpy as np
from typing import Optional

class CPGController:
    """Central Pattern Generator for rhythmic locomotion"""
    
    def __init__(self, genome: np.ndarray, num_joints: int):
        """
        Genome encodes:
        - Frequency for each joint (Hz)
        - Amplitude for each joint (0-1)
        - Phase offset for each joint (0-2π)
        - Coupling weights between joints (NxN matrix)
        """
        self.num_joints = num_joints
        idx = 0
        
        # Decode CPG parameters from genome
        self.frequencies = genome[idx:idx+num_joints] * 2.0 + 1.0  # 1-3 Hz
        idx += num_joints
        
        self.amplitudes = np.abs(genome[idx:idx+num_joints])  # 0-1
        idx += num_joints
        
        self.phase_offsets = genome[idx:idx+num_joints] * np.pi  # 0-π
        idx += num_joints
        
        # Coupling matrix (how joints influence each other)
        coupling_size = num_joints * num_joints
        self.coupling = genome[idx:idx+coupling_size].reshape(num_joints, num_joints)
        self.coupling *= 0.5  # Scale down coupling strength
        
        # State
        self.phases = self.phase_offsets.copy()
        self.time = 0.0
    
    def step(self, dt: float = 0.01) -> np.ndarray:
        """Generate next control signal"""
        # Update phases with coupling
        phase_dots = 2 * np.pi * self.frequencies
        
        # Add coupling: joints influence each other
        for i in range(self.num_joints):
            coupling_influence = 0.0
            for j in range(self.num_joints):
                if i != j:
                    # Phase difference influences frequency
                    phase_diff = np.sin(self.phases[j] - self.phases[i])
                    coupling_influence += self.coupling[i, j] * phase_diff
            
            phase_dots[i] += coupling_influence
        
        # Update phases
        self.phases += phase_dots * dt
        self.phases = np.mod(self.phases, 2 * np.pi)  # Wrap to [0, 2π]
        self.time += dt
        
        # Generate sinusoidal output
        output = self.amplitudes * np.sin(self.phases)
        
        return output * (np.pi / 2)  # Scale to joint limits
    
    def reset(self):
        """Reset oscillator state"""
        self.phases = self.phase_offsets.copy()
        self.time = 0.0


class HybridCPGController:
    """CPG + MLP modulation based on sensory feedback"""
    
    def __init__(self, genome: np.ndarray, num_joints: int, state_size: int):
        """
        Split genome into:
        1. CPG parameters (base rhythm)
        2. MLP parameters (modulation based on state)
        """
        # Calculate sizes
        cpg_size = num_joints * (3 + num_joints)  # freq, amp, phase, coupling
        mlp_input_size = state_size
        mlp_hidden_size = 32
        mlp_output_size = num_joints
        mlp_size = (mlp_input_size * mlp_hidden_size + mlp_hidden_size + 
                   mlp_hidden_size * mlp_output_size + mlp_output_size)
        
        # Split genome
        cpg_genome = genome[:cpg_size]
        mlp_genome = genome[cpg_size:cpg_size + mlp_size]
        
        # Initialize components
        self.cpg = CPGController(cpg_genome, num_joints)
        self.mlp_weights = self._decode_mlp(mlp_genome, mlp_input_size, 
                                           mlp_hidden_size, mlp_output_size)
    
    def _decode_mlp(self, genome, input_size, hidden_size, output_size):
        """Decode MLP weights from genome"""
        idx = 0
        w1 = genome[idx:idx+input_size*hidden_size].reshape(input_size, hidden_size)
        idx += input_size * hidden_size
        b1 = genome[idx:idx+hidden_size]
        idx += hidden_size
        w2 = genome[idx:idx+hidden_size*output_size].reshape(hidden_size, output_size)
        idx += hidden_size * output_size
        b2 = genome[idx:idx+output_size]
        
        return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    
    def _mlp_forward(self, state):
        """MLP forward pass"""
        w = self.mlp_weights
        h = np.tanh(state @ w['w1'] + w['b1'])
        out = np.tanh(h @ w['w2'] + w['b2'])
        return out
    
    def control(self, state: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        Generate control signal combining:
        - CPG base rhythm
        - MLP modulation based on sensory feedback
        """
        # Base rhythmic pattern
        cpg_output = self.cpg.step(dt)
        
        # Modulation based on state
        modulation = self._mlp_forward(state)
        
        # Combine: CPG provides rhythm, MLP adjusts amplitude
        output = cpg_output * (1.0 + 0.5 * modulation)
        
        return np.clip(output, -np.pi/2, np.pi/2)
    
    def reset(self):
        self.cpg.reset()


def cpg_genome_length(num_joints: int) -> int:
    """Calculate required genome length for CPG"""
    return num_joints * (3 + num_joints)  # freq, amp, phase, coupling


def hybrid_cpg_genome_length(num_joints: int, state_size: int) -> int:
    """Calculate genome length for Hybrid CPG+MLP"""
    cpg_size = cpg_genome_length(num_joints)
    mlp_hidden = 32
    mlp_size = (state_size * mlp_hidden + mlp_hidden + 
               mlp_hidden * num_joints + num_joints)
    return cpg_size + mlp_size