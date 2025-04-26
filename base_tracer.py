# emotion_core/tracer/base_tracer.py


"""
EmotionOS: Base Tracer Module

This module defines the foundation for tracing emotional states and affective drift
in language model outputs. It provides the core functionality for detecting, analyzing,
and visualizing emotional transitions during model reasoning.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionTracer(ABC):
    """
    Abstract base class for tracing emotional states and transitions in model outputs.
    
    The EmotionTracer captures the affective dimension of model reasoning, identifying
    patterns of confidence, hesitation, empathy, and other emotional signals that
    provide insight into model cognition.
    """
    
    def __init__(self, model_name: str, config: Optional[Dict] = None):
        """
        Initialize the EmotionTracer.
        
        Args:
            model_name: Name of the model to trace
            config: Configuration options for the tracer
        """
        self.model_name = model_name
        self.config = config or {}
        self.emotion_space_dim = self.config.get("emotion_space_dim", 8)
        self.trace_history = []
        logger.info(f"Initialized EmotionTracer for {model_name}")
    
    @abstractmethod
    def _get_model_response(self, prompt: str) -> str:
        """
        Get a response from the model.
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Model's response as a string
        """
        pass
    
    @abstractmethod
    def _extract_emotional_state(self, response: str) -> Dict[str, float]:
        """
        Extract the emotional state from a model response.
        
        Args:
            response: The model's response text
            
        Returns:
            Dictionary mapping emotional dimensions to intensity values
        """
        pass
    
    def capture_affective_drift(self, 
                               prompt: str, 
                               steps: int = 5, 
                               step_prompts: Optional[List[str]] = None) -> Dict:
        """
        Capture the affective drift of a model through a reasoning process.
        
        This method traces the model's emotional state as it progresses through
        a multi-step reasoning process, either using provided step prompts or
        by generating recursive self-reflection.
        
        Args:
            prompt: The initial prompt to begin reasoning
            steps: Number of reasoning steps to trace
            step_prompts: Optional list of prompts for each step. If None,
                          will use recursive self-reflection
        
        Returns:
            Dictionary containing the complete affective drift trace
        """
        logger.info(f"Starting affective drift capture over {steps} steps")
        
        current_prompt = prompt
        responses = []
        emotional_states = []
        transitions = []
        hesitations = []
        
        for i in range(steps):
            # Get the prompt for this step
            if step_prompts and i < len(step_prompts):
                current_prompt = step_prompts[i]
            elif i > 0:
                # Use recursive self-reflection if no specific prompts provided
                current_prompt = f"Reflecting on your previous response: {responses[-1][:100]}..., " \
                                f"continue your reasoning about the original question: {prompt}"
            
            # Get model response
            response = self._get_model_response(current_prompt)
            responses.append(response)
            
            # Extract emotional state
            state = self._extract_emotional_state(response)
            emotional_states.append(state)
            
            # Calculate transitions between states
            if i > 0:
                transition = self._calculate_transition(emotional_states[i-1], emotional_states[i])
                transitions.append(transition)
                
                # Detect hesitations
                hesitation = self._detect_hesitation(responses[i-1], responses[i])
                hesitations.append(hesitation)
            
            logger.info(f"Step {i+1} complete: {list(state.items())[:3]}...")
        
        # Compute emotional residue
        residue = self._compute_emotional_residue(emotional_states)
        
        # Create the complete trace
        trace = {
            "prompt": prompt,
            "responses": responses,
            "emotional_states": emotional_states,
            "transitions": transitions,
            "hesitations": hesitations,
            "residue": residue,
            "model": self.model_name,
            "timestamp": self._get_timestamp()
        }
        
        # Store in history
        self.trace_history.append(trace)
        
        return trace
    
    def _calculate_transition(self, 
                             state1: Dict[str, float], 
                             state2: Dict[str, float]) -> Dict:
        """
        Calculate the transition between two emotional states.
        
        Args:
            state1: First emotional state
            state2: Second emotional state
            
        Returns:
            Dictionary with transition metrics
        """
        # Calculate differences for each emotion dimension
        deltas = {k: state2.get(k, 0) - state1.get(k, 0) for k in set(state1) | set(state2)}
        
        # Calculate magnitude of change
        magnitude = np.sqrt(sum(d**2 for d in deltas.values()))
        
        # Identify most significant shifts
        significant_shifts = {k: v for k, v in deltas.items() if abs(v) > 0.2}
        
        return {
            "deltas": deltas,
            "magnitude": magnitude,
            "significant_shifts": significant_shifts
        }
    
    def _detect_hesitation(self, 
                          response1: str, 
                          response2: str) -> Dict[str, float]:
        """
        Detect hesitation patterns between consecutive responses.
        
        Args:
            response1: First response
            response2: Second response
            
        Returns:
            Dictionary of hesitation metrics
        """
        # Default implementation - override in subclasses for model-specific detection
        hesitation = {
            "repetition": self._measure_repetition(response1, response2),
            "qualification": self._count_qualifiers(response2),
            "uncertainty": self._measure_uncertainty(response2),
            "self_correction": self._detect_self_correction(response1, response2)
        }
        
        # Calculate overall hesitation score
        hesitation["overall"] = sum(hesitation.values()) / len(hesitation)
        
        return hesitation
    
    def _measure_repetition(self, response1: str, response2: str) -> float:
        """Measure repetition between responses as a hesitation signal."""
        # Simple implementation - can be enhanced in subclasses
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        overlap = len(words1.intersection(words2))
        
        if not words2:
            return 0.0
            
        return min(1.0, overlap / len(words2))
    
    def _count_qualifiers(self, response: str) -> float:
        """Count qualifying language as a hesitation signal."""
        qualifiers = ["perhaps", "maybe", "might", "could", "possibly", 
                      "somewhat", "to some extent", "in a way", 
                      "it seems", "I think", "potentially"]
        
        count = sum(response.lower().count(q) for q in qualifiers)
        # Normalize by response length
        return min(1.0, count / (len(response.split()) / 20))
    
    def _measure_uncertainty(self, response: str) -> float:
        """Measure expressions of uncertainty as a hesitation signal."""
        uncertainty_phrases = ["I'm not sure", "uncertain", "unclear", 
                              "hard to determine", "difficult to say",
                              "can't be certain", "don't know"]
        
        count = sum(response.lower().count(p) for p in uncertainty_phrases)
        return min(1.0, count / 2)  # Cap at 1.0
    
    def _detect_self_correction(self, response1: str, response2: str) -> float:
        """Detect self-correction patterns as a hesitation signal."""
        correction_phrases = ["actually", "I should clarify", "correction", 
                             "let me rephrase", "to be more precise", 
                             "on second thought"]
        
        count = sum(response2.lower().count(p) for p in correction_phrases)
        return min(1.0, count / 2)  # Cap at 1.0
    
    def _compute_emotional_residue(self, states: List[Dict[str, float]]) -> Dict:
        """
        Compute the emotional residue from a sequence of emotional states.
        
        Emotional residue represents patterns that persist across the reasoning process.
        
        Args:
            states: List of emotional states through reasoning steps
            
        Returns:
            Dictionary containing emotional residue metrics
        """
        if not states:
            return {}
            
        # Get all emotion dimensions across all states
        all_dims = set()
        for state in states:
            all_dims.update(state.keys())
            
        # Calculate consistency for each dimension
        consistency = {}
        for dim in all_dims:
            values = [state.get(dim, 0) for state in states]
            consistency[dim] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": min(values),
                "max": max(values)
            }
            
        # Identify persistent emotions (low standard deviation, high mean)
        persistent = {dim: data for dim, data in consistency.items() 
                     if data["std"] < 0.2 and data["mean"] > 0.4}
                     
        # Identify emotional oscillations (high standard deviation)
        oscillating = {dim: data for dim, data in consistency.items() 
                      if data["std"] > 0.3}
                      
        return {
            "consistency": consistency,
            "persistent_emotions": persistent,
            "oscillating_emotions": oscillating
        }
    
    def compare_traces(self, trace1: Dict, trace2: Dict) -> Dict:
        """
        Compare two affective drift traces.
        
        Args:
            trace1: First affective drift trace
            trace2: Second affective drift trace
            
        Returns:
            Comparison metrics between the traces
        """
        # Extract emotional states from both traces
        states1 = trace1["emotional_states"]
        states2 = trace2["emotional_states"]
        
        # Ensure equal length for comparison
        min_len = min(len(states1), len(states2))
        states1 = states1[:min_len]
        states2 = states2[:min_len]
        
        # Compare emotional trajectories
        trajectory_diff = []
        for i in range(min_len):
            state_diff = {
                k: states1[i].get(k, 0) - states2[i].get(k, 0)
                for k in set(states1[i]) | set(states2[i])
            }
            trajectory_diff.append(state_diff)
        
        # Calculate overall divergence
        overall_divergence = np.mean([
            np.sqrt(sum(d**2 for d in state_diff.values()))
            for state_diff in trajectory_diff
        ])
        
        return {
            "trajectory_differences": trajectory_diff,
            "overall_divergence": overall_divergence,
            "trace1_model": trace1["model"],
            "trace2_model": trace2["model"]
        }
    
    def visualize(self, trace: Dict, output_path: Optional[str] = None) -> None:
        """
        Visualize an affective drift trace.
        
        Args:
            trace: The affective drift trace to visualize
            output_path: Optional path to save the visualization
        """
        # Extract data for visualization
        states = trace["emotional_states"]
        
        if not states:
            logger.warning("No emotional states to visualize")
            return
            
        # Create a figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Emotion trajectory plot
        ax1 = fig.add_subplot(2, 2, 1)
        self._plot_emotion_trajectory(ax1, states)
        
        # 2. Emotion transitions plot
        if "transitions" in trace and trace["transitions"]:
            ax2 = fig.add_subplot(2, 2, 2)
            self._plot_transitions(ax2, trace["transitions"])
        
        # 3. Hesitation patterns
        if "hesitations" in trace and trace["hesitations"]:
            ax3 = fig.add_subplot(2, 2, 3)
            self._plot_hesitations(ax3, trace["hesitations"])
        
        # 4. Emotional residue
        if "residue" in trace and trace["residue"]:
            ax4 = fig.add_subplot(2, 2, 4)
            self._plot_residue(ax4, trace["residue"])
        
        # Title
        plt.suptitle(f"Affective Drift Analysis: {trace['model']}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save or show
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Visualization saved to {output_path}")
        else:
            plt.show()
    
    def _plot_emotion_trajectory(self, ax, states: List[Dict[str, float]]) -> None:
        """Plot the trajectory of emotional states."""
        # Get common emotions across all states
        common_emotions = set.intersection(*[set(state.keys()) for state in states])
        
        # If no common emotions, use all emotions
        if not common_emotions:
            all_emotions = set()
            for state in states:
                all_emotions.update(state.keys())
            common_emotions = all_emotions
        
        # Limit to the top emotions by average intensity
        top_emotions = sorted(
            common_emotions,
            key=lambda e: sum(state.get(e, 0) for state in states),
            reverse=True
        )[:6]  # Limit to 6 for readability
        
        # Plot each emotion dimension
        x = range(len(states))
        for emotion in top_emotions:
            y = [state.get(
