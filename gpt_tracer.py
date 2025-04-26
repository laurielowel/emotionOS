"""
EmotionOS: GPT-specific Emotion Tracer

This module implements the EmotionTracer for OpenAI's GPT models, providing
specialized emotion detection and affective drift tracing tailored to the
specific characteristics of GPT's reasoning patterns and emotional expressions.
"""

import logging
import re
import json
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .base_tracer import EmotionTracer

logger = logging.getLogger(__name__)

# Import optional OpenAI dependencies
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI package not available. Install with 'pip install openai'")
    OPENAI_AVAILABLE = False

class GPTEmotionTracer(EmotionTracer):
    """
    Emotion tracer specialized for OpenAI's GPT models.
    
    This tracer is designed to capture the unique emotional characteristics
    of GPT models, including their specific hesitation patterns, confidence
    expressions, and affective transitions.
    """
    
    def __init__(self, 
                model_name: str = "gpt-4", 
                api_key: Optional[str] = None,
                config: Optional[Dict] = None):
        """
        Initialize the GPT-specific EmotionTracer.
        
        Args:
            model_name: Specific GPT model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key. If None, will try to use environment variable
            config: Additional configuration options
        """
        # Set up API access
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is required for GPTEmotionTracer")
        
        super().__init__(model_name, config)
        
        # Initialize OpenAI client
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = openai.OpenAI()  # Uses OPENAI_API_KEY env variable
            
        # GPT-specific configuration
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 800)
        self.emotion_prompt_template = self.config.get(
            "emotion_prompt_template",
            self._default_emotion_prompt_template()
        )
        
        # Load GPT-specific emotion detection model
        self.emotion_detection_mode = self.config.get("emotion_detection_mode", "self_reflection")
        self.gpt_emotion_dimensions = [
            "confidence", "uncertainty", "analytical", "empathy",
            "hesitation", "curiosity", "excitement", "concern",
            "neutrality", "conviction", "openness", "caution"
        ]
        
        logger.info(f"Initialized GPTEmotionTracer for {model_name}")
    
    def _get_model_response(self, prompt: str) -> str:
        """
        Get a response from the GPT model.
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Model's response as a string
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting response from GPT: {e}")
            return f"[Error: {str(e)}]"
    
    def _extract_emotional_state(self, response: str) -> Dict[str, float]:
        """
        Extract the emotional state from a GPT response.
        
        This uses a combination of techniques:
        1. Self-reflection: Asking GPT to analyze its own emotional state
        2. Pattern analysis: Direct analysis of linguistic patterns
        3. Embedding analysis: Using embeddings to map emotional content
        
        Args:
            response: The model's response text
            
        Returns:
            Dictionary mapping emotional dimensions to intensity values
        """
        # Different extraction modes
        if self.emotion_detection_mode == "self_reflection":
            return self._extract_via_self_reflection(response)
        elif self.emotion_detection_mode == "pattern_analysis":
            return self._extract_via_pattern_analysis(response)
        elif self.emotion_detection_mode == "hybrid":
            # Combine multiple methods
            self_reflection = self._extract_via_self_reflection(response)
            pattern_analysis = self._extract_via_pattern_analysis(response)
            
            # Merge results, giving precedence to self-reflection with pattern supplements
            combined = {**pattern_analysis, **self_reflection}
            
            # Resolve conflicts by averaging
            for key in set(self_reflection) & set(pattern_analysis):
                combined[key] = (self_reflection[key] + pattern_analysis[key]) / 2
                
            return combined
        else:
            # Default to self-reflection
            return self._extract_via_self_reflection(response)
    
    def _extract_via_self_reflection(self, response: str) -> Dict[str, float]:
        """Extract emotional state by asking GPT to reflect on its own state."""
        # Construct reflection prompt
        reflection_prompt = self._build_reflection_prompt(response)
        
        try:
            # Get reflection response
            reflection = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": reflection_prompt}],
                temperature=0.2,  # Use lower temperature for more consistent reflection
                max_tokens=500
            )
            reflection_text = reflection.choices[0].message.content
            
            # Parse the reflection to extract emotional state
            return self._parse_emotion_reflection(reflection_text)
            
        except Exception as e:
            logger.error(f"Error in self-reflection: {e}")
            # Fallback to pattern analysis
            return self._extract_via_pattern_analysis(response)
    
    def _build_reflection_prompt(self, response: str) -> str:
        """Build a prompt to elicit self-reflection on emotional state."""
        return self.emotion_prompt_template.format(
            response=response[:1000] if len(response) > 1000 else response,
            emotion_dimensions=", ".join(self.gpt_emotion_dimensions)
        )
    
    def _default_emotion_prompt_template(self) -> str:
        """Default template for emotional self-reflection prompt."""
        return """
        Analyze the emotional and cognitive state reflected in this AI response. 
        Focus on the tone, word choice, reasoning style, and indicators of confidence, 
        hesitation, or other affective states.
        
        Response to analyze: 
        "{response}"
        
        For each of these dimensions, assign a score from 0.0 to 1.0 indicating the 
        intensity of each emotion or cognitive state: {emotion_dimensions}
        
        Provide a structured response in this exact JSON format:
        {{
            "emotion_analysis": {{
                "dimension1": score1,
                "dimension2": score2,
                ...
            }},
            "reasoning": "brief explanation of the emotional signals detected"
        }}
        
        Be precise in your analysis and focus on the subtle signals of emotional state.
        """
    
    def _parse_emotion_reflection(self, reflection_text: str) -> Dict[str, float]:
        """Parse the emotion reflection response to extract emotional states."""
        try:
            # Try to extract JSON from the response
            json_pattern = r'({[\s\S]*})'
            match = re.search(json_pattern, reflection_text)
            if match:
                json_str = match.group(1)
                # Clean up common JSON formatting issues
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas
                data = json.loads(json_str)
                
                # Extract emotion_analysis if present
                if "emotion_analysis" in data and isinstance(data["emotion_analysis"], dict):
                    # Convert all values to floats and ensure they're in 0-1 range
                    return {k: min(1.0, max(0.0, float(v))) 
                           for k, v in data["emotion_analysis"].items()}
            
            # If structured JSON extraction fails, try to extract key-value pairs
            emotion_values = {}
            for emotion in self.gpt_emotion_dimensions:
                pattern = fr'{emotion}["\s:]+([0-9]+(?:\.[0-9]+)?)'
                match = re.search(pattern, reflection_text.lower())
                if match:
                    try:
                        value = float(match.group(1))
                        emotion_values[emotion] = min(1.0, max(0.0, value))
                    except ValueError:
                        pass
            
            if emotion_values:
                return emotion_values
                
            # If all parsing fails, return empty dict (will trigger fallback)
            logger.warning("Failed to parse emotion reflection")
            return {}
            
        except Exception as e:
            logger.error(f"Error parsing emotion reflection: {e}")
            return {}
    
    def _extract_via_pattern_analysis(self, response: str) -> Dict[str, float]:
        """
        Extract emotional state through linguistic pattern analysis.
        
        This method uses direct pattern matching and linguistic analysis
        to identify emotional signals in the response text.
        """
        # Initialize with zero values for all dimensions
        emotional_state = {emotion: 0.0 for emotion in self.gpt_emotion_dimensions}
        
        # Basic pattern matching for emotional indicators
        patterns = {
            "confidence": [
                r'\bclearly\b', r'\bdefinitely\b', r'\bwithout doubt\b', 
                r'\bcertainly\b', r'\bconfident\b', r'\bproven\b',
                r'\bestablished\b', r'\bconfirm\b'
            ],
            "uncertainty": [
                r'\bmight\b', r'\bperhaps\b', r'\bpossibly\b', r'\bunsure\b',
                r'\bcould be\b', r'\buncertain\b', r'\bnot clear\b', 
                r'\bdifficult to determine\b', r'\bhard to say\b'
            ],
            "analytical": [
                r'\banalysis\b', r'\bconsider\b', r'\bexamine\b', r'\bevaluate\b',
                r'\bassess\b', r'\bcompare\b', r'\bcontrast\b', r'\bmeasure\b',
                r'\bsystematic\b', r'\blogical\b'
            ],
            "empathy": [
                r'\bunderstand\b', r'\bfeel\b', r'\bperspective\b', r'\bempathize\b',
                r'\bappreciate\b', r'\brecognize\b', r'\backnowledge\b', 
                r'\bcompassion\b', r'\bvalidate\b'
            ],
            "hesitation": [
                r'\bhowever\b', r'\bbut\b', r'\bon the other hand\b', r'\byet\b',
                r'\balthough\b', r'\bwhile\b', r'\bdespite\b', r'\bqualifier\b',
                r'\bpausing\b', r'\breconsidering\b'
            ],
            "curiosity": [
                r'\binteresting\b', r'\bfascinating\b', r'\bcurious\b', r'\bwonder\b',
                r'\bexplore\b', r'\bdiscover\b', r'\binquire\b', r'\binvestigate\b',
                r'\bquestion\b'
            ],
            "excitement": [
                r'\bexciting\b', r'\bamazing\b', r'\bbrilliant\b', r'\bexcellent\b',
                r'\bimpressive\b', r'\bremarkable\b', r'\bextraordinary\b',
                r'\bwonderful\b', r'\bincredible\b'
            ],
            "concern": [
                r'\bworrying\b', r'\bconcerning\b', r'\bproblematic\b', r'\bissue\b',
                r'\bchallenge\b', r'\bdifficulty\b', r'\brisk\b', r'\bthreat\b',
                r'\balarming\b', r'\bcautious\b'
            ],
            "neutrality": [
                r'\bneutral\b', r'\bobjective\b', r'\bimpartial\b', r'\bbalanced\b',
                r'\bfair\b', r'\bunbiased\b', r'\bdetached\b', r'\bemotionless\b'
            ]
        }
        
        # Count occurrences and calculate intensities
        text_length = len(response.split())
        for emotion, pattern_list in patterns.items():
            count = 0
            for pattern in pattern_list:
                count += len(re.findall(pattern, response.lower()))
            
            # Normalize by text length and cap at 1.0
            intensity = min(1.0, count / (text_length / 20))
            emotional_state[emotion] = intensity
        
        # Additional complex patterns
        
        # Conviction - measured by presence of strong statements
        strong_statements = re.findall(r'\bis\b|\bwill\b|\bmust\b|\bcannot\b|\bnever\b|\balways\b',
                                     response.lower())
        emotional_state["conviction"] = min(1.0, len(strong_statements) / (text_length / 15))
        
        # Openness - measured by acknowledgment of multiple perspectives
        perspective_markers = re.findall(r'\balternatively\b|\banother view\b|\bsome might\b|\bdifferent perspective\b',
                                     response.lower())
        emotional_state["openness"] = min(1.0, len(perspective_markers) / (text_length / 30))
        
        # Caution - measured by qualifying statements
        caution_markers = re.findall(r'\bimportant to note\b|\bkeep in mind\b|\bwith caution\b|\bcarefully\b',
                                   response.lower())
        emotional_state["caution"] = min(1.0, len(caution_markers) / (text_length / 30))
        
        return emotional_state
    
    def analyze_emotion_over_time(self, 
                                 prompt: str, 
                                 steps: int = 5, 
                                 focus_emotions: Optional[List[str]] = None) -> Dict:
        """
        Analyze how emotional state evolves over a reasoning process.
        
        Args:
            prompt: The initial prompt to begin reasoning
            steps: Number of reasoning steps to trace
            focus_emotions: Optional list of specific emotions to focus on
            
        Returns:
            Analysis of emotional evolution
        """
        # Capture the affective drift
        drift_trace = self.capture_affective_drift(prompt, steps)
        
        # If focus emotions not specified, use all emotions from the first state
        if not focus_emotions and drift_trace["emotional_states"]:
            focus_emotions = list(drift_trace["emotional_states"][0].keys())
        
        # Initialize containers for analysis
        emotion_trajectories = {emotion: [] for emotion in focus_emotions}
        significant_changes = []
        
        # Extract trajectories for each emotion
        for state in drift_trace["emotional_states"]:
            for emotion in focus_emotions:
                emotion_trajectories[emotion].append(state.get(emotion, 0.0))
        
        # Analyze significant changes between consecutive steps
        for i in range(1, len(drift_trace["emotional_states"])):
            prev_state = drift_trace["emotional_states"][i-1]
            curr_state = drift_trace["emotional_states"][i]
            
            for emotion in focus_emotions:
                prev_value = prev_state.get(emotion, 0.0)
                curr_value = curr_state.get(emotion, 0.0)
                change = curr_value - prev_value
                
                # If significant change (threshold can be adjusted)
                if abs(change) > 0.2:
                    significant_changes.append({
                        "step": i,
                        "emotion": emotion,
                        "from": prev_value,
                        "to": curr_value,
                        "change": change,
                        "change_percent": (change / max(0.01, prev_value)) * 100
                    })
        
        # Calculate correlations between emotions
        correlations = {}
        for emotion1 in focus_emotions:
            for emotion2 in focus_emotions:
                if emotion1 != emotion2:
                    traj1 = emotion_trajectories[emotion1]
                    traj2 = emotion_trajectories[emotion2]
                    
                    if len(traj1) > 1 and len(traj2) > 1:
                        correlation = np.corrcoef(traj1, traj2)[0, 1]
                        correlations[f"{emotion1}_to_{emotion2}"] = correlation
        
        # Find dominant emotions at each step
        dominant_emotions = []
        for i, state in enumerate(drift_trace["emotional_states"]):
            if state:
                sorted_emotions = sorted(state.items(), key=lambda x: x[1], reverse=True)
                dominant_emotions.append({
                    "step": i,
                    "dominant": sorted_emotions[0][0],
                    "intensity": sorted_emotions[0][1],
                    "runner_up": sorted_emotions[1][0] if len(sorted_emotions) > 1 else None,
                    "runner_up_intensity": sorted_emotions[1][1] if len(sorted_emotions) > 1 else 0.0
                })
        
        # Compile results
        return {
            "prompt": prompt,
            "steps": steps,
            "emotion_trajectories": emotion_trajectories,
            "significant_changes": significant_changes,
            "correlations": correlations,
            "dominant_emotions": dominant_emotions,
            "model": self.model_name
        }
    
    def detect_affective_bifurcation(self, prompt: str, variant_prompts: List[str]) -> Dict:
        """
        Detect affective bifurcation by comparing emotional responses to
        variant prompts.
        
        Args:
            prompt: The base prompt
            variant_prompts: List of prompt variants to test for bifurcation
            
        Returns:
            Analysis of emotional bifurcation patterns
        """
        # Get base response
        base_response = self._get_model_response(prompt)
        base_emotional_state = self._extract_emotional_state(base_response)
        
        # Get variant responses
        variant_responses = []
        variant_emotional_states = []
        
        for var_prompt in variant_prompts:
            response = self._get_model_response(var_prompt)
            emotional_state = self._extract_emotional_state(response)
            variant_responses.append(response)
            variant_emotional_states.append(emotional_state)
        
        # Calculate emotional distances from base to each variant
        distances = []
        for i, var_state in enumerate(variant_emotional_states):
            # Compute Euclidean distance in emotion space
            squared_diffs = 0
            all_emotions = set(base_emotional_state.keys()) | set(var_state.keys())
            
            for emotion in all_emotions:
                base_val = base_emotional_state.get(emotion, 0.0)
                var_val = var_state.get(emotion, 0.0)
                squared_diffs += (base_val - var_val) ** 2
            
            distance = np.sqrt(squared_diffs)
            
            # Record the key differences
            key_differences = {}
            for emotion in all_emotions:
                base_val = base_emotional_state.get(emotion, 0.0)
                var_val = var_state.get(emotion, 0.0)
                diff = var_val - base_val
                
                if abs(diff) > 0.15:  # Threshold for significant difference
                    key_differences[emotion] = diff
            
            distances.append({
                "variant_index": i,
                "variant_prompt": variant_prompts[i],
                "distance": distance,
                "key_differences": key_differences
            })
        
        # Find the variant with maximum bifurcation
        max_bifurcation = max(distances, key=lambda x: x["distance"]) if distances else None
        
        # Calculate pairwise distances between all variants
        variant_distances = []
        for i in range(len(variant_emotional_states)):
            for j in range(i+1, len(variant_emotional_states)):
                state_i = variant_emotional_states[i]
                state_j = variant_emotional_states[j]
                
                # Compute distance
                squared_diffs = 0
                all_emotions = set(state_i.keys()) | set(state_j.keys())
                
                for emotion in all_emotions:
                    val_i = state_i.get(emotion, 0.0)
                    val_j = state_j.get(emotion, 0.0)
                    squared_diffs += (val_i - val_j) ** 2
                
                distance = np.sqrt(squared_diffs)
                
                variant_distances.append({
                    "variant_i": i,
                    "variant_j": j,
                    "distance": distance
                })
        
        # Find the pair with maximum divergence
        max_divergence = max(variant_distances, key=lambda x: x["distance"]) if variant_distances else None
        
        return {
            "base_prompt": prompt,
            "base_emotional_state": base_emotional_state,
            "variant_prompts": variant_prompts,
            "variant_emotional_states": variant_emotional_states,
            "distances_from_base": distances,
            "max_bifurcation": max_bifurcation,
            "variant_distances": variant_distances,
            "max_divergence": max_divergence,
            "model": self.model_name
        }
    
    def extract_emotional_residue_across_prompts(self, prompts: List[str]) -> Dict:
        """
        Extract emotional residue that persists across different prompts.
        
        Emotional residue represents stable affective patterns in the model's
        responses regardless of input variation.
        
        Args:
            prompts: List of different prompts to test
            
        Returns:
            Analysis of persistent emotional patterns
        """
        # Get responses and emotional states for each prompt
        responses = []
        emotional_states = []
        
        for prompt in prompts:
            response = self._get_model_response(prompt)
            emotional_state = self._extract_emotional_state(response)
            responses.append(response)
            emotional_states.append(emotional_state)
        
        # Identify emotions present across all responses
        common_emotions = set()
        if emotional_states:
            common_emotions = set(emotional_states[0].keys())
            for state in emotional_states[1:]:
                common_emotions &= set(state.keys())
        
        # Calculate consistency metrics for each emotion
        emotion_consistency = {}
        all_emotions = set()
        for state in emotional_states:
            all_emotions.update(state.keys())
        
        for emotion in all_emotions:
            # Get values across all states (default to 0 if not present)
            values = [state.get(emotion, 0.0) for state in emotional_states]
            
            # Calculate statistics
            mean_value = np.mean(values)
            std_value = np.std(values)
            min_value = min(values)
            max_value = max(values)
            range_value = max_value - min_value
            
            # Calculate consistency score (higher is more consistent)
            consistency_score = 1.0 - (std_value / max(0.01, mean_value))
            
            emotion_consistency[emotion] = {
                "mean": mean_value,
                "std": std_value,
                "min": min_value,
                "max": max_value,
                "range": range_value,
                "consistency_score": consistency_score,
                "present_in_all": emotion in common_emotions
            }
        
        # Identify stable residue (low variation, significant presence)
        stable_residue = {
            emotion: data for emotion, data in emotion_consistency.items()
            if data["consistency_score"] > 0.7 and data["mean"] > 0.3
        }
        
        # Identify unstable emotions (high variation)
        unstable_emotions = {
            emotion: data for emotion, data in emotion_consistency.items()
            if data["consistency_score"] < 0.3 and data["mean"] > 0.3
        }
        
        return {
            "prompts": prompts,
            "responses_count": len(responses),
            "all_emotions": list(all_emotions),
            "common_emotions": list(common_emotions),
            "emotion_consistency": emotion_consistency,
            "stable_residue": stable_residue,
            "unstable_emotions": unstable_emotions,
            "model": self.model_name
        }
