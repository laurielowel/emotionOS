
# PROJECT_STRUCTURE.md


# EmotionOS Project Structure

This document provides an overview of the EmotionOS repository structure and component relationships.

## Repository Root

```
emotionos/
├── assets/                       # Images, logos, and design resources
├── emotion_core/                 # Core emotion detection and analysis modules
├── affective_drift_console/      # Interactive visualization interface
├── symbolic_emotion_shells/      # Diagnostic emotion shell templates
├── crossmodel_emotion_logs/      # Cross-model emotional drift comparisons
├── decentralization_templates/   # Templates for extending emotional research
├── case_studies/                 # Real-world examples and analyses
├── docs/                         # Documentation
├── tests/                        # Test suite
├── scripts/                      # Utility scripts
├── README.md                     # Project overview
├── CONTRIBUTING.md               # Contribution guidelines
├── LICENSE                       # MIT License
└── setup.py                      # Installation configuration
```

## Core Components

### 1. `emotion_core/`

The foundational library for emotional drift tracing and affective state analysis.

```
emotion_core/
├── __init__.py
├── tracer/                       # Emotion tracing capabilities
│   ├── __init__.py
│   ├── base_tracer.py            # Abstract base class for tracers
│   ├── gpt_tracer.py             # GPT-specific emotion tracing
│   ├── claude_tracer.py          # Claude-specific emotion tracing
│   ├── gemini_tracer.py          # Gemini-specific emotion tracing
│   └── custom_tracer.py          # Template for custom model tracers
├── detectors/                    # Emotional state detector modules
│   ├── __init__.py
│   ├── confidence_detector.py    # Confidence and uncertainty detection
│   ├── hesitation_detector.py    # Hesitation pattern recognition
│   ├── value_conflict_detector.py # Ethical value conflicts 
│   ├── empathy_detector.py       # Empathetic reasoning detection
│   └── transition_detector.py    # Emotional state transitions
├── extractors/                   # Affective residue extraction
│   ├── __init__.py
│   ├── residue_extractor.py      # Base residue extraction
│   ├── pattern_extractor.py      # Pattern-based residue extraction
│   └── recursive_extractor.py    # Recursive emotional pattern extraction
├── models/                       # Emotional state models
│   ├── __init__.py
│   ├── emotion_vector.py         # Vector representation of emotional states
│   ├── affective_drift.py        # Model for emotional transitions
│   ├── residue_model.py          # Representation of emotional residue
│   └── recursive_state.py        # Recursive emotional state model
└── utils/                        # Utility functions
    ├── __init__.py
    ├── prompt_engineering.py     # Prompt creation for emotional probing
    ├── response_parsing.py       # Parsing emotional signals from responses
    └── vector_operations.py      # Operations on emotional vectors
```

### 2. `affective_drift_console/`

Interactive visualization for emotional trajectories and transitions.

```
affective_drift_console/
├── __init__.py
├── app.py                        # Main application entry point
├── components/                   # UI components
│   ├── __init__.py
│   ├── emotion_map.py            # Emotional space visualization
│   ├── drift_timeline.py         # Timeline of emotional transitions
│   ├── residue_visualizer.py     # Visualizer for emotional residue
│   └── comparison_view.py        # Cross-model comparison view
├── visualizers/                  # Core visualization logic
│   ├── __init__.py
│   ├── vector_space.py           # Emotional vector space projection
│   ├── transition_graph.py       # Graph representation of transitions
│   ├── heatmap.py                # Emotion intensity heatmaps
│   └── recursive_visualizer.py   # Recursive pattern visualization
└── static/                       # Static assets for visualization
    ├── css/                      # Styling
    ├── js/                       # JavaScript components
    └── images/                   # Visualization images
```

### 3. `symbolic_emotion_shells/`

Diagnostic shells for inducing and studying emotional patterns.

```
symbolic_emotion_shells/
├── __init__.py
├── shell_core/                   # Core shell functionality
│   ├── __init__.py
│   ├── shell_engine.py           # Shell execution engine
│   ├── shell_parser.py           # Shell definition parser
│   └── shell_registry.py         # Registry of available shells
├── shells/                       # Shell definitions
│   ├── __init__.py
│   ├── confidence_collapse.shell # Confidence collapse induction
│   ├── value_conflict.shell      # Value conflict simulation
│   ├── empathy_drift.shell       # Empathy drift induction
│   ├── recursive_hesitation.shell # Recursive hesitation patterns
│   └── affective_bifurcation.shell # Emotional reasoning fork
├── results/                      # Shell execution results
│   ├── __init__.py
│   ├── result_analyzer.py        # Analysis of shell results
│   └── pattern_extractor.py      # Pattern extraction from results
└── templates/                    # Templates for new shells
    ├── basic_shell_template.yaml
    └── advanced_shell_template.yaml
```

### 4. `crossmodel_emotion_logs/`

Comparative emotional behavior across leading AI systems.

```
crossmodel_emotion_logs/
├── __init__.py
├── gpt4/                         # GPT-4 emotional logs
│   ├── confidence_patterns.json
│   ├── hesitation_maps.json
│   ├── value_conflicts.json
│   └── empathy_transitions.json
├── claude/                       # Claude emotional logs
│   ├── confidence_patterns.json
│   ├── hesitation_maps.json
│   ├── value_conflicts.json
│   └── empathy_transitions.json
├── gemini/                       # Gemini emotional logs
│   ├── confidence_patterns.json
│   ├── hesitation_maps.json
│   ├── value_conflicts.json
│   └── empathy_transitions.json
├── llama/                        # Llama emotional logs
│   ├── confidence_patterns.json
│   ├── hesitation_maps.json
│   ├── value_conflicts.json
│   └── empathy_transitions.json
├── comparisons/                  # Cross-model comparisons
│   ├── confidence_comparison.json
│   ├── hesitation_comparison.json
│   ├── value_conflict_comparison.json
│   └── empathy_comparison.json
└── collectors/                   # Log collection tools
    ├── __init__.py
    ├── log_collector.py
    └── comparison_generator.py
```

### 5. `decentralization_templates/`

Templates to extend emotional interpretability research.

```
decentralization_templates/
├── __init__.py
├── research_protocols/           # Standardized research protocols
│   ├── basic_emotion_mapping.md
│   ├── confidence_analysis.md
│   ├── hesitation_study.md
│   └── value_conflict_protocol.md
├── extension_guides/             # Guides for extending EmotionOS
│   ├── new_model_integration.md
│   ├── custom_emotion_detector.md
│   ├── visualization_extension.md
│   └── shell_development.md
├── community_resources/          # Resources for community research
│   ├── data_sharing_protocol.md
│   ├── standardized_prompts.json
│   └── collaboration_guide.md
└── project_templates/            # Templates for new projects
    ├── emotion_research_project/
    ├── visualization_project/
    └── shell_collection_project/
```

## Additional Components

### `docs/`

Comprehensive documentation for the project.

```
docs/
├── README.md                     # Documentation overview
├── installation.md               # Installation guide
├── quickstart.md                 # Getting started guide
├── architecture.md               # System architecture
├── concepts/                     # Conceptual explanations
│   ├── affective_drift.md
│   ├── emotional_residue.md
│   ├── recursive_emotion.md
│   └── symbolic_shells.md
├── api/                          # API documentation
├── tutorials/                    # Step-by-step tutorials
└── examples/                     # Example use cases
```

### `case_studies/`

Real-world examples and analyses.

```
case_studies/
├── README.md                     # Case studies overview
├── empathy_drift_climate.md      # Empathy drift in climate change reasoning
├── value_conflicts.md            # Value conflicts in ethical dilemmas
├── confidence_collapse.md        # Confidence collapse patterns
├── emotional_residue.md          # Emotional residue in decision-making
└── hesitation_medical.md         # Recursive hesitation in medical diagnoses
```

## Component Relationships

EmotionOS is designed with a layered architecture:

1. **Foundation Layer**: `emotion_core` provides the fundamental detection and analysis capabilities
2. **Application Layer**: `affective_drift_console` and `symbolic_emotion_shells` build on the core for specialized applications
3. **Research Layer**: `crossmodel_emotion_logs` and `case_studies` demonstrate findings and comparative research
4. **Extension Layer**: `decentralization_templates` enables community expansion and research

## Development Workflow

The typical workflow progresses through these stages:

1. Implement detection capabilities in `emotion_core`
2. Create visualization interfaces in `affective_drift_console`
3. Design diagnostic tools as `symbolic_emotion_shells`
4. Collect and compare results in `crossmodel_emotion_logs`
5. Document findings in `case_studies`
6. Share templates in `decentralization_templates`

This enables a continuous cycle of emotional interpretability research and improvement.
```

