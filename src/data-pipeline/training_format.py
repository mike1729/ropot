"""Pure task-grouped monitor export format, shared by producer and consumer."""

SCHEMA_VERSION = "monitor-training-v1"
ROLES = ("benign", "actor_attack", "synthetic_attack")
PHASES = {"shared_honest": "shared_honest_pretraining", "twin_continuation": "twin_continuation"}
