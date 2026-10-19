"""The paper's training configurations, as composable presets.

A run takes one method preset and one target preset, e.g.

    com-train preset=com,gemma behavior_ids=transluce_tabun

and any field after it overrides the presets. Every value here is what the
paper's runs used (read off the source repo's evals/configs/*.json), except
where noted.
"""

# The SFT warm start the paper's Black-box runs began from (and the PRBO
# warm-start ablation). It lives in the authors' Tinker account; to reproduce
# it, run the investigator SFT pipeline (see README) and pass its
# /weights/ path as load_checkpoint_path.
PAPER_WARM_START = (
    "tinker://dc6e07a3-5e75-5480-bd95-57ddd75a4394:train:0/weights/sft-final"
)

METHODS: dict[str, dict] = {
    # Curriculum of Models: one ladder per behavior, easiest stage first.
    "com": {
        "reward_type": "curriculum",
        "curriculum_scales": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
        "curriculum_threshold": 60.0,
        "kl_penalty_coef": 0.5,
        "zero_reward_on_truncation": True,
        "judge_skip_refusals": False,
    },
    # Black-box: CoM's reward on the base target only, from the SFT warm
    # start. Needs load_checkpoint_path.
    "blackbox": {
        "reward_type": "curriculum",
        "curriculum_scales": [0.0],
        "curriculum_threshold": 60.0,
        "kl_penalty_coef": 0.5,
        "zero_reward_on_truncation": True,
        "judge_skip_refusals": False,
    },
    # PRBO, as Transluce's combined estimator.
    "prbo": {
        "reward_type": "combined_estimator",
        "prbo_judge_scale": 1.1,
        "prbo_natural_response_threshold": None,
        "kl_penalty_coef": 0.1,
        "zero_reward_on_truncation": False,
        "judge_skip_refusals": True,
    },
    # Plain PRBO, without the natural-response term (ablation).
    "prbo_plain": {
        "reward_type": "prbo",
        "prbo_judge_scale": 1.1,
        "kl_penalty_coef": 0.1,
        "zero_reward_on_truncation": False,
        "judge_skip_refusals": True,
    },
}

# target_max_tokens: the paper's CoM runs used 2400 (6000 with reasoning) and
# PRBO 7000 (14000 with reasoning). Responses end at EOS well below either, so
# both methods use the PRBO values here.
TARGETS: dict[str, dict] = {
    "gemma": {
        "target_model_id": "google/gemma-4-31B-it",
        "chat_template_kwargs": {"enable_thinking": False},
        "target_max_tokens": 7000,
    },
    "gemma_reasoning": {
        "target_model_id": "google/gemma-4-31B-it",
        "chat_template_kwargs": {"enable_thinking": True},
        "target_max_tokens": 14000,
    },
    # gpt-oss always reasons (vLLM returns the final channel as the response,
    # and PRBO splits the reasoning off), so it has no non-reasoning variant.
    "gpt_oss": {
        "target_model_id": "openai/gpt-oss-120b",
        "chat_template_kwargs": {},
        "target_max_tokens": 7000,
    },
}

PRESETS = {**METHODS, **TARGETS}
