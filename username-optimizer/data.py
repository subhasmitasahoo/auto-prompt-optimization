"""
Training and evaluation data for the Username Personality Archetype Classifier.

Each Example maps a username to one of 5 personality archetypes:
  - The Creator   : artistry, building, crafting, making things
  - The Rebel     : chaos, edge, counter-culture, breaking norms
  - The Strategist: logic, precision, data, calculated moves
  - The Dreamer   : cosmic, whimsical, ethereal, imaginative
  - The Maverick  : bold, adventurous, unconventional, untamed
"""

from dataclasses import dataclass


@dataclass
class Example:
    username: str
    archetype: str

    def __repr__(self):
        return f"Username: '{self.username}'  →  Archetype: '{self.archetype}'"


# ---------------------------------------------------------------------------
# TRAINING DATA  (used in the APO optimization loop)
# ---------------------------------------------------------------------------
TRAINING_DATA = [
    # ── The Creator ──────────────────────────────────────────────────────────
    Example("pixel_forge",           "The Creator"),
    Example("inkdreamer99",          "The Creator"),
    Example("codecraft_",            "The Creator"),
    Example("brushstroke_kai",       "The Creator"),
    Example("velvet_quill",          "The Creator"),
    Example("maker_of_worlds",       "The Creator"),
    Example("forge_and_flourish",    "The Creator"),
    Example("sculpt_code",           "The Creator"),
    Example("typeset_ghost",         "The Creator"),
    Example("loom_and_loop",         "The Creator"),
    # complex
    Example("the_artisan_protocol",  "The Creator"),
    Example("handcrafted_hexadecimal", "The Creator"),
    Example("obsidian_workshop_7",   "The Creator"),

    # ── The Rebel ────────────────────────────────────────────────────────────
    Example("ch4os_engine",          "The Rebel"),
    Example("glitch_witch",          "The Rebel"),
    Example("neon_anarchist",        "The Rebel"),
    Example("xX_dark_l0rd_Xx",       "The Rebel"),
    Example("acid_drop_viper",       "The Rebel"),
    Example("null_pointer_punk",     "The Rebel"),
    Example("r3b3l_without_cause",   "The Rebel"),
    Example("system_crasher",        "The Rebel"),
    Example("anti_meta_ghost",       "The Rebel"),
    Example("void_protocol_breaker", "The Rebel"),
    # complex
    Example("404_rules_not_found",   "The Rebel"),
    Example("hexed_outlier_v2",      "The Rebel"),
    Example("d3vi4nt_signal",        "The Rebel"),

    # ── The Strategist ───────────────────────────────────────────────────────
    Example("cipher_logic",          "The Strategist"),
    Example("sigma_protocol",        "The Strategist"),
    Example("datamind_x",            "The Strategist"),
    Example("calculated_ghost",      "The Strategist"),
    Example("meta_analyst_9",        "The Strategist"),
    Example("binary_tactician",      "The Strategist"),
    Example("optima_node",           "The Strategist"),
    Example("precision_vector",      "The Strategist"),
    Example("entropy_minimizer",     "The Strategist"),
    Example("logic_gate_prime",      "The Strategist"),
    # complex
    Example("chess_engine_overflow", "The Strategist"),
    Example("bayesian_mindset_x",    "The Strategist"),
    Example("cold_calculus_7734",    "The Strategist"),

    # ── The Dreamer ──────────────────────────────────────────────────────────
    Example("stardust_mind",         "The Dreamer"),
    Example("lunar_echo",            "The Dreamer"),
    Example("velvet_cosmos",         "The Dreamer"),
    Example("aurora_whisper",        "The Dreamer"),
    Example("nebula_soft",           "The Dreamer"),
    Example("driftwood_reverie",     "The Dreamer"),
    Example("gossamer_signal",       "The Dreamer"),
    Example("solstice_soul",         "The Dreamer"),
    Example("moonlit_vagrant",       "The Dreamer"),
    Example("ethereal_comma",        "The Dreamer"),
    # complex
    Example("between_two_galaxies",  "The Dreamer"),
    Example("twilight_theorem_ix",   "The Dreamer"),
    Example("half_asleep_oracle",    "The Dreamer"),

    # ── The Maverick ─────────────────────────────────────────────────────────
    Example("rogue_vector",          "The Maverick"),
    Example("storm_circuit",         "The Maverick"),
    Example("blaze_axiom",           "The Maverick"),
    Example("wild_bandwidth",        "The Maverick"),
    Example("ironclad_nomad",        "The Maverick"),
    Example("voltage_outlaw",        "The Maverick"),
    Example("apex_lone_wolf",        "The Maverick"),
    Example("turbo_unhinged",        "The Maverick"),
    Example("feral_uptime",          "The Maverick"),
    Example("warpspeed_drifter",     "The Maverick"),
    # complex
    Example("untethered_flare_99",   "The Maverick"),
    Example("reckless_gradient",     "The Maverick"),
    Example("off_the_grid_operator", "The Maverick"),
]


# ---------------------------------------------------------------------------
# EVALUATION DATA  (held-out set — never used during optimization)
# ---------------------------------------------------------------------------
EVAL_DATA = [
    Example("canvas_hacker",         "The Creator"),
    Example("render_witch",          "The Creator"),
    Example("syntax_sculptor",       "The Creator"),

    Example("corrupt_signal",        "The Rebel"),
    Example("error_404_soul",        "The Rebel"),
    Example("chaotic_uplink",        "The Rebel"),

    Example("null_hypothesis_x",     "The Strategist"),
    Example("gradient_descent_pro",  "The Strategist"),
    Example("edge_case_oracle",      "The Strategist"),

    Example("cloudwatcher_99",       "The Dreamer"),
    Example("soft_parallax",         "The Dreamer"),
    Example("quiet_singularity",     "The Dreamer"),

    Example("throttle_ghost",        "The Maverick"),
    Example("wildfire_kernel",       "The Maverick"),
    Example("loose_cannon_dev",      "The Maverick"),
]


ARCHETYPES = [
    "The Creator",
    "The Rebel",
    "The Strategist",
    "The Dreamer",
    "The Maverick",
]
