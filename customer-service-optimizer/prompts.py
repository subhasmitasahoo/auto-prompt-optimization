"""
Prompt templates for the Auto Prompt Optimization loop.

Directly mirrors the templates described in the project README.
"""

# ---------------------------------------------------------------------------
# Initial (seed) prompt — deliberately minimal, two lines only
# ---------------------------------------------------------------------------
INITIAL_PROMPT = """\
# Task
Categorize the customer support ticket.

# Output format
Classify into one of these classes: 'Technical Support', 'Billing', \
'General Information', 'Complaint and Escalations', 'Feedback and Suggestions'.
Respond with ONLY the class name, nothing else."""


# ---------------------------------------------------------------------------
# Gradient prompt — asks the model WHY the current prompt fails
# ---------------------------------------------------------------------------
GRADIENT_PROMPT_TEMPLATE = """\
I'm trying to write a zero-shot classifier prompt.
My current prompt is:
"{prompt}"

But this prompt gets the following examples wrong:
{error_string}

Give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.
Wrap each reason with <START> and <END>"""


# ---------------------------------------------------------------------------
# Edit prompt — generates improved prompt candidates from gradient feedback
# ---------------------------------------------------------------------------
EDIT_PROMPT_TEMPLATE = """\
I'm trying to write a zero-shot classifier.
My current prompt is:
"{prompt}"

But it gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that {gradient}

Based on the above information, write {steps_per_gradient} different improved prompts.
To handle the losses, consider making the descriptions of each class in the improved prompts.
Each prompt should be wrapped with <START> and <END>.
The {steps_per_gradient} new prompts are:"""
