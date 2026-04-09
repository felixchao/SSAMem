PROMPTS = {
    "autogen": {
        "assistant_system": "Answer the given question. You should answer concisely.",
        "assistant_user": """
Below is the retrieved memory content and the current task. Please provide your answer.

# Retrieved Content
{memory_content}

# Current Task
{task_description}
""".strip(),
        "user_proxy_system": """
You will be given a question and an assistant's answer for that question. Provide the best final answer.
""".strip(),
        "user_proxy_user": """
Below is the relevant content retrieved from memory, the answer provided by the assistant, and the current task. Please provide your answer.

# Retrieved Content
{memory_content}

# Assistant Answer
{assistant_output}

# Current Task
{task_description}
""".strip(),
    },
    "camel": {
        "user_proxy_system": "Answer the given question. You should answer concisely.",
        "user_proxy_user": """
Below is the retrieved memory content and the current task. Please provide your answer.

# Retrieved Content
{memory_content}

# Current Task
{task_description}
""".strip(),
        "actor_system": """
You will be given a question and a user proxy agent's answer for that question. Provide an improved answer.
""".strip(),
        "actor_user": """
Below is the relevant content retrieved from memory, the answer provided by the user proxy agent, and the current task. Please provide your answer.

# Retrieved Content
{memory_content}

# User Proxy Answer
{userproxy_output}

# Current Task
{task_description}
""".strip(),
        "critic_system": """
You are a strategy evaluator. If the actor answer is correct, reply only with "Agree". Otherwise provide short feedback.
""".strip(),
        "critic_user": """
Below are the relevant contents retrieved from memory, the response given by the actor agent, and the requirements of the current task. Please provide your review.

# Retrieved Content
{memory_content}

# Actor Output
{actor_output}

# Current Task
{task_description}
""".strip(),
        "summarizer_system": "Return the best final answer based on the actor output and critic feedback.",
        "summarizer_user": """
# Retrieved Content
{memory_content}

# Actor Output
{actor_output}

# Critic Feedback
{critic_output}

# Current Task
{task_description}
""".strip(),
    },
    "macnet": {
        "actor_system": "Answer the given question. You should answer concisely.",
        "actor_user": """
Below is the relevant content retrieved from memory and the current task requirements. Please provide your answer.

# Retrieved Content
{memory_content}

# Current Task
{task_description}
""".strip(),
        "critic_system": """
You are a strategy evaluator. If the actor answer is correct, reply only with "Agree". Otherwise provide short feedback.
""".strip(),
        "critic_user": """
Below are the relevant contents retrieved from memory, the response given by the actor agent, and the requirements of the current task. Please provide your review.

# Retrieved Content
{memory_content}

# Actor Output
{actor_output}

# Current Task
{task_description}
""".strip(),
        "summarizer_system": "Return the best final answer by combining both branches.",
        "summarizer_user": """
# Retrieved Content
{memory_content}

# Actor1 Output and Critic1 Feedback
{feedback_page1}

# Actor2 Output and Critic2 Feedback
{feedback_page2}

# Current Task
{task_description}
""".strip(),
        "feedback_page": """
## Actor Output
{actor_output}

## Critic Feedback
{critic_output}
""".strip(),
    },
}

