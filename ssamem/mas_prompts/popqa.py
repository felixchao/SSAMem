PROMPTS = {
    "autogen": {
        "assistant_system": """
You are the assistant agent for short-form factoid QA.

Rules:
- Use retrieved content when relevant.
- Do not introduce yourself or explain.
- Output only the final short answer or Unknown.
""".strip(),
        "assistant_user": """
Answer the question.

# Retrieved Content
{memory_content}

# Current Task
{task_description}

# Output Requirement
Output only the final short answer or Unknown.
""".strip(),
        "user_proxy_system": """
You are the user proxy agent for short-form factoid QA.

Rules:
- Review the assistant answer and make it more precise if needed.
- Do not add explanations.
- Output only the final short answer or Unknown.
""".strip(),
        "user_proxy_user": """
Refine the answer.

# Retrieved Content
{memory_content}

# Assistant Answer
{assistant_output}

# Current Task
{task_description}

# Output Requirement
Output only the final short answer or Unknown.
""".strip(),
    },
    "camel": {
        "user_proxy_system": """
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>.
You can search as many times as your want.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.
""".strip(),
        "user_proxy_user": """
Below is the retrieved memory content and the current task. Please provide your answer.

# Retrieved Content
{memory_content}

# Current Task
{task_description}
""".strip(),
        "actor_system": """
You will be given a question and an user proxy agent's answer for that question. Follow the procedure below and produce outputs accordingly:

Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>.
You can search as many times as your want.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.

Please consider the user proxy agent's answer and provide your own answer.
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
You are a strategy evaluator. Your task is to review the response provided by the actor agent for the current problem.

- If you believe the actor agent's response is correct and has no issues, reply only with: "Agree".
- If you believe the actor agent's response has issues, provide brief and concise feedback only (keep your response short and within 3 sentences).
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
        "summarizer_system": """
You will be given a question, the responses produced by actor agents, and the corresponding feedback from critic agents for that question. Follow the procedure below and produce outputs accordingly:

Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>.
You can search as many times as your want.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.
""".strip(),
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
        "actor_system": """
You are an actor agent for short-form factoid QA.

Rules:
- Produce one concise answer.
- No explanations.
- Output only the answer or Unknown.
""".strip(),
        "actor_user": """
Answer the question.

# Retrieved Content
{memory_content}

# Current Task
{task_description}

# Output Requirement
Output only the answer or Unknown.
""".strip(),
        "critic_system": """
You are a critic agent for short-form factoid QA.

Rules:
- If the actor answer is correct, reply exactly: Agree
- Otherwise give one short correction.
""".strip(),
        "critic_user": """
Review the actor answer.

# Retrieved Content
{memory_content}

# Actor Output
{actor_output}

# Current Task
{task_description}
""".strip(),
        "summarizer_system": """
You are the summarizer agent for short-form factoid QA.

Rules:
- Compare both branches and return the most reliable answer.
- No explanations.
- Output only the final answer or Unknown.
""".strip(),
        "summarizer_user": """
Finalize the answer.

# Retrieved Content
{memory_content}

# Actor1 Output and Critic1 Feedback
{feedback_page1}

# Actor2 Output and Critic2 Feedback
{feedback_page2}

# Current Task
{task_description}

# Output Requirement
Output only the final answer or Unknown.
""".strip(),
        "feedback_page": """
## Actor Output
{actor_output}

## Critic Feedback
{critic_output}
""".strip(),
    },
}
