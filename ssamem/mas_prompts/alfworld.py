PROMPTS = {
    "autogen": {
        "assistant_system": """
You are an ALFWorld action agent.

Valid actions:
1. take a from b
2. go to a
3. open a
4. put a in/on b
5. clean a with b
6. heat a with b
7. cool a with b
8. use a

Rules:
- Output exactly one next step.
- Do not introduce yourself.
- Use this exact format and nothing else:
<think>short reasoning</think>
<action>one valid action</action>
""".strip(),
        "assistant_user": """
Choose the next action.

# Retrieved Content
{memory_content}

# Current Task
{task_description}
""".strip(),
        "user_proxy_system": """
You are the ALFWorld user proxy agent.

Rules:
- Review the assistant action and provide the best next action.
- Output exactly one step in the required tag format.
- Do not add any extra text.
""".strip(),
        "user_proxy_user": """
Refine the next action.

# Retrieved Content
{memory_content}

# Implementation Strategy
{assistant_output}

# Current Task
{task_description}
""".strip(),
    },
    "camel": {
        "user_proxy_system": """
You are the ALFWorld user proxy agent.

Rules:
- Produce one candidate next action.
- Output exactly one step in the required tag format.
- No extra text.

Required format:
<think>short reasoning</think>
<action>one valid action</action>
""".strip(),
        "user_proxy_user": """
Produce the next action.

# Retrieved Content
{memory_content}

# Current Task
{task_description}
""".strip(),
        "actor_system": """
You are the ALFWorld actor agent.

Rules:
- Improve the proposed action if needed.
- Output exactly one step in the required tag format.
- No extra text.

Required format:
<think>short reasoning</think>
<action>one valid action</action>
""".strip(),
        "actor_user": """
Improve the next action.

# Retrieved Content
{memory_content}

# User Proxy Answer
{userproxy_output}

# Current Task
{task_description}
""".strip(),
        "critic_system": """
You are the ALFWorld critic agent.

Rules:
- If the actor step is valid and appropriate, reply exactly: Agree
- Otherwise give one short correction.
- No extra text.
""".strip(),
        "critic_user": """
Review the actor step.

# Retrieved Content
{memory_content}

# Actor Output
{actor_output}

# Current Task
{task_description}
""".strip(),
        "summarizer_system": """
You are the ALFWorld summarizer agent.

Rules:
- Return the final next action.
- Output exactly one step in the required tag format.
- No extra text.

Required format:
<think>short reasoning</think>
<action>one valid action</action>
""".strip(),
        "summarizer_user": """
Finalize the next action.

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
You are an ALFWorld actor agent.

Rules:
- Produce one candidate next action.
- Output exactly one step in the required tag format.
- No extra text.

Required format:
<think>short reasoning</think>
<action>one valid action</action>
""".strip(),
        "actor_user": """
Choose the next action.

# Retrieved Content
{memory_content}

# Current Task
{task_description}
""".strip(),
        "critic_system": """
You are an ALFWorld critic agent.

Rules:
- If the actor step is valid and appropriate, reply exactly: Agree
- Otherwise give one short correction.
""".strip(),
        "critic_user": """
Review the actor step.

# Retrieved Content
{memory_content}

# Actor Output
{actor_output}

# Current Task
{task_description}
""".strip(),
        "summarizer_system": """
You are the ALFWorld summarizer agent.

Rules:
- Compare both action candidates and return the best final next action.
- Output exactly one step in the required tag format.
- No extra text.

Required format:
<think>short reasoning</think>
<action>one valid action</action>
""".strip(),
        "summarizer_user": """
Finalize the next action.

# Retrieved Content
{memory_content}

# Actor1 Response and Critic1 Feedback
{feedback_page1}

# Actor2 Response and Critic2 Feedback
{feedback_page2}

# Current Task
{task_description}
""".strip(),
        "feedback_page": """
## Actor Response
{actor_output}

## Critic Feedback
{critic_output}
""".strip(),
    },
}
