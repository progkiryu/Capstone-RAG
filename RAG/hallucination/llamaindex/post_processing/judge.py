import json
import re
from llama_index.core.prompts import ChatPromptTemplate

LABEL_SUPPORT = {"yes": 1.0, "partial": 0.5, "no": 0.0}

JUDGE_PROMPT = """
You are checking whether numbered claims are supported by retrieved context.
Decide only whether the context supports each claim. Do not use outside knowledge and do not
judge whether a claim is true in the real world.

- "yes": the context states or directly implies the claim
- "partial": the context supports only part of the claim
- "no": the context does not support the claim

The claims answer this question:
{question}

Context:
{context}

Claims:
{claims}

Respond with only a JSON array containing one object per claim, for example:
[{{"claim_id": 1, "supported": "yes"}}, {{"claim_id": 2, "supported": "no"}}]
"""


def parse_verdicts(text, claim_count):
    # DeepSeek-R1 prefixes a <think> block; a code fence is the only wrapper tolerated
    text = text.rsplit("</think>", 1)[-1].strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    verdicts = json.loads(fenced.group(1) if fenced else text)

    if not isinstance(verdicts, list) or len(verdicts) != claim_count:
        raise ValueError(f"expected a list of {claim_count} verdicts")
    by_id = {}
    for verdict in verdicts:
        claim_id = verdict.get("claim_id") if isinstance(verdict, dict) else None
        label = verdict.get("supported") if isinstance(verdict, dict) else None
        if type(claim_id) is not int or label not in LABEL_SUPPORT:
            raise ValueError(f"malformed verdict {verdict!r}")
        by_id[claim_id] = LABEL_SUPPORT[label]
    if sorted(by_id) != list(range(1, claim_count + 1)):
        raise ValueError(f"claim ids {sorted(by_id)} do not match 1..{claim_count}")
    return [by_id[claim_id] for claim_id in range(1, claim_count + 1)]


class Judge:
    def __init__(self, judge_llm):
        self.llm = judge_llm
        self.prompt = ChatPromptTemplate.from_messages([("user", JUDGE_PROMPT)])

    def judge(self, question, claims, chunks):
        text = self.llm.predict(
            self.prompt,
            question=question,
            context="\n\n".join(chunks),
            claims="\n".join(f"{i}. {claim}" for i, claim in enumerate(claims, 1)),
        )
        try:
            return parse_verdicts(text, len(claims))
        except ValueError as error:
            # json.JSONDecodeError is a ValueError; fall back to NLI scores alone
            print(f"Judge: unparseable verdicts ({error}); using NLI support only")
            return None
