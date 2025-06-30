import warnings

import openai
from attrs import field, frozen

from crew_algorithms.multimodal_feedback.gpt_constants import SYSTEM_CONTENT

openai.api_key = "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"


def _convert_feedback_to_template_message(feedback1, feedback2):
    return f"""
<trajectories>
    <trajectory>
        <overall-feedback>{feedback1.overall_feedback}</overall-feedback>
        <improvement>{feedback1.improvement}</improvement>
        <good>{feedback1.good}</good>
        <rating>{feedback1.rating}</rating>
    </trajectory>
    <trajectory>
        <overall-feedback>{feedback2.overall_feedback}</overall-feedback>
        <improvement>{feedback2.improvement}</improvement>
        <good>{feedback2.good}</good>
        <rating>{feedback2.rating}</rating>
    </trajectory>
</trajectories>
""".strip()


def _rank_feedback(t1, t2):
    print(_convert_feedback_to_template_message(t1, t2))
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_CONTENT},
            {
                "role": "user",
                "content": _convert_feedback_to_template_message(t1, t2),
            },
        ],
    )
    try:
        output = completion.choices[0].message.content
        print(output)
        ranking = int(output[-1])
        is_valid_ranking = ranking in [1, 2]
        if not is_valid_ranking:
            raise ValueError("Invalid ranking!")
        return True if ranking == 1 else False
    except ValueError:
        warnings.warn(f"ChatGPT gave an invalid response: {output}!")
        return 0


@frozen
class TrajectoryFeedback:
    id: int = field(eq=True)
    overall_feedback: str = field(eq=False)
    improvement: str = field(eq=False)
    good: str = field(eq=False)
    rating: int = field(eq=False)

    def __lt__(self, other):
        if not isinstance(other, TrajectoryFeedback):
            raise ValueError(f"Unexpected object of type {type(other).__name__}")
        return _rank_feedback(self, other)

    @classmethod
    def from_id_and_str(cls, id, feedback):
        print(feedback)
        before_overall_feedback, _, after_overall_feedback = feedback.partition(
            "Overall Feedback:"
        )
        before_improvement, _, after_improvement = after_overall_feedback.partition(
            "Improvement:"
        )
        before_good, _, after_good = after_improvement.partition("What went well?:")
        before_rating, _, after_rating = after_good.partition("Rating (1-10):")
        try:
            rating = int(after_rating.strip("n").strip())
        except ValueError:
            rating = 5
        return cls(
            id,
            before_improvement.strip("n").strip(),
            before_good.strip("n").strip(),
            before_rating.strip("n").strip(),
            rating,
        )
