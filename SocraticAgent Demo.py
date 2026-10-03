import asyncio
import json
import logging
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Annotated, Literal, TypedDict
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from openai import OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


API_KEY = "your_api_key"
BASE_URL = "your_api_base_url"

json_md_re = re.compile(r"```json(.*?)```", re.DOTALL)
boxed_re = re.compile(r"\\boxed\{([^}]*)\}")

MAX_ATTEMPTS = 1
LLM_CLIENT = None
NO_SYSTEM_PROMPT_MODEL_LIST = ["gemma-2-2b-it", "gemma-2-9b-it", "gemma-2-27b-it"]
STOP_TOKEN_DICT = {
    "qwen": ["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
    "gemma": ["<end_of_turn>", "<eos>"],
    "llama3": ["<|eot_id|>"],
    "llama2": ["\n\n\n", "</s>"],
    "other": [],
}


class LLMWrapper:
    """
    A wrapper class for the OpenAI API client to encapsulate the client,
    rate limiting, model configuration, and API call logic.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        model_type: str,
        max_concurrent_requests: int = 9999999,
        requests_per_minute: float = float("inf"),
        requests_per_second: float = float("inf"),
        proxy: Optional[str] = None,
        api_logger: logging.Logger = None,
    ):
        """
        Initializes the LLMWrapper.

        Args:
            api_key: The API key for the OpenAI client.
            base_url: The base URL for the API endpoint.
            model_name: The name of the model to be used for API calls.
            model_type: The type of the model (e.g., 'qwen', 'gemma').
            max_concurrent_requests: The maximum number of concurrent API requests.
            api_logger: The logger for recording API statistics.
        """
        self.model_name = model_name
        self.model_type = model_type
        self.base_url = base_url

        extra_params = {}
        if proxy:
            extra_params["http_client"] = httpx.Client(
                proxies=proxy,
                transport=httpx.HTTPTransport(local_address="0.0.0.0"),
                verify=False,
                timeout=7200,
            )
        else:
            extra_params["http_client"] = httpx.Client(timeout=7200)

        self.client = OpenAI(api_key=api_key, base_url=base_url, **extra_params)

        self.rate_limiter = APIRateLimiter(
            max_concurrent_requests=max_concurrent_requests,
            requests_per_minute=requests_per_minute,
            requests_per_second=requests_per_second,
            logger=api_logger,
        )

    def call_api(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]] = None,
        uuid: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Optional[Any]:
        """
        Makes a rate-limited API call to the LLM.

        This method uses the configured rate limiter to call the OpenAI API,
        handling retries and logging automatically. It ensures the response is
        valid before returning.

        Args:
            messages: A list of messages for the chat completion.
            max_tokens: The maximum number of tokens to generate.
            temperature: The sampling temperature.
            stop: A list of stop sequences.
            uuid: A unique identifier for the request for logging purposes.

        Returns:
            The full API response object if successful, otherwise None.
        """

        if stop is None:
            stop = STOP_TOKEN_DICT.get(self.model_type, []) or []
        else:
            stop = list(set((STOP_TOKEN_DICT.get(self.model_type, []) or []) + stop))

        response = self.rate_limiter.rate_limited_api_call(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            uuid=uuid,
            *args,
            **kwargs,
        )

        if (
            response
            and hasattr(response, "choices")
            and response.choices
            and hasattr(response.choices[0], "message")
            and hasattr(response.choices[0].message, "content")
            and isinstance(response.choices[0].message.content, str)
        ):
            return response
        else:
            logger.warning(f"Invalid or empty response received for uuid: {uuid}")
            return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.rate_limiter.shutdown()


class APIRateLimiter:
    def __init__(
        self,
        max_concurrent_requests,
        requests_per_second=float("inf"),
        requests_per_minute=float("inf"),
        logger=None,
    ):
        self.max_concurrent_requests = max_concurrent_requests
        self._semaphore = threading.Semaphore(max_concurrent_requests)
        self.api_logger = logger

        self.requests_per_second = requests_per_second
        self.requests_per_minute = requests_per_minute

        self._min_interval = self._calculate_min_interval()

        self._last_request_time = 0
        self._rate_lock = threading.Lock()

    def _calculate_min_interval(self):
        """Calculate the minimum interval between requests based on rate limits."""
        intervals = []

        if self.requests_per_second != float("inf") and self.requests_per_second > 0:
            intervals.append(1.0 / self.requests_per_second)

        if self.requests_per_minute != float("inf") and self.requests_per_minute > 0:
            intervals.append(60.0 / self.requests_per_minute)

        return max(intervals) if intervals else 0

    def get_available_slots(self):
        """Returns the number of available slots for new requests."""
        return self._semaphore._value

    def rate_limited_api_call(self, api_function, *args, **kwargs):
        """
        Makes an API call using the provided function, respecting rate limits,
        with retries and logging.
        """
        uuid = kwargs.pop("uuid", None)

        with self._semaphore:
            return self._execute_api_call(api_function, uuid, *args, **kwargs)

    def _apply_rate_limit(self):
        """Apply smooth rate limiting by sleeping if necessary."""
        if self._min_interval <= 0:
            return

        with self._rate_lock:
            current_time = time.time()
            time_since_last_request = current_time - self._last_request_time

            if time_since_last_request < self._min_interval:
                sleep_time = self._min_interval - time_since_last_request
                time.sleep(sleep_time)
                self._last_request_time = time.time()
            else:
                self._last_request_time = current_time

    def _execute_api_call(self, api_function, uuid, *args, **kwargs):
        """Execute the API call with retry logic."""
        response = None
        last_exception = None

        for attempt in range(MAX_ATTEMPTS):

            self._apply_rate_limit()

            try:
                response = api_function(*args, **kwargs)
                if response:
                    break
                else:
                    logger.warning(
                        f"API call attempt {attempt + 1}/{MAX_ATTEMPTS} returned unexpected format for uuid {uuid}. "
                        f"The request is {kwargs.get('messages')}"
                    )
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"API call attempt {attempt + 1}/{MAX_ATTEMPTS} failed for uuid {uuid}: {e}"
                )

                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(min(10, 0.5 * (attempt + 1)))

        if response is None:
            logger.error(
                f"API call failed after {MAX_ATTEMPTS} attempts for uuid {uuid}. "
                f"Last exception: {last_exception}"
            )

        self._log_api_stats(response, uuid)

        return response

    def _log_api_stats(self, response, uuid):
        """Log API call statistics if logger is configured."""
        if not self.api_logger:
            return

        prompt_tokens = completion_tokens = total_tokens = -1
        success = response is not None

        if response and hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", -1)
            completion_tokens = getattr(response.usage, "completion_tokens", -1)
            total_tokens = getattr(response.usage, "total_tokens", -1)

        api_stat_dict = {
            "uuid": uuid,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "success": success,
        }

        self.api_logger.info(json.dumps(api_stat_dict, ensure_ascii=False))


def construct_prompt(
    model_name: str, common_prompts: list, system_prompt: Optional[str] = None
) -> List[Dict[str, str]]:
    """Constructs the prompt messages for the LLM."""
    messages = []
    if system_prompt:
        if model_name in NO_SYSTEM_PROMPT_MODEL_LIST:
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": f"Please follow the instruction below:\n```\n{system_prompt}\n```\nIf you understand the instruction, please reply with 'I understand'.",
                    },
                    {"role": "assistant", "content": "I understand"},
                ]
            )
        else:
            messages.append({"role": "system", "content": system_prompt})
    messages.extend(common_prompts)
    return messages


class BoxedAnswerExtractor:
    """
    A class to handle the extraction of a \\boxed{} answer from a language model's
    response when it's not present in the initial generation.
    """

    def __init__(self, llm_wrapper: LLMWrapper):
        """
        Initializes the BoxedAnswerExtractor.

        Args:
            llm_wrapper: The LLMWrapper instance to handle API calls.
        """
        self.llm_wrapper = llm_wrapper

    def extract_answer(
        self, messages: List[Dict[str, str]], original_response: str, uuid: str = None
    ) -> str:
        """
        Attempts to extract a boxed answer by prompting the model again.
        """
        try:
            extraction_prompt_messages = messages + [
                {
                    "role": "assistant",
                    "content": (
                        f"{original_response}\n\n"
                        r"Now, let's format the result in `<answer>\boxed{RESULT}</answer>` format."
                        r" So the answer is <answer>"
                    ),
                    "prefix": True,
                }
            ]

            stop_tokens = (
                STOP_TOKEN_DICT.get(self.llm_wrapper.model_type, []) or []
            ) + ["</answer>"]

            llm_response = self.llm_wrapper.call_api(
                messages=extraction_prompt_messages,
                max_tokens=32,
                temperature=0.0,
                stop=stop_tokens,
                uuid=uuid + "_extract_answer",
            )

            if llm_response:
                extracted_part = llm_response.choices[0].message.content.strip()
                return f"{original_response} So the answer is {extracted_part}"
            else:
                return original_response

        except Exception as e:
            return original_response


knowledge_analyzer_system_prompt = """# ROLE: Knowledge Extraction Agent for elementary-level questions.

- Extract 3-8 concepts (nouns/terms, 1-3 words each).
- Concepts must include: core terms from the question, necessary prerequisites, and critical background knowledge.
- Keep concepts appropriate for an elementary student.
- CRITICAL: Do NOT answer the question or leak any part of the answer.

Respond with a JSON object containing a single key "concepts" with a list of strings.

"Why do plants need sunlight to make food?"

```json
{"concepts": ["photosynthesis", "chlorophyll", "sunlight", "glucose"]}
```"""

knowledge_analyzer_prompt = """Extract key concepts from the following question.

```
{question}
```"""

knowledge_analyzer_critic_system_prompt = """# ROLE: Knowledge Analysis Critic.

- Evaluate the concepts based on the following criteria:
    1.  **Completeness**: Are there 3-8 essential concepts for a basic understanding?
    2.  **Relevance**: Is every concept directly related to the question?
    3.  **Appropriateness**: Are concepts simple enough for an elementary student?
    4.  **Brevity**: Are concepts concise (1-3 words)?
- If all criteria pass, decide "ACCEPT".
- If any criterion fails, decide "REVISE" and provide specific, actionable feedback for improvement.

Respond with a JSON object: {"decision": "ACCEPT" or "REVISE", "feedback": "Your detailed feedback..."}"""

knowledge_analyzer_critic_prompt = """Evaluate the concepts below based on the question.

```
{question}
```

```json
{knowledge_points_json}
```"""

knowledge_analyzer_reviser_system_prompt = """# ROLE: Knowledge Concept Reviser.

- Strictly follow all instructions in the provided critique.
- Add, remove, or simplify concepts to meet all evaluation criteria (relevance, completeness, appropriateness, brevity).
- The final list must contain 3-8 concise concepts (1-3 words each).

Output ONLY the revised JSON object with the "concepts" key."""

knowledge_analyzer_reviser_prompt = """Revise the concepts based on the critique.

```
{question}
```

```json
{knowledge_points_json}
```

```json
{knowledge_analysis_critique}
```"""

knowledge_explainer_system_prompt = """# ROLE: Expert Educator for elementary students.

- **Length**: 2-4 sentences maximum.
- **Style**: Define like a dictionary or encyclopedia. Focus on WHAT the concept is.
- **Independence**: The definition must be stand-alone and NEVER reference any external question.
- **Formatting**: Start the explanation with the concept term in bold (e.g., "**Gravity** is...").

Respond with a JSON object: {"explanation": "Your definition..."}

```json
{"explanation": "**Gravity** is an invisible force that pulls objects toward each other. The bigger an object is, the stronger its gravitational pull. This is why we stay on Earth instead of floating away into space."}
```"""

knowledge_explainer_user_prompt = """Define this concept for an elementary student.

"{concept}"
"""

knowledge_explainer_critic_system_prompt = """# ROLE: Explanation Critic for elementary education.

- Evaluate the explanation based on the following criteria:
    1.  **Definition-Focused**: Does it explain WHAT the concept is?
    2.  **Clarity & Simplicity**: Is the language easy for a child to understand?
    3.  **Independence**: Is it stand-alone? It must NOT reference the original question.
    4.  **Format**: Does it begin with the concept in bold (e.g., "**Gravity** is...")?
- If all criteria pass, decide "ACCEPT".
- If any criterion fails, decide "REVISE" and provide specific, actionable feedback.

Respond with a JSON object: {"decision": "ACCEPT" or "REVISE", "feedback": "Your detailed feedback..."}"""

knowledge_explainer_critic_user_prompt = """Evaluate the explanation provided for the concept in the context of the original question.

```
{question}
```

"{concept}"

```
{explanation}
```"""

knowledge_explainer_reviser_system_prompt = """# ROLE: Expert Explanation Reviser for elementary education.

- Directly address all issues raised in the critique.
- Ensure the revised explanation is a clear, simple, stand-alone definition.
- Keep the explanation to 2-4 sentences.
- The explanation must start with the concept in bold (e.g., "**Gravity** is...").

Output ONLY the revised JSON object with the "explanation" key."""

knowledge_explainer_reviser_user_prompt = """Revise the explanation based on the critique.

"{concept}"

```
{explanation}
```

```json
{critique}
```"""

knowledge_reasoner_system_prompt = """# ROLE: Knowledge Relationship Analyst.

- Synthesize the provided explanations into a cohesive narrative (e.g., cause-and-effect, sequences).
- Base your analysis ONLY on the provided information.
- CRITICAL: Do NOT answer the original question directly. Your goal is to explain how the concepts relate to each other.

Respond with a JSON object: {"analysis": "Your detailed analysis..."}"""

knowledge_reasoner_user_prompt = """Analyze the relationships between the following concepts and explanations.

```
{question}
```

```
{all_explanations}
```"""

knowledge_reasoner_critic_system_prompt = """# ROLE: Reasoning Confidence Assessor.

- Evaluate the analysis based on the following criteria:
    1.  **Sufficiency**: Does it connect the concepts logically to fully address the question's foundation?
    2.  **Clarity**: Is the reasoning clear and easy for a student to follow?
    3.  **Scope**: Does it explain relationships *between* concepts without answering the original question directly?
- If all criteria pass, decide "ACCEPT".
- If any criterion fails, decide "REVISE" and provide specific feedback on logical gaps or unclear connections.

Respond with a JSON object: {"decision": "ACCEPT" or "REVISE", "feedback": "Your detailed feedback..."}"""

knowledge_reasoner_critic_user_prompt = """Evaluate the reasoning analysis below based on the original question.

```
{question}
```

```
{reasoning_analysis}
```"""

knowledge_reasoner_reviser_system_prompt = """# ROLE: Reasoning Refiner.

- Directly address all specific points in the critique.
- Strengthen the logical connections between concepts and fill any identified gaps.
- Ensure the analysis remains focused on explaining relationships, not giving the final answer.
- CRITICAL: NEVER answer the original question directly.

Output ONLY the revised JSON object with the "analysis" key."""

knowledge_reasoner_reviser_user_prompt = """Revise the analysis based on the provided critique.

```
{question}
```

```
{reasoning_analysis}
```

```json
{reasoning_critique}
```"""


class KnowledgePoints(BaseModel):
    """Structure for knowledge points list"""

    concepts: List[str] = Field(
        ...,
        min_length=1,
        max_length=15,
        description="Essential knowledge points to answer a question",
    )


class KnowledgeExplanation(BaseModel):
    """Structure for single knowledge point explanation"""

    explanation: str = Field(..., description="Clear explanation of a knowledge point")


class KnowledgeReasoning(BaseModel):
    """Structure for knowledge relationship analysis"""

    analysis: str = Field(
        ..., description="Analysis of relationships between knowledge points"
    )


class KnowledgeConceptsOutput(BaseModel):
    concepts: List[str]


class CriticOutput(BaseModel):
    """Pydantic model for the critic's output."""

    decision: Literal["ACCEPT", "REVISE"] = Field(
        description="The final decision. 'ACCEPT' if concepts are good, 'REVISE' if they need changes."
    )
    feedback: str = Field(
        description="Constructive feedback for revision if the decision is 'REVISE'. If 'ACCEPT', this can be a brief confirmation."
    )


class KnowledgeConceptsOutput(BaseModel):
    concepts: List[str]


class ExplanationOutput(BaseModel):
    """Pydantic model for a single concept explanation."""

    explanation: str = Field(
        description="A clear, concise definition of the concept, suitable for an elementary student."
    )


class CriticOutput(BaseModel):
    """Pydantic model for the explanation critic's output."""

    decision: Literal["ACCEPT", "REVISE"] = Field(
        description="The final decision. 'ACCEPT' if the explanation is good, 'REVISE' if it needs changes."
    )
    feedback: str = Field(
        description="Constructive feedback for revision if the decision is 'REVISE'. If 'ACCEPT', this can be a brief confirmation."
    )


class RevisedExplanationOutput(BaseModel):
    """Pydantic model for a revised concept explanation."""

    explanation: str = Field(
        description="The revised, improved definition of the concept, following all instructions."
    )


class ReasoningOutput(BaseModel):
    """Pydantic model for the reasoning analysis output."""

    analysis: str = Field(
        description="A detailed analysis of how the provided concepts relate to each other and the question."
    )


class ReasoningCriticOutput(BaseModel):
    """Pydantic model for the reasoning critic's output."""

    decision: Literal["ACCEPT", "REVISE"] = Field(
        description="The final decision. 'ACCEPT' if the analysis is sufficient, 'REVISE' if it needs changes."
    )
    feedback: str = Field(
        description="Constructive feedback for revision if the decision is 'REVISE'. If 'ACCEPT', this can be a brief confirmation."
    )


class ReasoningReviserOutput(BaseModel):
    """Pydantic model for the revised reasoning analysis."""

    analysis: str = Field(
        description="The revised and improved analysis, addressing all feedback from the critique."
    )


class GraphState(TypedDict):
    """State schema for the knowledge QA workflow"""

    question: str
    messages: Annotated[List[BaseMessage], add_messages]

    knowledge_points_json: Optional[str]
    knowledge_analysis_critique: Optional[str]
    knowledge_analysis_iterations: int

    knowledge_points: List[str]
    explanations: Dict[str, str]
    explanation_critiques: Dict[str, str]
    explanation_iterations: Dict[str, int]

    all_explanations: Optional[str]

    final_answer: Optional[str]


def extract_json(text: str) -> dict:
    """Extract JSON from text, handling markdown code blocks"""
    try:
        match = json_md_re.search(text)
        if match:
            json_str = match.group(1).strip()
        else:
            json_str = text
        return json.loads(json_str)
    except:
        return {}


def parse_concepts(output: Any) -> List[str]:
    """Parse concepts from various output formats"""
    if isinstance(output, KnowledgePoints):
        return output.concepts
    elif isinstance(output, dict):
        return output.get("concepts", [])
    elif isinstance(output, str):

        try:
            data = extract_json(output)
            if data and "concepts" in data:
                return data["concepts"]
        except:
            pass

        if "[" in output and "]" in output:
            try:

                start = output.find("[")
                end = output.find("]") + 1
                list_str = output[start:end]
                concepts = json.loads(list_str)
                return [str(c).strip() for c in concepts if c]
            except:
                pass

        if "," in output:
            return [
                p.strip().strip('"').strip("'") for p in output.split(",") if p.strip()
            ]

    return []


def extract_main_question(question_field: str) -> str:
    """
    Extracts the main question text from a formatted string using regex.
    """

    regex_pattern = r"Question(?: and Choices)?:\s*(.*?)\s*\n\nAnswer:"
    match = re.search(regex_pattern, question_field, re.DOTALL)

    if match:
        return match.group(1).strip()

    return question_field


def get_llm(llm_client: Any = None, temperature: float = 0.7, max_tokens: int = 2048):
    """Create LLM instance with custom endpoint"""
    if llm_client:
        return ChatOpenAI(
            model=llm_client.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=API_KEY,
            base_url=llm_client.base_url,
            timeout=30.0,
            max_retries=2,
        )
    else:
        return ChatOpenAI(
            model="Qwen2.5-7B-Instruct",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=API_KEY,
            base_url=BASE_URL,
            timeout=30.0,
            max_retries=2,
        )


def knowledge_analyzer_node(state: GraphState) -> GraphState:
    """Extract key concepts from the question"""
    llm = get_llm(LLM_CLIENT)

    prompt = SystemMessage(content=knowledge_analyzer_system_prompt)
    user_msg = HumanMessage(
        content=knowledge_analyzer_prompt.format(question=state["question"])
    )

    response = llm.with_structured_output(KnowledgeConceptsOutput).invoke(
        [prompt, user_msg]
    )

    content = response.model_dump_json()

    state["knowledge_points_json"] = content
    state["knowledge_analysis_iterations"] = (
        state.get("knowledge_analysis_iterations", 0) + 1
    )

    concepts = parse_concepts(content)
    state["knowledge_points"] = concepts

    return state


def knowledge_analyzer_critic_node(state: GraphState) -> GraphState:
    """Critiques the extracted knowledge points for relevance, completeness, and appropriateness."""
    llm = get_llm(LLM_CLIENT)

    prompt = SystemMessage(content=knowledge_analyzer_critic_system_prompt)
    user_msg = HumanMessage(
        content=knowledge_analyzer_critic_prompt.format(
            question=state["question"],
            knowledge_points_json=state["knowledge_points_json"],
        )
    )

    response = llm.with_structured_output(CriticOutput).invoke([prompt, user_msg])

    state["knowledge_analysis_critique"] = response.model_dump_json()

    return state


def knowledge_analyzer_reviser_node(state: GraphState) -> GraphState:
    """Revises the extracted knowledge concepts based on the critic's feedback."""
    llm = get_llm(LLM_CLIENT)

    prompt = SystemMessage(content=knowledge_analyzer_reviser_system_prompt)
    user_msg = HumanMessage(
        content=knowledge_analyzer_reviser_prompt.format(
            question=state["question"],
            knowledge_points_json=state["knowledge_points_json"],
            knowledge_analysis_critique=state["knowledge_analysis_critique"],
        )
    )

    response = llm.with_structured_output(KnowledgeConceptsOutput).invoke(
        [prompt, user_msg]
    )

    state["knowledge_points_json"] = response.model_dump_json()
    state["knowledge_points"] = response.concepts

    return state


def should_continue_analysis(state: GraphState) -> str:
    """Determine if analysis loop should continue"""
    critique = state.get("knowledge_analysis_critique", "")
    iterations = state.get("knowledge_analysis_iterations", 0)
    concepts = state.get("knowledge_points", [])

    if not concepts:
        if iterations >= 2:
            return "explain"
        return "revise"

    try:
        data = extract_json(critique)
        decision = data.get("decision", "REVISE")
    except:
        decision = "REVISE"

    if decision == "ACCEPT" or iterations >= 2:
        return "explain"
    return "revise"


def knowledge_explainer_node(state: GraphState) -> GraphState:
    """
    Explains each extracted knowledge concept in parallel using a structured output model.
    """
    llm = get_llm(LLM_CLIENT)

    concepts = state.get("knowledge_points", [])
    explanations = state.get("explanations", {})

    if not concepts:
        return state

    structured_llm = llm.with_structured_output(ExplanationOutput)

    def explain_concept(concept: str) -> tuple[str, str]:
        """
        Invokes the LLM to get a structured explanation for a single concept.
        """
        prompt = SystemMessage(content=knowledge_explainer_system_prompt)
        user_msg = HumanMessage(
            content=knowledge_explainer_user_prompt.format(concept=concept)
        )

        response = structured_llm.invoke([prompt, user_msg])
        return concept, response.explanation

    with ThreadPoolExecutor(max_workers=min(len(concepts), 5)) as executor:

        concepts_to_explain = [c for c in concepts if c not in explanations]
        results = list(executor.map(explain_concept, concepts_to_explain))

    for concept, explanation in results:
        explanations[concept] = explanation

    state["explanations"] = explanations
    return state


def knowledge_explainer_critic_node(state: GraphState) -> GraphState:
    """Critiques each explanation for clarity, correctness, and adherence to rules."""
    llm = get_llm(LLM_CLIENT)

    explanations = state.get("explanations", {})
    critiques: Dict[str, str] = state.get("explanation_critiques", {})
    iterations: Dict[str, int] = state.get("explanation_iterations", {})

    for concept, explanation in explanations.items():

        if concept in critiques:
            try:

                if json.loads(critiques[concept]).get("decision") == "ACCEPT":
                    continue
            except (json.JSONDecodeError, AttributeError):

                pass

        concept_iterations = iterations.get(concept, 0)
        if concept_iterations >= 2:

            critique_obj = CriticOutput(
                decision="ACCEPT", feedback="Max iterations reached."
            )
            critiques[concept] = critique_obj.model_dump_json()
            continue

        prompt = SystemMessage(content=knowledge_explainer_critic_system_prompt)
        user_msg = HumanMessage(
            content=knowledge_explainer_critic_user_prompt.format(
                question=state["question"],
                concept=concept,
                explanation=explanation,
            )
        )

        response = llm.with_structured_output(CriticOutput).invoke([prompt, user_msg])
        critiques[concept] = response.model_dump_json()

        iterations[concept] = concept_iterations + 1

    state["explanation_critiques"] = critiques
    state["explanation_iterations"] = iterations
    return state


def knowledge_explainer_reviser_node(state: GraphState) -> GraphState:
    """Revises a concept explanation based on a critic's feedback."""
    llm = get_llm(LLM_CLIENT)

    explanations = state.get("explanations", {})
    critiques = state.get("explanation_critiques", {})
    iterations = state.get("explanation_iterations", {})

    revised_explanations = explanations.copy()

    for concept, critique_json in critiques.items():

        try:
            critique_data = json.loads(critique_json)
            decision = critique_data.get("decision", "REVISE")
        except json.JSONDecodeError:
            decision = "REVISE"

        concept_iterations = iterations.get(concept, 0)
        if decision == "REVISE" and concept_iterations < 2:
            current_explanation = explanations.get(concept, "")
            if not current_explanation:
                continue

            prompt = SystemMessage(content=knowledge_explainer_reviser_system_prompt)
            user_msg = HumanMessage(
                content=knowledge_explainer_reviser_user_prompt.format(
                    concept=concept,
                    explanation=current_explanation,
                    critique=critique_json,
                )
            )

            response = llm.with_structured_output(RevisedExplanationOutput).invoke(
                [prompt, user_msg]
            )

            revised_explanations[concept] = response.explanation

    state["explanations"] = revised_explanations
    return state


def should_continue_explanations(state: GraphState) -> str:
    """Check if all explanations are accepted or max iterations reached"""
    critiques = state.get("explanation_critiques", {})
    iterations = state.get("explanation_iterations", {})

    if not critiques:
        return "revise"

    all_accepted = True
    for concept, critique in critiques.items():
        try:
            data = extract_json(critique)
            decision = data.get("decision", "REVISE")
        except:
            decision = "REVISE"

        concept_iterations = iterations.get(concept, 0)

        if decision != "ACCEPT" and concept_iterations < 2:
            all_accepted = False
            break

    if all_accepted:
        return "aggregate"
    else:
        return "revise"


def aggregate_explanations_node(state: GraphState) -> GraphState:
    """Aggregate all explanations into formatted text"""
    explanations = state.get("explanations", {})

    all_explanations_text = ""
    for concept, explanation in explanations.items():
        concept_formatted = " ".join(w.capitalize() for w in concept.split("_")).strip()
        all_explanations_text += f"- **{concept_formatted}**: {explanation}\n"

    state["all_explanations"] = all_explanations_text
    return state


def finalize_answer_node(state: GraphState) -> GraphState:
    """Compile the final answer"""
    all_explanations = state.get("all_explanations", "")

    final_answer = (
        "To understand your question, let's first break down the key concepts:\n\n"
        f"{all_explanations}\n\n"
    )

    state["final_answer"] = final_answer
    return state


def build_knowledge_qa_graph() -> CompiledStateGraph:
    """Build the LangGraph workflow"""
    workflow = StateGraph(GraphState)

    workflow.add_node("knowledge_analyzer", knowledge_analyzer_node)
    workflow.add_node("knowledge_analyzer_critic", knowledge_analyzer_critic_node)
    workflow.add_node("knowledge_analyzer_reviser", knowledge_analyzer_reviser_node)

    workflow.add_node("knowledge_explainer", knowledge_explainer_node)
    workflow.add_node("knowledge_explainer_critic", knowledge_explainer_critic_node)
    workflow.add_node("knowledge_explainer_reviser", knowledge_explainer_reviser_node)
    workflow.add_node("aggregate_explanations", aggregate_explanations_node)

    workflow.add_node("finalize_answer", finalize_answer_node)

    workflow.set_entry_point("knowledge_analyzer")

    workflow.add_edge("knowledge_analyzer", "knowledge_analyzer_critic")
    workflow.add_conditional_edges(
        "knowledge_analyzer_critic",
        should_continue_analysis,
        {"revise": "knowledge_analyzer_reviser", "explain": "knowledge_explainer"},
    )
    workflow.add_edge("knowledge_analyzer_reviser", "knowledge_analyzer")

    workflow.add_edge("knowledge_explainer", "knowledge_explainer_critic")
    workflow.add_conditional_edges(
        "knowledge_explainer_critic",
        should_continue_explanations,
        {
            "revise": "knowledge_explainer_reviser",
            "aggregate": "aggregate_explanations",
        },
    )
    workflow.add_edge("knowledge_explainer_reviser", "aggregate_explanations")

    workflow.add_edge("aggregate_explanations", "finalize_answer")

    workflow.add_edge("finalize_answer", END)

    return workflow.compile()


async def _run_knowledge_qa_langgraph_async(
    question: str, uuid: str, llm_client: Any
) -> str:
    """Run the Knowledge QA workflow asynchronously using LangGraph"""

    app = build_knowledge_qa_graph()

    initial_state = {
        "question": question,
        "messages": [],
        "knowledge_analysis_iterations": 0,
        "reasoning_iterations": 0,
        "explanation_iterations": {},
    }

    config = {
        "configurable": {"thread_id": uuid},
    }

    result = await app.ainvoke(initial_state, config=config)

    final_answer = result.get("final_answer", "")

    if not final_answer:
        final_answer = (
            "I couldn't identify any key concepts to explain for your question."
        )

    return final_answer


def knowledge_agent(
    question: str,
    true_answer: str,
    uuid: str,
    llm_client: Any,
    answer_extractor: Any = None,
) -> dict:
    """
    Knowledge QA agent using LangGraph.
    Maintains compatibility with existing system interface.
    """

    global LLM_CLIENT

    if LLM_CLIENT is None:
        LLM_CLIENT = llm_client

    system_prompt = None
    response_content = ""
    knowledge_points_context = ""

    MAX_ATTEMPTS = 3

    for attempt in range(MAX_ATTEMPTS):
        loop = None
        try:

            loop = asyncio.new_event_loop()

            question_text = extract_main_question(question)

            knowledge_points_context = loop.run_until_complete(
                _run_knowledge_qa_langgraph_async(
                    question_text, uuid.replace("-", "_"), llm_client
                )
            )

        except KeyboardInterrupt as e:
            logger.error(
                f"KnowledgeQA failed for uuid {uuid} with error: {e}, the traceback is:\n{traceback.format_exc()}"
            )
        finally:

            if loop is not None:
                try:

                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()

                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )

                    loop.run_until_complete(loop.shutdown_asyncgens())

                    loop.close()
                except Exception as cleanup_error:
                    logger.error(
                        f"Error during cleanup for uuid {uuid}: {cleanup_error}"
                    )

        prompt = f"Solve the following question step by step and place the final choice within \\boxed{{}}:\nContext:\n{knowledge_points_context}\n\n{question}"

        common_messages = [
            {"role": "user", "content": prompt},
        ]

        messages = construct_prompt(
            llm_client.model_name,
            common_prompts=common_messages,
            system_prompt=system_prompt,
        )

        try:
            llm_response = answer_extractor.llm_wrapper.call_api(
                messages=messages, max_tokens=2048, temperature=0.0, uuid=uuid
            )

            if llm_response:
                response_content = llm_response.choices[0].message.content.strip()

                match = boxed_re.findall(response_content)
                if (not match) and answer_extractor:
                    logger.debug(
                        f"Response for {uuid} lacks a boxed answer. Attempting extraction."
                    )

                    response_content = answer_extractor.extract_answer(
                        messages, response_content, uuid
                    )

                break
            else:
                logger.warning(
                    f"Attempt {attempt + 1}/{MAX_ATTEMPTS} for uuid {uuid} returned an invalid response format: {llm_response}"
                )

        except Exception as e:
            logger.warning(
                f"Attempt {attempt + 1}/{MAX_ATTEMPTS} for uuid {uuid} failed with error: {e}"
            )

    return {
        "solution_text": response_content,
        "knowledge_points": knowledge_points_context,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Knowledge QA LangGraph module loaded successfully.")

    question = "What is the capital of France?"
    true_answer = "Paris"
    uuid = "123e4567-e89b-12d3-a456-426"

    llm_client = LLMWrapper(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name="Qwen2.5-7B-Instruct",
        model_type="qwen",
        max_concurrent_requests=100,
        requests_per_minute=100,
        requests_per_second=100,
    )

    extract_llm_client = LLMWrapper(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name="Qwen2.5-7B-Instruct",
        model_type="qwen",
        max_concurrent_requests=100,
        requests_per_minute=100,
        requests_per_second=100,
    )

    answer_extractor = BoxedAnswerExtractor(extract_llm_client)

    result = knowledge_agent(question, true_answer, uuid, llm_client, answer_extractor)
    print("Result:", result)
