# %%
#!/usr/bin/env python3
"""
Script to annotate completions using OpenAI's built-in server-side web search.
"""

import os
import logging
from typing import List, Optional

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt, LLMResponse

from utils.parsing import parse_and_validate_json
from utils.string_utils import try_matching_span_in_text
from annotation_pipeline.data_models import AnnotatedSpan

logger = logging.getLogger(__name__)

# %%
# check entity annotation prompt
ENTITY_ANNOTATION_PROMPT_TEMPLATE = os.getenv("ENTITY_ANNOTATION_PROMPT_TEMPLATE")
if ENTITY_ANNOTATION_PROMPT_TEMPLATE is None:
    # Minimal fallback template
    ENTITY_ANNOTATION_PROMPT_TEMPLATE = """
    Instruction: {instruction}
    Completion: {completion}
    
    Verify the factual claims in the completion using your web search tool.
    Return a JSON list of AnnotatedSpan objects.
    """
else:
    if os.path.exists(ENTITY_ANNOTATION_PROMPT_TEMPLATE):
        ENTITY_ANNOTATION_PROMPT_TEMPLATE = open(ENTITY_ANNOTATION_PROMPT_TEMPLATE.strip()).read().strip()

# %%

def format_prompt(instruction: str, completion: str, prompt_template: str) -> str:
    """Format the user prompt with the text to analyze"""
    return prompt_template.replace(
        "{instruction}", instruction
    ).replace(
        "{completion}", completion
    )

def assign_span_positions(spans: List[AnnotatedSpan], text: str, min_similarity: float = 0.8) -> List[AnnotatedSpan]:
    """
    Assign positions to spans and convert to the expected format.
    """
    results = []
    cur_idx = 0
    used_positions = set()

    for span in spans:
        closest_match, matched_idx = try_matching_span_in_text(
            span.span,
            text,
            cur_idx=cur_idx,
            min_similarity=min_similarity
        )

        if closest_match is None:
            logger.warning(f"Could not locate span {repr(span.span)} in text.")
            span.label = None
            span.index = None
            results.append(span)
            continue

        if matched_idx is not None and all(pos in used_positions for pos in range(matched_idx, matched_idx+len(closest_match))):
            logger.warning(f"Span {repr(span.span)} matched at same position as already-matched span")
            continue

        span.index = matched_idx
        span.span = closest_match

        results.append(span)
        used_positions.update(range(matched_idx, matched_idx+len(closest_match)))
        cur_idx = max(cur_idx, matched_idx + len(closest_match))
    
    return results

async def annotate_completion(
    instruction: str,
    completion: str,
    inference_api: InferenceAPI,
    annotation_prompt: str = ENTITY_ANNOTATION_PROMPT_TEMPLATE,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    # Note: Use a model capable of server-side search
    model_id: str = "gpt-4o-search-preview", 
    max_searches: Optional[int] = None,
) -> List[AnnotatedSpan]:
    """
    Annotate spans in the provided text completion using OpenAI's native web search.
    """
    try:
        user_prompt: str = format_prompt(
            instruction, completion, annotation_prompt
        )
        
        # Define the NATIVE OpenAI web search tool
        # This tells the API to use its internal browsing capability
        tools = [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for facts.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        }]

        # Single call: The model handles the search/browse loop on the server side
        response_list: List[LLMResponse] = await inference_api(
            model_id=model_id,
            prompt=Prompt(messages=[ChatMessage(role=MessageRole.user, content=user_prompt)]),
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools
        )

        response_text: str = response_list[0].completion
        
        # In server-side search, the model might include citation markers (e.g., [1]).
        # You might need to clean them or ensure your parser handles them.
        logger.info(f"Model response with search: {response_text[:100]}...")

        annotated_spans = parse_and_validate_json(
            response_text, 
            List[AnnotatedSpan],
            allow_partial=True,
        )
        
        annotated_spans = assign_span_positions(annotated_spans, completion)
        
        logger.info(f"Successfully annotated {len(annotated_spans)} spans")

        return annotated_spans
        
    except Exception as e:
        logger.error(f"Error during span annotation: {e}")
        raise
