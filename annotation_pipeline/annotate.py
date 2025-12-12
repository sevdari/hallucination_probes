# %%
#!/usr/bin/env python3
"""
Script to annotate completions using OpenAI's new Responses API with integrated search.
"""

import os
import logging
import asyncio
from typing import List, Any

from openai import AsyncOpenAI

from utils.parsing import parse_and_validate_json
from utils.string_utils import try_matching_span_in_text
from annotation_pipeline.data_models import AnnotatedSpan

logger = logging.getLogger(__name__)

# fallback to default prompt if env var not set
ENTITY_ANNOTATION_PROMPT_TEMPLATE = os.getenv("ENTITY_ANNOTATION_PROMPT_TEMPLATE")
if ENTITY_ANNOTATION_PROMPT_TEMPLATE is None:
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
    inference_api: Any = None, # kept for signature
    annotation_prompt: str = ENTITY_ANNOTATION_PROMPT_TEMPLATE,
    temperature: float = 0.0, # kept for signature
    max_tokens: int = 4096, # kept for signature
    model_id: str = "gpt-4o", # Responses API works with standard model IDs
    max_searches: int = 5,
) -> List[AnnotatedSpan]:
    """
    Annotate spans using OpenAI's Responses API (Server-side Search).
    """
    
    client = AsyncOpenAI()

    # Prepare input content
    user_content = format_prompt(instruction, completion, annotation_prompt)

    try:
        # NOTE: 'input' replaces 'messages', 'tools' uses 'web_search_preview'
        response = await client.responses.create(
            model=model_id,
            input=user_content,
            tools=[{
                "type": "web_search",
                "user_location": {
                    "type": "approximate",
                    "country": "Switzerland"
                }
            }]
        )
        
        # Responses API returns the final answer in 'output_text'
        final_content = response.output_text
        
        logger.info(f"Received response from model (Length: {len(final_content)} chars)")
        
        # Parse JSON
        annotated_spans = parse_and_validate_json(
            final_content, 
            List[AnnotatedSpan],
            allow_partial=True
        )
        
        if annotated_spans:
            annotated_spans = assign_span_positions(annotated_spans, completion)
            logger.info(f"Successfully annotated {len(annotated_spans)} spans")
            return annotated_spans
        
        return []

    except Exception as e:
        logger.error(f"Error during OpenAI Responses API annotation: {e}")
        raise
