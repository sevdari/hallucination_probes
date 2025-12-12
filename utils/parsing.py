"""Utilities for parsing and validating JSON from LLM responses."""

import re
import logging
from typing import Any, List, Type, TypeVar, Union

from pydantic import BaseModel, TypeAdapter
from pydantic_core import from_json

from utils.string_utils import normalize_text

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

def parse_and_validate_json(
    llm_response: str, 
    schema: Union[Type[BaseModel], Any], 
    allow_partial: bool = False
) -> Any:
    """
    Parse and validate JSON from an LLM response with robust cleanup.
    
    This function:
    1. Normalizes the text to handle encoding issues
    2. Removes markdown code fences
    3. Extracts the JSON structure
    4. Parses and validates against the provided schema
    
    Args:
        llm_response: Raw LLM response that may contain JSON with surrounding text
        schema: Pydantic model or type hint for validation (e.g., MyModel or List[MyModel])
        allow_partial: Whether to allow partial JSON parsing (useful for truncated responses)
        
    Returns:
        Parsed and validated object(s) according to the schema
        
    Raises:
        ValueError: If no valid JSON found or validation fails
    """
    # Normalize text to handle control characters
    cleaned_response = normalize_text(llm_response)
    
    # Remove markdown code fences (``````)
    # Using specific split logic is often safer than regex substitution for multi-line blocks
    if "```" in cleaned_response:
        parts = cleaned_response.split("```")
        # Usually the code block is the second part (index 1) if it's wrapped
        # e.g., "Here is JSON:\n``````" -> ["Here...", "json\n[...]\n", ""]
        # We look for the part that actually looks like JSON
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("[") or part.startswith("{"):
                cleaned_response = part
                break
    
    cleaned_response = cleaned_response.strip()

    # IMPROVED JSON EXTRACTION
    # We look specifically for the start of a JSON array containing objects: `[ {` or `[{`
    # This avoids matching citations like `[1]` or `[source.com]`
    
    # 1. Try to find start of a JSON List of Objects (most common for this pipeline)
    json_list_start = re.search(r'\[\s*\{', cleaned_response)
    
    # 2. Try to find start of a JSON Object
    json_obj_start = re.search(r'^\s*\{', cleaned_response)
    
    start_index = -1
    
    if json_list_start:
        start_index = json_list_start.start()
    elif json_obj_start:
        start_index = json_obj_start.start()
    elif cleaned_response.startswith("["):
        # Fallback: starts with [ but maybe not immediately followed by {
        # Check if it looks like a citation (short content) vs array
        if not re.match(r'^\[[\w\.\-]+\]', cleaned_response): 
             start_index = 0

    if start_index != -1:
        # Slice from the detected start
        candidate = cleaned_response[start_index:]
        
        # Find the last valid closing bracket ']' or '}'
        # This helps ignore trailing text/citations after the JSON
        last_square = candidate.rfind(']')
        last_curly = candidate.rfind('}')
        
        end_index = max(last_square, last_curly)
        
        if end_index != -1:
            json_str = candidate[:end_index+1]
        else:
            json_str = candidate # Try parsing what we have
            
    else:
        # Fallback to original greedy search if heuristic fails
        # This catches cases where it might be a simple list [1, 2] not objects
        json_match = re.search(r'(\{.*\}|\[.*\])', cleaned_response, flags=re.DOTALL)
        if not json_match:
             raise ValueError(
                f"No valid JSON object or array found in response: {llm_response[:200]}..."
            )
        json_str = json_match.group(0).strip()

    try:
        # Parse JSON
        parsed = from_json(json_str, allow_partial=allow_partial)
        
        # Validate against schema
        # Note: parse_obj_as is deprecated in Pydantic V2, using TypeAdapter if available
        try:
            validator = TypeAdapter(schema)
            validated_data = validator.validate_python(parsed)
        except ImportError:
            # Fallback for Pydantic V1
            from pydantic import parse_obj_as
            validated_data = parse_obj_as(schema, parsed)
        except NameError:
             # Fallback if TypeAdapter not imported/available
             from pydantic import parse_obj_as
             validated_data = parse_obj_as(schema, parsed)
        
        return validated_data
        
    except Exception as e:
        raise ValueError(
            f"Error parsing/validating JSON: {e}\n"
            f"Extracted string: {json_str[:200]}..."
        ) from e


def validate_dicts_to_pydantic(
    dicts: List[dict],
    model: Type[T],
    skip_invalid: bool = False
) -> List[T]:
    """
    Validate a list of dictionaries against a Pydantic model.
    
    Args:
        dicts: List of dictionaries to validate
        model: Pydantic model class to validate against
        skip_invalid: If True, skip invalid items instead of raising an error
        
    Returns:
        List of validated Pydantic model instances
        
    Raises:
        ValueError: If skip_invalid is False and validation fails for any item
    """
    validated = []
    
    for i, item_dict in enumerate(dicts):
        try:
            # Pydantic V2 uses model_validate, V1 uses parse_obj
            if hasattr(model, 'model_validate'):
                validated_item = model.model_validate(item_dict)
            else:
                validated_item = model.parse_obj(item_dict)
                
            validated.append(validated_item)
        except Exception as e:
            if skip_invalid:
                # Silently skip invalid items
                continue
            else:
                raise ValueError(
                    f"Validation failed for item {i}: {e}"
                ) from e
    
    return validated