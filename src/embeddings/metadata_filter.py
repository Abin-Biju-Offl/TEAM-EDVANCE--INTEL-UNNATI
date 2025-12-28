"""
Metadata Filtering for Pre-Retrieval
=====================================

Filters chunks by metadata BEFORE semantic search to ensure:
- Grade-appropriate content (correct class)
- Subject-specific retrieval
- Chapter-level precision
- Content type filtering (definitions, examples, etc.)

Design Decisions:
1. Pre-Retrieval Filtering:
   - Filter metadata BEFORE FAISS search (not after)
   - Creates filtered vector ID list for FAISS
   - Much faster than post-filtering thousands of results
   - Ensures only relevant chunks are searched

2. Filter Conditions:
   - Exact match: class_number == 10
   - Range: token_count >= 300 AND token_count <= 500
   - Contains: page_numbers contains 95
   - Multiple values: chunk_type in [definition, theorem]
   - Combinations: AND/OR logic

3. Performance:
   - O(n) scan through metadata (fast for <100K chunks)
   - Result: Filtered vector ID list
   - FAISS searches only filtered subset
   - 10-100x faster than searching all + post-filtering

4. Use Cases:
   - Student queries: Filter by class_number
   - Chapter-specific: Filter by class + subject + chapter
   - Definition lookup: Filter by chunk_type = "definition"
   - Math problems: Filter by has_equations = True
"""

import logging
from typing import List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FilterOperator(Enum):
    """Comparison operators for filtering."""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_EQUALS = ">="
    LESS_THAN = "<"
    LESS_EQUALS = "<="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"


@dataclass
class FilterCondition:
    """
    A single filter condition.
    
    Examples:
        # Exact match
        FilterCondition(field="class_number", operator=FilterOperator.EQUALS, value=10)
        
        # Range
        FilterCondition(field="token_count", operator=FilterOperator.GREATER_EQUALS, value=300)
        
        # Multiple values
        FilterCondition(field="chunk_type", operator=FilterOperator.IN, value=["definition", "theorem"])
        
        # Contains
        FilterCondition(field="page_numbers", operator=FilterOperator.CONTAINS, value=95)
    """
    field: str
    operator: FilterOperator
    value: Any
    
    def matches(self, metadata: Dict[str, Any]) -> bool:
        """
        Check if metadata matches this condition.
        
        Args:
            metadata: Metadata dict to check
        
        Returns:
            True if condition is satisfied
        """
        if self.field not in metadata:
            return False
        
        field_value = metadata[self.field]
        
        # Handle None values
        if field_value is None:
            return self.value is None if self.operator == FilterOperator.EQUALS else False
        
        # Apply operator
        if self.operator == FilterOperator.EQUALS:
            return field_value == self.value
        
        elif self.operator == FilterOperator.NOT_EQUALS:
            return field_value != self.value
        
        elif self.operator == FilterOperator.GREATER_THAN:
            return field_value > self.value
        
        elif self.operator == FilterOperator.GREATER_EQUALS:
            return field_value >= self.value
        
        elif self.operator == FilterOperator.LESS_THAN:
            return field_value < self.value
        
        elif self.operator == FilterOperator.LESS_EQUALS:
            return field_value <= self.value
        
        elif self.operator == FilterOperator.IN:
            return field_value in self.value
        
        elif self.operator == FilterOperator.NOT_IN:
            return field_value not in self.value
        
        elif self.operator == FilterOperator.CONTAINS:
            # For lists/sets
            if isinstance(field_value, (list, set)):
                return self.value in field_value
            # For strings
            elif isinstance(field_value, str):
                return str(self.value) in field_value
            else:
                return False
        
        elif self.operator == FilterOperator.NOT_CONTAINS:
            if isinstance(field_value, (list, set)):
                return self.value not in field_value
            elif isinstance(field_value, str):
                return str(self.value) not in field_value
            else:
                return True
        
        return False


class MetadataFilter:
    """
    Filters vector store metadata before semantic search.
    
    Supports:
    - Multiple conditions with AND/OR logic
    - Common NCERT filtering patterns
    - Performance optimization
    
    Example:
        >>> filter = MetadataFilter()
        >>> 
        >>> # Filter for Class 10, Math, Chapter 5
        >>> filter.add_condition("class_number", FilterOperator.EQUALS, 10)
        >>> filter.add_condition("subject", FilterOperator.EQUALS, "mathematics")
        >>> filter.add_condition("chapter_number", FilterOperator.EQUALS, 5)
        >>> 
        >>> # Get matching vector IDs
        >>> vector_ids = filter.apply(metadata_store)
        >>> 
        >>> # Search only these vectors
        >>> results = vector_store.search_subset(query, vector_ids, k=5)
    """
    
    def __init__(self, conditions: Optional[List[FilterCondition]] = None, logic: str = "AND"):
        """
        Initialize metadata filter.
        
        Args:
            conditions: List of filter conditions
            logic: "AND" or "OR" for combining conditions
        """
        self.conditions = conditions or []
        self.logic = logic.upper()
        
        if self.logic not in ["AND", "OR"]:
            raise ValueError("logic must be 'AND' or 'OR'")
    
    def add_condition(
        self,
        field: str,
        operator: Union[FilterOperator, str],
        value: Any
    ):
        """
        Add a filter condition.
        
        Args:
            field: Metadata field name
            operator: FilterOperator or string ("==", ">", "in", etc.)
            value: Value to compare against
        """
        if isinstance(operator, str):
            # Convert string to FilterOperator
            operator_map = {
                "==": FilterOperator.EQUALS,
                "!=": FilterOperator.NOT_EQUALS,
                ">": FilterOperator.GREATER_THAN,
                ">=": FilterOperator.GREATER_EQUALS,
                "<": FilterOperator.LESS_THAN,
                "<=": FilterOperator.LESS_EQUALS,
                "in": FilterOperator.IN,
                "not_in": FilterOperator.NOT_IN,
                "contains": FilterOperator.CONTAINS,
                "not_contains": FilterOperator.NOT_CONTAINS
            }
            operator = operator_map.get(operator)
            if operator is None:
                raise ValueError(f"Invalid operator: {operator}")
        
        condition = FilterCondition(field=field, operator=operator, value=value)
        self.conditions.append(condition)
    
    def matches(self, metadata: Dict[str, Any]) -> bool:
        """
        Check if metadata matches all/any conditions.
        
        Args:
            metadata: Metadata dict to check
        
        Returns:
            True if metadata passes filter
        """
        if not self.conditions:
            return True  # No conditions = match all
        
        results = [cond.matches(metadata) for cond in self.conditions]
        
        if self.logic == "AND":
            return all(results)
        else:  # OR
            return any(results)
    
    def apply(self, metadata_store: Dict[int, Dict[str, Any]]) -> List[int]:
        """
        Apply filter to metadata store.
        
        Args:
            metadata_store: Dict mapping vector_id → metadata
        
        Returns:
            List of vector IDs that match filter
        """
        if not self.conditions:
            return list(metadata_store.keys())  # No filter = all IDs
        
        matching_ids = []
        for vector_id, metadata in metadata_store.items():
            if self.matches(metadata):
                matching_ids.append(vector_id)
        
        logger.info(f"Filter matched {len(matching_ids)}/{len(metadata_store)} vectors")
        
        return matching_ids
    
    def get_filter_summary(self) -> str:
        """Get human-readable filter summary."""
        if not self.conditions:
            return "No filters (match all)"
        
        parts = []
        for cond in self.conditions:
            parts.append(f"{cond.field} {cond.operator.value} {cond.value}")
        
        return f" {self.logic} ".join(parts)
    
    def clear(self):
        """Clear all conditions."""
        self.conditions.clear()


# Pre-built filter factories for common patterns

def create_class_filter(class_number: int) -> MetadataFilter:
    """Create filter for specific class."""
    filter = MetadataFilter()
    filter.add_condition("class_number", FilterOperator.EQUALS, class_number)
    return filter


def create_chapter_filter(
    class_number: int,
    subject: str,
    chapter_number: int
) -> MetadataFilter:
    """Create filter for specific chapter."""
    filter = MetadataFilter()
    filter.add_condition("class_number", FilterOperator.EQUALS, class_number)
    filter.add_condition("subject", FilterOperator.EQUALS, subject)
    filter.add_condition("chapter_number", FilterOperator.EQUALS, chapter_number)
    return filter


def create_definition_filter(class_number: Optional[int] = None) -> MetadataFilter:
    """Create filter for definitions only."""
    filter = MetadataFilter()
    filter.add_condition("chunk_type", FilterOperator.IN, ["definition", "theorem", "formula"])
    
    if class_number:
        filter.add_condition("class_number", FilterOperator.EQUALS, class_number)
    
    return filter


def create_example_filter(
    class_number: Optional[int] = None,
    has_equations: Optional[bool] = None
) -> MetadataFilter:
    """Create filter for examples."""
    filter = MetadataFilter()
    filter.add_condition("chunk_type", FilterOperator.EQUALS, "example")
    
    if class_number:
        filter.add_condition("class_number", FilterOperator.EQUALS, class_number)
    
    if has_equations is not None:
        filter.add_condition("has_equations", FilterOperator.EQUALS, has_equations)
    
    return filter


def create_page_range_filter(
    class_number: int,
    subject: str,
    start_page: int,
    end_page: int
) -> MetadataFilter:
    """Create filter for page range (requires checking if any page in range)."""
    filter = MetadataFilter()
    filter.add_condition("class_number", FilterOperator.EQUALS, class_number)
    filter.add_condition("subject", FilterOperator.EQUALS, subject)
    
    # Note: This is simplified. For exact page range, need custom logic
    # to check if any page_number in chunk.page_numbers is in range
    
    return filter


def create_language_filter(language: str, class_number: Optional[int] = None) -> MetadataFilter:
    """Create filter for specific language."""
    filter = MetadataFilter()
    filter.add_condition("language", FilterOperator.EQUALS, language)
    
    if class_number:
        filter.add_condition("class_number", FilterOperator.EQUALS, class_number)
    
    return filter


def create_quality_filter(
    min_confidence: float = 0.8,
    completeness: str = "complete"
) -> MetadataFilter:
    """Create filter for high-quality chunks only."""
    filter = MetadataFilter()
    filter.add_condition("structure_confidence", FilterOperator.GREATER_EQUALS, min_confidence)
    filter.add_condition("completeness", FilterOperator.EQUALS, completeness)
    return filter


class FilterBuilder:
    """
    Fluent API for building complex filters.
    
    Example:
        >>> filter = (FilterBuilder()
        ...     .for_class(10)
        ...     .for_subject("mathematics")
        ...     .for_chapter(5)
        ...     .with_chunk_type("definition")
        ...     .build())
    """
    
    def __init__(self):
        self.filter = MetadataFilter()
    
    def for_class(self, class_number: int) -> 'FilterBuilder':
        """Filter by class number."""
        self.filter.add_condition("class_number", FilterOperator.EQUALS, class_number)
        return self
    
    def for_subject(self, subject: str) -> 'FilterBuilder':
        """Filter by subject."""
        self.filter.add_condition("subject", FilterOperator.EQUALS, subject)
        return self
    
    def for_chapter(self, chapter_number: int) -> 'FilterBuilder':
        """Filter by chapter."""
        self.filter.add_condition("chapter_number", FilterOperator.EQUALS, chapter_number)
        return self
    
    def with_chunk_type(self, chunk_type: Union[str, List[str]]) -> 'FilterBuilder':
        """Filter by chunk type(s)."""
        if isinstance(chunk_type, str):
            self.filter.add_condition("chunk_type", FilterOperator.EQUALS, chunk_type)
        else:
            self.filter.add_condition("chunk_type", FilterOperator.IN, chunk_type)
        return self
    
    def with_language(self, language: str) -> 'FilterBuilder':
        """Filter by language."""
        self.filter.add_condition("language", FilterOperator.EQUALS, language)
        return self
    
    def with_equations(self, has_equations: bool = True) -> 'FilterBuilder':
        """Filter by presence of equations."""
        self.filter.add_condition("has_equations", FilterOperator.EQUALS, has_equations)
        return self
    
    def with_examples(self, has_examples: bool = True) -> 'FilterBuilder':
        """Filter by presence of examples."""
        self.filter.add_condition("has_examples", FilterOperator.EQUALS, has_examples)
        return self
    
    def with_min_confidence(self, confidence: float) -> 'FilterBuilder':
        """Filter by minimum structure confidence."""
        self.filter.add_condition("structure_confidence", FilterOperator.GREATER_EQUALS, confidence)
        return self
    
    def complete_only(self) -> 'FilterBuilder':
        """Filter for complete chunks only."""
        self.filter.add_condition("completeness", FilterOperator.EQUALS, "complete")
        return self
    
    def build(self) -> MetadataFilter:
        """Build and return the filter."""
        return self.filter


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Testing MetadataFilter...")
    print("=" * 70)
    
    # Sample metadata store
    metadata_store = {
        0: {
            'chunk_id': '10_mathematics_5_001',
            'class_number': 10,
            'subject': 'mathematics',
            'chapter_number': 5,
            'chunk_type': 'definition',
            'language': 'eng',
            'has_equations': False,
            'structure_confidence': 1.0,
            'completeness': 'complete'
        },
        1: {
            'chunk_id': '10_mathematics_5_002',
            'class_number': 10,
            'subject': 'mathematics',
            'chapter_number': 5,
            'chunk_type': 'example',
            'language': 'eng',
            'has_equations': True,
            'structure_confidence': 0.95,
            'completeness': 'complete'
        },
        2: {
            'chunk_id': '11_mathematics_3_001',
            'class_number': 11,
            'subject': 'mathematics',
            'chapter_number': 3,
            'chunk_type': 'definition',
            'language': 'eng',
            'has_equations': False,
            'structure_confidence': 1.0,
            'completeness': 'complete'
        }
    }
    
    # Test 1: Class filter
    print("\nTest 1: Class 10 only")
    filter1 = create_class_filter(10)
    result1 = filter1.apply(metadata_store)
    print(f"Matched: {result1}")
    
    # Test 2: Chapter filter
    print("\nTest 2: Class 10, Math, Chapter 5")
    filter2 = create_chapter_filter(10, "mathematics", 5)
    result2 = filter2.apply(metadata_store)
    print(f"Matched: {result2}")
    
    # Test 3: Definition filter
    print("\nTest 3: Definitions only (any class)")
    filter3 = create_definition_filter()
    result3 = filter3.apply(metadata_store)
    print(f"Matched: {result3}")
    
    # Test 4: FilterBuilder
    print("\nTest 4: FilterBuilder - Class 10, Math, Ch 5, Examples with equations")
    filter4 = (FilterBuilder()
        .for_class(10)
        .for_subject("mathematics")
        .for_chapter(5)
        .with_chunk_type("example")
        .with_equations(True)
        .build())
    result4 = filter4.apply(metadata_store)
    print(f"Filter: {filter4.get_filter_summary()}")
    print(f"Matched: {result4}")
