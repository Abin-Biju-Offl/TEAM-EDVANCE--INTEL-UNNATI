"""
Confidence Scoring and Hallucination Detection
===============================================

Estimates confidence in retrieval results and detects low-confidence cases
that may lead to hallucinations.

Confidence Signals:
1. Similarity Score: Cosine similarity from bi-encoder
   - High (>0.8): Strong semantic match
   - Medium (0.6-0.8): Reasonable match
   - Low (<0.6): Weak match → hallucination risk

2. Rerank Score: Cross-encoder relevance score
   - High (>0.8): Very relevant
   - Medium (0.6-0.8): Moderately relevant
   - Low (<0.6): Not relevant → reject

3. Score Gap: Difference between top-1 and top-2
   - Large gap (>0.2): Clear winner
   - Small gap (<0.1): Ambiguous → lower confidence

4. Metadata Consistency: Query intent matches chunk type
   - Query "What is..." + chunk type "definition" → high confidence
   - Query "Example of..." + chunk type "exercise" → medium confidence
   - Mismatch → lower confidence

5. Content Quality: Chunk completeness and structure confidence
   - completeness="complete" + structure_confidence>0.9 → bonus
   - completeness="partial" → penalty

Thresholds (Empirically Tuned):
- High confidence: ≥0.8 (safe to answer)
- Medium confidence: 0.6-0.8 (answer with disclaimer)
- Low confidence: <0.6 (reject or ask for clarification)

Hallucination Detection:
- Low similarity + Low rerank → REJECT
- Ambiguous top results (small gap) → "I'm not certain..."
- Metadata mismatch → "The closest match is..."
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceThresholds:
    """Confidence score thresholds."""
    
    # Overall confidence levels
    high_confidence: float = 0.8
    medium_confidence: float = 0.6
    low_confidence: float = 0.4
    
    # Individual signal thresholds
    min_similarity: float = 0.5      # Below this → reject
    min_rerank: float = 0.5          # Below this → reject
    ambiguous_gap: float = 0.1       # Small gap → uncertainty
    clear_winner_gap: float = 0.2    # Large gap → confidence
    
    # Quality thresholds
    min_structure_confidence: float = 0.7
    
    def to_dict(self) -> dict:
        return {
            'high_confidence': self.high_confidence,
            'medium_confidence': self.medium_confidence,
            'low_confidence': self.low_confidence,
            'min_similarity': self.min_similarity,
            'min_rerank': self.min_rerank,
            'ambiguous_gap': self.ambiguous_gap,
            'clear_winner_gap': self.clear_winner_gap,
            'min_structure_confidence': self.min_structure_confidence
        }


class ConfidenceScorer:
    """
    Computes confidence scores and detects hallucination risks.
    
    Usage:
        >>> scorer = ConfidenceScorer()
        >>> 
        >>> confidence = scorer.compute_confidence(
        ...     query="What is an arithmetic progression?",
        ...     retrieved_content="Definition: An arithmetic progression is...",
        ...     similarity_score=0.85,
        ...     rerank_score=0.92,
        ...     metadata={'chunk_type': 'definition', 'completeness': 'complete'}
        ... )
        >>> 
        >>> print(f"Confidence: {confidence:.2f}")
        >>> 
        >>> if confidence < 0.6:
        ...     print("⚠ Low confidence - may hallucinate")
    """
    
    def __init__(self, thresholds: Optional[ConfidenceThresholds] = None):
        """
        Initialize confidence scorer.
        
        Args:
            thresholds: Custom thresholds (default: standard thresholds)
        """
        self.thresholds = thresholds or ConfidenceThresholds()
        logger.info("ConfidenceScorer initialized")
    
    def compute_confidence(
        self,
        query: str,
        retrieved_content: str,
        similarity_score: float,
        rerank_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        score_gap: Optional[float] = None
    ) -> float:
        """
        Compute overall confidence score (0-1).
        
        Args:
            query: User query
            retrieved_content: Retrieved chunk content
            similarity_score: Cosine similarity from bi-encoder
            rerank_score: Cross-encoder score (if available)
            metadata: Chunk metadata dict
            score_gap: Gap between top-1 and top-2 results
        
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from scores
        confidence = self._score_based_confidence(similarity_score, rerank_score)
        
        # Adjust for score gap (if available)
        if score_gap is not None:
            confidence = self._adjust_for_gap(confidence, score_gap)
        
        # Adjust for query-chunk alignment
        if metadata:
            confidence = self._adjust_for_metadata(confidence, query, metadata)
        
        # Adjust for content quality
        if metadata:
            confidence = self._adjust_for_quality(confidence, metadata)
        
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def _score_based_confidence(
        self,
        similarity_score: float,
        rerank_score: Optional[float]
    ) -> float:
        """
        Base confidence from similarity and rerank scores.
        
        Strategy:
        - If both available: 40% similarity + 60% rerank (trust rerank more)
        - If only similarity: Use similarity directly
        """
        if rerank_score is not None:
            # Weighted combination (favor rerank)
            confidence = 0.4 * similarity_score + 0.6 * rerank_score
        else:
            confidence = similarity_score
        
        return confidence
    
    def _adjust_for_gap(self, confidence: float, gap: float) -> float:
        """
        Adjust confidence based on score gap.
        
        Large gap → Clear winner → Boost confidence
        Small gap → Ambiguous → Reduce confidence
        """
        if gap >= self.thresholds.clear_winner_gap:
            # Clear winner: boost confidence
            boost = min(0.1, gap * 0.5)
            return confidence + boost
        
        elif gap <= self.thresholds.ambiguous_gap:
            # Ambiguous: reduce confidence
            penalty = 0.15
            return confidence - penalty
        
        else:
            # Normal gap: no adjustment
            return confidence
    
    def _adjust_for_metadata(
        self,
        confidence: float,
        query: str,
        metadata: Dict[str, Any]
    ) -> float:
        """
        Adjust confidence based on query-chunk type alignment.
        
        Examples:
        - Query "What is..." + chunk "definition" → boost
        - Query "Example of..." + chunk "example" → boost
        - Query "Solve..." + chunk "note" → penalty
        """
        chunk_type = metadata.get('chunk_type', '')
        query_lower = query.lower()
        
        # Detect query intent
        intent = self._detect_query_intent(query_lower)
        
        # Check alignment
        if intent and chunk_type:
            if self._is_aligned(intent, chunk_type):
                # Good match: boost confidence
                return confidence + 0.05
            else:
                # Mismatch: penalty
                return confidence - 0.1
        
        return confidence
    
    def _detect_query_intent(self, query: str) -> Optional[str]:
        """
        Detect query intent from query text.
        
        Returns:
            Intent type: "definition", "example", "explanation", "proof", etc.
        """
        # Definition queries
        if re.search(r'\b(what is|define|definition of|meaning of)\b', query):
            return 'definition'
        
        # Example queries
        if re.search(r'\b(example|show me|demonstrate|sample)\b', query):
            return 'example'
        
        # Explanation queries
        if re.search(r'\b(how|why|explain|understand)\b', query):
            return 'explanation'
        
        # Problem-solving queries
        if re.search(r'\b(solve|calculate|find|compute)\b', query):
            return 'problem_solving'
        
        # Proof queries
        if re.search(r'\b(prove|proof|derive|derivation)\b', query):
            return 'proof'
        
        return None
    
    def _is_aligned(self, intent: str, chunk_type: str) -> bool:
        """Check if query intent aligns with chunk type."""
        
        # Define alignment rules
        alignments = {
            'definition': ['definition', 'theorem', 'formula'],
            'example': ['example', 'solution'],
            'explanation': ['explanation', 'note', 'context'],
            'problem_solving': ['example', 'solution', 'exercise'],
            'proof': ['proof', 'theorem']
        }
        
        expected_types = alignments.get(intent, [])
        return chunk_type in expected_types
    
    def _adjust_for_quality(
        self,
        confidence: float,
        metadata: Dict[str, Any]
    ) -> float:
        """
        Adjust confidence based on chunk quality signals.
        
        Quality indicators:
        - completeness: "complete" > "partial" > "fragment"
        - structure_confidence: Higher is better
        """
        # Completeness bonus/penalty
        completeness = metadata.get('completeness', 'complete')
        if completeness == 'complete':
            confidence += 0.05
        elif completeness == 'fragment':
            confidence -= 0.1
        
        # Structure confidence bonus/penalty
        structure_conf = metadata.get('structure_confidence', 1.0)
        if structure_conf < self.thresholds.min_structure_confidence:
            confidence -= 0.05
        elif structure_conf >= 0.95:
            confidence += 0.03
        
        return confidence
    
    def detect_hallucination_risk(
        self,
        confidence: float,
        similarity_score: float,
        rerank_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detect if retrieval is at risk of hallucination.
        
        Args:
            confidence: Overall confidence score
            similarity_score: Bi-encoder similarity
            rerank_score: Cross-encoder score
        
        Returns:
            dict with:
                - risk_level: "low", "medium", "high"
                - should_reject: bool
                - reason: str explaining risk
        """
        # High risk conditions
        if confidence < self.thresholds.low_confidence:
            return {
                'risk_level': 'high',
                'should_reject': True,
                'reason': f'Low confidence ({confidence:.2f}). No confident match found.'
            }
        
        if similarity_score < self.thresholds.min_similarity:
            return {
                'risk_level': 'high',
                'should_reject': True,
                'reason': f'Low similarity ({similarity_score:.2f}). Query may be out of scope.'
            }
        
        if rerank_score is not None and rerank_score < self.thresholds.min_rerank:
            return {
                'risk_level': 'high',
                'should_reject': True,
                'reason': f'Low rerank score ({rerank_score:.2f}). Retrieved content not relevant.'
            }
        
        # Medium risk
        if confidence < self.thresholds.medium_confidence:
            return {
                'risk_level': 'medium',
                'should_reject': False,
                'reason': f'Medium confidence ({confidence:.2f}). Answer with disclaimer.'
            }
        
        # Low risk
        return {
            'risk_level': 'low',
            'should_reject': False,
            'reason': f'High confidence ({confidence:.2f}). Safe to answer.'
        }
    
    def get_confidence_level(self, confidence: float) -> str:
        """
        Get confidence level category.
        
        Returns:
            "high", "medium", or "low"
        """
        if confidence >= self.thresholds.high_confidence:
            return "high"
        elif confidence >= self.thresholds.medium_confidence:
            return "medium"
        else:
            return "low"
    
    def compute_batch_confidence(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Compute confidence for a batch of retrieval results.
        
        Also computes score gaps between consecutive results.
        
        Args:
            results: List of retrieval result dicts
        
        Returns:
            Results with confidence scores added
        """
        if not results:
            return []
        
        # Compute score gaps
        for i in range(len(results) - 1):
            current_score = results[i].get('final_score', 0)
            next_score = results[i + 1].get('final_score', 0)
            results[i]['score_gap'] = current_score - next_score
        
        # Last result has no gap
        if len(results) > 0:
            results[-1]['score_gap'] = None
        
        # Compute confidence for each result
        for i, result in enumerate(results):
            query = result.get('query', '')
            content = result.get('content', '')
            similarity = result.get('similarity_score', 0)
            rerank = result.get('rerank_score')
            metadata = result.get('metadata', {})
            gap = result.get('score_gap')
            
            confidence = self.compute_confidence(
                query=query,
                retrieved_content=content,
                similarity_score=similarity,
                rerank_score=rerank,
                metadata=metadata,
                score_gap=gap
            )
            
            result['confidence'] = confidence
            result['confidence_level'] = self.get_confidence_level(confidence)
            
            # Check hallucination risk (only for top result)
            if i == 0:
                risk = self.detect_hallucination_risk(confidence, similarity, rerank)
                result['hallucination_risk'] = risk
        
        return results


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ConfidenceScorer...")
    print("=" * 70)
    
    scorer = ConfidenceScorer()
    
    # Test case 1: High confidence
    print("\nTest 1: High Confidence Case")
    conf1 = scorer.compute_confidence(
        query="What is an arithmetic progression?",
        retrieved_content="Definition: An arithmetic progression is a sequence...",
        similarity_score=0.88,
        rerank_score=0.94,
        metadata={'chunk_type': 'definition', 'completeness': 'complete', 'structure_confidence': 1.0},
        score_gap=0.25
    )
    print(f"Confidence: {conf1:.3f} ({scorer.get_confidence_level(conf1)})")
    risk1 = scorer.detect_hallucination_risk(conf1, 0.88, 0.94)
    print(f"Risk: {risk1['risk_level']} - {risk1['reason']}")
    
    # Test case 2: Low confidence
    print("\nTest 2: Low Confidence Case")
    conf2 = scorer.compute_confidence(
        query="What is an arithmetic progression?",
        retrieved_content="Exercise 5.2: Solve the following problems...",
        similarity_score=0.52,
        rerank_score=0.48,
        metadata={'chunk_type': 'exercise', 'completeness': 'partial', 'structure_confidence': 0.6},
        score_gap=0.05
    )
    print(f"Confidence: {conf2:.3f} ({scorer.get_confidence_level(conf2)})")
    risk2 = scorer.detect_hallucination_risk(conf2, 0.52, 0.48)
    print(f"Risk: {risk2['risk_level']} - {risk2['reason']}")
    print(f"Should reject: {risk2['should_reject']}")
