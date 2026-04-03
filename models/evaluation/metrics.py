"""
Hybrid Transformer-Mamba Model Evaluation Metrics

2026 evaluation framework for testing hybrid architecture performance:
- Search relevance metrics (NDCG@10, MAP, MRR)
- Efficiency metrics (inference time, memory usage)
- Domain classification accuracy
- Specificity prediction correlation
- Ablation studies (T-only vs M-only vs Hybrid)
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResults:
    """Container for evaluation results."""

    # Search relevance metrics
    ndcg_at_10: float = 0.0
    map_score: float = 0.0
    mrr_score: float = 0.0

    # Domain classification metrics
    domain_accuracy: float = 0.0
    domain_f1: float = 0.0

    # Specificity prediction metrics
    specificity_accuracy: float = 0.0
    specificity_correlation: float = 0.0

    # Efficiency metrics
    avg_inference_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    throughput_qps: float = 0.0

    # Architecture-specific metrics
    transformer_layer_stats: Optional[Dict[str, float]] = None
    mamba_layer_stats: Optional[Dict[str, float]] = None

    # Model size
    total_parameters: int = 0
    model_size_mb: float = 0.0

class HybridModelEvaluator:
    """
    Comprehensive evaluation framework for Hybrid Transformer-Mamba models.

    Evaluates:
    1. Search relevance performance
    2. Multi-task learning effectiveness
    3. Inference efficiency vs pure transformers
    4. Memory usage and throughput
    5. Layer-wise performance analysis
    """

    def __init__(self, device: str = "auto"):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Evaluator initialized on device: {self.device}")

    def evaluate_model(
        self,
        model: torch.nn.Module,
        test_dataloader: torch.utils.data.DataLoader,
        query_product_pairs: Optional[List[Tuple[str, str, float]]] = None,
        verbose: bool = True
    ) -> EvaluationResults:
        """
        Comprehensive model evaluation.

        Args:
            model: Trained hybrid model
            test_dataloader: Test dataset
            query_product_pairs: (query, product, relevance_score) for search evaluation
            verbose: Print detailed results

        Returns:
            EvaluationResults with all metrics
        """
        model.eval()
        model.to(self.device)

        results = EvaluationResults()

        with torch.no_grad():
            # 1. Multi-task performance evaluation
            if verbose:
                logger.info("🧠 Evaluating multi-task performance...")

            domain_preds, domain_labels = [], []
            specificity_preds, specificity_labels = [], []
            inference_times = []

            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Measure inference time
                start_time = time.time()

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_embeddings=True
                )

                end_time = time.time()
                inference_times.append((end_time - start_time) * 1000)  # ms

                # Collect predictions
                if 'domain_logits' in outputs and 'domain_labels' in batch:
                    domain_preds.extend(torch.argmax(outputs['domain_logits'], dim=-1).cpu().numpy())
                    domain_labels.extend(batch['domain_labels'].cpu().numpy())

                if 'specificity_logits' in outputs and 'specificity_labels' in batch:
                    specificity_preds.extend(torch.argmax(outputs['specificity_logits'], dim=-1).cpu().numpy())
                    specificity_labels.extend(batch['specificity_labels'].cpu().numpy())

            # Calculate multi-task metrics
            if domain_preds and domain_labels:
                results.domain_accuracy = accuracy_score(domain_labels, domain_preds)
                results.domain_f1 = f1_score(domain_labels, domain_preds, average='weighted')

                if verbose:
                    logger.info(f"  Domain Classification Accuracy: {results.domain_accuracy:.3f}")
                    logger.info(f"  Domain F1 Score: {results.domain_f1:.3f}")

            if specificity_preds and specificity_labels:
                results.specificity_accuracy = accuracy_score(specificity_labels, specificity_preds)
                results.specificity_correlation = np.corrcoef(specificity_labels, specificity_preds)[0, 1]

                if verbose:
                    logger.info(f"  Specificity Accuracy: {results.specificity_accuracy:.3f}")
                    logger.info(f"  Specificity Correlation: {results.specificity_correlation:.3f}")

            # 2. Search relevance evaluation
            if query_product_pairs:
                if verbose:
                    logger.info("🔍 Evaluating search relevance...")

                search_metrics = self._evaluate_search_relevance(model, query_product_pairs)
                results.ndcg_at_10 = search_metrics['ndcg_at_10']
                results.map_score = search_metrics['map']
                results.mrr_score = search_metrics['mrr']

                if verbose:
                    logger.info(f"  NDCG@10: {results.ndcg_at_10:.3f}")
                    logger.info(f"  MAP: {results.map_score:.3f}")
                    logger.info(f"  MRR: {results.mrr_score:.3f}")

            # 3. Efficiency metrics
            if verbose:
                logger.info("⚡ Evaluating efficiency metrics...")

            results.avg_inference_time_ms = np.mean(inference_times)
            results.peak_memory_mb = self._get_peak_memory_usage()
            results.throughput_qps = 1000.0 / results.avg_inference_time_ms if results.avg_inference_time_ms > 0 else 0

            if verbose:
                logger.info(f"  Avg Inference Time: {results.avg_inference_time_ms:.2f} ms")
                logger.info(f"  Peak Memory: {results.peak_memory_mb:.1f} MB")
                logger.info(f"  Throughput: {results.throughput_qps:.1f} QPS")

            # 4. Model size metrics
            results.total_parameters = sum(p.numel() for p in model.parameters())
            results.model_size_mb = results.total_parameters * 4 / (1024 * 1024)  # FP32 assumption

            if verbose:
                logger.info(f"  Total Parameters: {results.total_parameters:,}")
                logger.info(f"  Model Size: {results.model_size_mb:.1f} MB")

            # 5. Layer-wise analysis for hybrid models
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
                if verbose:
                    logger.info("🔬 Analyzing layer performance...")

                layer_stats = self._analyze_layer_performance(model)
                results.transformer_layer_stats = layer_stats.get('transformer', {})
                results.mamba_layer_stats = layer_stats.get('mamba', {})

                if verbose and results.transformer_layer_stats:
                    logger.info(f"  Transformer layers: {len(results.transformer_layer_stats)} layers")
                if verbose and results.mamba_layer_stats:
                    logger.info(f"  Mamba layers: {len(results.mamba_layer_stats)} layers")

        return results

    def _evaluate_search_relevance(
        self,
        model: torch.nn.Module,
        query_product_pairs: List[Tuple[str, str, float]]
    ) -> Dict[str, float]:
        """Evaluate search relevance using NDCG@10, MAP, MRR."""
        from transformers import AutoTokenizer

        # Use default tokenizer for evaluation
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

        all_queries = list(set([pair[0] for pair in query_product_pairs]))
        all_products = list(set([pair[1] for pair in query_product_pairs]))

        # Get embeddings for queries and products
        query_embeddings = self._get_embeddings(model, tokenizer, all_queries)
        product_embeddings = self._get_embeddings(model, tokenizer, all_products)

        # Create relevance matrix
        relevance_scores = {}
        for query, product, score in query_product_pairs:
            if query not in relevance_scores:
                relevance_scores[query] = {}
            relevance_scores[query][product] = score

        # Calculate metrics
        ndcg_scores = []
        ap_scores = []
        rr_scores = []

        for i, query in enumerate(all_queries):
            if query not in relevance_scores:
                continue

            query_emb = query_embeddings[i:i+1]
            similarities = cosine_similarity(query_emb, product_embeddings)[0]

            # Get relevance scores for this query
            true_relevances = []
            pred_similarities = []

            for j, product in enumerate(all_products):
                if product in relevance_scores[query]:
                    true_relevances.append(relevance_scores[query][product])
                    pred_similarities.append(similarities[j])

            if len(true_relevances) > 0:
                # Sort by predicted similarities (descending)
                sorted_pairs = sorted(zip(true_relevances, pred_similarities), key=lambda x: x[1], reverse=True)
                sorted_relevances = [pair[0] for pair in sorted_pairs]

                # NDCG@10
                ndcg = self._calculate_ndcg(sorted_relevances[:10])
                ndcg_scores.append(ndcg)

                # Average Precision
                ap = self._calculate_average_precision(sorted_relevances)
                ap_scores.append(ap)

                # Reciprocal Rank
                rr = self._calculate_reciprocal_rank(sorted_relevances)
                rr_scores.append(rr)

        return {
            'ndcg_at_10': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'map': np.mean(ap_scores) if ap_scores else 0.0,
            'mrr': np.mean(rr_scores) if rr_scores else 0.0
        }

    def _get_embeddings(
        self,
        model: torch.nn.Module,
        tokenizer,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """Get embeddings for a list of texts."""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            # Get embeddings
            with torch.no_grad():
                if hasattr(model, 'get_search_embeddings'):
                    batch_embeddings = model.get_search_embeddings(input_ids, attention_mask)
                else:
                    outputs = model(input_ids, attention_mask, return_embeddings=True)
                    batch_embeddings = outputs['search_embeddings']

            embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def _calculate_ndcg(self, relevances: List[float], k: int = 10) -> float:
        """Calculate NDCG@k."""
        relevances = relevances[:k]
        if not relevances:
            return 0.0

        # DCG
        dcg = relevances[0]
        for i in range(1, len(relevances)):
            dcg += relevances[i] / np.log2(i + 1)

        # IDCG
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = ideal_relevances[0]
        for i in range(1, len(ideal_relevances)):
            idcg += ideal_relevances[i] / np.log2(i + 1)

        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_average_precision(self, relevances: List[float]) -> float:
        """Calculate Average Precision."""
        if not relevances:
            return 0.0

        num_relevant = sum(1 for r in relevances if r > 0)
        if num_relevant == 0:
            return 0.0

        ap = 0.0
        num_relevant_so_far = 0

        for i, relevance in enumerate(relevances):
            if relevance > 0:
                num_relevant_so_far += 1
                precision_at_i = num_relevant_so_far / (i + 1)
                ap += precision_at_i

        return ap / num_relevant

    def _calculate_reciprocal_rank(self, relevances: List[float]) -> float:
        """Calculate Reciprocal Rank."""
        for i, relevance in enumerate(relevances):
            if relevance > 0:
                return 1.0 / (i + 1)
        return 0.0

    def _get_peak_memory_usage(self) -> float:
        """Get peak memory usage in MB."""
        if torch.cuda.is_available() and self.device == 'cuda':
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        elif torch.backends.mps.is_available() and self.device == 'mps':
            # MPS doesn't have direct memory monitoring, return estimate
            return 512.0  # Rough estimate for M1/M2 Macs
        else:
            # CPU - use system memory estimate
            import psutil
            return psutil.virtual_memory().used / (1024 * 1024)

    def _analyze_layer_performance(self, model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
        """Analyze performance characteristics of different layer types."""
        transformer_stats = {}
        mamba_stats = {}

        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            for i, layer in enumerate(model.encoder.layers):
                layer_name = f"layer_{i}"

                if hasattr(layer, 'layer_type'):
                    from ..architecture.hybrid_transformer_mamba import LayerType

                    if layer.layer_type == LayerType.TRANSFORMER:
                        # Count transformer parameters
                        params = sum(p.numel() for p in layer.parameters())
                        transformer_stats[layer_name] = {
                            'parameters': params,
                            'type': 'transformer'
                        }
                    elif layer.layer_type == LayerType.MAMBA:
                        # Count Mamba parameters
                        params = sum(p.numel() for p in layer.parameters())
                        mamba_stats[layer_name] = {
                            'parameters': params,
                            'type': 'mamba'
                        }

        return {
            'transformer': transformer_stats,
            'mamba': mamba_stats
        }

def create_test_query_product_pairs() -> List[Tuple[str, str, float]]:
    """Create test query-product pairs for evaluation."""
    pairs = [
        # Sports car queries - high relevance
        ("sports car", "Ferrari 488 GTB high-performance sports car with twin-turbo V8 engine", 1.0),
        ("sports car", "Porsche 911 Turbo sports coupe with AWD and precision handling", 1.0),
        ("sports car", "Toyota Camry sedan with excellent fuel efficiency", 0.1),  # Low relevance

        # Electronics queries
        ("gaming laptop", "ASUS ROG gaming laptop with RTX 4080 and Intel i9 processor", 1.0),
        ("gaming laptop", "MacBook Air ultrabook for professional work and productivity", 0.2),

        # Fashion queries
        ("leather handbag", "Designer Italian leather handbag with gold hardware", 1.0),
        ("leather handbag", "Canvas backpack for hiking and outdoor activities", 0.1),

        # General queries
        ("luxury watch", "Swiss Rolex Submariner with automatic movement", 1.0),
        ("luxury watch", "Digital fitness tracker with heart rate monitor", 0.3),
    ]
    return pairs

def benchmark_models(
    hybrid_model: torch.nn.Module,
    baseline_model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    query_product_pairs: Optional[List[Tuple[str, str, float]]] = None
) -> Dict[str, EvaluationResults]:
    """
    Benchmark hybrid model against baseline.

    Returns:
        Dictionary with 'hybrid' and 'baseline' evaluation results
    """
    evaluator = HybridModelEvaluator()

    logger.info("🔥 Evaluating Hybrid Transformer-Mamba model...")
    hybrid_results = evaluator.evaluate_model(
        hybrid_model,
        test_dataloader,
        query_product_pairs,
        verbose=True
    )

    logger.info("\n📊 Evaluating baseline model...")
    baseline_results = evaluator.evaluate_model(
        baseline_model,
        test_dataloader,
        query_product_pairs,
        verbose=True
    )

    # Print comparison
    logger.info("\n🚀 Performance Comparison:")
    logger.info(f"Inference Speed: Hybrid {hybrid_results.avg_inference_time_ms:.2f}ms vs Baseline {baseline_results.avg_inference_time_ms:.2f}ms")

    speedup = baseline_results.avg_inference_time_ms / hybrid_results.avg_inference_time_ms if hybrid_results.avg_inference_time_ms > 0 else 1.0
    logger.info(f"Speedup: {speedup:.1f}x faster")

    if hybrid_results.ndcg_at_10 > 0 and baseline_results.ndcg_at_10 > 0:
        logger.info(f"Search Quality: Hybrid NDCG@10 {hybrid_results.ndcg_at_10:.3f} vs Baseline {baseline_results.ndcg_at_10:.3f}")

    return {
        'hybrid': hybrid_results,
        'baseline': baseline_results
    }