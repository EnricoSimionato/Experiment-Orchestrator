from __future__ import annotations

import logging

import transformers

import lm_eval


benchmark_id_metric_name_mapping = {
    "arc_challenge": "acc,none",
    "hellaswag": "acc,none",
    "gsm8k": "acc,none",
    "mmlu": "acc,none",
    "truthfulqa_mc1": "acc,none",
    "winogrande": "acc,none"
}


def evaluate_model_on_benchmark(
        model: [transformers.AutoModel | transformers.PreTrainedModel],
        tokenizer: transformers.AutoTokenizer,
        benchmark_id: str,
        evaluation_args: dict,
        device: str
) -> dict:
    """
    Evaluates the model in the benchmark.

    Args:
        model (transformers.AutoModel | transformers.PreTrainedModel):
            The model to be evaluated.
        tokenizer (transformers.AutoTokenizer):
            The tokenizer of the model.
        benchmark_id (str):
            The benchmark on which the model has to be evaluated.
        evaluation_args (dict):
            The arguments to use in the benchmark.
        device (str):
            The device to use in the evaluation.
    Returns:
        dict:
            Dictionary of the results.
    """

    logger = logging.getLogger(__name__)
    logger.info("Running the function evaluate_model_on_benchmark in the file lm_eval_pipeline.py.")

    # Defining the evaluation parameters
    #default_evaluation_args = (benchmark_id_eval_args_default_mapping[benchmark_id]
    #                           if benchmark_id in benchmark_id_eval_args_default_mapping.keys() else {})
    #default_evaluation_args.update(evaluation_args)
    #evaluation_args = default_evaluation_args
    logger.info(f"Evaluation args: {evaluation_args}")

    # Evaluating the model
    logger.info(f"Starting the evaluation for the benchmark: {benchmark_id}.")
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args={"pretrained": model, "tokenizer": tokenizer, "backend": "causal"},
        tasks=[benchmark_id],
        device=device,
        **evaluation_args
    )
    logger.info(f"Model evaluated.")

    return results["results"]
