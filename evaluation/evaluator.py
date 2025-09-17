import copy
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import evaluate
import spacy
from tqdm import tqdm

from utils.io import save_json, load_json
from utils.text_processing import get_lemma_word


def _invert_binary_predictions(data_list: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """ Invert binary prediction (0 or 1) in a list of dict on the 'key' field """
    data_new = copy.deepcopy(data_list)
    for e in data_new:
        if e[key] == 1:
            e[key] = 0
        else:
            e[key] = 1
    return data_new


def _score_keywords(true_wp, pred_wp, true_wa, pred_wa, nlp, removed_stopwords: bool) -> int:
    w_p_lemma, w_p = get_lemma_word(true_wp, nlp, removed_stopwords, True)
    w_a_lemma, w_a = get_lemma_word(true_wa, nlp, removed_stopwords, True)
    w_a = None if w_a == "" else w_a
    w_p = None if w_p == "" else w_p
    w_a_lemma = None if w_a_lemma == "" else w_a_lemma
    w_p_lemma = None if w_p_lemma == "" else w_p_lemma
    true_keywords_lemma = [w for w in [w_p_lemma, w_a_lemma] if w]
    true_keywords = [w for w in [w_p, w_a] if w]

    w_p_lemma, w_p, w_a_lemma, w_a = None, None, None, None
    if pred_wp:
        w_p_lemma, w_p = get_lemma_word(pred_wp, nlp, removed_stopwords, True)
    if pred_wa:
        w_a_lemma, w_a = get_lemma_word(pred_wa, nlp, removed_stopwords, True)
    w_a = None if w_a == "" else w_a
    w_p = None if w_p == "" else w_p
    w_a_lemma = None if w_a_lemma == "" else w_a_lemma
    w_p_lemma = None if w_p_lemma == "" else w_p_lemma
    pred_keywords = {w_p, w_a, w_p_lemma, w_a_lemma}

    lemma_overlap = len([k for k in true_keywords_lemma if k in pred_keywords])
    overlap = len([k for k in true_keywords if k in pred_keywords])
    assert 0 <= lemma_overlap <= 2
    assert 0 <= overlap <= 2
    return max(lemma_overlap, overlap)


class Evaluator:
    def __init__(self, logger: logging.Logger, outdir: Path = None):
        self.logger = logger
        self._nlp = spacy.load("en_core_web_lg")
        self._outdir: Optional[Path] = outdir
        self._binary_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
        self._recall_metrics = evaluate.load("recall")

    def evaluate_rationale(self, truth: List[Dict], pred: List[Dict]) -> Tuple[Dict[int, float], float]:
        """
        Assumes two lists containing the example and prediction already parsed, listed should be sorted with indexes in same way.
        I.e. truth[i]["id"] == pred[i]["id"] for all i

        :param truth: reference values (original examples)
        :param pred: parsed predictions, containing keys "pred_w_p", "pred_w_a", "pred_label", "id"
        :return: percentage of misses (0), partial (1) or total matches (2)
        """
        matches = list()
        for true_example, pred_example in tqdm(zip(truth, pred), desc="Evaluating rationales"):
            # # ******************* COMMENT THESE 3 LINES BELOW TO EVALUATE ALL PREDICTION CATEGORIES
            # if true_example["label"] != pred_example["pred_label"] or true_example["label"] == 0:
            #     print("Evaluating only TP!!!!")
            #     continue

            if true_example["label"] != pred_example["pred_label"]:
                matches.append(0)
            elif true_example["label"] == 1:
                pred_wp = pred_example["pred_w_p"].strip() if isinstance(pred_example["pred_w_p"], str) else \
                    pred_example["pred_w_p"]
                true_wp = true_example["w_p"].strip() if isinstance(true_example["w_p"], str) else true_example["w_p"]
                pred_wa = pred_example["pred_w_a"].strip() if isinstance(pred_example["pred_w_a"], str) else \
                    pred_example["pred_w_a"]
                true_wa = true_example["w_a"].strip() if isinstance(true_example["w_a"], str) else true_example["w_a"]

                lemma_score = 0
                match_score = int(pred_wp == true_wp) + int(pred_wa == true_wa)
                if match_score < 2:
                    lemma_score = _score_keywords(true_wp, pred_wp, true_wa, pred_wa, self._nlp, False)
                matches.append(max(lemma_score, match_score))
            else:
                matches.append(2)  # includes the case where both labels are 0
        counter = Counter(matches)
        counter = Counter({key: counter.get(key, 0) for key in [0, 1, 2]})
        total = len(matches)
        scores = {num: count / total for num, count in counter.items()}
        unified_score = sum(matches) / (2 * len(matches))
        return scores, unified_score

    @staticmethod
    def _search_rationale(s):
        groups = re.findall(r"<([^>]*)>", s)
        wp = groups[0].strip() if len(groups) > 0 else None
        wa = groups[1].strip() if len(groups) > 1 else None
        sp = groups[2].strip() if len(groups) > 2 else None
        sa = groups[3].strip() if len(groups) > 3 else None
        return wp, wa, sp, sa

    def parse_prediction(self, predictions: List[Dict], require_rationale: bool) -> Tuple[List[Dict], List[str]]:
        wrongs = list()
        parsed_predictions = list()
        pat = r"\b\W{,2}(yes|no)\W{,2}\b(?:[^<>]{,5}<([^>]*)>)" + ("+" if require_rationale else "*")
        match_pattern = re.compile(pat, flags=re.IGNORECASE)
        for pred in tqdm(predictions, desc="Parsing predictions"):
            output = pred["output"]
            wp, wa, sp, sa = None, None, None, None
            if output is not None:
                output = output.strip()
                matches = re.findall(match_pattern, output)
                if matches:
                    label = 1 if matches[-1][0].lower() == "yes" else 0
                    if label == 1:
                        wp, wa, sp, sa = self._search_rationale(output)
                else:
                    msg = f"Could not match yes or no in line: '{output}'. Setting label to 0."
                    self.logger.warning(msg)
                    wrongs.append(pred["id"])
                    label = 0
            else:
                msg = f"Could not match yes or no in line: '{output}'. Setting label to 0."
                self.logger.warning(msg)
                wrongs.append(pred["id"])
                label = 0
            wp = None if wp == "" else wp
            wa = None if wa == "" else wa
            sp = None if sp == "" else sp
            sa = None if sa == "" else sa
            parsed_predictions.append(
                {**pred, "pred_label": label, "pred_w_p": wp, "pred_w_a": wa, "pred_s_p": sp, "pred_s_a": sa})
        return parsed_predictions, wrongs

    def evaluate_metrics(self, examples: List[Dict], results: List[Dict], keyword_evaluation: bool,
                         require_rationale: bool, pos_class: int = 1, hom_het_evaluation: bool = True, re_parse_results: bool = True) -> Dict[str, float]:
        """
        Binary and rationale metrics evaluation.

        Args:
            examples: examples from the dataset
            results: a model's predictions for the same examples
            keyword_evaluation: whether to evaluate rationales or not. If False, rationale metrics are not computed.
            require_rationale: whether to enforce the presence of at least one rationale in the output. If False, yes|no is sufficient to compute metrics. If True, examples without rationale are discarded.
            pos_class: positive class for binary metrics. Can be 0 or 1. Default is 1.
            hom_het_evaluation: whether to perform an additional separate evaluation of only heterographic/homographic examples. Default is True.
            re_parse_results: whether to redo the parsing of passed results. Default is True. If False, the previously parsed results are used (parsed_output.json). Wrong ids are also reloaded.

        Returns: a dictionary with all the computed metrics rounded to 4 decimals.
        """
        # Sort examples and results in the same fashion
        examples = list(sorted(examples, key=lambda e: e["id"]))
        id_index_map = {e["id"]: index for index, e in enumerate(examples)}
        results = list(sorted(results, key=lambda r: id_index_map[r["id"]]))
        self.logger.info("Sorted examples and results")
        parsed_results_file = self._outdir / "parsed_output.json"
        wrong_ids_file = self._outdir / "wrong.json"

        # confidences = [item['confidence'] for item in results]
        # threshold = np.percentile(confidences, 90)
        # results = [item for item in results if item['confidence'] > threshold]
        # examples = [e for e in examples if e["id"] in {r["id"] for r in results}]

        # PARSE
        if re_parse_results:
            parsed_results, wrongs = self.parse_prediction(results, require_rationale)
        else:
            parsed_results = load_json(parsed_results_file)
            if wrong_ids_file.exists():
                wrongs = load_json(wrong_ids_file)
            else:
                wrongs = list()

        # Avoid evaluating wrongly formatted predictions
        if len(wrongs) > 0:
            examples = [e for e in examples if e["id"] not in wrongs]
            results = [e for e in results if e["id"] not in wrongs]
            parsed_results = [e for e in parsed_results if e["id"] not in wrongs]

        if self._outdir is not None:
            if len(wrongs) > 0:
                if re_parse_results:
                    save_json(wrong_ids_file, wrongs)
            elif wrong_ids_file.exists():
                os.remove(wrong_ids_file)

            if re_parse_results:
                save_json(parsed_results_file, parsed_results)
        self.logger.info("Parsed results")

        # EVALUATE BINARY METRICS
        metrics_results = dict()

        if pos_class == 0:
            parsed_results = _invert_binary_predictions(parsed_results, "pred_label")
            parsed_results = _invert_binary_predictions(parsed_results, "label")
            results = _invert_binary_predictions(results, "label")

        y_pred = [p["pred_label"] for p in parsed_results]
        y_true = [p["label"] for p in parsed_results]
        assert y_true == [p["label"] for p in results]
        assert len(y_pred) == len(y_true)
        assert len(parsed_results) == len(results) == len(examples)
        bin_metric_results = self._binary_metrics.compute(predictions=y_pred, references=y_true)
        self.logger.info("Computed metrics")

        # EVALUATE BINARY METRICS on HET/HOM
        if hom_het_evaluation:
            het_ids = {p["id"] for p in examples if p["is_het"] == True}
            hom_ids = {p["id"] for p in examples if p["is_het"] == False}

            het_y_true = [o["label"] for o in results if o["id"] in het_ids]
            het_y_pred = [o["pred_label"] for o in parsed_results if o["id"] in het_ids]

            hom_y_true = [o["label"] for o in results if o["id"] in hom_ids]
            hom_y_pred = [o["pred_label"] for o in parsed_results if o["id"] in hom_ids]

            assert len(het_y_true) == len(het_y_pred)
            assert len(hom_y_true) == len(hom_y_pred)
            assert len(het_ids) == len(het_y_true)
            assert len(hom_ids) == len(hom_y_true)

            het_metrics = self._recall_metrics.compute(predictions=het_y_pred, references=het_y_true)
            hom_metrics = self._recall_metrics.compute(predictions=hom_y_pred, references=hom_y_true)
            bin_metric_results[f"het_recall"] = het_metrics["recall"]
            bin_metric_results[f"hom_recall"] = hom_metrics["recall"]
            bin_metric_results[f"het_support"] = len(het_ids)
            bin_metric_results[f"hom_support"] = len(hom_ids)
            self.logger.info("Computed het/hom metrics")

        for k, v in bin_metric_results.items():
            metrics_results[k] = float(round(v, 4))

        # EVALUATE RATIONALE
        keyword_evaluation &= all(e["w_p"] is not None and e["w_a"] is not None for e in examples if e["label"] == 1)
        if keyword_evaluation:
            kw_scores, unified_score = self.evaluate_rationale(examples, parsed_results)
            kw_scores = {k: round(v, 4) for k, v in kw_scores.items()}
            metrics_results["kw_agreement_2"] = kw_scores[2]
            metrics_results["kw_agreement_1"] = kw_scores[1]
            metrics_results["kw_agreement_0"] = kw_scores[0]
            metrics_results["kw_agreement"] = round(unified_score, 4)
            self.logger.info("Computed rationale metrics")

        return metrics_results
