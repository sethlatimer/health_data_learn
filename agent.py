import json
import re
from datetime import date

import requests

import api_data
import embedding
import pdf
from analysis import analyze_oecd_time_series
from data_sources.oecd import OECDDataSource
from data_sources.retrieval import DataSourceRetrieval
from database import get_recent_messages, save_message


OLLAMA_URL = "http://localhost:11434/api/generate"
ROUTES = {
    "general_chat",
    "document_search",
    "oecd_discovery",
    "oecd_analysis",
}


class Agent:
    def __init__(self, model="qwen-fast:latest"):
        self.model = model
        self.oecd_data_source = OECDDataSource()

    def retrieve_data_question(self, question, **options):
        return DataSourceRetrieval().retrieve(
            question,
            intent_resolver=self._resolve_retrieval_intent,
            **options,
        )

    def _resolve_retrieval_intent(self, question, fields):
        allowed_values = {
            "topic": ["utilization", "spending", None],
            "measure_type": ["count", "rate", "amount", None],
            "desired_analysis": [
                "trend",
                "comparison",
                "ranking",
                "snapshot",
                None,
            ],
        }
        properties = {
            field: {
                "type": ["string", "null"],
                "enum": allowed_values[field],
            }
            for field in fields
            if field in allowed_values
        }
        if not properties:
            return {}

        schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": self.model,
                "prompt": (
                    "Resolve only the ambiguities in the user's data request. "
                    "Choose the best matching intent from the allowed JSON "
                    "values; use null when the user has not specified it. "
                    "Do not select datasets or codes, and do not answer the "
                    "question. Return only the JSON object.\n\n"
                    f"Ambiguous fields: {', '.join(properties)}\n"
                    f"User request: {question}"
                ),
                "format": schema,
                "stream": False,
                "think": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        content = response.json().get("response", "")
        try:
            resolution = json.loads(content)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                "Retrieval intent resolver returned invalid JSON"
            ) from error
        if (
            not isinstance(resolution, dict)
            or set(resolution) != set(properties)
        ):
            raise ValueError(
                "Retrieval intent resolver returned an invalid object"
            )
        for field, value in resolution.items():
            if value not in allowed_values[field]:
                raise ValueError(
                    f"Retrieval intent resolver returned invalid {field}"
                )
        return resolution

    def ask_corpus(self, question, top_k=6):
        route = self.route_request(question)["route"]
        if route == "oecd_analysis":
            return self.ask_oecd_analysis(question)
        if route == "oecd_discovery":
            return self.ask_oecd_discovery(question)
        if route == "document_search":
            return self.ask_document_corpus(question, top_k=top_k)
        return self.ask(question)

    @staticmethod
    def _deterministic_route(question):
        text = question.strip()
        lowered = text.casefold()

        document_signal = re.search(
            r"\b(document|pdf|file|uploaded|attached|report)\b",
            lowered,
        )
        discovery_signal = re.search(
            r"\b(oecd|dataset|datasets|measure|measures|indicator|"
            r"indicators|catalog|available data|data source)\b",
            lowered,
        )
        analysis_signal = re.search(
            r"\b(changed?|change|trend|increased?|decreased?|"
            r"grew|fell|since|between|from \d{4}|over time)\b",
            lowered,
        )
        analysis_topic = re.search(
            r"\b(asthma|diabetes|hospital|admission|inpatient|event|"
            r"health|mortality|physician|patient|primary care|healthcare|"
            r"expenditure)\b",
            lowered,
        )
        has_year = re.search(r"\b(?:19|20)\d{2}\b", lowered)
        requests_measurement = re.search(
            r"\b(how many|how much|number of|count of)\b",
            lowered,
        )
        recent_period = re.search(
            r"\b(last|previous|past)\s+(year|12 months)\b",
            lowered,
        )

        if document_signal and not (
            discovery_signal and analysis_signal and "oecd" in lowered
        ):
            return "document_search"
        if (
            analysis_signal
            and has_year
            and analysis_topic
        ) or (
            requests_measurement
            and recent_period
            and analysis_topic
        ) or (discovery_signal and analysis_signal):
            return "oecd_analysis"
        if discovery_signal:
            return "oecd_discovery"
        if re.match(
            r"^(hi|hello|hey|good morning|good afternoon|good evening|"
            r"thanks(?: for your help)?|thank you(?: for your help)?|"
            r"goodbye|bye)\b[\s!.?]*$",
            lowered,
        ):
            return "general_chat"
        return None

    def route_request(self, question):
        deterministic_route = self._deterministic_route(question)
        if deterministic_route is not None:
            return {"route": deterministic_route}

        schema = {
            "type": "object",
            "properties": {
                "route": {
                    "type": "string",
                    "enum": sorted(ROUTES),
                },
            },
            "required": ["route"],
            "additionalProperties": False,
        }
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": self.model,
                "prompt": (
                    "Classify the user's request into exactly one route. "
                    "Use document_search for questions answered from the "
                    "local indexed documents, oecd_discovery for finding "
                    "OECD datasets or measures, oecd_analysis for questions "
                    "asking for OECD observations or quantitative trends, "
                    "and general_chat otherwise. Return only the JSON object "
                    "matching the supplied schema. Do not answer the request.\n\n"
                    f"USER REQUEST:\n{question}"
                ),
                "format": schema,
                "stream": False,
                "think": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        content = response.json().get("response", "")
        try:
            routing = json.loads(content)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                "Intent router returned invalid JSON"
            ) from error
        if (
            not isinstance(routing, dict)
            or set(routing) != {"route"}
            or routing["route"] not in ROUTES
        ):
            raise ValueError(
                "Intent router returned an invalid routing object"
            )
        return routing

    def ask_document_corpus(self, question, top_k=6):
        results = embedding.search_corpus(
            question,
            top_k=top_k,
            include_oecd=False,
        )

        if not results:
            answer = (
                "I couldn't find relevant information "
                "in the indexed document corpus."
            )
            save_message("user", question)
            save_message("assistant", answer)
            return answer

        context_parts = []

        for result in results:
            source_parts = [
                f"Source: {result.get('source') or 'LOCAL'}",
                result.get("title") or result["file_name"],
            ]
            if result.get("dataset_id"):
                source_parts.append(
                    f"dataset {result['dataset_id']}"
                )
            if result.get("page_number") is not None:
                source_parts.append(f"page {result['page_number']}")
            if result.get("source_url"):
                source_parts.append(result["source_url"])
            context_parts.append(
                "[" + "; ".join(source_parts) + "]\n" + result["content"]
            )

        context = "\n\n".join(context_parts)
        prompt = (
            "Answer the user's question using the "
            "retrieved document evidence below.\n\n"
            "Base factual claims on the supplied evidence. "
            "Include the supplied source, document title, page, and URL "
            "references when available. "
            "If the evidence is insufficient, say so.\n\n"
            "RETRIEVED EVIDENCE:\n\n"
            + context
            + "\n\nQUESTION:\n\n"
            + question
        )

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "think": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        answer = response.json()["response"].strip()

        save_message("user", question)
        save_message("assistant", answer)
        return answer

    def ask_oecd_discovery(self, question, top_k=10):
        results = self.oecd_data_source.search_concepts(
            question,
            top_k=top_k,
        )
        if not results:
            answer = (
                "I couldn't find matching concepts in the indexed "
                "OECD datasets."
            )
        else:
            lines = ["Matching indexed OECD measures:"]
            for result in results:
                lines.append(
                    f"- {result['label']} "
                    f"(code {result['code']}; "
                    f"dimension {result['dimension_id']}; "
                    f"dataset {result['dataset_id']}; "
                    f"semantic score {result['score']:.3f})"
                )
            answer = "\n".join(lines)

        save_message("user", question)
        save_message("assistant", answer)
        return answer

    @staticmethod
    def _question_periods(question):
        if re.search(
            r"\b(last|previous)\s+year\b",
            question,
            flags=re.IGNORECASE,
        ):
            previous_year = str(date.today().year - 1)
            return previous_year, previous_year

        years = re.findall(r"\b(?:19|20)\d{2}\b", question)
        if not years:
            return None, None

        since_match = re.search(
            r"\b(?:since|from|starting(?:\s+in)?)\s+((?:19|20)\d{2})\b",
            question,
            flags=re.IGNORECASE,
        )
        if since_match:
            start_period = since_match.group(1)
            end_match = re.search(
                r"\b(?:to|through|until|ending(?:\s+in)?)\s+((?:19|20)\d{2})\b",
                question,
                flags=re.IGNORECASE,
            )
            return start_period, end_match.group(1) if end_match else None

        if len(years) >= 2:
            return years[0], years[1]
        return years[0], years[0]

    @staticmethod
    def _text_occurs(text, question):
        return bool(re.search(
            r"(?<!\w)" + re.escape(text.strip()) + r"(?!\w)",
            question,
            flags=re.IGNORECASE,
        ))

    @classmethod
    def _resolve_filter_code(
        cls,
        dimension,
        options,
        question,
        has_time_range,
        measure_label=None,
    ):
        mentioned = [
            option
            for option in options
            if option["label"] and cls._text_occurs(option["label"], question)
        ]
        if mentioned:
            exact = [
                option
                for option in mentioned
                if option["label"].casefold() in question.casefold()
            ]
            selected = exact or mentioned
            unique_codes = {option["code"] for option in selected}
            if len(unique_codes) == 1:
                return selected[0]["code"]

        if len(options) == 1:
            return options[0]["code"]

        dimension_name = dimension["name"].casefold()
        labels = {}
        for option in options:
            labels.setdefault(option["label"].strip().casefold(), option)

        preferred_labels = []
        is_per_capita = re.search(
            r"\b(per capita|per person|per inhabitant|per head)\b",
            question,
            flags=re.IGNORECASE,
        )
        if (
            is_per_capita
            and (
                dimension["id"].upper() == "UNIT_MEASURE"
                or "unit of measure" in dimension_name
            )
        ):
            measure_terms = {
                term
                for term in re.findall(
                    r"[a-z]+",
                    (measure_label or "").casefold(),
                )
                if len(term) > 3 and term not in {"hospital"}
            }
            population_units = [
                option for option in options
                if re.search(
                    r"\bper\s+[\d\s,]+\s*(?:persons?|people|inhabitants?)\b",
                    option["label"],
                    flags=re.IGNORECASE,
                )
            ]
            matching_units = [
                option for option in population_units
                if measure_terms
                and any(
                    term in option["label"].casefold()
                    for term in measure_terms
                )
            ]
            if len(matching_units) == 1:
                return matching_units[0]["code"]
            if len(population_units) == 1:
                return population_units[0]["code"]

        if is_per_capita and "age" in dimension_name:
            total_age = labels.get("total") or labels.get("all ages")
            if total_age is not None:
                return total_age["code"]

        if "frequen" in dimension_name and has_time_range:
            preferred_labels.append("annual")
        if "diagnostic" in dimension_name:
            preferred_labels.append("all causes")
        if "age" in dimension_name:
            if not is_per_capita:
                common_adult_group = labels.get("15 years or over")
                if common_adult_group is not None:
                    return common_adult_group["code"]
            age_options = [
                option for option in options
                if re.search(r"\byears?\s+or\s+over\b", option["label"], re.I)
            ]
            if not is_per_capita and len(age_options) == 1:
                return age_options[0]["code"]
        if "sex" in dimension_name:
            preferred_labels.append("total")
        if "statistical operation" in dimension_name:
            preferred_labels.append("observed")
        if any(term in dimension_name for term in ("income", "profession")):
            preferred_labels.append("not applicable")

        preferred_labels.extend(("total", "not applicable", "observed"))
        for label in preferred_labels:
            if label in labels:
                return labels[label]["code"]

        return None

    def analyze_oecd_question(self, question):
        search_query = question
        inpatient_events_query = re.search(
            r"\bin[\s-]?patient\s+events?\b",
            question,
            flags=re.IGNORECASE,
        ) and not re.search(
            r"\bhospital\s+discharg",
            question,
            flags=re.IGNORECASE,
        )
        if inpatient_events_query:
            search_query = "Hospital discharges"

        candidates = self.oecd_data_source.search_concepts(
            search_query,
            top_k=10,
        )
        if not candidates:
            raise ValueError(
                "No indexed OECD semantic concepts matched the question"
            )
        if inpatient_events_query:
            candidates = candidates[:1]

        not_found_error = None
        for measure_candidate in candidates:
            try:
                return self._analyze_oecd_candidate(
                    question,
                    measure_candidate,
                )
            except requests.HTTPError as error:
                if error.response is None or error.response.status_code != 404:
                    raise
                not_found_error = error
            except LookupError as error:
                not_found_error = error

        if not_found_error is not None:
            raise not_found_error
        raise ValueError("No matching OECD dataset returned observations")

    def _analyze_oecd_candidate(self, question, measure_candidate):
        dataset_id = measure_candidate["dataset_id"]
        structure = api_data.get_indexed_dataset_structure(dataset_id)
        dimensions = structure["data_structure"]["dimensions"]
        measure_dimension = next(
            (
                dimension
                for dimension in dimensions
                if dimension["id"] == measure_candidate["dimension_id"]
            ),
            None,
        )
        if measure_dimension is None:
            raise ValueError(
                "The selected OECD measure is not present in its dataset structure"
            )
        valid_measure_codes = {
            code["id"] for code in measure_dimension["codes"]
        }
        if measure_candidate["code"] not in valid_measure_codes:
            raise ValueError(
                "The selected OECD measure code is not valid for its dataset"
            )

        start_period, end_period = self._question_periods(question)
        has_time_range = start_period is not None or end_period is not None
        filter_options = self.oecd_data_source.find_filters(dataset_id)
        options_by_dimension = {}
        for option in filter_options:
            options_by_dimension.setdefault(
                option["dimension_id"], []
            ).append(option)

        geographic_dimensions = [
            dimension
            for dimension in dimensions
            if dimension["id"] != measure_dimension["id"]
            and any(
                term in dimension["name"].casefold()
                for term in ("reference area", "geography", "country", "region")
            )
        ]
        resolved_filters = {}
        is_hospital_discharge_measure = (
            measure_candidate["label"].strip().casefold()
            == "hospital discharges"
        )

        for dimension in dimensions:
            dimension_id = dimension["id"]
            if dimension_id in {
                measure_dimension["id"],
                next(
                    (
                        item["id"]
                        for item in dimensions
                        if item["type"] == "TimeDimension"
                    ),
                    None,
                ),
            }:
                continue

            options = options_by_dimension.get(dimension_id, [])
            if not options:
                continue

            if (
                is_hospital_discharge_measure
                and dimension_id.upper() not in {
                    "UNIT_MEASURE",
                    "DIAGNOSTIC_TYPE",
                }
                and dimension not in geographic_dimensions
            ):
                continue

            if dimension in geographic_dimensions:
                geographic_matches = [
                    option
                    for option in options
                    if option["label"]
                    and self._text_occurs(option["label"], question)
                ]
                if not geographic_matches:
                    continue
                longest_label = max(
                    len(option["label"]) for option in geographic_matches
                )
                geographic_matches = [
                    option
                    for option in geographic_matches
                    if len(option["label"]) == longest_label
                ]
                codes = {option["code"] for option in geographic_matches}
                if len(codes) != 1:
                    raise ValueError(
                        f"Geography is ambiguous for dimension {dimension_id}"
                    )
                resolved_filters[dimension_id] = next(iter(codes))
                continue

            code = self._resolve_filter_code(
                dimension,
                options,
                question,
                has_time_range,
                measure_label=measure_candidate["label"],
            )
            if code is not None:
                resolved_filters[dimension_id] = code

        if not any(
            dimension_id in resolved_filters
            for dimension_id in (
                item["id"] for item in geographic_dimensions
            )
        ):
            geography_names = ", ".join(
                dimension["name"] for dimension in geographic_dimensions
            )
            raise ValueError(
                "Could not resolve a geography from the question"
                + (f" for dimensions: {geography_names}" if geography_names else "")
            )

        analysis_args = {
            "dataset_id": dataset_id,
            "measure_code": measure_candidate["code"],
            "filters": resolved_filters,
            "start_period": start_period,
            "end_period": end_period,
        }
        last_no_results_error = None
        try:
            return analyze_oecd_time_series(**analysis_args)
        except requests.HTTPError as error:
            requested_last_year = re.search(
                r"\b(last|previous)\s+year\b",
                question,
                flags=re.IGNORECASE,
            )
            if (
                not requested_last_year
                or not self._is_oecd_no_results(error)
                or start_period != end_period
            ):
                raise
            last_no_results_error = error

        requested_year = int(start_period)
        for year in range(requested_year - 1, requested_year - 4, -1):
            analysis_args["start_period"] = str(year)
            analysis_args["end_period"] = str(year)
            try:
                result = analyze_oecd_time_series(**analysis_args)
            except requests.HTTPError as fallback_error:
                if not self._is_oecd_no_results(fallback_error):
                    raise
                last_no_results_error = fallback_error
                continue

            result["availability"] = {
                "requested_period": str(requested_year),
                "returned_period": str(year),
                "note": (
                    f"No observations were available for {requested_year}; "
                    f"the latest available observation found was {year}."
                ),
            }
            return result

        raise last_no_results_error

    @staticmethod
    def _is_oecd_no_results(error):
        response = error.response
        return (
            response is not None
            and response.status_code == 404
            and response.text.strip().casefold()
            in {"noresultsfound", "norecordsfound"}
        )

    def _answer_oecd_analysis(self, question, analysis):
        statistics = analysis["statistics"]
        if (
            statistics["observation_count"] == 1
            and re.search(
                r"\b(how many|how much|what (?:was|were) the (?:number|total))\b",
                question,
                flags=re.IGNORECASE,
            )
        ):
            period = str(statistics["last_period"])
            value = statistics["last_value"]
            measure = analysis["measure"]
            filters = analysis["filters"]
            geography = filters.get("REF_AREA", {}).get("label")
            unit = filters.get("UNIT_MEASURE", {}).get("label")
            diagnosis = filters.get("DIAGNOSTIC_TYPE", {}).get("label")
            availability = analysis.get("availability")

            if availability:
                period_text = (
                    f"No observations were available for "
                    f"{availability['requested_period']}; the latest "
                    f"available observation was for {period}."
                )
            else:
                period_text = f"For {period},"

            details = [
                f"{geography}: {value:,} {unit or measure['label']}"
            ]
            if diagnosis:
                details.append(diagnosis.casefold())
            provenance = analysis["provenance"]
            return (
                f"{period_text} {', '.join(details)}. "
                f"OECD measure: {measure['label']} ({measure['code']}); "
                f"dataset: {provenance['dataset_title']} "
                f"({provenance['dataset_id']})."
            )

        answer_data = {
            "dataset_id": analysis["provenance"]["dataset_id"],
            "dataset_title": analysis["provenance"]["dataset_title"],
            "measure": analysis["measure"],
            "filters": analysis["filters"],
            "period": {
                "first": statistics["first_period"],
                "last": statistics["last_period"],
            },
            "availability": analysis.get("availability"),
            "statistics": {
                "first_value": statistics["first_value"],
                "last_value": statistics["last_value"],
                "absolute_change": statistics["absolute_change"],
                "percent_change": statistics["percent_change"],
                "mean": statistics["mean"],
                "year_over_year_changes": statistics[
                    "year_over_year_changes"
                ],
            },
        }
        prompt = (
            "Write a concise answer to the user's question using only the "
            "deterministic OECD analysis JSON below. Do not recalculate, "
            "estimate, add outside facts, change codes, or infer missing "
            "values. State the selected OECD measure and label, geography "
            "and other filters, period, first and last values, calculated "
            "absolute and percent change, and OECD dataset provenance. "
            "If percent change is null, say it is undefined because the "
            "starting value is zero. If availability reports that the "
            "requested period has no observations, clearly state that and "
            "identify the actual period returned instead.\n\n"
            f"QUESTION:\n{question}\n\n"
            "ANALYSIS JSON:\n"
            + json.dumps(answer_data, ensure_ascii=False)
        )
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "think": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    def ask_oecd_analysis(self, question):
        try:
            analysis = self.analyze_oecd_question(question)
            answer = self._answer_oecd_analysis(question, analysis)
            return {**analysis, "route": "oecd_analysis", "answer": answer}
        except requests.HTTPError as error:
            status_code = (
                error.response.status_code
                if error.response is not None
                else None
            )
            if status_code == 429:
                answer = (
                    "The OECD data service is still rate-limiting this "
                    "request after retries. Please try again shortly."
                )
            elif status_code == 404:
                answer = (
                    "The OECD data service did not return observations for "
                    "the selected measure, filters, and period. Try a "
                    "different year or specify a more precise measure."
                )
            else:
                raise

        save_message("user", question)
        save_message("assistant", answer)
        return answer

    def index_folder(
        self,
        folder_path,
        embedding_model="nomic-local",
        recursive=False,
    ):
        return pdf.index_folder(
            folder_path,
            embedding_model=embedding_model,
            recursive=recursive,
        )

    def search_corpus(
        self,
        query,
        top_k=5,
        embedding_model="nomic-local",
    ):
        return embedding.search_corpus(
            query,
            top_k=top_k,
            embedding_model=embedding_model,
        )

    def ask_document(self, document_id, question, top_k=5):
        results = self.search_document(
            document_id=document_id,
            query=question,
            top_k=top_k,
        )

        if not results:
            return (
                "I could not find any embedded document "
                "chunks for that document."
            )

        context_parts = []

        for result in results:
            source_label = (
                f"[Source: {result['file_name']}, "
                f"page {result['page_number']}]"
            )
            context_parts.append(
                source_label + "\n" + result["content"]
            )

        context = "\n\n".join(context_parts)
        full_prompt = (
            "Answer the user's question using only the "
            "document excerpts below.\n\n"
            "Do not use outside knowledge.\n"
            "If the excerpts do not contain enough information, "
            "say that the indexed document does not provide "
            "enough information.\n"
            "Include the filename and page number for each "
            "important claim.\n\n"
            "DOCUMENT EXCERPTS:\n\n"
            + context
            + "\n\nUSER QUESTION:\n\n"
            + question
        )

        save_message("user", question)
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "think": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        answer = response.json()["response"].strip()
        save_message("assistant", answer)
        return answer

    def search_document(
        self,
        document_id,
        query,
        top_k=5,
        embedding_model="nomic-local",
    ):
        return embedding.search_document(
            document_id,
            query,
            top_k=top_k,
            embedding_model=embedding_model,
        )

    def embed_document(self, document_id, embedding_model="nomic-local"):
        return embedding.embed_document(
            document_id,
            embedding_model=embedding_model,
        )

    def _build_prompt(self, prompt):
        history = get_recent_messages(20)
        history_text = "".join(
            f"{role}: {content}\n" for role, content in history
        )
        return (
            "Previous conversation:\n\n"
            + history_text
            + "\nCurrent user message:\n\n"
            + prompt
            + "\n\nRespond naturally and directly."
        )

    def ask(self, prompt):
        save_message("user", prompt)
        full_prompt = self._build_prompt(prompt)
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "think": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        answer = response.json()["response"].strip()
        save_message("assistant", answer)
        return answer

    def ask_about_pdf(self, pdf_path, question):
        return pdf.ask_about_pdf(self.ask, pdf_path, question)

    def index_pdf(self, pdf_path):
        return pdf.index_pdf(pdf_path)

    def discover_oecd_dataflows(self):
        return api_data.discover_dataflows()

    def find_healthcare_datasets(self, limit=10):
        return api_data.find_healthcare_datasets(limit=limit)

    def get_oecd_dataset_metadata(
        self,
        dataset_id,
        embedding_model=None,
    ):
        return api_data.get_dataset_metadata(
            dataset_id,
            embedding_model=embedding_model,
        )

    def index_oecd_dataset_semantics(
        self,
        dataset_id,
        embedding_model="nomic-local",
    ):
        return api_data.index_dataset_semantics(
            dataset_id,
            embedding_model=embedding_model,
        )


if __name__ == "__main__":
    agent = Agent()
    print("Agent ready. Type quit to exit.")

    while True:
        prompt = input("\nYou: ").strip()

        if prompt.lower() in ["quit", "exit"]:
            print("Goodbye.")
            break

        if not prompt:
            continue

        try:
            answer = agent.ask(prompt)
            print("\nAgent:")
            print(answer)
        except Exception as error:
            print(f"\nERROR: {error}")
