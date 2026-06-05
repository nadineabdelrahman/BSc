
# AI Pinocchio Detector

This repository contains the implementation prototype for the bachelor thesis:

**The AI Pinocchio Detector: Automated Fact-Checking**

The project implements a neuro-symbolic hallucination auditing pipeline for LLM-generated answers. Instead of trusting an LLM answer directly, the system extracts factual claims, represents them as subject--predicate--object triples, links entities to Wikidata, verifies claims using graph-based evidence, and applies an evidence-constrained correction loop when unsafe claims are detected.

## Project Overview

The system follows this general pipeline:

```text
User Question
    ↓
LLM-generated answer
    ↓
Hybrid claim extraction
    ↓
Predicate normalization and entity linking
    ↓
Wikidata/SPARQL verification
    ↓
SUPPORTED / REFUTED / NEI labeling
    ↓
Evidence-constrained correction
````

The goal of the project is not only to detect hallucinations, but also to make the verification process inspectable. The system stores extracted triples, linked entities, verification labels, graph evidence, NEI reasons, and correction metadata.

## Main Features

* LLM-based factual triple extraction.
* Rule-based spaCy extraction using dependency patterns.
* Hybrid merging, normalization, filtering, and deduplication of extracted claims.
* Predicate schema mapping between natural-language relations and Wikidata properties.
* Wikidata entity linking.
* SPARQL-based verification using direct edges and graph-based reasoning.
* Typed NEI handling for unresolved claims.
* Evidence-constrained correction loop.
* JSON-style logging of extracted claims, verification outputs, and correction results.

## Repository Structure

The main implementation files are organized as follows. Some auxiliary files may vary depending on the submitted prototype version.

```text
.
├── main.py                  # Main orchestration script
├── generation.py            # Initial LLM answer generation
├── extraction.py            # LLM-based triple extraction
├── extraction_spacy.py      # Rule-based spaCy extraction
├── hybrid_extraction.py     # Hybrid extraction, filtering, normalization, and deduplication
├── predicate_schema.py      # Canonical predicates, Wikidata mappings, and schema rules
├── verification.py          # Wikidata entity linking, SPARQL verification, and evidence retrieval
├── correction.py            # Evidence-constrained correction loop
├── logger.py                # Logging utilities
└── README.md                # Project documentation
```

## Requirements

This project requires Python 3.10 or later.

Install the required Python packages:


```bash
pip install openai python-dotenv requests spacy
```

The spaCy extractor uses the English model `en_core_web_sm`. Install it with:

```bash
python -m spacy download en_core_web_sm
```

## Environment Variables

The project uses the OpenAI API for LLM-based extraction, answer generation, and correction.

Create a local `.env` file in the project root:

 add your OpenAI API key inside `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```


## Example `.env.example`

```env
OPENAI_API_KEY=your_openai_api_key_here
```



## Running the Prototype

The exact execution command depends on the submitted prototype version.

A typical run may look like:

```bash
python main.py
```


The system expects an input question or prompt, generates or receives an LLM answer, extracts factual claims, verifies them against Wikidata, and optionally applies the correction loop.

## Main Components

### `extraction.py`

Performs LLM-based claim extraction. It prompts an LLM to extract candidate factual triples from generated answers and normalizes them into a structured format.

### `extraction_spacy.py`

Performs rule-based extraction using spaCy dependency parsing. This component provides deterministic syntactic extraction for high-confidence factual patterns.

### `hybrid_extraction.py`

Combines the LLM-based and spaCy-based extractors. It normalizes entities and predicates, filters noisy claims, removes duplicates, and returns the final claim set used by the verifier.

### `predicate_schema.py`

Defines the schema-guided verification layer. It maps canonical predicates to Wikidata properties, relation directions, type expectations, and verification strategies.

### `verification.py`

Links extracted entities to Wikidata identifiers and verifies claims using SPARQL queries. It supports direct edge checks, graph evidence retrieval, and typed handling of unresolved claims.

### `correction.py`

Implements the evidence-constrained correction loop. It preserves supported claims, removes or rewrites unsafe claims, and uses graph-supported replacement evidence when available.

## Output

The system  produce outputs such as:

* extracted triples,
* Wikidata QIDs,
* verification verdicts,
* final labels,
* graph evidence,
* NEI reasons,
* correction attempts,
* corrected answers.

Example labels include:

```text
SUPPORTED
REFUTED
NEI
TRUE
FALSE
HALLUCINATION
UNVERIFIABLE
```



```
