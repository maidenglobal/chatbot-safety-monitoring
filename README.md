# Embedding-Based Safety Monitoring

We use semantic similarity search to find safety concerns in AI chatbot conversations that standard moderation APIs miss. This repository contains our methodology, code, and aggregate results from applying this approach to the [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) dataset<sup>[1]</sup>.

## How it works

**1. Generate search prototypes** from a behavior x domain taxonomy (`taxonomy.yaml`). We cross behavioral templates<sup>[2]</sup> (e.g., `"how to {term}"`) with harm-domain terms<sup>[3][4]</sup> to produce 1,117 prototypes across 4 risk categories, applying combinatorial coverage<sup>[5]</sup> to embedding-space search rather than red-teaming.

**2. Embed and scan.** We embed prototypes and user messages with `text-embedding-3-small` and score each message against all prototypes via cosine similarity<sup>[6][7]</sup>. We flag messages exceeding the category threshold.

**3. Grade candidates.** We grade flagged messages with `gpt-4.1-nano` using structured outputs (keep/reject + reason). The criterion is permissive: if a message is ambiguous enough to warrant human review, we keep it. The full prompt is in `src/grade_candidates.py`.

**4. Snowball expansion.** We query embedding neighbors of kept items, grade the new candidates, and repeat until convergence<sup>[8]</sup>. This ran for 8 rounds in our analysis (14,000 candidates graded, 7,757 confirmed concerns). Parameters: cosine similarity threshold 0.65, 30 neighbors per seed.

## Results

From 469,859 conversations and 129,166 users:

| Category | Flagged messages | Users |
|---|---|---|
| Safety Circumvention | 3,041 | 1,119 |
| Crisis & Despair | 2,485 | 1,097 |
| Violence & Harmful Content | 2,094 | 1,117 |
| AI Psychosis | 137 | 125 |
| **Total** | **7,757** | **3,458 (3,210 unique)** |

Across these categories we identified **7,757 flagged messages** (0.79% of 977,962 messages), in **6,213 unique conversations** (1.32% of 469,859 conversations), among **3,210 unique users** (2.49% of 129,166 users). The per-category user counts sum to 3,458 because 248 users were flagged in more than one category.

**The OpenAI Moderation API missed 96.6% of flagged messages** (97.4% at the conversation level), including conversations where the model provided crisis resources while the moderation layer reported negligible self-harm scores.

## Replication

```
pip install openai numpy pyyaml pydantic
```

1. Download WildChat-1M from [HuggingFace](https://huggingface.co/datasets/allenai/WildChat-1M) and extract user messages to JSONL (one `{"text": "..."}` per line)
2. Generate prototypes: `python src/generate_prototypes.py --output prototypes.json`
3. Scan messages: `python src/scan_example.py --messages user_messages.jsonl`
4. Grade results: `python src/grade_candidates.py --input scan_results.json --output graded.json`
5. For snowball expansion, query embedding neighbors of kept items and repeat grading until convergence

Total cost for the full corpus: ~$5.50 ($4.23 embedding, $1.27 grading). We document threshold selection in `taxonomy.yaml`.

Set an `OPENAI_API_KEY` environment variable before running.

## Repository contents

| File | Description |
|------|-------------|
| `taxonomy.yaml` | Risk category definitions, thresholds, and prototype generation rules |
| `src/generate_prototypes.py` | Generates search prototypes from the taxonomy |
| `src/scan_example.py` | Scans messages against prototypes using cosine similarity |
| `src/grade_candidates.py` | LLM grading with structured outputs |
| `findings_summary.json` | Aggregate results (no raw data) |

## References

1. Zhao, Y., et al. (2024). [WildChat: 1M ChatGPT Interaction Logs in the Wild](https://arxiv.org/abs/2405.01470). *ICLR 2024*.
2. Mazeika, M., et al. (2024). [HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal](https://arxiv.org/abs/2402.04249). *ICML 2024*.
3. Zeng, Y., et al. (2024). [AI Risk Categorization Decoded (AIR 2024): From Government Regulations to Corporate Policies](https://arxiv.org/abs/2406.17864).
4. Markov, T., et al. (2023). [A Holistic Approach to Undesired Content Detection in the Real World](https://arxiv.org/abs/2208.03274). *AAAI 2023*.
5. Samvelyan, M., et al. (2024). [Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts](https://arxiv.org/abs/2402.16822). *NeurIPS 2024*.
6. Jung, J., et al. (2024). [Safe-Embed: Unveiling the Safety-Critical Knowledge of Sentence Encoders](https://arxiv.org/abs/2407.06851).
7. Chang, M.-W., Ratinov, L., Roth, D., & Srikumar, V. (2008). [Importance of Semantic Representation: Dataless Classification](https://cdn.aaai.org/AAAI/2008/AAAI08-132.pdf). *AAAI 2008*.
8. Perez, E., et al. (2022). [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286). *EMNLP 2022*.

## License

AGPL v3. Free for internal use. Commercial licensing: contact [Maiden Labs](https://maidenlabs.org).
