# LangChain Text Splitters

This module covers **Text Splitters** — a critical component in the RAG (Retrieval-Augmented Generation) pipeline that sits right after Document Loaders.

---

## Where Text Splitters Fit in a RAG Application

A typical RAG pipeline follows this sequence:

```
Raw Data          Document      Document        Text         Embeddings   Vector Store   Retriever   LLM
(txt, pdf,   -->   Loader   -->  Object(s)  -->  Splitter -->             -->          -->          -->
csv, website)
```

Once a **Document Loader** pulls raw data (a `.txt` file, a PDF, a CSV, or a webpage) into LangChain-compatible Document objects, that content is often too large, too unstructured, or too inconsistent for an LLM or embedding model to use directly. This is where the **Text Splitter** comes in — it takes those raw Document objects and breaks them down into smaller, well-formed chunks before they are embedded and stored in a vector database.

Without this step, embeddings would be generated from huge blocks of text, retrieval would return noisy and unfocused results, and the LLM would struggle to produce accurate, grounded answers. Text Splitters are what make the rest of the RAG pipeline actually usable.

---

## 1. What is Text Splitting?

> **Text Splitting** is the process of breaking large chunks of text (like articles, PDFs, HTML pages, or books) into smaller, manageable pieces (**chunks**) that an LLM can handle effectively.

```
                              ┌──────────────┐
                        ┌────▶│   Chunk 1    │
                        │     └──────────────┘
┌──────────────────┐    │     ┌──────────────┐
│                   │    │
│    Large Text     │────┼────▶│   Chunk 2    │
│                   │    │     └──────────────┘
└──────────────────┘    │     ┌──────────────┐
                        └────▶│   Chunk 3    │
                              └──────────────┘
```

A large document is fed in on the left, and the Text Splitter produces several smaller, independent chunks on the right — each small enough for a model to process, but still meaningful on its own.

---

## 2. Why Text Splitting Matters (Benefits)

**Overcoming model limitations:** Many embedding models and language models have maximum input size constraints. Splitting allows us to process documents that would otherwise exceed these limits.

**Downstream tasks:** Text Splitting improves nearly every LLM-powered task.

| Task | Why Splitting Helps |
|---|---|
| Embedding | Short chunks yield more accurate vectors |
| Semantic Search | Search results point to focused info, not noise |
| Summarization | Prevents hallucination and topic drift |

**Optimizing computational resources:** Working with smaller chunks of text can be more memory-efficient and allow for better parallelization of processing tasks.

---

## 3. Types of Text Splitters

```
                         ┌────────────────┐
                         │  Text Splitters │
                         └────────┬────────┘
              ┌───────────┬───────┴───────┬───────────────┐
              ▼           ▼               ▼               ▼
        ┌──────────┐┌──────────────┐┌──────────────┐┌───────────────┐
        │  Length  ││ Text         ││ Document      ││ Semantic      │
        │  Based   ││ Structure    ││ Structure     ││ Meaning       │
        │          ││ Based        ││ Based         ││ Based         │
        └──────────┘└──────────────┘└──────────────┘└───────────────┘
```

There are four broad families of text splitters, each using a different strategy to decide *where* to cut the text: **Length Based**, **Text Structure Based**, **Document Structure Based**, and **Semantic Meaning Based**. Below we cover the first two in detail.

---

## 4. Length Based Text Splitter

The simplest type of splitter. It splits text or Document objects purely based on a fixed **length** you provide — for example, if you set the length to 100 characters, the splitter will cut the text into chunks of 100 characters each, without any regard to sentence or paragraph boundaries.

- **Implementation used:** `CharacterTextSplitter`
- **How it works:** Counts characters (or tokens) and cuts once the limit is reached.

### Disadvantage

Because it only counts length, it completely ignores:
- Grammar
- Linguistic sentence structure
- Meaning / semantics

This means a chunk can end mid-sentence or even mid-word, splitting a single idea across two disconnected chunks. This problem is largely mitigated by **chunk overlap**.

### Chunk Overlap

**Chunk overlap** is the number of characters (or tokens) that consecutive chunks share in common.

When long text is divided into chunks, splitting a sentence or thought right down the middle can cause the model to lose key context. Chunk overlap creates a sliding window that duplicates a portion of text from the end of one chunk at the start of the next, preserving the semantic relationship between contiguous sentences.

```
Without Overlap:
[ Chunk 1: "Machine learning algorithms learn patterns from data." ]
[ Chunk 2: "They are widely used in predictive modeling applications." ]

With Overlap (e.g., overlap of ~20 characters):
[ Chunk 1: "Machine learning algorithms learn patterns from data." ]
[ Chunk 2: "learn patterns from data. They are widely used in predictive modeling applications." ]
```

By repeating a small window of text at the boundary, the model reading Chunk 2 still has the context from the end of Chunk 1, instead of starting cold.

#### Key Parameters & Best Practices

- **Recommended Ratio:** Set `chunk_overlap` to roughly **10% to 20%** of your `chunk_size` (e.g., `chunk_size=1000` with `chunk_overlap=150`).

---

## 5. Text Structure Based Text Splitter

This approach follows the idea that text naturally has an inherent **structure** — it is organized into paragraphs, sentences, words, and characters, in that order of granularity. Instead of blindly cutting at a fixed length, this splitter respects that structure and only breaks it down further when it has no other choice.

- **Implementation used:** `RecursiveCharacterTextSplitter`

`RecursiveCharacterTextSplitter` uses a predefined hierarchy of separators to keep semantically related text together for as long as possible. It only moves down to smaller units when a piece of text exceeds the `chunk_size`.

### Default Separator Order

```python
separators = ["\n\n", "\n", " ", ""]
```

1. **`"\n\n"` (Paragraphs):** Tries to split by double line breaks first, to keep whole paragraphs intact.
2. **`"\n"` (Lines/Sentences):** If a single paragraph is larger than `chunk_size`, it falls back to splitting line by line or sentence by sentence.
3. **`" "` (Words):** If a single line is still too long, it splits by spaces to keep complete words intact.
4. **`""` (Characters):** If an individual word exceeds `chunk_size`, it forces a hard split character by character, as a last resort.

This recursive, hierarchical fallback is what makes `RecursiveCharacterTextSplitter` the **recommended default** for most general-purpose text — it tries hard to preserve meaning before ever resorting to a raw character split.

---

## 6. Document Structure Based Text Splitter

This is essentially an **extension of Text Structure Based splitting**. Plain human language (English, Urdu, etc.) naturally breaks into paragraphs and sentences, so splitting by those units makes sense. But not all content is plain text — things like **Markdown files** or **code snippets** don't follow a "paragraph → sentence → word" structure. You can't just split code on periods or blank lines and expect it to stay meaningful.

So for this kind of content, we still use the same underlying idea as `RecursiveCharacterTextSplitter` — a hierarchy of separators — but the **separators themselves are different**, and tailored to the structure of the document type being split.

- For **Markdown**, separators are based on Markdown syntax — headers, code blocks, bullet points — instead of plain sentences.
- For **code** (Python, JS, etc.), separators are based on code structure — class definitions, function definitions, blocks — instead of paragraphs.

In short: same recursive splitting logic, different predefined separator set, chosen based on the format of the document.

---

## 7. Semantic Meaning Based Text Splitter

This is the most advanced approach. Instead of splitting by length or by structural separators, this method makes its splitting decision based on the **meaning (semantics)** of the text.

### How it works

1. An **embedding** is generated for each sentence.
2. The **similarity** between consecutive sentence embeddings is computed.
3. If two consecutive sentences are similar, they belong together and stay in the same chunk.
4. If similarity drops significantly, that point is treated as a natural **breakpoint** — a new chunk starts there.

### The key question: how low is "too low"?

The hard part is deciding how much of a similarity drop is significant enough to count as a breakpoint. Rather than picking an arbitrary fixed number, this is typically decided using statistical criteria over the distribution of similarity scores across the document, such as:

- **Standard Deviation** — e.g., mark a breakpoint wherever the similarity drop is **1 SD (or 2 SD) above** the average drop.
- **Percentile** — mark a breakpoint at similarity drops beyond a certain percentile threshold.
- **Interquartile Range (IQR)** — use the spread between the 25th and 75th percentile of similarity scores to flag unusually large drops as breakpoints.

This makes the splitting adaptive to each specific document, rather than using one fixed similarity threshold for everything.

> **Note:** Semantic Meaning Based splitting is currently available in LangChain's **experimental** library, since it's still under active development.

---

## Summary

| Splitter Type | Splits By | Cares About Meaning? | Best For |
|---|---|---|---|
| Length Based (`CharacterTextSplitter`) | Fixed character/token count | ❌ (mitigated by chunk overlap) | Simple, fast, generic splitting |
| Text Structure Based (`RecursiveCharacterTextSplitter`) | Paragraph → Line → Word → Character | ✅ | General-purpose default, preserves structure |
| Document Structure Based | Format-specific separators (Markdown headers, code blocks/functions) | ✅ | Markdown files, code snippets |
| Semantic Meaning Based | Embedding similarity between sentences (SD / percentile / IQR thresholds) | ✅✅ | Highest-quality semantic chunking (experimental in LangChain) |
