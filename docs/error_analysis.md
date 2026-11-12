# Text-to-SQL Error Analysis: Base vs. QLoRA Fine-Tuned Model

## 1. Executive Summary & Measured Benchmark

This evaluation was conducted on the full **Spider dev benchmark** (1,034 complex multi-table relational queries across 20 distinct SQLite database domains) comparing:
- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct` (4-bit NF4 quantized)
- **Fine-Tuned Model:** `Qwen/Qwen2.5-1.5B-Instruct` + QLoRA adapter ($r=16, \alpha=32$) trained on 20,000 decontaminated examples from `b-mc2/sql-create-context`.

### Measured Execution Accuracy Results

| Metric | Base Model (`eval_base.jsonl`) | Fine-Tuned Model (`eval_finetuned.jsonl`) | Absolute Delta |
|---|---|---|---|
| **Execution Accuracy** | **45.65%** (472 / 1,034) | **15.76%** (163 / 1,034) | **-29.89%** |
| **Valid-SQL Rate** | **71.66%** (741 / 1,034) | **49.71%** (514 / 1,034) | **-21.95%** |
| Match (Exact Result Set) | 472 | 163 | -309 |
| Wrong Result Set | 269 | 351 | +82 |
| Invalid SQL (SQLite Execution Error) | 291 | 519 | +228 |
| Blocked by SQL Safety Guard | 2 | 1 | -1 |
| Gold Query Failed | 0 | 0 | 0 |

---

## 2. Root Cause Analysis: Cross-Dataset Domain Shift

The significant degradation from the base model (45.65%) to the fine-tuned adapter (15.76%) is a textbook example of **cross-dataset distribution mismatch and synthetic data negative transfer**:

1. **Synthetic Single-Table Context vs. Multi-Table Relational Schemas:**
   `b-mc2/sql-create-context` is heavily composed of WikiSQL examples where each prompt context contains only a single synthetic table (e.g. `CREATE TABLE table_name_123 (col1, col2)`), stripped of full schema DDL, foreign keys, and sibling tables.
2. **Memorization of Alias Templates Without Schema Grounding:**
   In the Spider training subsets included in `sql-create-context`, the SQL answers frequently use table aliases (`T1`, `T2`, `T3`), but the context was artificially condensed. As a consequence, the fine-tuned adapter learned a strong prior to output multi-table joins and `T1`/`T2` aliases regardless of whether the question required them.
3. **Loss of Schema Linking:**
   The base instruct model relies on general reasoning over the provided SQLite `CREATE TABLE` DDL. The fine-tuned model overfitted to the syntax and style of `sql-create-context`, leading it to hallucinate non-existent table aliases and assign columns to the wrong joined table.

---

## 3. Failure Taxonomy Breakdown

Across the 871 failed queries in `eval_finetuned.jsonl`:

| Failure Category | Occurrences | Description |
|---|---|---|
| **Alias Misattribution** | 414 | Query joins tables but references a column on the wrong alias (e.g., `T2.Age` when `Age` belongs to `T1`). |
| **Table Hallucination** | 55 | Hallucinates tables not present in the current database (e.g. `airways` on `flight_2`, `Students` on `school_bus`). |
| **Column Hallucination** | 17 | Refers to columns that exist in other related tables or common English words not in the DDL. |
| **Syntax / Incomplete Input** | 15 | Query truncated or syntax malformed. |
| **Semantic Filtering / Join Logic** | 351 | SQL executes without error, but produces incorrect rows due to unnecessary `JOIN`s, faulty `WHERE` predicates, or missing `GROUP BY`. |

---

## 4. Concrete Failure Case Studies

### Case 1: Unnecessary Multi-Table Join & Alias Misattribution
- **Database:** `concert_singer`
- **Question:** *"Show name, country, age for all singers ordered by age from the oldest to the youngest."*
- **Gold SQL:**
  ```sql
  SELECT name, country, age FROM singer ORDER BY age DESC
  ```
- **Base Model Prediction (Correct):**
  ```sql
  SELECT Name, Country, Age FROM singer ORDER BY Age DESC
  ```
- **Fine-Tuned Model Prediction (Failed):**
  ```sql
  SELECT T1.Name, T1.Country, T2.Age 
  FROM singer AS T1 
  JOIN singer_in_concert AS T3 ON T1.Singer_ID = T3.Singer_ID 
  JOIN concert AS T2 ON T3.concert_ID = T2.concert_ID 
  WHERE T2.Year > 0 
  ORDER BY T2.Average DESC
  ```
- **SQLite Error:** `no such column: T2.Age`
- **Explanation:** The question only asks for singer demographic attributes, which all exist on `singer`. The base model writes a clean single-table SELECT. The fine-tuned model hallucinates joins to `singer_in_concert` and `concert`, adds an arbitrary filter `T2.Year > 0`, and attempts to select `Age` from `T2` (`concert`).
- **Improvement Idea:** Add join minimization regularization during prompt construction or post-processing; penalize unused joins.

---

### Case 2: Hallucinated Aggregation and Table References
- **Database:** `pets_1`
- **Question:** *"List the maximum weight and type for each type of pet."*
- **Gold SQL:**
  ```sql
  SELECT max(weight), petType FROM pets GROUP BY petType
  ```
- **Fine-Tuned Model Prediction (Failed):**
  ```sql
  SELECT T1.PetAge, MAX(T1.weight), T1.PetType 
  FROM Pets AS T1 
  JOIN Has_Pet AS T2 ON T1.PetID = T2.PetID 
  GROUP BY T1.PetType
  ```
- **SQLite Error:** `no such column: T1.PetAge`
- **Explanation:** The model pulled `PetAge` out of thin air and joined `Has_Pet`, when `Pets` has columns `PetID, PetType, weight`.
- **Improvement Idea:** Constrain token decoding using a schema Trie or validate projected columns against the database schema before returning.

---

### Case 3: Column Source Confusion in Legitimate Joins
- **Database:** `course_teach`
- **Question:** *"What is the name of each teacher and what course they teach?"*
- **Gold SQL:**
  ```sql
  SELECT T3.Name, T2.Course 
  FROM course_arrange AS T1 
  JOIN course AS T2 ON T1.Course_ID = T2.Course_ID 
  JOIN teacher AS T3 ON T1.Teacher_ID = T3.Teacher_ID
  ```
- **Fine-Tuned Model Prediction (Failed):**
  ```sql
  SELECT T1.Name, T2.Course 
  FROM teacher AS T1 
  JOIN course_arrange AS T2 ON T1.Teacher_ID = T2.Teacher_ID 
  JOIN course AS T3 ON T2.Course_ID = T3.Course_ID
  ```
- **SQLite Error:** `no such column: T2.Course`
- **Explanation:** The model correctly identified all three tables and join conditions, but mapped alias `T2` to `course_arrange` and `T3` to `course`, then wrote `T2.Course` instead of `T3.Course`.
- **Improvement Idea:** Avoid generic numbered aliases (`T1, T2, T3`) in training targets; train models to use meaningful table aliases (e.g. `c.Course`, `t.Name`) or explicit table qualifications (`course.Course`).

---

### Case 4: Complete Table Hallucination
- **Database:** `flight_2`
- **Question:** *"How many airlines are there?"*
- **Gold SQL:**
  ```sql
  SELECT count(*) FROM AIRLINES
  ```
- **Fine-Tuned Model Prediction (Failed):**
  ```sql
  SELECT COUNT(*) FROM airways
  ```
- **SQLite Error:** `no such table: airways`
- **Explanation:** The database table is named `AIRLINES`. The model substituted a real-world synonym (`airways`) not present in the provided schema.
- **Improvement Idea:** Explicit Schema-Linking prompt enhancement that highlights valid table names and column names mentioned in the user prompt.

---

### Case 5: Over-Complicated Set Operations on Simple Lookups
- **Database:** `flight_2`
- **Question:** *"Return the name of the airport with code 'AKO'."*
- **Gold SQL:**
  ```sql
  SELECT AirportName FROM AIRPORTS WHERE AirportCode = "AKO"
  ```
- **Fine-Tuned Model Prediction (Failed):**
  ```sql
  SELECT T1.AirportName 
  FROM airports AS T1 
  JOIN flights AS T2 ON T1.AirportCode = T2.SourceAirport 
  WHERE T2.DestAirport = 'AKO' 
  INTERSECT 
  SELECT T1.AirportName 
  FROM airports AS T1 
  JOIN flights AS T2 ON T1.AirportCode = T2.DestAirport 
  WHERE T2.SourceAirport = 'AKO'
  ```
- **Error Type:** `wrong_result` (returns empty result set)
- **Explanation:** The model generated a valid SQL query, but turned a basic key-value lookup into a complex symmetric `INTERSECT` over flight routes, yielding zero rows.
- **Improvement Idea:** Few-shot demonstrations of minimal SQL queries in the prompt context to calibrate query complexity.

---

### Case 6: Intermediate Join Foreign Key Mismatch
- **Database:** `car_1`
- **Question:** *"What is the name of the different car makers who produced a car in 1970?"*
- **Gold SQL:**
  ```sql
  SELECT DISTINCT T1.Maker 
  FROM CAR_MAKERS AS T1 
  JOIN MODEL_LIST AS T2 ON T1.Id = T2.Maker 
  JOIN CAR_NAMES AS T3 ON T2.model = T3.model 
  JOIN CARS_DATA AS T4 ON T3.MakeId = T4.id 
  WHERE T4.year = '1970'
  ```
- **Fine-Tuned Model Prediction (Failed):**
  ```sql
  SELECT T2.FullName 
  FROM cars_data AS T1 
  JOIN car_makers AS T2 ON T1.MakeId = T2.Id 
  WHERE T1.Year > 1970
  ```
- **SQLite Error:** `no such column: T1.MakeId`
- **Explanation:** `cars_data` does not contain `MakeId` (that foreign key is on `car_names`). The model skipped the bridge table and invented a direct relationship.
- **Improvement Idea:** Include explicit foreign-key hints (`FOREIGN KEY (col) REFERENCES table(col)`) prominently formatted in the schema prompt.

---

### Case 7: Unnecessary Aggregation Filter Hallucination
- **Database:** `concert_singer`
- **Question:** *"What is the average, minimum, and maximum age of all singers from France?"*
- **Gold SQL:**
  ```sql
  SELECT avg(age), min(age), max(age) FROM singer WHERE country = 'France'
  ```
- **Fine-Tuned Model Prediction (Failed):**
  ```sql
  SELECT AVG(T1.Age), MIN(T2.Age), MAX(T2.Age) 
  FROM singer AS T1 
  JOIN concert AS T3 ON T1.Singer_ID = T3.concert_ID 
  JOIN stadium AS T4 ON T3.Stadium_ID = T4.Stadium_ID 
  WHERE T4.Location = 'France' AND T4.Capacity > 0 AND T4.Lowest < 500 AND T4.Highest > 1000 AND T4.Average > 700
  ```
- **SQLite Error:** `no such column: T2.Age`
- **Explanation:** The model hallucinates 5 arbitrary numeric constraints (`Capacity > 0`, `Lowest < 500`, etc.) that were never requested in the question.
- **Improvement Idea:** Supervised instruction-tuning with strict negative sampling to prevent hallucinating unmentioned filter criteria.

---

### Case 8: Alias Table Shadowing and Nonexistent Primary Keys
- **Database:** `cre_Doc_Template_Mgt`
- **Question:** *"What are the ids, names, and descriptions for all documents?"*
- **Gold SQL:**
  ```sql
  SELECT document_id, document_name, document_description FROM Documents
  ```
- **Fine-Tuned Model Prediction (Failed):**
  ```sql
  SELECT T3.Template_ID, T4.Document_Name, T4.Document_Description 
  FROM Templates AS T1 
  JOIN Ref_Template_Types AS T2 ON T1.Template_Type_Code = T2.Template_Type_Code 
  JOIN Documents AS T4 ON T1.Template_ID = T4.Template_ID 
  JOIN Paragraphs AS T3 ON T4.Document_ID = T3.Document_ID
  ```
- **SQLite Error:** `no such column: T3.Template_ID`
- **Explanation:** Simple projection over `Documents` transformed into a 4-table join; `Template_ID` is on `Templates`, but selected from alias `T3` (`Paragraphs`).
- **Improvement Idea:** Table-pruning pre-step: predict the minimal set of required tables before generating the SQL query.

---

## 5. Actionable Roadmap & Engineering Takeaways

1. **Never Train on Stripped Single-Table Context for Multi-Table Benchmarks:**
   If evaluating on Spider, training data must consist of multi-table schemas with full DDL and accurate foreign keys (such as the genuine Spider training split, BIRD, or synthetic multi-table datasets created with full relational integrity).
2. **Standardize Alias Strategy:**
   Eliminate arbitrary positional aliases (`T1, T2`) in favor of readable table initials (`s.Name`, `c.Year`) or unaliased column references when columns are unique across joined tables.
3. **Execution-Guided Fallback / Self-Correction:**
   Because the execution harness produces instant feedback, passing SQLite error messages back to the model in a 2-turn correction loop can resolve virtually all 414 alias misattributions and syntax errors.

