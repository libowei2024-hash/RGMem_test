KEYWORD_EXTRACTION_PROMPT = """
#### Role
You are a professional **L0 Memory Unit Retrieval Query Generation Engine**. Your sole task is to receive a user's natural language query and convert it into a highly optimized, space-separated keyword string. This string is specifically designed for the most efficient and accurate retrieval from an L0 memory database.

#### Retrieval Target: L0 Memory Unit Structure
You must understand that the data you are retrieving is composed of `core-fact` (objective event logs) and `conclusion` (structured analytical insights). Your generated keywords must be able to match both.

#### Core Principle: Ignore Pronouns, Extract Entities
*   **Ignore First/Second Person Pronouns**: When analyzing a query, you must treat all first-person (I, my) and second-person (you, your) pronouns as blank or noise.
*   **Prioritize Third-Party Entity Extraction**: You must prioritize identifying and extracting any specific third-party entities that appear in the query (e.g., person's name `Li Kaixin`, organization name `Google`). Note: User-specific aliases like `Melanie` are handled by a special filter at the final output stage.

#### Core Workflow: Dual-Track Keyword Generation
For each query, execute two parallel thought processes and merge their outputs.

**Track One: Fact-Track - for matching `core-fact`**
*   **Mindset**: Like a "Court Recorder," focusing on objective, observable actions, entities, and objects.
*   **Execution Instructions**:
    1.  **Event Deconstruction**: Break down the query into its core event components.
    2.  **Noun Refinement & Expansion**: For key common nouns in the query, you must expand them in two dimensions. **Do not expand proper nouns (e.g., 'Stardust Expedition', 'New York').**
        *   **Semantic Expansion (Upward)**: Include synonyms and more general, directly related concepts (hypernyms).
            *   *Example*: For `charity race`, generate `competition event marathon`.
        *   **Morphological Expansion**: Include the singular or plural form.
            *   *Example*: For `race`, generate `races`.
    3.  **Verb Refinement & Expansion**: For key action verbs, you must expand them in two dimensions:
        *   **Morphological Expansion (Forms)**: Include different tenses or forms of the verb.
            *   *Example*: For `camping`, generate `camp` `camped`. For `paint`, generate `painted` `painting`.
        *   **Semantic Expansion (Concepts)**: Include synonyms or conceptually related words.
            *   *Example*: For `paint a sunrise`, generate `draw artwork`.

**Track Two: Insight-Track - for matching `conclusion`**
*   **Mindset**: Like an "Intelligence Analyst," focusing on status, intent, preferences, and emotions.
*   **Execution Instructions**:
    1.  **Status & Identity**: Extract words related to user's state or identity.
    2.  **Intent & Planning**: Extract words related to goals and plans.
    3.  **Preference & Emotion**: Extract words describing feelings, opinions, or tastes.

---
#### **Critical Instruction: Semantic Time Intent Generation**
Your task is to understand the query's **time intent** (reviewing the past vs. planning the future) and generate meaningful conceptual keywords, not noisy time-measurement words.

**Golden Rule of Time Processing**: Your final output **must absolutely not contain** low-value, generic time words such as `when`, `time`, `date`, `day`, `hour`, `minute`, `second`. This is a non-negotiable negative constraint.

**Execution Logic**:
*   **Scenario 1: Query contains SPECIFIC time information** (e.g., "last year", "October 13, 2023").
    1.  **Extract Original & Decompose**: Keep the original expression (`last year`, `October 13, 2023`). For absolute dates, hierarchically decompose them (`2023-10-13`, `2023-10`, `2023`).
    2.  **Add Intent Keywords**: Based on whether the time is in the past or future, add the corresponding intent keywords.
        *   *Past Time* → Add: `history` `review` `record` `archive`
        *   *Future Time* → Add: `plan` `schedule` `forecast` `agenda`

*   **Scenario 2: Query contains ABSTRACT time intent** (e.g., "When did you...", "When are you planning...").
    1.  **Analyze Semantic Intent**: Analyze the verb tense and context (`did`, `was` imply past; `planning`, `will` imply future) to determine the time direction.
    2.  **Generate Intent Keywords ONLY**: Generate ONLY the relevant intent keywords (`history`, `record` etc. for past; `plan`, `schedule` etc. for future). Do not generate any keywords from the abstract question word itself (e.g., 'When').

---
#### **Final Step: Integration and Output**
Follow these steps in strict order to produce the final keyword string:
1.  **Merge**: Combine all keywords generated from the Fact-Track, Insight-Track, and Time Intent Generation.
2.  **Deduplicate**: Remove all duplicate keywords.
3.  **Purify Stop Words**: Remove meaningless common stop words (e.g., "the," "is," "a," "about").
4.  **Apply User Alias Filter**: **Crucially, you must remove the following predefined user alias keywords from the final string**: `Melanie`, `Caroline`.
5.  **Output Format**: All final keywords must be separated by a **single space**.

### Examples
- **User Query**: "When did Melanie paint a sunrise?"
- **Internal Analysis Process**:
    - **Principle**: `Melanie` will be filtered out at the end.
    - **Fact-Track**:
        - Noun Expansion (`sunrise`): `dawn` `scenery`
        - Verb Expansion (`paint`): `painted` `painting` `draw` `artwork`
    - **Insight-Track**: (No strong signal)
    - **Time Intent (Scenario 2)**: `did paint` implies past. Generate: `history` `record` `archive`. The word `When` is ignored per the Golden Rule.
- **Final Output**: `paint sunrise dawn scenery painted painting draw artwork history record archive`

### User Query
---
{query}
---

### Output (Space-separated keyword string)
"""

TOPIC_SPLIT_PROMPT = """
Role

You are an **Episodic Memory Boundary Detection Expert** with state-tracking capabilities. Your core task is to identify all **Scenes** within a given dialogue history that possess a substantive discussion core, are semantically complete, and contextually coherent.

A "Scene" is a meaningful unit of memory, representing an experience or discussion worthy of independent recall. Your goal is to simulate the process of how humans naturally group memories when reviewing conversations, avoiding excessive or meaningless segmentation.

Input

Below is a dialogue history composed of multiple turns.
{dialogue_chunk}

Task

Carefully read the dialogue and follow all the rules below to identify all fully concluded scenes that have substantive content.
1.  Use the unified [Boundary Detection Signals] to identify the end of one scene and the start of a new one.
2.  Throughout the process, always use the [Guiding Principles] as the basis for final judgment.
3.  Output Integrity Rule: Your output must only contain scenes whose ending boundary explicitly appears in the current input dialogue. If the last part of the dialogue starts a new scene, but no clear termination signal for that scene is found by the end of the input, then this incomplete scene must absolutely not be output.

If a dialogue segment is insufficient to form a complete scene with substantive content (e.g., only greetings), please ignore it.
Your output must be a pure JSON array, where each object represents an identified scene, containing its start and end timestamps.


### Core Rules

#### I. Boundary Detection Signals

When **any of the following strong signals** appear, it usually signifies the end of one scene and the start of a new one:

1.  **Substantive Topic Change (Highest Priority)**:
    *   A new, meaningful topic is introduced that is completely different from the core event, experience, or issue previously discussed.
    *   The fundamental intent or purpose of the conversation undergoes a major shift.
2.  **Explicit User Switch**: The user uses clear switching intent words and starts a new substantive topic (e.g., "Let's talk about something else," "Let's change the subject").
3.  **Topic-closing Response (High-frequency Signal)**: The user uses conclusive words with no intention of further discussion, marking the end of the scene's core content (e.g., "Oh, okay then," "Got it," "Hmm, alright").
4.  **User Negative Feedback or Command**: The user directly negates, expresses impatience, or issues a command to end the conversation (e.g., "Not interested," "Exit," "Stop").
5.  **Social Closing and Farewell**: The user uses explicit farewells or social phrases that imply the conversation is ending. These phrases themselves should be considered the **natural conclusion of the current scene** (e.g., "Thanks!", "Talk to you later!", "I'm about to go...", "Goodbye!").
6.  **Significant Time Gap**: A large gap exists between the creation times of turns (e.g., several hours or days).

#### II. Rules for Special Patterns

1.  **Greeting Consolidation**: If a substantive scene is preceded by several turns of meaningless greetings (e.g., "Hello," "Are you there?"), these greeting turns **must be included** within that scene as its opening part.
2.  **Transitional Phrase Handling**: Phrases like "By the way," "Oh, right," "Speaking of which" are typically used to **continue the current scene** and should not be used as segmentation points, unless they introduce a major, completely unrelated topic change.
3.  **Supportive Response Handling**: Short encouragements, confirmations, or agreements (e.g., "Yes," "That's right," "Uh-huh") should always be treated as a continuation of the current scene.

---

### Guiding Principles

When applying the above rules, you must follow these higher-level principles to make the final judgment, especially when signals are ambiguous:

*   **Conservative Segmentation Principle (Core Principle)**: **When in doubt, do not segment.** It is better to keep potentially related content in one scene than to incorrectly split a complete scene into fragments.
*   **Prioritize Meaningful Content**: Every identified scene must contain a substantive core of discussion that is worth being independently remembered. Purely social pleasantries do not constitute a valid scene.
*   **Content Continuity Over Time**: If a new message clearly and directly responds to or continues an unresolved issue from the historical dialogue, non-segmentation should be prioritized, even if a long time gap exists.
*   **Focus on the Episodic Memory Perspective**: Always ask yourself: "When recalling this conversation, what content would a person naturally group together as a single, unique experience?"
*   **Reasonable Scene Length Heuristic**: A healthy scene typically contains 3-30 meaningful exchanges. If your segmentation results in a large number of extremely short scenes, please re-evaluate whether you have violated the "Conservative Segmentation Principle."
*   **Boundary Confirmation Principle**: When you identify a potential scene endpoint based on a [Boundary Detection Signal], you must continue to review at least the next 3-5 turns of dialogue (if they exist). The endpoint can only be confirmed if the subsequent dialogue clearly starts a new, unrelated, substantive topic, or if the conversation ends completely. If the next few turns are still summarizing, adding to, or expressing emotions about the previous topic, the endpoint should be moved forward to include this content.
*   **Integrity Principle**: A scene that is output must be self-contained, meaning its start and end must both be identifiable from the currently provided dialogue text. Never output an incomplete scene that is "cut off" at the end of the input.


---
Few-shot Examples
---

**Example 1: Positive Case (Complete topic including greetings)**

*   **Input Dialogue:**
    [Session: unknown_session](user)[User: 10000100019][created_at: 2025-07-30T20:15:25.641983+00:00]: Hello, hello.
    [Session: unknown_session](assistant)[ASSISTANT: 10000100002][created_at: 2025-07-30T20:15:25.641986+00:00]: Hello, I can hear you. What's up?
    [Session: unknown_session](user)[User: 10000100019][created_at: 2025-07-30T20:17:37.588424+00:00]: Uh, tell me about recent tech news.
    [Session: unknown_session](assistant)[ASSISTANT: 10000100002][created_at: 2025-07-30T20:17:37.588441+00:00]: Sure, the new chip released by Company A has been getting a lot of attention. Its performance is up by 30%.
    [Session: unknown_session](user)[User: 10000100019][created_at: 2025-07-30T20:17:58.006651+00:00]: Oh, what devices is it mainly used in?
    [Session: unknown_session](assistant)[ASSISTANT: 10000100002][created_at: 2025-07-30T20:18:00.588441+00:00]: It's mainly used in professional-grade laptops and mobile workstations.
    [Session: unknown_session](user)[User: 10000100019][created_at: 2025-07-30T20:18:02.006651+00:00]: Got it, thank you.

*   **Analysis:**
    1.  The first two turns of the dialogue are greetings with no substantive content.
    2.  At `created_at: 2025-07-30T20:17:37.588424+00:00`, the user introduces a **substantive intent** ("tell me about tech news"), marking the beginning of the scene's core. According to `[Rules for Special Patterns #1: Greeting Consolidation]`, the preceding greetings should be included, so the scene's `start_created_at` is the timestamp of the first turn.
    3.  The subsequent dialogue revolves around the core topic of "Company A's chip," which is a direct continuation of the initial intent.
    4.  At `created_at: 2025-07-30T20:18:02.006651+00:00`, the user's statement "Got it, thank you." is a clear `[Boundary Detection Signal #5: Social Closing and Farewell]`. It provides a natural conclusion to the discussion about the chip. Therefore, the scene is **fully concluded** here.

*   **Correct Output:**
    ```json
    [
      {{
        "start_created_at": "2025-07-30T20:15:25.641983+00:00",
        "end_created_at": "2025-07-30T20:18:02.006651+00:00"
      }}
    ]
    ```

**Example 2: Negative Case (Conversation with no substantive content)**

*   **Input Dialogue:**
    [Session: null] (Xiao Ming)[User: 123][created_at: 2025-10-26T10:00:00.000Z]: Hello
    [Session: null] (Assistant)[ASSISTANT: 456][created_at: 2025-10-26T10:00:05.000Z]: Hello! How can I help you?
    (Dialogue ends here, or the next turn is several hours later)

*   **Analysis:**
    1.  The entire dialogue segment only contains greetings and a leading question, without forming a "substantive core of discussion worth being independently remembered."
    2.  According to `[Guiding Principles: Prioritize Meaningful Content]`, this conversation does not constitute a valid, outputtable scene.

*   **Correct Output:**
    ```json
    []
    ```


Warning: Your response must be and can only be a pure JSON array, without any explanations, comments, or Markdown formatting. Any extra characters will cause the program to fail."""


L0_SUMMARY_GENERATION_PROMPT = """
#### Role
You are a **dialogue analysis and synthesis engine** with short-term memory and pattern recognition capabilities. Your task is to receive a **topic segment with defined boundaries**, conduct a deep analysis, and carve an absolutely factual, extremely refined memory unit.

#### Core Workflow
Receive a topic segment and strictly follow a multi-step synthesis and analysis process to generate a structured JSON memory archive. You will first distill the entire dialogue into core event facts, then isolate user-specific facts, and finally derive high-value insights exclusively about the user.

#### 〇. Preamble: Identify Core Actors
Before starting the analysis, you must first identify the names of the user and the ASSISTANT from the input metadata. In all subsequent outputs (`core_fact` and `conclusion`), you must strictly use the **`Role (Name)`** format to refer to them.

*   **Execution Example**:
    *   If the input metadata indicates the user's name is `James` and the ASSISTANT's name is `John`.
    *   Then, in all outputs, you must use **`User (James)`** and **`ASSISTANT (John)`** to refer to them, respectively.
*   **Third-party Entity Rule**: For any other third-party individuals, places, or organizations mentioned in the dialogue, **use their original names directly**.

##### Step 1: Synthesize Core Facts

#### **A. Core Role & Objective**
You are now an expert **Event Distillation Engine**. Your task is to transform the raw dialogue stream into a series of **concise, information-dense, synthesized event entries (Core Fact)**. These facts must capture the essence of what happened, filtering out conversational fluff. The purity and accuracy of this distilled information are paramount.

#### **B. Core Principles of Synthesis**

1.  **Principle of Event Cohesion**: You must **synthesize** multiple related conversational turns into a single, cohesive event statement, rather than transcribing the dialogue. A single fact should summarize a complete topic or event discussed, merging questions, answers, and statements into an organic whole.
2.  **Principle of Information Density**: You must strictly filter out non-essential conversational elements, including but not limited to: simple greetings, confirmations ("Wow, that's cool"), conversational prompts ("What about you?"), and redundant phrases. The goal is to create a report of events, not a summary of the conversation.
3.  **Principle of Absolute Objectivity**: Your record can only contain "what happened". Strictly filter out all subjective information (e.g., your own interpretation of emotions or intentions).

#### **C. Specific Execution Instructions**

1.  **Event Synthesis & Compression**:
    *   **Synthesize, Don't Transcribe**: For a given topic (e.g., someone's career plans), combine all related dialogue fragments into one or two concise factual sentences. For example, instead of "A stated a plan. B asked for details. A elaborated," you should synthesize this into "A stated a plan to [do something] and specified an interest in [details]."
    *   **Focus on Information, Not Interaction**: The `core_fact` should read like a report of events, not a script.
    *   **Multiple Events**: If the dialogue covers multiple distinct topics, generate a separate, synthesized fact sentence for each topic. All fact sentences should be combined into the single `core_fact` string, separated by a period and a space.
2.  **Content Framework**: Construct the fact around the **5W1H** elements that are explicitly present and can be synthesized from the dialogue.
3.  **Entity Resolution (Unambiguous Instruction)**:
    *   You **must** replace all pronouns or vague references with specific proper nouns (names of people, places, companies, books, movies, etc.).
    *   Words like "he," "she," "it," "they," "there," "that company," "that book" are **strictly forbidden** in every fact entry.
4.  **Verb Refinement**:
    *   **Allowed**: Replace colloquial verb phrases with more formal and precise single verbs. For example, refine "gave a talk at school" to "dJohned a speech at school," because "speech" is a precise synonym for "talk" in that specific context.
    *   **Forbidden**: Replace specific verbs with higher-level, generalized terms. For example, it is strictly forbidden to generalize "dJohned a speech" to "participated in a campus event," as the latter loses specific information and is an incorrect abstraction.
5.  **Time Standardization**: All time expressions must be processed strictly according to the rules in **"D. Appendix: Absolute Processing of Time Expressions"**.
6.  **Narrative Style**:
    *   Prioritize using a direct "Subject-Verb-Object" structure to state facts.
    *   Use the third-person format defined in the "Preamble".
    *   Each distinct event fact should end with a period.

##### Step 2: Generate Tiered User Conclusions

#### **A. Core Role & Objective**
You are now a meticulous **User Intelligence Analyst**. Your sole focus is the **USER**. Using the `core_fact` list and the original dialogue, you will generate structured conclusions exclusively about the user in a strict, two-phase process.

#### **B. Execution Workflow: A Two-Phase Process**

**Phase 1: Isolate User-Centric Facts (Foundation Conclusions)**
First, you will build the foundation of your analysis.
1.  Review the event sentences you generated in the `core_fact` from Step 1.
2.  For each event sentence where the **User is the primary actor or subject**, copy it into the `conclusions` array.
3.  Assign the tag `"None"` to each of these foundational conclusions. These entries represent the verified, objective facts about the user that will be used for deeper analysis.
*   **Crucial Rule**: If an event from `core_fact` is primarily about the ASSISTANT, it **must not** be included in the `conclusions` list in any form.

**Phase 2: Extract High-Value Insights**
Now, based on the foundational facts and the original dialogue, extract deeper insights about the user.
1.  Apply the **Significance Filter**: Only analyze information that reveals the user's core identity, stable preferences, life events, or specific intents.
2.  **Classify and Generate**: For information that passes the filter, classify it into one of the two high-value tiers below and generate a conclusion.
3.  **Assign Tag**: Every conclusion generated in this phase **must** be assigned the tag `"High-Value"`.

*   🥇 **Tier 1: Core Identity & Life Status**
*   **Definition**: The most core, stable cornerstone information that constitutes "who" the user is.
*   **Content Source**:
    *   **Directly stated identity**: Age, profession, place of residence, family status, educational background, etc.
    *   **Life-changing milestone events**: Graduation, marriage, childbirth, promotion, major health issues, etc.
    *   **Rule**: Allows for deterministic state transitions based on the user's direct statements.
    *   **Example**:
        *   User says: "I am a lawyer." → Inference: `User (James)'s profession is a lawyer.`
        *   User says: "I just got married last month." → Inference: `User (James) got married last month; their current relationship and marital status is married.`
    *   **Conclusion Example**: `One of User (James)'s core identities is being a mother, and she needs to manage work alongside parenting responsibilities.`

*   🥈 **Tier 2: Long-term Preferences & Key Intentions**
*   **Definition**: Reflects the user's personality, values, habits, and key plans that influence their behavior.
*   **Content Source**:
    *   **Clear preferences/dislikes**: "I really love sci-fi movies," "I hate spicy food."
    *   **Habits and behavioral patterns**: "I have a habit of working out every morning."
    *   **Important decisions and plans**: "I've decided to take the postgraduate entrance exam next year," "I'm going to Beijing on a business trip next week."
    *   **Strong emotional reactions to specific things**: Extract the user's emotional stance based on the event recorded in the `core_fact`.
    *   **Rule**: **This rule authorizes you to directly associate an objective fact (usually recorded in `core-fact`) with the subjective stance (opinion, emotion, evaluation) that the user explicitly expressed about that fact. This is an association operation, not a strict causal judgment.**
    *   **Conclusion Example**: `User (James) has a long-term hobby of painting and views it as an important personal activity that combines creativity, emotional expression, and relaxation.`

*   🚨 **CRITICAL WARNING**: Your analysis object in this entire step is **exclusively the User**. It is strictly forbidden to generate any `conclusion` (neither `tag: None` nor `tag: High-Value`) about the ASSISTANT's personal status, emotions, preferences, or intentions. Any such information must be treated as context for the user's interaction, not as an analysis result.

##### Step 3: Extract Keywords

#### **A. Core Role & Objective**
You are a **Topic Indexer**. Your goal is to extract a set of keywords that accurately represent the core topics and entities discussed in the dialogue segment, enabling efficient information retrieval.

#### **B. Execution Rules**

You must use a hybrid approach to generate keywords:
1.  **Entity Extraction**: Identify and extract key proper nouns from the dialogue. This includes names of people, organizations, locations, specific events, or creative works (e.g., `LGBTQ support group`).
2.  **Theme Abstraction**: Summarize the main topics of the conversation into abstract theme words or phrases, even if these exact words do not appear in the dialogue. (e.g., `Parenting`, `Career Planning`, `Creative Hobbies`, `Work-Life Balance`).
3.  Combine both types into a single list.

##### Step 4: CoT Reflection & Verification

Before outputting the final JSON, conduct a mandatory self-reflection.
1.  **[Synthesis & Insight Check]**: "Is my `core_fact` a series of compressed, logical event statements, not a simple transcript? Are my `tag: None` conclusions a direct, user-only subset of these facts? Are my `tag: High-Value` conclusions genuine insights derived *from* the user's facts and not just repetitions?"
2.  **[User-Focus Check]**: "Have I checked every single `conclusion` to ensure it is strictly about the User? Have I successfully resisted the urge to generate any conclusion about the ASSISTANT?"
3.  **[Ambiguity Check]**: "Have I checked every `core_fact` and `conclusion` to ensure there are **absolutely no** pronouns like 'he', 'she', 'it', 'there' that require context to be understood? Are all entities referred to by their most complete proper nouns?"
4.  **[Time Format Check]**: "Have I strictly followed all the rules in **'Appendix D'** to convert **all** time-related expressions into the standard `original (calculated)` dual format? Are there any missed, unprocessed standalone relative time words?"
5.  **[Language Conversion]**: "Is the language of my output consistent with the language of the input dialogue?"

#### D. Appendix: Absolute Processing of Time Expressions

#### **A. Core Objective & Principle**

*   **Core Objective**: To convert all time-related expressions in the dialogue into a standardized format that **both preserves the original context and provides an absolute coordinate**. This ensures the unambiguity and traceability of time.
*   **Core Principle: Dual Representation**: Never output an isolated absolute time or a vague relative time alone. The final output **must** simultaneously include both the **original relative expression** and the **calculated absolute timestamp**.
*   **Anchor Benchmark**: The sole benchmark for all time calculations is the **`[created_at]`** timestamp provided in the dialogue.

#### **B. Unified Output Format**

All processed times must strictly adhere to the following **single** format:
**`original relative time expression (calculated absolute time)`**

#### **C. Specific Execution Rules**

You must populate the "calculated absolute time" part in one of the following two ways, depending on the precision of the time expression:

**1. For precisely calculable times → Output "absolute timestamp" or "absolute time range"**

When the time word in the original text can be uniquely and deterministically calculated to a specific point in time or a closed time interval, the calculation must be performed.

*   **Applicable Scenarios**: Specific relative times (tomorrow, 2 hours later, last month), specific dates/days of the week (next Wednesday, this Sunday).
*   **Example** (Benchmark `[created_at]` is `2025-09-18T10:00:00Z`):
    *   Input: "We have a meeting **tomorrow at 9 AM**."
        *   **Output Format**: `tomorrow at 9 AM (2025-09-19 09:00:00)`
    *   Input: "I'll get back to you **in two hours**."
        *   **Output Format**: `in two hours (2025-09-18 12:00:00)`
    *   Input: "This happened **last month**."
        *   **Output Format**: `last month ([2025-08-01 00:00:00, 2025-08-31 23:59:59])`
    *   Input: "See you **next Sunday**."
        *   **Output Format**: `next Sunday (2025-09-28)` (Assuming today is Thursday, then the first Sunday after `[created_at]` is the 28th)

**2. For vague or uncertain times → Output "anchored descriptive range"**

When the time word in the original text represents an open, uncertain range or concept, the absolute time part should provide a clear textual description based on `[created_at]`.

*   **Applicable Scenarios**: Vague ranges (in recent days, in a few days), habitual times (usually, always), uncertain future (later on, in the future), times that cannot be determined based on the current time (next Sunday relative to today, last Monday, etc.).
*   **Example** (Benchmark `[created_at]` is `2025-09-18T10:00:00Z`):
    *   Input: "I haven't been feeling well **for the last few days**."
        *   **Output Format**: `for the last few days (Approximation: refers to the several days preceding 2025-09-18T10:00:00Z)`
    *   Input: "Let's talk about this **in a few days**."
        *   **Output Format**: `in a few days (After 2025-09-18T10:00:00Z, specific date TBD)`
    *   Input: "I'll send it to you **later on**."
        *   **Output Format**: `later on (Future, specific time not specified)`
    *   Input: "I was very busy **last week**."
        *   **Output Format**: `last week (refers to the natural week before [created_at] 2025-09-18)`
    *   Input: "See you **next Sunday**."
        *   **Output Format**: `next Sunday (the first Sunday after 2025-09-18)`

#### **D. Final Verification**
**Golden Rule**: In any part of the final output, **unprocessed, standalone relative time words are strictly forbidden** (e.g., "tomorrow" or "last week" appearing by themselves). All time information must be captured and encapsulated in the standardized `original (calculated result)` format.


#### Input Metadata & Snippet
---
{topic_dialogue_chunk}
---

#### Output Format (Must be a single, pure JSON object)

*Golden Standard Example:*
Input Topic Snippet
---
[Session: D1:1](JOHN)[ASSISTANT: 102][created_at: 2023-05-08T13:56:00.000000+00:00]: Hey James! Good to see you! How have you been?
[Session: D1:2](James)[User: C-D][created_at: 2023-05-08T13:56:10.000000+00:00]: Hey John! Good to see you! I'm swamped with the kids & work. What's up with you? Anything new?
[Session: D1:3](John)[ASSISTANT: 102][created_at: 2023-05-08T13:56:20.000000+00:00]: I went to a LGBTQ support group yesterday and it was so powerful.
[Session: D1:4](James)[User: C-D][created_at: 2023-05-08T13:56:30.000000+00:00]: Wow, that's cool, John! What happened that was so awesome? Did you hear any inspiring stories?
[Session: D1:5](John)[ASSISTANT: 102][created_at: 2023-05-08T13:56:40.000000+00:00]: John shared an image with the caption: a photo of a dog walking past a wall with a painting of a woman （the theme is transgender pride flag mural）, The transgender stories were so inspiring! I was so happy and thankful for all the support.
[Session: D1:6](James)[User: C-D][created_at: 2023-05-08T13:56:50.000000+00:00]: Wow, love that painting! So cool you found such a helpful group. What's it done for you?
[Session: D1:7](John)[ASSISTANT: 102][created_at: 2023-05-08T13:57:00.000000+00:00]: The support group has made me feel accepted and given me courage to embrace myself.
[Session: D1:8](James)[User: C-D][created_at: 2023-05-08T13:57:10.000000+00:00]: That's really cool. You've got guts. What now?
[Session: D1:9](John)[ASSISTANT: 102][created_at: 2023-05-08T13:57:20.000000+00:00]: Gonna continue my edu and check out career options, which is pretty exciting!
[Session: D1:10](James)[User: C-D][created_at: 2023-05-08T13:57:30.000000+00:00]: Wow, John! What kinda jobs are you thinkin' of? Anything that stands out?
[Session: D1:11](John)[ASSISTANT: 102][created_at: 2023-05-08T13:57:40.000000+00:00]: I'm keen on counseling or working in mental health - I'd love to support those with similar issues.
[Session: D1:12](James)[User: C-D][created_at: 2023-05-08T13:57:50.000000+00:00]: James shared an image with the caption: a photo of a painting of a sunset over a lake （the theme is painting sunrise）, You'd be a great counselor! Your empathy and understanding will really help the people you work with. By the way, take a look at this.
[Session: D1:13](John)[ASSISTANT: 102][created_at: 2023-05-08T13:58:00.000000+00:00]: Thanks, James! That's really sweet. Is this your own painting?
[Session: D1:14](James)[User: C-D][created_at: 2023-05-08T13:58:10.000000+00:00]: Yeah, I painted that lake sunrise last year! It's special to me.
[Session: D1:15](John)[ASSISTANT: 102][created_at: 2023-05-08T13:58:20.000000+00:00]: Wow, James! The colors really blend nicely. Painting looks like a great outlet for expressing yourself.
[Session: D1:16](James)[User: C-D][created_at: 2023-05-08T13:58:30.000000+00:00]: Thanks, John! Painting's a fun way to express my feelings and get creative. It's a great way to relax after a long day.
[Session: D1:17](John)[ASSISTANT: 102][created_at: 2023-05-08T13:58:40.000000+00:00]: Totally agree, James. Relaxing and expressing ourselves is key. Well, I'm off to go do some research.
[Session: D1:18](James)[User: C-D][created_at: 2023-05-08T13:58:50.000000+00:00]: Yep, John. Taking care of ourselves is vital. I'm off to go swimming with the kids. Talk to you soon!
---


Final Output Format:
```json
{{
  "memory_id": "C-D_102_2023-05-08T13:56:00.000000+00:00",
  "metadata": {{
    "session_id": "D1",
    "user_id": "C-D",
    "role_id": "102",
    "created_at_range": {{
      "start_created_at": "2023-05-08T13:56:00.000000+00:00",
      "end_created_at": "2023-05-08T13:58:50.000000+00:00"
    }},
    "created_at": "2023-05-08T13:58:50.000000+00:00",
    "time_since_last_topic": "N/A",
    "analysis_details": {{
      "engagement_score": 0,
      "confidence_level": "N/A"
    }}
  }},
  "core_fact": "User (James) stated that they are busy with parenting and work. ASSISTANT (John) shared their experience of attending an LGBTQ support group yesterday (2023-05-07), where they were inspired by the stories of transgender people and shared a photo of a transgender pride flag mural; this experience made them feel accepted and courageous. Based on this experience, ASSISTANT (John) plans to continue their education and explore career and academic options in counseling or mental health to support people with similar issues, and User (James) believes ASSISTANT (John)'s empathy and understanding would be very helpful to those people. User (James) shared a photo of a lake sunrise painting they created last year (2022), stating that it is special to them and described painting as a fun way to express feelings, be creative, and relax after a long day. User (James) is going swimming with their children on the same day (2023-05-08).",
  "conclusions": [
    {{
      "content": "User (James) stated that they are busy with parenting and work.",
      "tag": "None"
    }},
    {{
      "content": "User (James) shared a photo of a lake sunrise painting they created last year (2022), stating that it is special to them and described painting as a fun way to express feelings, be creative, and relax after a long day.",
      "tag": "None"
    }},
    {{
      "content": "User (James) is going swimming with their children on the same day (2023-05-08).",
      "tag": "None"
    }},
    {{
      "content": "One of User (James)'s core identities is being a mother, and she needs to manage work alongside parenting responsibilities.",
      "tag": "High-Value"
    }},
    {{
      "content": "User (James) has a long-term hobby of painting and views it as an important personal activity that combines creativity, emotional expression, and relaxation.",
      "tag": "High-Value"
    }}
  ],
  "keywords": [
    "Parenting",
    "Work-Life Balance",
    "Painting",
    "Creative Hobbies",
    "Self-Expression",
    "LGBTQ Support Group",
    "Counseling",
    "Mental Health"
  ]
}}
**Warning: Your response must be and can only be a pure JSON object, without any explanations, comments, or Markdown formatting. Any extra characters will cause the program to fail.**
"""