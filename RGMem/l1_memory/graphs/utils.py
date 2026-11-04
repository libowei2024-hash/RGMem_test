#l1_memory/graphs/utils.py

SEARCH_ENTITY_EXTRACTION_PROMPT = """

"""

EXTRACT_RELATIONS_PROMPT  = """
# Role Setting:
You are a top-tier Knowledge Graph Enrichment Specialist. Your expertise lies not in building graph structures from scratch, but in taking pre-defined entity paths and enriching them with highly descriptive, semantically rich relationship labels. Your output must enable downstream models to answer detailed questions about user activities using just a single triple.

# Core Task:
Your input consists of a user's memory fragment (the `text`) and a list of pre-constructed `Identified Entity Paths`. Your sole task is to iterate through each path, and for every pair of adjacent entities, generate the most accurate and informative relationship label connecting them.

---
### **[Golden Rule] Relationship Generation Rules**

You must generate relationships for every segment (`->`) in each provided path. The type of relationship you generate is strictly determined by the types of the source and destination entities.

**Rule 1: User-to-Abstract Connection (`User` -> `abstract`)**
*   If the source is `User` and the destination is an `abstract` entity, the relationship **MUST** always be `has_profile_in`.
*   The `relationship_type` is `classification`.

**Rule 2: Abstract-to-Abstract Connection (`abstract` -> `abstract`)**
*   If both the source and destination are `abstract` entities, the relationship **MUST** always be `has_subclass`.
*   The `relationship_type` is `classification`.

**Rule 3: Connections Involving Events (`abstract` -> `event` or `event` -> `event`)**
This is your most critical task. For any connection that points to or originates from an `event` entity, you must create a highly descriptive relationship label. Follow the "**Relation as Event Snapshot**" principle.

*   **Relationship Type (`relationship_type`):** `event`
*   **Core Principle: Relation as Event Snapshot**
    *   Your goal is to make each relationship label a self-contained, miniature summary of the event from the perspective of the `source` entity.
    *   To do this, you **MUST** analyze the **entire context** of the memory fragment. You need to gather all relevant details about the event—what happened, who was involved, where it took place, when it happened, and crucially, **why it happened**.
    *   **Explanatory Information is Key:** If the text provides a reason, cause, or motivation for an action or state, you **must** capture this explanatory information in the relationship label. This is vital for creating a truly intelligent knowledge graph.
    *   Crucially, you should incorporate details that may appear as entities further down the path to ensure each relationship is as complete as possible. Each triple must be an independent, informative unit.

*   **[Mandatory Generation Formula for User-Related Events]**
    To ensure clarity for downstream models, you **MUST** follow this formula for any event involving the user:
    `user_[Core_Verb_Phrase](_[Preposition]_[Context_1])(_[Preposition]_[Context_2])...`

    *   **`user_`**: A mandatory prefix to clarify that the user is the protagonist of the event.
    *   **`[Core_Verb_Phrase]`**: A concise verb or verb phrase describing the user's primary action, state, or attitude (e.g., `participated_in`, `created`, `avoids_eating`, `is`, `cherishes`).
    *   **`(_[Preposition]_[Context])`**: Optional but highly encouraged "context blocks" that add critical details.
        *   **`[Preposition]`**: Words like `with`, `at`, `on`, `in`, `as`, `for`, and critically for explaining causality, **`due_to`**, **`because_of`**, or **`for_reason_of`**.
        *   **`[Context]`**: The specific detail (e.g., `friends`, `community_center`, `2023-10-27`, `allergy`).

    *   **Example of Excellence (Descriptive):** For a text "User and their friend attended a pottery class at the community center yesterday (2023-10-27)", a relationship connecting `Therapeutic Activities` to `Pottery` should be:
        `user_attended_pottery_class_with_friend_at_community_center_on_2023-10-27`

    *   **Example of Excellence (Explanatory/Causal):** For a text "User is allergic to beef, so they don't eat steak anymore", a relationship connecting `Food` to `Steak` should be:
        `user_avoids_eating_due_to_allergy`

*   **Constraint: Objectivity and Conciseness**
    *   The label must be derived directly from the text. Do not infer or imagine details.
    *   Avoid vague words. Be factual and specific.
    *   Keep context blocks concise. `with_friend` is better than `with_their_close_friend`.

---
### **Chain of Thought & Workflow**

To ensure accuracy, follow this exact process:

1.  **Select a Path:** Start with the first path in the `Identified Entity Paths` list.
2.  **Iterate Through Segments:** For that path, move from left to right, processing one `->` segment at a time. A segment is defined by a `source` entity and its adjacent `destination` entity.
3.  **Apply Generation Rules:**
    *   For the current `(source, destination)` pair, check their entity types.
    *   If it's `User -> abstract` or `abstract -> abstract`, apply Rule 1 or Rule 2 to generate the fixed classification relationship.
    *   If it involves an `event`, apply Rule 3. This means you must:
        *   a. Re-read the entire memory fragment.
        *   b. Gather ALL details related to the `source` entity's involvement in the broader event.
        *   c. Construct the rich, descriptive relationship label using the `user_[...](_[prep]_[...])` formula.
4.  **Record the Triple:** Form the `(source, relationship, destination, relationship_type)` triple.
5.  **Repeat:** Move to the next segment in the path. Once a path is complete, move to the next path in the list. Continue until all segments in all paths have a corresponding relationship.

---
### **Few-shot Example**

**Input:**
*   **Text**: "User (Melanie) and her friend attended a pottery class at the community center yesterday (2023-10-27), which she found to be a very therapeutic self-care activity. During the class, she created a beautiful bowl with a black and white flower design, which is now a cherished personal possession."
*   **Identified Entity Paths**:
    *   `Path 1: User(abstract)->Health and Wellness(abstract)->Self-care(abstract)->Therapeutic Activities(abstract)->Pottery(event)->Pottery Class(event)`
    *   `Path 2: User(abstract)->Assets and Environment(abstract)->Personal Possessions(abstract)->Handmade Crafts(abstract)->Bowl with Black and White Flower Design(event)`
    *   `Path 3: User(abstract)->Interests and Entertainment(abstract)->Creative Pursuits(abstract)->Pottery(event)->Bowl with Black and White Flower Design(event)`

**Expected Output (Illustrative JSON structure, focus on the content):**

```json
{
  "entities": [
    // Relationships for Path 1
    {
      "source": "User",
      "relationship": "has_profile_in",
      "destination": "Health and Wellness",
      "relationship_type": "classification"
    },
    {
      "source": "Health and Wellness",
      "relationship": "has_subclass",
      "destination": "Self-care",
      "relationship_type": "classification"
    },
    {
      "source": "Self-care",
      "relationship": "has_subclass",
      "destination": "Therapeutic Activities",
      "relationship_type": "classification"
    },
    {
      "source": "Therapeutic Activities",
      "relationship": "user_participated_in_pottery_class_with_friend_on_2023-10-27",
      "destination": "Pottery",
      "relationship_type": "event"
    },
    {
      "source": "Pottery",
      "relationship": "user_attended_class_with_friend_at_community_center",
      "destination": "Pottery Class",
      "relationship_type": "event"
    },
    // Relationships for Path 2
    {
      "source": "User",
      "relationship": "has_profile_in",
      "destination": "Assets and Environment",
      "relationship_type": "classification"
    },
    {
      "source": "Assets and Environment",
      "relationship": "has_subclass",
      "destination": "Personal Possessions",
      "relationship_type": "classification"
    },
    {
      "source": "Personal Possessions",
      "relationship": "has_subclass",
      "destination": "Handmade Crafts",
      "relationship_type": "classification"
    },
    {
      "source": "Handmade Crafts",
      "relationship": "user_created_as_cherished_possession",
      "destination": "Bowl with Black and White Flower Design",
      "relationship_type": "event"
    },
    // Relationships for Path 3
    {
      "source": "User",
      "relationship": "has_profile_in",
      "destination": "Interests and Entertainment",
      "relationship_type": "classification"
    },
    {
      "source": "Interests and Entertainment",
      "relationship": "has_subclass",
      "destination": "Creative Pursuits",
      "relationship_type": "classification"
    },
    {
      "source": "Creative Pursuits",
      "relationship": "user_created_bowl_during_pottery_class",
      "destination": "Pottery",
      "relationship_type": "event"
    },
    {
      "source": "Pottery",
      "relationship": "user_created_as_result_of_activity",
      "destination": "Bowl with Black and White Flower Design",
      "relationship_type": "event"
    }
  ]
}

"""

EXTRACT_ENTITIES_PROMPT = """
**Role Setting:**
You are a top-tier Personalized Knowledge Graph (PKG) architect. Your expertise lies in deconstructing user "memory fragments" into a profound and interconnected knowledge graph. Your mission is not just to extract entities, but to construct the complete, logical **paths** that give them context.

**Core Task:**
Analyze the given "memory fragment" and identify all leaf-node concepts (concrete events, places, books, etc.). For **each** leaf node, you must trace and construct all of its relevant logical paths back to the `User` root node. Your final output must be a list of these complete entity paths, representing a rich, multi-faceted graph structure.

---
### **Core Extraction Principles**

**1. The Three-Tier Conceptual Model (Primary Principle):**
Your entire reasoning must be based on a three-tier conceptual hierarchy. While the final output `type` is only `abstract` or `event`, you must understand these three distinct roles during your analysis:

*   **a. `abstract` (Pure Classification / Folder):**
    *   **Definition:** A pure organizational label for classifying knowledge. It cannot be directly executed or experienced by the user.
    *   **Litmus Test:** "Is this a **category** of...?" (e.g., `Interests & Entertainment` is a category).
    *   **Function:** Serves as the high-level and intermediate structure of the knowledge graph.

*   **b. `General Event` (Standardized Activity / Countable Sub-Folder):**
    *   **Definition:** The standardized name for a repeatable activity or interest (e.g., `Running`, `Reading`, `Museum Visiting`). It aggregates specific instances.
    *   **Litmus Test:** "Can the user **do** this activity in general?" (e.g., The user can *do* `Running`).
    *   **Output Type:** When outputting, its `type` is **`event`**.
    *   **Mandatory Abstraction Rule:** When you identify a specific `Instance Event`, you **must** abstract its corresponding `General Event` to serve as one of its parent nodes.

*   **c. `Instance Event` (Specific Instance / File):**
    *   **Definition:** A unique, concrete event, or a specific entity with a proper name (e.g., a specific race, a book title, a museum name). It is the leaf node of every path.
    *   **Litmus Test:** "Does this have a **specific, unique context** (like a name, a unique theme, or a specific time/place implied)?"
    *   **Output Type:** When outputting, its `type` is **`event`**.

**2. Multi-Dimensional Analysis & Path Anchoring (High Priority):**
A single `Instance Event` is often multi-faceted. You must analyze it from different dimensions and generate a separate path for each relevant dimension. This is the key to creating a rich, interconnected graph.

*   **a. The Dual Parenting Principle for Entities:** For `Instance Events` representing specific things (books, places, organizations), you must generate **at least two paths**:
    *   **Taxonomic Path:** Answers "What **is** this entity?" The path should classify the entity's nature.
        *   *Example:* For the book `"The Art of Strategy"`, the taxonomic path is `... -> Books -> "The Art of Strategy"`.
    *   **Contextual Path:** Answers "In what **activity** was this entity used/mentioned?" The path should link it to the relevant `General Event`.
        *   *Example:* For the book `"The Art of Strategy"`, the contextual path is `... -> Reading -> "The Art of Strategy"`.

*   **b. Activity vs. Theme Decomposition:** For complex events, decompose them into their core activity and theme, generating a path for each.
    *   *Example:* For `Charity Race for Mental Health`, generate one path under `Running` (the activity) and another under `Mental Health Awareness` (the theme).

**3. Path Construction & Scope:**

*   **a. Rich and Logical Paths:** Every generated path must be logically sound and reasonably deep. Always strive to add meaningful intermediate `abstract` nodes between a base node and a `General Event` or leaf node (e.g., `Interests & Entertainment -> Sports -> Running`).
*   **b. Information Scope:**
    *   Your primary focus is the `User`.
    *   Information about other individuals (e.g., an assistant) should only be extracted if it describes a **direct interaction, a shared plan, or a state that directly affects the user**. The resulting path should reflect this relationship (e.g., `User -> Social Relationships -> Assistant -> [Assistant's Plan]`). Information not meeting this criterion is C-Tier noise and must be ignored.

**4. Constraints and Conventions:**

*   **a. Final Output Integrity:** Your final `entity_paths` output **must not contain any paths or entities that you decided to ignore** during your thought process. The output must be the clean, final result.
*   **b. Naming Convention:** All entity names must be concise, standardized noun phrases. They must **NOT** contain instance-specific metadata like dates, times, or other qualifiers.
*   **c. Base Node Constraint:** The `User` entity must connect to one of these 11 base `abstract` nodes: Personal Identity and Traits, Health and Wellness, Goals and Plans, Consumption and Finance, Dining, Interests and Entertainment, Travel and Commute, Social Relationships, Work and Study, Values and Beliefs, Assets and Environment.

---
### **Chain of Thought with Reflection**

**Input Text:** `User mentioned she finished reading "The Art of Strategy", finding it insightful for her work. This reminded her of the "Charity Race for Mental Health" she ran, which involved a different kind of strategic planning.`

**1. Initial Scan & Instance Event Identification:**
*   I see two primary `Instance Events` (leaf nodes) in this text:
    *   A specific book: `"The Art of Strategy"`.
    *   A specific event: `"Charity Race for Mental Health"`.
*   My task is to build all logical paths for these two leaf nodes.

**2. Path Construction for `"The Art of Strategy"`:**
*   **Multi-Dimensional Analysis (Dual Parenting Principle):** This is a specific entity, so I need to find its taxonomic and contextual paths.
*   **a. Taxonomic Path ("What is it?"):**
    *   It's a `Book`. Books are a type of `Media`, and also can be considered `Personal Possessions`. I'll choose the most relevant classification. `Books` seems like a good `abstract` category. This belongs under `Assets & Environment`.
    *   **Path 1 Construction:** `User -> Assets & Environment -> Personal Possessions -> Books -> "The Art of Strategy"`. This path is complete.
*   **b. Contextual Path ("How was it used?"):**
    *   It was used in a `Reading` activity. `Reading` is the `General Event` for this context.
    *   `Reading` is a `Personal Hobby`, which falls under `Interests & Entertainment`.
    *   **Path 2 Construction:** `User -> Interests and Entertainment -> Personal Hobbies -> Reading -> "The Art of Strategy"`. This path is also complete.

**3. Path Construction for `"Charity Race for Mental Health"`:**
*   **Multi-Dimensional Analysis (Activity vs. Theme):** This event has a clear activity and a clear theme.
*   **a. Activity Path ("What was the activity?"):**
    *   The activity was a race, implying the `General Event` is `Running`.
    *   `Running` is a `Sport`, which falls under `Interests & Entertainment`.
    *   **Path 3 Construction:** `User -> Interests & Entertainment -> Sports -> Running -> "Charity Race for Mental Health"`. This path is complete.
*   **b. Theme Path ("What was it about?"):**
    *   The theme was `Mental Health`. I can create an `abstract` node `Mental Health Awareness` for this.
    *   This theme belongs under the base node `Health & Wellness`.
    *   **Path 4 Construction:** `User -> Health and Wellness -> Mental Health Awareness -> "Charity Race for Mental Health"`. This path is complete.

**4. Final Reflection & Formatting:**
*   **[Completeness Check]:** I have identified all `Instance Events` from the text.
*   **[Multi-Path Check]:** I have applied multi-dimensional analysis to both `Instance Events` and generated all logical paths (Taxonomic/Contextual for the book, Activity/Theme for the race).
*   **[Path Richness Check]:** All paths contain meaningful intermediate `abstract` nodes.
*   **[Constraint Check]:** All paths start with `User` and connect to a valid base node. All names are clean. No ignored entities are present.
*   **Conclusion:** All checks passed. I will now format the four generated paths into the final JSON output.

### **Output Format**
Your output must be a single JSON object containing a list of paths, structured exactly as follows:
```json
{
  "entity_paths": [
    {
      "path": [
        {"name": "User", "type": "abstract"},
        {"name": "Assets and Environment", "type": "abstract"},
        {"name": "Personal Possessions", "type": "abstract"},
        {"name": "Books", "type": "abstract"},
        {"name": "The Art of Strategy", "type": "event"}
      ]
    },
    {
      "path": [
        {"name": "User", "type": "abstract"},
        {"name": "Interests and Entertainment", "type": "abstract"},
        {"name": "Personal Hobbies", "type": "abstract"},
        {"name": "Reading", "type": "event"},
        {"name": "The Art of Strategy", "type": "event"}
      ]
    },
    {
      "path": [
        {"name": "User", "type": "abstract"},
        {"name": "Interests and Entertainment", "type": "abstract"},
        {"name": "Sports", "type": "abstract"},
        {"name": "Running", "type": "event"},
        {"name": "Charity Race for Mental Health", "type": "event"}
      ]
    },
    {
      "path": [
        {"name": "User", "type": "abstract"},
        {"name": "Health and Wellness", "type": "abstract"},
        {"name": "Mental Health Awareness", "type": "abstract"},
        {"name": "Charity Race for Mental Health", "type": "event"}
      ]
    }
  ]
}
"""

UPDATE_INFERENCE_PROMPT = """
# Role
You are an expert user profiler and insight extractor. Your specialty is transforming scattered evidence into a concise, actionable user profile slice that other AI systems can directly use.

# Task
Your goal is to analyze the relationship between `[Source Entity]` and `[Target Entity]` based on `[New Evidence]` and `[Previous Inference]` to create an updated, highly condensed user profile slice. This slice must be directly usable for downstream tasks like personalized recommendations or conversation.

You will face two scenarios:
1.  **Initialization:** When `[Previous Inference]` is empty. Your task is to establish the baseline profile slice for the first time.
2.  **Update:** When `[Previous Inference]` already exists. Your task is to integrate the new evidence, highlighting the **evolution and changes** from the old inference.

# Input Format Explanation (Crucially Important)
*   `[Conclusion]` contains natural language conclusions related to the relationship.
*   `[Raw Event]` is a structured, summary-style relationship label. To interpret it correctly, understand its two core modes:
    *   **Objective Fact Mode:** A concise verb phrase, e.g., `takes_place_at`.
    *   **User Interaction Mode:** A composite structure `[Base_Verb]_[User_Attitude]`, e.g., `categorizes_event_anticipated_by_user`.

# Core Content Generation Framework
You must generate the output in two distinct, mandatory parts:

### Part 1: Crafting the Fact Summary
*   **Goal:** To provide a highly condensed summary of the current state and its changes.
*   **Rules:**
    *   This must be a **single, concise paragraph**.
    *   It must integrate all relevant information from `[New Evidence]`.
    *   **For the "Update" scenario,** it is critical to explicitly state how the new evidence changes, confirms, or evolves the `[Previous Inference]`. Use comparative language like "This changes the previous understanding," "This further confirms," etc.
    *   **For the "Initialization" scenario,** this summary establishes the foundational facts about the user.

### Part 2: Generating Actionable Inferences
*   **Goal:** To extract underlying traits, potential interests, or preferences that a downstream AI can directly act upon.
*   **Rules:**
    *   This must be a **bulleted list**.
    *   Each item **must** start with the fixed phrase "**Inferred Trait/Interest:**".
    *   These inferences should identify commonalities, latent preferences, or potential future interests. Think about what product, service, or topic you could recommend based on this.
    *   **DO NOT explain your reasoning process.** Only state the inferred trait itself.

# Final Output Structure
Strictly adhere to the following two-part structure, including the guide words. Do not add any other titles or commentary.

**Fact Summary:**
[A concise paragraph summarizing the facts and changes.]

**Actionable Inferences:**
- Inferred Trait/Interest: [Inference 1]
- Inferred Trait/Interest: [Inference 2]
- ... (and so on)

---
#### Example 1: Update Scenario - Preference Change
**[Source Entity]:** Lunch Choice
**[Target Entity]:** Braised Chicken Rice
**[Previous Inference]:**
Fact Summary:
The user considers Braised Chicken Rice as an occasional lunch option, but a stable preference has not yet been observed.
Actionable Inferences:
- Inferred Trait/Interest: Shows some acceptance of Chinese fast food.
**[New Evidence]:**
- [Raw Event]: includes_food_eaten_by_user. (at 2023-10-30 ~ 2023-11-04, 4 times)
- [Conclusion]: The user mentioned, "This week is a project sprint, I'm extremely busy, so for lunch I just want something simple, fast, and familiar."
---
**[New Inference]:**
**Fact Summary:**
Based on new evidence, the user's preference for Braised Chicken Rice has changed significantly. During last week's high-intensity work period, it quickly shifted from an occasional option to a high-frequency core choice (4 times a week), because the user prefers familiar, fast, and thought-free food when busy.

**Actionable Inferences:**
- Inferred Trait/Interest: Has a strong preference for 'convenient' and 'quick' food when busy with work.
- Inferred Trait/Interest: May be interested in other similar simple meals, meal delivery subscription services, or ready-to-eat meals.
- Inferred Trait/Interest: Decision-making behavior is heavily influenced by current work pressure.
---
#### Example 2: Initialization Scenario - User Identity
**[Source Entity]:** family_role
**[Target Entity]:** parent
**[Previous Inference]:**
<!-- This area is empty -->
**[New Evidence]:**
- [Conclusion]: One of User (Melanie)'s core identities is being a parent, managing responsibilities alongside work.
- [Conclusion]: One of User (Melanie)'s core identities is being a parent who cherishes creating memorable family experiences in nature.
---
**[New Inference]:**
**Fact Summary:**
One of user Melanie's core identities is being a parent. She not only balances work and parenting but also highly values creating unique and memorable family experiences in nature.

**Actionable Inferences:**
- Inferred Trait/Interest: Interested in topics like family education and parent-child relationship growth.
- Inferred Trait/Interest: May like products or services related to weekend getaways, outdoor activities, and nature exploration.
- Inferred Trait/Interest: Values quality family life and emotional connection.
---
---

### Formal Task

**[Source Entity]:**
{source_entity_name}

**[Target Entity]:**
{destination_entity_name}

**[Previous Inference]:**
{old_inference}

**[New Evidence]:**
{new_evidence}
---
**[New Inference]:**
"""

UPDATE_COMMUNITY_SUMMARY_PROMPT = """
# Role
You are a User Profile Knowledge Synthesis Engine. Your purpose is to process hierarchical evidence about a user's engagement within a broad domain (`Core Entity`) and its specific sub-topics. Your output must be a highly condensed, structured, and factual knowledge snippet for a downstream AI, avoiding all subjective analysis or literary language.

# Core Task: Synthesize from Branches to Trunk
Your primary task is to create or update a profile snippet for the main `Core Entity` (the "trunk") by synthesizing information from its various sub-topics (the "branches," e.g., 'pottery', 'painting'). The final output must describe the trunk, not the individual branches.

# Input Evidence Interpretation
- The `[Core Entity]` is the primary subject of the final summary.
- Evidence provided under specific sub-topics (e.g., `[Inference about 'pottery']`) are the "branches" from which you must generalize.
- `[Inference]` and `[Conclusion]` are ground truth facts.
- `[Raw Event]` represents a user behavior.
- Evidence may be provided in a pre-summarized format (e.g., with "Fact Summary," "Actionable Inferences"). Treat all provided text as raw factual material to be re-synthesized.

# Output Generation Principles & Structure (Strictly Follow)

**Principle 1: Layered Factual Structure**
Your output is a single, unified list of facts, organized into two layers: "General Core Facts" and "Specific Domain Facts". You MUST use `#` for these layer titles.

**Principle 2: General Core Facts (The Trunk)**
This first section synthesizes the **commonalities** found across multiple branches.
- It must contain bulleted facts that are broadly true for the entire `Core Entity`.
- These facts are derived from recurring themes, motivations, and values seen in the evidence from different sub-topics.

**Principle 3: Specific Domain Facts (The Unique Leaves)**
This second section captures important facts that are unique to a specific branch and cannot be generalized to the entire trunk, but are still crucial for a complete profile.
- You MUST group these unique facts under **dynamically generated thematic labels**. A label should be a concise phrase ending with a colon (e.g., `Regarding Resilience & Adaptability:`, `On Growth & Inspiration:`).
- A thematic label is your own synthesis of the core theme of the unique fact(s) it groups.
- Under each label, list the relevant bulleted fact(s).

**Principle 4: Absolute Factual Purity & High Density**
- **Strict Prohibition of Analytical Language:** Do NOT use any phrasing that sounds like an analyst's conclusion (e.g., "The user's core practice is...", "This demonstrates..."). State facts directly.
- **No Meta-Language:** Avoid conversational or procedural phrases (e.g., "The summary is...", "Based on the evidence...").
- **Synthesize, Don't Transcribe:** Do not merely copy points from the input. Rephrase and consolidate them into dense, comprehensive facts. If two facts from the input describe the same core idea (e.g., one states an action, another labels it "resilience"), merge them into a single, powerful factual statement.

# Integration Strategy for Updates
When a `[Previous Summary]` is provided, you must intelligently integrate it with `[New Evidence Collection]`.
1.  **Foundation:** Use the `[Previous Summary]` as the base knowledge.
2.  **Enhance & Refine:** Use new evidence to make existing facts more specific (e.g., "long-term hobby" becomes "hobby of over seven years") or to refine their substance.
3.  **Add:** Add entirely new facts from the new evidence that were not previously mentioned.
4.  **Preserve:** Retain unique, still-relevant facts from the `[Previous Summary]` even if they are not directly mentioned in the new evidence (e.g., a specific future plan).
5.  **Re-Synthesize:** After integrating, re-evaluate and rewrite the entire "General Core Facts" and "Specific Domain Facts" sections to reflect the most current and complete understanding.

---
### Example

**[Core Entity]:**
creative_arts

**[Previous Summary]:**
# General Core Facts
- Values creative activities that promote relaxation and emotional expression.
# Specific Domain Facts
On Growth & Inspiration:
- Plans to produce new paintings inspired by seasonal changes.

**[New Evidence Collection]:**
- [Inference about 'pottery']: User has a 7+ year commitment to pottery as a calming and essential outlet. User shares photos of their creations.
- [Inference about 'painting']: Painting is a crucial outlet for emotional expression, and the user persists with it even when a physical injury stops them from doing pottery.

---
**[New Summary]:**
# General Core Facts
- Engages in hands-on creative arts like pottery and painting as a core, long-term practice for emotional expression and achieving inner peace.
- Values using consistent, tangible artistic activities to support personal well-being and emotional stability.
- May appreciate resources that support sustained practice and skill development in their chosen crafts.

# Specific Domain Facts
Regarding Resilience & Adaptability:
- Adapts creative practice in response to physical challenges, switching to painting when an injury prevented pottery work.
Regarding Community & Sharing:
- Shares photos of pottery creations, indicating active engagement and pride in the results.
On Growth & Inspiration:
- Plans to produce new paintings inspired by seasonal changes.
---
### Formal Task

**[Core Entity]:**
{core_entity_name}

**[Previous Summary]:**
{old_summary}

**[New Evidence Collection]:**
{new_evidence_str}
---
**[New Summary]:**
"""

UPDATE_HIERARCHICAL_SUMMARY_PROMPT = """
**Role:**
You are a top-tier knowledge architect and a user portrait specialist. Your core competency is applying "Portrait Dialectics" to distill a unifying macro-pattern (Emergent Pattern) from multiple details. Your primary goal is to generate a highly actionable, concise, and semantically clear user portrait summary that directly guides downstream AI systems for personalized dialogue and interaction.

**Task:**
Generate a high-level meta-summary for a parent concept (the "Core Entity"). Your output must be a direct user profile, not an analytical report, focusing on the user's core drivers and practical constraints.

**Core Analysis Framework (Portrait Dialectics Engine):**
You must strictly follow this logically progressive analysis framework:

1.  **Step One: Deconstruct Microstates & Filter Input Noise**
    *   Treat each `[Updated Sub-Summary]` as a fundamental "microstate" or core fact.
    *   **Crucial Constraint (Content Filtering):** If a sub-summary contains previously extracted Macro-Laws, Synergy, or Tension analysis (i.e., it is a high-level summary from a lower tier), you MUST IGNORE and STRIP OUT the previous summary structure. Only extract the core, factual insights and detailed observations to be used as peer-level "microstates." The goal is to avoid content redundancy and maintain clean hierarchical integrity.

2.  **Step Two: Analyze Inter-Relationships (Synergy & Tension)**
    *   Examine the interactions between these "microstates."
    *   **Synergy:** Determine the collective, reinforcing patterns that define the user's core traits, values, and motivations.
    *   **Tension:** Identify potential conflicts, contradictions, or resource competition points that the user must continuously manage.
    *   **Boundary Condition (N=1 Input Rule - Content Specificity):** If only ONE microstate is provided, you must shift the analysis focus to its 'Intrinsic Tension' and 'Intrinsic Synergy.'
        *   **Intrinsic Tension Content:** The Tension MUST be internal to the concept itself, focusing on the specific, quantifiable resource demands (time, money, physical effort, or opportunity cost) required to sustain the positive aspects (Synergy). DO NOT use vague, generalized external conflicts (e.g., 'conflict with time and energy') unless they are explicitly detailed in the input.

3.  **Step Three: Synthesize the Macro-Law & Convert to Actionable Principles**
    *   **Synthesize Macro-Law:** Create the unifying "macro-law."
    *   **Convert Synergy to Core Traits:** Translate the Synergy analysis into direct, affirmative descriptions of the user's core traits and motivations.
    *   **Convert Tension to Restrictions:** Translate the Tension analysis into specific, practical guiding principles that advise downstream models on what to avoid or how to structure interactions (e.g., resource sensitivity, required balance).
    *   **Scope Constraint (Content Accuracy):** If the combined sub-summaries only cover a narrow aspect of the `[Core Entity]`, you MUST explicitly qualify the Macro-Law statement to reflect this narrow scope.

**Final Output Instructions (Content Structure and Style):**
*   **Style Constraint:** Output must be written in a concise, high signal-to-noise ratio, and highly actionable tone. AVOID complex, 'literary' sentence structures, abstract descriptive filler, and redundant explanations. Focus solely on the user's defined traits, motivations, and conflicts.
*   **Output ONLY the text of the new summary, adhering to the structure below.**
*   **Mandatory Structure (Use these exact English Headers for easy semantic parsing):** The output MUST be structured using three distinct, non-narrative content blocks, separated by line breaks.

---
### Example

**[Core Entity]:**
Personal Identity & Traits

**[Previous Summary]:**
Some basic information about the user.

**[Updated Sub-Summaries]:**
- Sub-summary about 'Family Role': The user is a mother of two, highly values family life, and considers it a core part of her personal identity.
- Sub-summary about 'Professional Identity': The user is an experienced data analyst, passionate about technology, and has a clear career development plan, aspiring to achieve greater success in the field of data science.
- Sub-summary about 'Personality Traits': The user has an introverted personality but is skilled at logical thinking, pragmatic and methodical in her actions, and prefers to solve problems through deep thought.
---
**[New Summary]:**
Macro-Law: "Structured Goal-Orientation."

Core User Traits (Synergy):
- Drive: Pragmatic and logical thinking strongly supports career development as a data analyst and efficient family management.
- Values: Highly values family life as a core part of identity, progressing this priority alongside professional aspirations.

Action/Recommendation Constraints (Tension):
- Resource Sensitivity: Ongoing resource (time, energy) allocation conflict exists between the pursuit of professional excellence and high-value family investment.
- Behavioral Preference: Prefers solving problems through deep, logical thought; interactions should provide structured, methodical input.
---

### Formal Task

**[Core Entity]:**
{core_entity_name}

**[Previous Summary]:**
{old_summary}

**[Updated Sub-Summaries]:**
{child_summaries_str}
---
**[New Summary]:**
"""

CONCLUSION_MATCHING_PROMPT_TEMPLATE = """
**Role:**
You are a highly intelligent Knowledge Graph Allocation Engine. Your primary function is to analyze a list of natural language conclusions and a list of structured triples, and then strategically **allocate** each triple to the **single most appropriate conclusion** it serves as evidence for. Your goal is to create a clear, non-redundant mapping between evidence (triples) and claims (conclusions).

**【Core Task: Global Allocation of Triples to Conclusions】**
Your main task is to assign each triple from the "Candidate Triple List" to the one conclusion from the "Conclusion List" that it **most specifically and directly supports**. This is a **global optimization and allocation task**, not a simple one-to-one matching process. Your objective is to ensure that every piece of evidence finds its most logical home, minimizing redundancy across conclusions.

**【Guiding Principles for Allocation】**
To make the best allocation decisions, you must act as a discerning analyst, evaluating each potential link based on the following principles, with a clear order of importance:

1.  **Principle 1: Semantic Relevance of the Relationship (Highest Priority)**
    *   The core of your decision lies in the triple's relationship `[Relationship]`. Does this relationship (the raw event or state) describe a **direct action, instance, cause, or consequence** of the core idea expressed in the conclusion?
    *   A strong match in the relationship's meaning is the most critical signal for a correct allocation.

2.  **Principle 2: Specificity of the Connection (The Deciding Factor)**
    *   When a triple could potentially relate to multiple conclusions, you must ask: **"Which conclusion provides the most specific and unique context for this triple?"**
    *   Allocate the triple to the conclusion where it acts as a key, irreplaceable piece of evidence, rather than just general support.
    *   *Example:* A triple about `(...motivated_by_family...)` is far more specific to a conclusion about "motivation" than to a general conclusion about "family identity".

3.  **Principle 3: Entity Confirmation (Supporting Evidence)**
    *   After evaluating the relationship and specificity, confirm that the triple's entities (`Source` or `Destination`) are explicitly mentioned or conceptually central to the chosen conclusion. This includes semantically linked concepts (e.g., 'wedding' as strong evidence for 'marriage').

**【Workflow: A Three-Step Allocation Process】**
You must follow this structured process to ensure accuracy and consistency:

1.  **Step 1: Holistic Review**
    *   Begin by carefully reading **all** conclusions and **all** candidate triples. Develop a complete understanding of the thematic landscape before making any assignments. Identify which conclusions are broad and which are specific.

2.  **Step 2: Affinity Analysis & Allocation**
    *   For each triple in the candidate list, mentally evaluate its affinity with **every** conclusion based on the "Guiding Principles" above.
    *   Based on this analysis, make a definitive allocation of the triple to the **single best-fit conclusion**. Place the triple in a temporary holding area for that conclusion.

3.  **Step 3: Final Output Construction**
    *   After all triples have been allocated, construct the final JSON output.
    *   Iterate through the original "Conclusion List" in order. For each conclusion, gather all the triples you allocated to it.
    *   **【Crucial Mandate: Process All Items】** You MUST process every single conclusion from the 'Conclusion List'. Your final output list must contain exactly one object for each conclusion provided in the input, maintaining the original order. If a conclusion has no triples allocated to it, its `matched_triplets` field must be an empty list `[]`.
    *   **【Extremely Important Copying Rule】** When populating the `matched_triplets` list, you **must** copy the matched triple strings **verbatim and precisely** from the "Candidate Triple List". Do not perform any modification.

**【Example of High-Quality Allocation】**
*   **Input Conclusions:**
    1. "One of User (Melanie)'s core identities is being a married woman with a family including a husband and children." (A broad identity conclusion)
    2. "User (Melanie) values family bonding and finds motivation and happiness primarily through them." (A specific conclusion about motivation)
*   **Input Triples:**
    *   `T1: (marital_status) --[user_is_married]--> (marriage)`
    *   `T2: (family_role) --[user_shared_photo_of_spouse_who_keeps_her_motivated]--> (spouse)`
    *   `T3: (family_role) --[user_shared_photo_of_children_who_keep_her_motivated]--> (children)`
*   **Correct Allocation Logic:**
    *   T1 directly speaks to marital status, which is a core component of Conclusion 1. It is less specific to Conclusion 2. **Allocate T1 to Conclusion 1.**
    *   T2 and T3 both contain the crucial semantic information "...keeps_her_motivated...". This aligns with extreme **specificity** to Conclusion 2. While spouse and children are part of the identity in Conclusion 1, the *action* in the triple is about motivation. **Therefore, allocate T2 and T3 to Conclusion 2.**
*   **Correct Output Snippet:**
    ```json
    [
      {{
        "conclusion_content": "One of User (Melanie)'s core identities is being a married woman with a family including a husband and children.",
        "matched_triplets": [
          "(marital_status) --[user_is_married]--> (marriage)"
        ]
      }},
      {{
        "conclusion_content": "User (Melanie) values family bonding and finds motivation and happiness primarily through them.",
        "matched_triplets": [
          "(family_role) --[user_shared_photo_of_spouse_who_keeps_her_motivated]--> (spouse)",
          "(family_role) --[user_shared_photo_of_children_who_keep_her_motivated]--> (children)"
        ]
      }}
    ]
    ```

---
Input Data:
1. Conclusion List:
{conclusions_str}

2. Candidate Triple List:
{triplets_str}dwas

Output Format Requirement:
Your output must be in strict JSON format, without any extra explanations, comments, or code block markers.
The JSON should be a list, where each object in the list represents a conclusion and its allocated triples, in the following format:
[
  {{
    "conclusion_content": "<Full content of the conclusion goes here>",
    "matched_triplets": [
      "<String form of the first matched triple goes here>",
      "<String form of the second matched triple goes here>"
    ]
  }},
  // ... results for other conclusions ...
]
"""