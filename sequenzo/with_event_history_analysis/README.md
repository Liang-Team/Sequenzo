# Event Sequence Analysis in Sequenzo

## Overview

Event sequence analysis is a powerful approach for analyzing temporal data that focuses on **when changes occur** rather than **what states persist**. This module provides tools for creating, analyzing, and comparing event sequences, complementing Sequenzo's state sequence analysis capabilities.

## Core Concepts: State Sequences vs Event Sequences

### State Sequences
- **Description**: "What state are you in at each time point?"
- **Example**: A person's employment status sequence
  ```
  Age 15-20: Student
  Age 21-25: Student  
  Age 26-30: Full-time
  Age 31-35: Full-time
  ```
- **Characteristics**: Each time point has a state; states can persist over time

### Event Sequences
- **Description**: "What happened and when?"
- **Example**: The same person's employment event sequence
  ```
  18 years: Enter university
  22 years: Graduate
  26 years: Start full-time job
  30 years: Promotion
  ```
- **Characteristics**: Only records "change moments"; does not record persistent states

## Why Event Sequence Analysis?

### 1. Focus on Changes, Not States
- **State sequences**: Focus on "what you are doing"
- **Event sequences**: Focus on "when you change"

### 2. Discover Frequent Patterns
- Find common event combinations across individuals
- Example: Many people follow the pattern "Graduate → Find job → Promotion"
- Identify the most common "life paths"

### 3. Compare Groups
- Example: Which group more frequently follows "Work first, then marry"?
- Identify patterns that distinguish different populations

## Main Functions

### 1. `seqecreate()` - Create Event Sequences
Creates an event sequence object from:
- **State sequences**: Converts state sequences to event sequences
- **Raw data**: Creates directly from timestamped event data (TSE format)

**Key Features**:
- Automatic conversion from state sequences using transition methods
- Support for multiple events at the same timestamp
- Weight preservation from state sequences

### 2. `seqefsub()` - Find Frequent Subsequences
Identifies frequently occurring event patterns in the dataset.

**Key Features**:
- Finds subsequences with minimum support threshold
- Supports time constraints (age windows, gaps, etc.)
- Returns support and count statistics
- Can search for specific user-defined subsequences

**Use Cases**:
- Discover common life trajectories
- Identify typical career paths
- Find frequent medical event sequences

### 3. `seqecmpgroup()` - Compare Group Differences
Identifies event patterns that discriminate between groups.

**Key Features**:
- Statistical tests (Chi-square) to compare group frequencies
- Returns p-values and test statistics
- Identifies patterns more common in one group than another
- Supports Bonferroni correction for multiple testing

**Use Cases**:
- Compare male vs female career patterns
- Identify patterns distinguishing successful vs unsuccessful cases
- Find event sequences characteristic of specific populations

### 4. `seqeapplysub()` - Apply Subsequences
Counts occurrences of specific subsequences in each sequence.

**Key Features**:
- Multiple counting methods (presence, count, distinct occurrences)
- Returns matrix of subsequence occurrences per sequence
- Supports time constraints

**Use Cases**:
- Count how many times each person experienced "promotion"
- Create features for predictive modeling
- Analyze subsequence frequency distributions

## Data Formats

### TSE Format (Time-Stamped Events)
A data frame with three columns:
- `id`: Individual identifier
- `timestamp` or `time`: Time when event occurred
- `event`: Event type/name

**Example**:
```
id  timestamp  event
1   18         EnterUniversity
1   22         Graduate
1   26         StartJob
2   17         EnterUniversity
2   20         Graduate
2   21         StartJob
```

### Event Sequence Object
An internal representation storing:
- List of events with timestamps for each individual
- Event dictionary (alphabet)
- Weights (if available)
- Sequence lengths

## Conversion Methods

When converting from state sequences, several methods are available:

1. **"transition"** (default): One event per state transition
   - Example: `A→B` becomes event `"A>B"`

2. **"state"**: One event when entering each new state
   - Example: Entering state `B` becomes event `"B"`

3. **"period"**: Pair of events (start and end) for each transition
   - Example: `A→B` becomes `"endA"` and `"beginB"`

## Time Constraints

Event sequence analysis supports various time constraints:

- **max.gap**: Maximum time gap between consecutive events in a subsequence
- **window.size**: Maximum time window for a subsequence
- **age.min / age.max**: Restrict to events occurring within age range
- **age.max.end**: Maximum age for subsequence end

## Counting Methods

Different ways to count subsequence occurrences:

1. **COBJ** (count.method=1): Count per sequence (presence/absence)
2. **CDIST_O** (count.method=2): Count distinct occurrences
3. **CWIN** (count.method=3): Count within time windows
4. **CMINWIN** (count.method=4): Count minimum windows
5. **CDIST** (count.method=5): Count with distance constraints

## Practical Examples

### Example 1: Career Trajectories

**State Sequence Approach**:
```
Person A: [Student, Student, Student, Full-time, Full-time, Full-time, Full-time]
Person B: [Student, Student, Full-time, Full-time, Full-time, Full-time, Full-time]
```
- Shows states but not "change moments"

**Event Sequence Approach**:
```
Person A: (18: EnterUniversity) → (22: Graduate) → (23: StartJob)
Person B: (17: EnterUniversity) → (20: Graduate) → (21: StartJob)
```
- Can analyze:
  - Most common paths
  - Average time from "Graduate" to "StartJob"
  - Who skipped certain steps

### Example 2: Medical Events

Analyze patient treatment sequences:
- Find frequent treatment patterns
- Compare treatment effectiveness between groups
- Identify critical event sequences

### Example 3: System Logs

Analyze system event logs:
- Discover common error sequences
- Identify patterns leading to failures
- Compare behavior across different system configurations

## Relationship to Other Sequenzo Modules

### vs State Sequence Analysis
- **State sequences**: Better for analyzing persistent states and durations
- **Event sequences**: Better for analyzing change moments and transitions

### vs Event History Analysis (SAMM)
- **Event sequences**: Focus on discrete events and their patterns
- **SAMM**: Focus on multi-state models and transition probabilities

Both approaches complement each other and can be used together for comprehensive analysis.

## Implementation Notes

This module implements TraMineR-compatible event sequence analysis functions:
- `seqecreate()`: Creates event sequence objects
- `seqefsub()`: Finds frequent subsequences
- `seqecmpgroup()`: Compares groups
- `seqeapplysub()`: Applies subsequences to sequences

Results are designed to be consistent with TraMineR outputs for compatibility and reproducibility.

## References

- Ritschard, G., Bürgin, R., and Studer, M. (2014). "Exploratory Mining of Life Event Histories", In McArdle, J.J. & Ritschard, G. (eds) *Contemporary Issues in Exploratory Data Mining in the Behavioral Sciences*. Series: Quantitative Methodology, pp. 221-253. New York: Routledge.

- TraMineR Documentation: http://traminer.unige.ch
