# Named Entities in the Spam Email Classification Dataset

This document explains the **named entities** identified and labeled in the spam email classification dataset. These entities are part of **Named Entity Recognition (NER)** and are crucial for understanding and analyzing the structure of the dataset.

## 1. CARDINAL
- **Definition**: Refers to cardinal numbers, i.e., numbers that represent quantities or counts.
- **Example**: `"I have 5 emails"` → `5` would be a CARDINAL.
- **In the Dataset**: This could represent any numbers found in the text that indicate amounts, like counts, or numbers of items.

## 2. PERSON
- **Definition**: Refers to names of people or individual human beings (or entities typically recognized as persons).
- **Example**: `"John"` or `"Mary"` would be labeled as PERSON.
- **In the Dataset**: This counts the number of times personal names appear in the emails (whether spam or not).

## 3. ORG (Organization)
- **Definition**: Refers to companies, institutions, or organizations.
- **Example**: `"Google"`, `"United Nations"`, `"NASA"` are all considered ORG.
- **In the Dataset**: This counts the number of times an organization name appears in the email text.

## 4. DATE
- **Definition**: Represents dates or time expressions.
- **Example**: `"January 1, 2023"` or `"yesterday"`.
- **In the Dataset**: This refers to any mention of dates in the email.

## 5. MONEY
- **Definition**: Refers to monetary values or financial figures.
- **Example**: `"$100"`, `"€50"`, `"3 dollars"`.
- **In the Dataset**: Counts how many times money-related values appear (for example, if an email talks about prices, payments, etc.).

## 6. GPE (Geopolitical Entity)
- **Definition**: Refers to names of countries, cities, or regions.
- **Example**: `"USA"`, `"Paris"`, `"California"`.
- **In the Dataset**: This counts mentions of places like countries or cities.

## 7. NORP (Nationalities or Religious/Political Groups)
- **Definition**: Refers to a person's or group's nationality, ethnicity, religion, or political affiliation.
- **Example**: `"American"`, `"Muslim"`, `"Democrat"`.
- **In the Dataset**: This would count references to nationalities, religions, or political terms.

## 8. TIME
- **Definition**: Refers to time-related expressions (time of day, etc.).
- **Example**: `"2 PM"`, `"morning"`.
- **In the Dataset**: This counts mentions of specific times or time-related expressions.

## 9. PERCENT
- **Definition**: Refers to percentage values.
- **Example**: `"50%"` or `"5 percent"`.
- **In the Dataset**: Counts how many times a percentage is mentioned in the text.

## 10. ORDINAL
- **Definition**: Refers to ordinal numbers (numbers that show position or rank).
- **Example**: `"1st"`, `"3rd"`, `"10th"`.
- **In the Dataset**: This would count references to positions, rankings, or ordering in the email text.

## 11. QUANTITY
- **Definition**: Refers to measurements, quantities, or units.
- **Example**: `"5 liters"`, `"2 miles"`.
- **In the Dataset**: This counts mentions of quantities with units (not just numbers, but numbers tied to units).