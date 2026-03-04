# Execution Guide

## Step 1

Run PDF parsing to convert pages into images.

## Step 2

Process pages in parallel.

Recommended:

workers = 4

## Step 3

Run LangChain multi-agent pipeline:

Vision Agent
→ Structuring Agent
→ Validator Agent

## Step 4

Parse answer key pages.

## Step 5

Merge answers with extracted questions.

## Step 6

Insert data into PostgreSQL.

## Expected Output

~20,000 structured questions stored in database.
