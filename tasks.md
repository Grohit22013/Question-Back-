# Tasks

The agent must implement the following steps.

## Step 1

Convert PDF pages into images.

Output:

pages/page_1.png
pages/page_2.png

## Step 2

Send each page image to the Vision Agent to extract questions.

## Step 3

Use Structuring Agent to convert raw output into JSON.

## Step 4

Parse answer key pages and map answers.

## Step 5

Merge answers with questions.

## Step 6

Assign topic tags based on page number.

Example:

Page 3–50 → number_system

## Step 7

Classify difficulty:

Easy
Medium
Hard

## Step 8

Validate extracted questions.

## Step 9

Insert questions into database using batch inserts.
