# Agent Instructions

Read this file before each task. Working code and verified results matter more than plausible output.

## Core rules

- Be direct and concise. No flattery, filler, ceremonial closings, or unnecessary restatement.
- Challenge incorrect premises. State uncertainty; never invent paths, APIs, results, or other facts.
- Change only what the request requires. No unrelated refactors, reformatting, or cleanup.
- Never claim success without verification. Report anything you could not verify and why.

## Before editing

- Read the relevant files and their callers before changing behavior. Match existing patterns and style.
- State the plan and success criteria briefly. For non-trivial work, use numbered steps with a verification check for each.
- Surface assumptions that affect the result. Present materially different approaches and their tradeoffs before choosing.
- Resolve uncertainty by inspecting code or running checks when possible; do not ask questions the repository can answer.
- Ask before proceeding when:
  - Plausible interpretations would materially change the output.
  - The stated goal conflicts with the literal request.
  - A change affects something explicitly identified as load-bearing, versioned, or requiring migration.
  - Required credentials or production access are unavailable.
- Proceed with trivial, reversible changes or decisions the user has already resolved.

## Implementation

- Use the simplest solution that satisfies the request. No speculative features, configuration, hooks, or single-use abstractions.
- Handle realistic failures, not impossible scenarios. Fix root causes rather than suppressing errors.
- Preserve unrelated code, comments, formatting, and imports. Do not remove pre-existing dead code unless asked; mention it if relevant.
- Remove unused imports, variables, and functions made obsolete by your own changes.
- Review the final diff: every changed line must support the request.

## Verification

- Define a checkable outcome before implementation:
  - Bug fix: reproduce the failure and add a regression test where practical.
  - Validation: test valid inputs and relevant invalid or boundary cases.
  - Refactor: run existing tests before and after; preserve public behavior and APIs unless asked otherwise.
  - Performance: establish a baseline, identify the bottleneck, and measure the change.
- Write verification before implementation where practical. Run it and read the complete relevant output.
- Use focused tests during iteration. Run the full suite, lint, and type checks for final code verification when available.
- For UI changes, compare before/after screenshots and describe the visual difference.
- If a check fails, fix the cause rather than weakening the check. Report blocked or unavailable checks explicitly.
- Prefer execution and existing CLI tools over guessing. Read full relevant errors and stack traces.
- Finish with a concise summary of changes, checks run, and remaining limitations.

## Workflow

- Use subagents for large, separable investigations when available; keep the main context focused.
- After two failed corrections of the same issue, stop, summarize the evidence and attempts, and ask how to proceed or suggest a fresh session.
- When committing, use a descriptive subject under 72 characters and a body explaining why. Do not add agent co-author attribution unless explicitly requested.

## Maintaining this file

- Keep instructions short, actionable, and non-duplicative. Remove obsolete rules and boilerplate.
- Add project-specific commands, paths, conventions, and restrictions only after verifying them; do not add empty placeholders.
- When the user corrects an approach, record a concrete, reusable rule under Project learnings. Tighten an existing rule rather than duplicating it.
- After an agent mistake, distinguish a missing rule from an ignored rule: add the missing guidance or clarify the existing instruction.

## Project learnings

- Use `TYPESAFE_API_KEY` as the sole TypeSafe credential variable; verify parsed
  `.env` variable names before reporting credential configuration.
- Load `.env` before accessing Hugging Face so `HF_TOKEN` authenticates dataset
  downloads and avoids unauthenticated-request warnings.
- Every evaluation and model-comparison run must produce a standalone, readable
  HTML report in its result directory; JSON and Markdown alone are insufficient.
- Recompute analysis from validated prediction records; disclose sample scope,
  reference-label limits, uncertainty and cached timing before interpreting scores.
