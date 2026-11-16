# Issue tracker: GitHub

Issues and specs live in GitHub Issues for `seansorek/DiffiScape`.
Use the `gh` CLI from this repository.

## Conventions

- Create: `gh issue create --title "..." --body "..."`.
- Read: `gh issue view <number> --json number,title,body,labels,comments`.
- List: `gh issue list --state open`; use `--label` and `--state` to filter.
- Comment: `gh issue comment <number> --body "..."`.
- Label: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`.
- Close: comment with the explanation, then `gh issue close <number>`.

Use GitHub sub-issues and issue dependencies for parent and blocking
relationships when available. Otherwise, put `Part of #<parent>` or
`Blocked by: #<number>, ...` at the top of the issue body. A ticket is
unblocked when every blocker is closed.

Before creating an issue, verify the target with
`gh repo view --json nameWithOwner`.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## Skill operations

- “Publish to the issue tracker” means create a GitHub issue.
- “Fetch the relevant ticket” means read the issue with `gh issue view`.
- For `/wayfinder`, use an issue labelled `wayfinder:map` as the map,
  with child issues as tickets. Label children `wayfinder:<type>`.
  Claim a ticket by assigning it to yourself; resolve it by commenting
  with the answer, closing it, and linking the decision from the map.
