# Acme Cloud — Engineering Onboarding Guide

Welcome to engineering. This guide covers your first 90 days. Owner: Engineering Enablement
(enablement@acmecloud.example). Ask your onboarding buddy anything — there are no silly questions.

## Before you start
You will receive an email 5 days before your start date with your laptop shipping details, your work email address
and a link to sign your final paperwork. Set up two-factor authentication on your phone as soon as the account is active.

## First day
On your first day you will receive a laptop, a security badge, and access to
the internal wiki. Your onboarding buddy will walk you through the office and
introduce you to the team.

At 11:00 you meet your manager to agree on goals for the first 30 days. At 15:00 the IT desk helps you sign in to the
VPN, email, chat, and the source code host. Lunch is on the company on your first day.

## First week
- Set up your development environment using the `dev-setup` script.
- Complete the security awareness training (about 45 minutes).
- Ship your first small pull request — a documentation fix is fine.
- Read the architecture overview and the glossary in the internal wiki.
- Meet each member of your team for a 20-minute one-to-one.

### Development environment
The `dev-setup` script installs the container runtime, the language toolchains, and our command line tools. It takes
about 20 minutes. If it fails, post the log in the #dev-help channel; the rotation answers within 1 hour in office
hours. Use the shared staging environment for testing; never test against production data.

## Tools we use
- Source code: Git with pull requests on our internal Git host.
- CI/CD: every merge to `main` runs tests and deploys to staging automatically; production deploys need a manual approval.
- Chat: team channels plus #incidents for outages. Email is for external communication.
- Tickets: each task has a ticket; reference its ID in the branch name and pull request title.

## Code review rules
Every pull request needs at least one approval from a code owner. Pull
requests should stay under 400 changed lines. Use squash merge for feature
branches.

Reviewers respond within one working day. Write tests for new behaviour and keep the CI pipeline green; do not merge
with failing checks. Commit messages use the imperative mood ("Add retry to uploader").

## Security basics
Never commit secrets; use the secrets manager. Lock your screen when you leave your desk. Report phishing to
security@acmecloud.example using the report button in your mail client. Laptops use full-disk encryption, which must not be disabled.

## On-call
New engineers join the on-call rotation after 3 months. On-call shifts last
one week and are paid an extra stipend of 200 USD per shift.

You shadow an on-call engineer for one shift first. The primary on-call acknowledges alerts within 15 minutes. After an
incident, the on-call writes a blameless post-incident review within 5 working days.

## Your first 90 days
**Days 1–30:** learn the codebase, ship small changes, and complete required training.
**Days 31–60:** own a small feature end to end with your buddy as reviewer.
**Days 61–90:** present your work in the engineering demo, prepare for on-call, and agree on goals with your manager.
A feedback conversation with your manager happens at the end of month 1, month 2, and month 3.

## Who to ask
Laptop and access issues: IT desk. Pay and benefits: People Operations. Process and tooling questions: your onboarding
buddy or the #dev-help channel.
