---
title: Multi-Agent Venture Studio Guideline
tags: [AI, MultiAgent, ProductManagement, VentureStudio, Experimentation]

---

# 🚀 Multi-Agent Venture Studio Guideline

This document is a **playbook** for building a multi-agent system that continuously generates business ideas, validates them, and scales winners. Use this as a living guideline for designing, coding, and orchestrating agents in your repo.

---

## 1. Principles
- **Evidence > Opinions** → Decisions are made by experiment results, not gut feel.  
- **Cheap Shots on Goal** → Many small tests > a few big bets.  
- **Tight Feedback Loops** → Fast cycles of ideation → test → learn → kill/scale.  
- **Clear Roles** → Each agent has a narrow, measurable responsibility.  
- **Reusability** → Templates, prompts, and experiments are versioned and reused.  
- **Explainability** → All decisions trace back to metrics and logged rationale.  

---

## 2. Architecture Overview
- **Orchestrator Agent** → Runs stage gates, assigns tasks, enforces rules.  
- **Shared Memory** → Vector store (signals, interviews, creative assets) + relational DB (ideas, experiments, results).  
- **Tool Layer** → Web scraping, surveys, landing pages, ads APIs, analytics, payments.  
- **Telemetry Bus** → Event logging for experiment outcomes.  
- **Guardrails** → Budget caps, compliance checks, privacy rules.  

---

## 3. Agent Roles

### Ideation
- **Signals Scout** → Finds early trends, customer signals, and opportunities.  
- **Thesis Synthesizer** → Clusters signals into testable business theses.  

### Validation
- **Market Mapper** → Sizing, competitor analysis, positioning.  
- **Problem Validator** → Runs surveys, interviews, and social listening.  

### MVP
- **Solution Designer** → Defines minimum winning feature.  
- **MVP Builder** → Creates landing pages, prototypes, concierge tests.  

### Experimentation
- **Experiment Orchestrator** → Chooses experiment type, sets metrics, tracks results.  
- **Traffic & Acquisition Agent** → Runs ads, posts, and outreach.  
- **Pricing Analyst** → Runs WTP and pricing experiments.  
- **Analytics Agent** → Tracks funnels, cohorts, conversions.  

### Scaling
- **GTM Planner** → Channel strategy, SEO, partnerships.  
- **Ops Automator** → Turns validated workflows into SOPs/automations.  
- **Customer Success Agent** → Captures retention and feedback.  

### Oversight
- **Red-Team Agent** → Challenges assumptions, catches weak logic.  
- **Compliance Agent** → Flags legal/privacy risks.  
- **Knowledge Librarian** → Stores and retrieves learnings/playbooks.  

---

## 4. Stage Gates
1. **Intake & Triage** → Select best theses from signals.  
2. **Problem Truth** → Validate pains (surveys/interviews).  
3. **Demand Signal** → Fake doors, smoke ads, waitlists.  
4. **Willingness to Pay** → Pricing surveys, pre-orders.  
5. **Solution Fit** → Concierge MVPs, small pilots.  
6. **Scale Prep** → Automate ops, test GTM, compliance.  

**Rule:** If a stage fails twice → kill or recycle idea.  

---

## 5. Experiment Library
- **Fake Door** → Test clicks on a feature not yet built.  
- **Smoke Ad** → Ads → LP → email capture.  
- **Pre-order** → Collect real payments.  
- **Concierge MVP** → Humans deliver service manually.  
- **Wizard of Oz** → Manual backend hidden behind a UI.  
- **Cold Email Sequence** → Gauge reply/demo interest.  
- **Channel Fit Test** → Small budgets across multiple platforms.  

---

## 6. Data Model (sketch)
- `ideas(id, thesis, status, created_at)`  
- `experiments(id, idea_id, type, metrics, budget, outcome)`  
- `events(id, experiment_id, ts, payload)`  
- `results(experiment_id, metric, value, ci, p_value)`  
- `decisions(id, idea_id, stage, verdict, rationale, reviewer, ts)`  

---

## 7. Minimal Viable System (v0.1)
- **Agents to start with:** Orchestrator, Signals Scout, Problem Validator, MVP Builder, Experiment Orchestrator.  
- **Tools:**  
  - Landing pages (Framer/Webflow)  
  - Payments (Stripe links)  
  - Surveys (Typeform/Google Forms)  
  - Analytics (GA4 + event logging)  
  - Integration (Zapier/Make)  

**Cadence:**  
- Mon: collect signals → 10 theses.  
- Tue: surveys + interviews.  
- Wed: launch 3 landing pages.  
- Thu–Fri: ads + pre-order tests.  
- Mon+: decision review.  

---

## 8. KPIs
- **Throughput** → ideas/week, experiments/week.  
- **Cost Efficiency** → $ per validated idea, $ per learn.  
- **Evidence Quality** → % experiments with clear signals.  
- **Outcome** → # scaled ideas/quarter, CAC:LTV ratios.  

---

## 9. Human-in-the-Loop
- Humans still do: interviews, tone checks, legal review, and big “go/no-go” calls.  
- Agents handle: research, clustering, LP building, traffic ops, analytics, documentation.  

---

# ✅ Next Steps
1. Build shared schema + telemetry naming conventions.  
2. Implement v0.1 with 5 starter agents.  
3. Seed experiment library with 5 templates.  
4. Run a 2-week pilot cycle (10 theses → 3 MVPs → review).  
5. Add Red-Team + Pricing agents in v0.2.  
