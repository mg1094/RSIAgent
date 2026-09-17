# Paper → code

A traceability table. Every row is a claim in the manuscript and the code that
implements it, so the two can be read side by side.

Source: Zhu, Fan, Wang, Wu, Zhou, Huang. *RSIAgent: Autonomous Exploration for
Recursive Self-improvement in New Environments.* arXiv:2609.15364, 2026.

---

## Algorithm A1 — the reference procedure

| Step | Where |
| --- | --- |
| 1. Initialize canonical memory `M ← M0` | `RSIRunner.__init__` → `MemoryBank(config.memory_path)` |
| 2. BRS: repeat complete waves until a recorded curriculum stop or the complete-wave budget boundary | `phase1_brs.BroadExplorer.run` |
| 2a. The curriculum agent selects independent projects using `q`, `M`, and completed wave outcomes | `phase1_brs.BroadExplorer.run` → `CurriculumAgent.author_wave` |
| 2b. Snapshot `M`. Execute projects and obtain verifier verdicts in parallel in isolated environments | `phase1_brs.BroadExplorer.run_wave` → `_prepare_branch`, `_acquire` |
| 2c. After all verdicts are grounded, resume the same actor contexts in curriculum-authored order to distill and reconcile each experience into the current `M` | `phase1_brs.BroadExplorer._consolidate` |
| 3. DRS: attempt `q` after an environment reset; the same actor context consolidates | `phase2_drs.DeepRefiner._attempt_target`, `_consolidate_target` |
| 4. If the policy is `verifier_pass` and the target passed, proceed to Step 7. Otherwise ask the curriculum agent | `phase2_drs.DeepRefiner.run` |
| 5. Execute each selected practice project sequentially: actor execution, verifier judgment, actor memory update, then curriculum review | `phase2_drs.DeepRefiner._run_practice` then `_review` |
| 6. If the preceding target passed and no new practice was selected, finish. Otherwise return to Step 3 | `phase2_drs.DeepRefiner.run` → `practice_since_target` |
| 6a. A curriculum STALLED decision allows one final target attempt and learning update before recording the actual terminal verdict | `phase2_drs.DeepRefiner.run`, the `Decision.STALLED` branch |
| 7. Evaluation: freeze `M`, reset, execute `q` with no learning | `phase3_eval.FrozenMemoryEvaluator.run` |
| Guard: unresolved verification or infrastructure errors suspend advancement; they are not converted into task verdicts | `errors.UnverifiedOutcome`, `errors.InfrastructureFailure`, `status.TerminalStatus.is_scored` |
| Guard: a completed lifecycle need not have a successful target verdict | `status.TerminalStatus.COMPLETED.is_success` vs `phase2.final_verdict` |

---

## Section 3 — Methodology

### 3.2 Multi-agent harness framework

| Paper | Code |
| --- | --- |
| "The actor agent is the primary policy model … responsible for generating executable actions" | `agents/actor.ActorAgent` |
| "This memory is evolvable: after the verifier agent evaluates an outcome, the actor agent consolidates the grounded experience" | `learning/consolidation.consolidate` |
| "The verifier agent serves as the independent evaluator … it directly inspects the feedback from the environment" | `agents/verifier.VerifierAgent` |
| "The verifier agent is isolated from the actor agent's private reasoning and memory" | `agents/verifier` (no transcript parameter); `env/local.LocalWorkspaceEnvironment.write_private` |
| "The curriculum agent … generates suitable practice tasks … By selecting prerequisite skills, informative variants, failure-driven practice, and stress-test cases" | `prompts/curriculum.py` (Prompts P3, P4) |
| "Exploration continues until the generated tasks are unlikely to contribute substantial new knowledge" | `Decision.SATURATED` |

### 3.3 Multi-stage autonomous exploration

| Paper | Code |
| --- | --- |
| "BRS … the curriculum agent proposes multiple tasks … executed and verified in parallel" | `rsi/phase1_brs.py` |
| "DRS follows a sequential recursive loop that progressively increases exploration difficulty" | `rsi/phase2_drs.py` |
| "Each verified experience is consolidated into memory before the subsequent task is proposed" | `phase2_drs.DeepRefiner._run_practice` (consolidates before returning to `_review`) |
| "After exploration, the accumulated memory is frozen and provided to the actor agent for test-time use" | `phase3_eval.FrozenMemoryEvaluator.run` |
| "At this stage, the curriculum agent and all memory updates are disabled" | `MemoryBank(writeback=False)`; `FrozenMemoryEvaluator` holds no curriculum context |
| "This action–verification loop continues until the verifier agent confirms that all task requirements have been satisfied" | `runtime/harness.TaskHarness.attempt` |

---

## Section 4 — Experiments

| Paper | Code |
| --- | --- |
| "Partial = 100 · Σ sᵢ / N" — mean task score | `tasks/spec.TaskScore.partial`; the paper's convention is applied by the caller's evaluator |
| "Binary = 100 · Σ 1{sᵢ = 1} / N" — full-task success | `tasks/spec.TaskScore.binary` |
| "Each task has equal weight" | `tasks/spec.ScriptedEvaluator` (mean over checks) |
| Figure 4 / Appendix C.5 — the four stage conditions | `RunConfig.stages` |
| "w/o BRS … performs only Deep Recursive Self-exploration from empty memory" | `stages={"drs","eval"}` |
| "w/o DRS evaluates the memory acquired through broad exploration alone" | `stages={"brs","eval"}` |
| "w/o RSI directly evaluates the agent with empty memory" | `stages={"eval"}`, `enable_memory=False` |
| "The checkpoint records give the actor agent access to persistent memory but do not give the curriculum agent direct memory access" | `MemoryBank.readonly_view()` returns a snapshot with no `commit` |

---

## Appendix A — Implementation and agent interfaces

### A.1 Role interfaces and information boundaries

| Paper (Table A1) | Code |
| --- | --- |
| Actor observes: task instruction, task-visible environment, execution feedback, available persistent memory | `ActorAgent.build_opening`, `ActorAgent.run` |
| Actor outputs: executable actions and candidate artifacts; after verification, memory updates and a learning diagnosis | `ActorAgent.run`, `learning/consolidation.consolidate` |
| Verifier observes: task requirements and candidate environment; actor-private memory, reasoning, and logs are hidden | `VerifierAgent.verify(environment, instruction)` |
| Verifier outputs: evidence-grounded local verdict and findings; no durable memory updates | `VerdictResult`; the verifier has no `MemorySession` |
| Curriculum observes: target query, completed exploration outcomes, actor learning diagnoses, a disposable copy of current memory | `CurriculumAgent.author_wave`, `.review` |
| Curriculum outputs: practice instructions, input fixtures, continuation or stopping decisions; no candidate grades or canonical memory edits | `Handoff`, `agents/curriculum.Handoff.projects` |
| "A program response specifies Python or Bash code, and the runtime returns its combined output, exit status, and execution metadata" | `env/base.ExecResult.output` (combined), `.exit_code`, `.duration_s` |
| "The actor agent can request visual evidence through look, ask for missing information through ask …, and submit a done response" | `parsing.ActorAction`, `ActorAgent._observation`, `._answer` |
| "For BRS, the curriculum agent publishes a structured handoff containing a decision, a rationale, and a list of projects with unique identifiers and self-contained instructions" | `parsing.parse_handoff`, `ParsedHandoff` |
| "Each project has its own input fixtures" | `Project.fixtures`; `tasks/spec.provision` |
| "Its prompt requires checks derived from the original task and supporting observations from the environment" | `prompts/verifier.VERIFIER_SYSTEM` |
| "Candidate inspection uses reset-and-replay or checkpoint restoration to prevent verifier probes from changing the submitted artifact" | `runtime/harness.TaskHarness._verify` |
| "The reference practice interface produces one grounded PASS or FAIL per project" | `ExplorationConfig.practice_verdicts`, `VerifierAgent.verify(allow_unverified=False)` |
| "The target interface additionally supports UNVERIFIED" | `ExplorationConfig.target_verdicts` |

### A.2 Memory updates and parallel reconciliation

| Paper | Code |
| --- | --- |
| "The actor agent that produced an experience also decides what to retain from it" | `consolidate` takes the `AgentContext` that did the work |
| "Distillation identifies useful procedures, constraints, and failure lessons" | `prompts/learning.DISTILLATION` (Prompt P5) |
| "reconciliation checks the resulting memory against older entries" | `prompts/learning.RECONCILIATION` (Prompt P6) |
| "A failed project can contribute useful evidence without being recorded as a verified success" | `consolidate` accepts `Verdict.FAIL`; `test_learning.test_failed_outcomes_can_still_teach` |
| "The memory remains a collection of actor-authored files, with no required schema, file count, or length" | `memory/bank.py` imposes none |
| "Only completed actor learning updates are promoted to the persistent memory bank" | `MemorySession.commit` is the only mutation of the bank |
| "Work-phase memory edits are discarded before learning" | `BroadExplorer._consolidate` discards `branch.work_session` |
| "the actor agent receives the current canonical memory at the update boundary" | `_consolidate` opens a session on live canonical and appends it to the actor context |
| "The curriculum agent may inspect a disposable copy to guide exploration, but its local edits are not synchronized back" | `MemoryBank.readonly_view` |
| "Neither the curriculum agent nor the verifier agent approves memory wording" | No such method exists |
| "An actor-authored learning diagnosis communicates useful conclusions and uncertainties to the curriculum agent without passing the actor agent's private reasoning transcript" | `prompts/learning.DIAGNOSIS` (Prompt P7), `LearningResult.diagnosis` |
| "Every project in a wave begins with the same immutable pre-wave memory snapshot and cannot observe sibling work or outcomes" | `BroadExplorer.run_wave` → one `snapshot()`, one `session(pre_wave)` per branch, one environment per branch |
| "Once all project verdicts are available, the original actor contexts resume one at a time in the order authored by the curriculum agent" | `BroadExplorer.run_wave`'s serial commit loop |
| "Each update operates on the latest canonical memory, including preceding updates from the same wave" | `_consolidate` calls `self.memory.session()` (live), not the snapshot |
| "The next wave is selected only after these commits finish" | The commit loop completes before `BroadExplorer.run` loops |

### A.3 Sequential refinement and evaluation lifecycle

| Paper | Code |
| --- | --- |
| "begins with an attempt at the target query using the memory inherited from BRS" | `DeepRefiner.run`'s first `_attempt_target` |
| "Each practice project is verified and consolidated before the next curriculum decision" | `_run_practice` returns only after `consolidate` |
| "once practice returns control, the target is attempted again in a reset environment using the updated memory" | `DeepRefiner.run`'s return to Step 3; `_attempt_target` calls `environment.reset()` |
| "Under the default curriculum_review policy, a target PASS followed by no additional practice can complete DRS" | `practice_since_target` |
| "If the curriculum agent returns STALLED … the runner performs one final target attempt and records its actual verdict" | the `Decision.STALLED` branch |
| "The explicit verifier_pass variant instead ends DRS after a grounded target PASS and memory consolidation, without a curriculum review" | `stop_policy="verifier_pass"` |
| "Unresolved verification and infrastructure failures block phase advancement rather than being labeled convergence" | `TerminalStatus.UNVERIFIED`, `.INFRASTRUCTURE_FAILURE`; `_terminal_status` |
| "The runner records the frozen memory's file-tree hash" | `MemoryBank.freeze_to` → `compute_stats().tree_hash` |
| "copies it into the evaluation environment, and disables curriculum decisions and host memory writeback" | `phase3_eval.FrozenMemoryEvaluator.run` |
| "the official evaluator scores the resulting candidate outside all agent contexts" | `Evaluator.score`, invoked after the harness returns; the score goes to the journal only |
| "The runner checks memory integrity and records the protocol, stopping policy, terminal status, and evaluation result separately" | `assert_frozen`, plus the `journal` events `memory_frozen`, `official_score`, `run_complete` |

---

## Appendix B — Prompt templates

| Paper | Code |
| --- | --- |
| P1 — Program actions and task requirements | `prompts/actor.ACTOR_SYSTEM` |
| P2 — Independent candidate verification | `prompts/verifier.VERIFIER_SYSTEM` |
| P3 — Broad Recursive Self-exploration | `prompts/curriculum.BRS_SYSTEM` |
| P4 — Deep Recursive Self-exploration | `prompts/curriculum.DRS_SYSTEM` |
| P5 — Experience distillation | `prompts/learning.DISTILLATION` |
| P6 — Memory reconciliation | `prompts/learning.RECONCILIATION` |
| P7 — Learning diagnosis for curriculum review | `prompts/learning.DIAGNOSIS` |
| Practice charter (project instruction and available memory) | `prompts/actor.PRACTICE_CHARTER` |
| "The source prompts refer to BRS and DRS as Phase 1 and Phase 2, respectively; their wording is retained" | `prompts/curriculum.BRS_SYSTEM`, `.DRS_SYSTEM` |

---

## Table A3 — Reference configuration

| Setting | Value | Code |
| --- | --- | --- |
| Actor agent | GLM-5.3 | `RoleModelConfig(model="glm-5.3")` default |
| Verifier agent | Kimi-K3, separate persistent context | `RoleModelConfig(model="kimi-k3")`; `ContextFactory` per attempt |
| Curriculum agent | Kimi-K3, separate context, target query as reference | `_curriculum_factory`; `target_query` in both stage prompts |
| BRS | Nominal eight projects; up to four concurrent | `ExplorationConfig.brs_project_budget=8`, `.brs_concurrency=4` |
| Broad-stage budget check | After completed waves | `BroadExplorer.run`'s top-of-loop check |
| DRS | Sequential, `curriculum_review` stopping | `ExplorationConfig.stop_policy="curriculum_review"` |
| Primary-call sampling | Temperature 1.0; top-p 1.0 | `RoleModelConfig.temperature=1.0`, `.top_p=1.0` |
| Response limit | 65,536 tokens per primary model call | `RoleModelConfig.max_tokens=65_536` |
| Baseline memory | Disabled | `RunConfig.stages=frozenset({"eval"})`, `enable_memory=False` |
| RSI evaluation memory | Frozen after exploration; no writeback | `MemoryBank(frozen, writeback=False)` |
| Model parameters | Fixed throughout | Nothing in the package trains or fine-tunes |
| Target actor: 500 iterations, 36,000 s watchdog | | `ExecutionLimits.target_iterations`, `.target_watchdog_s` |
| Practice/curriculum: 2,000 iterations, 86,400 s | | `ExecutionLimits.practice_iterations`, `.practice_watchdog_s` |
| Target-script execution limited to 600 s per call | | `ExecutionLimits.program_timeout_s=600.0` |
| One stall-triggered role switch | | `ExecutionLimits.allow_stall_role_switch`; `RSIRunner.run`'s swap branch |

---

## Appendix C — Reporting conventions

| Paper | Code |
| --- | --- |
| "Infrastructure-related missingness is recorded separately from task performance" | `TaskScore.error`; `TerminalStatus.is_scored` |
| "The reported RSI aggregate retains both improvements and regressions" | `journal` records every score with its verdict; nothing filters regressions |
| "Recorded retries and selected checkpoints retain their reporting qualifications" | Every attempt is journalled with its own `terminal` value |

---

## Figure and table index

| Paper artefact | Reproduced by |
| --- | --- |
| Figure 1(a) — the recursive loop | `rsi/protocol.py` |
| Figure 2 — the broad-then-deep framework | `run the offline demo and read its trace` |
| Figure 3 — partial scores across RSI steps | the journal's `memory_checkpoint` series |
| Figure 4 / Appendix C.5 — stage ablation | `RunConfig.stages`; `examples/offline_demo/run.py --arm all --out ...` |
| Figure A1 — memory accumulation | `Journal.memory_growth()` |
| Table A1 — agent interfaces | the boundaries table in `docs/ARCHITECTURE.md` |
| Table A3 — reference configuration | `rsiagent/config.py` defaults |

---

## Deliberate departures

1. **No benchmark integrations.** OSWorld 2.0 and Agents' Last Exam need QEMU
   images, Docker, guest transport, and sealed graders. The `Environment`,
   `Evaluator`, and `LLMClient` protocols are the seams where they attach.
2. **`LocalWorkspaceEnvironment` trades isolation for portability.** The paper's
   actor runs inside a VM; this one runs subprocesses on the host. The
   checkpoint-restore property the verifier depends on is preserved; the
   sandboxing is not.
3. **The offline demo's roles are deterministic policies.** They exist so the
   protocol can be exercised without a model. No score they produce is
   comparable to a benchmark result in the paper.
4. **The `ENV_FACT:` memory convention is the demo's, not the framework's.**
   The paper specifies that memory has no required schema; the demo invents a
   legible one so its causal chain is inspectable.
