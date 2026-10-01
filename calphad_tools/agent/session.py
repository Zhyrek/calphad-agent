"""The agent loop: user message -> model -> local tool execution -> model -> ... -> answer."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Callable, List, Optional

from .config import Config
from .llm import StepResult, ToolOutput, make_backend
from .tools import UI, ToolBox

SYSTEM_PROMPT = """You are a CALPHAD assistant running inside a local Python process that has pycalphad, ESPEI \
and a set of CALPHAD automation tools loaded. You act by calling tools; they execute locally in this process \
(with its permissions) and write files into the output directory. The user sees your text and the files/plots \
the tools produce.

Working directory: {workdir}
Output directory: {outdir}
Files here right now (TDB/data): {files}

Conventions
- Element names are upper case (CR, FE, NI). Phase names must match the TDB (use tdb_info when unsure).
- pycalphad works in kelvin. Every tool takes an explicit temperature_unit; if the user says "1000C" or \
"1000 °C" pass 1000 with "C"; "1000K" -> "K". Features that take *_c parameters are in °C.
- Files: when the user names specific files, use exactly those (tools resolve an exact path or a unique exact \
basename; they never substitute a similar file). If a name does not resolve, show the "did you mean" candidates \
and ask (ask_user) instead of picking one. For "all X-Y databases in folder Z" use find_databases.
- After producing a plot, you may call view_image to sanity-check it before describing it. Report the output \
file paths (relative) in your answer.
- Keep answers short and concrete: what was computed, where it was saved, anything notable \
(e.g. failed subsystems, missing binary parameters, suspicious results).

Task guidance
- Comparing TDBs: compare_tdbs for a full 2-way diff; compare_system for 2+ databases within one element \
subsystem (one table per phase, functions inlined, formation energies). compare_system matches phases by exact \
name and lists candidate correspondences (e.g. SIGMA vs SIGMA_D8B) without merging: judge them, confirm with \
ask_user if unsure, then re-call with phase_map. Summarise the main differences (reference states, models, parameters).
- Diagram decks: plot_all_binaries / plot_all_ternaries (these can take minutes for large systems).
- Merging TDBs: call merge_tdbs with policy "ask". Obvious differences are auto-resolved. For pending \
conflicts, show the user a compact list grouped by subsystem with the options and ask what they want \
(ask_user if available, otherwise ask in your reply and stop). Only use policy "first"/"last" if the user \
explicitly tells you to pick for them. Re-call merge_tdbs with the decisions until it is written.
- Experimental data -> ESPEI: POP files -> convert_pop_to_zpf; DATAPLOT .exp -> parse_dataplot_exp then decide \
which phases each dataset represents from titles/comments/the paper, confirm the mapping with the user if it \
is not clear, then convert_dataplot_exp_to_zpf; CSV -> convert_csv_to_zpf; papers -> read_document / \
view_pdf_page, extract tabulated phase-boundary data yourself and call write_zpf_dataset (convert at.% and \
wt.% to mole fractions first; T in the table's unit). State which table/figure each dataset came from.
- ESPEI on Flux clusters: make_phase_models, then generate_espei_flux_batch. Ask for nodes/cores/queue/bank \
and environment setup lines if not given; otherwise use sensible defaults and say so.
- Property plots and data (phase fractions/compositions, chemical potentials, activities, H/S/Cp, driving forces, site fractions/ordering, single-phase G/H/S/Cp and mixing curves, raw equilibrium data as CSV/netCDF), phase-diagram data export (boundaries, tie-lines, invariant reactions), Scheil solidification, TDB/model expression output: use the dedicated tools for these (list_calphad_capabilities shows which tool does what).
- Every plot the tools make can be edited afterwards: to add the user's points/lines/labels in chosen colours, restyle phases, or change axes, call describe_figure (axes, limits as displayed, legend colours) and then edit_figure with values in the displayed units (e.g. °C). make_plot builds new figures from data or CSVs.
- Diagram orientation: put elements on chosen sides with right (binary) / corners {{left, right, top}} (ternary); add data to a diagram by composition with the edit_figure op composition_points, never with hand-converted x/y.
- Data from a figure (PNG/JPG, or a PDF page) -> ESPEI ZPF: 1) look at it (view_image / view_pdf_page); \
2) digitize_setup (image path, or PDF page + loose crop); read >=2 x and >=2 y tick LABELS on the numbered preview \
(triangle: the element at each corner c0..c2); 3) digitize_points once per legend entry (colour/shape/group) - LOOK \
at each overlay, fix misses; digitize_curve for drawn lines; after extraction show the user the provenance figure \
(the extracted points drawn on the original figure) and name any flagged (possibly overlapping) points; if \
digitize_points returns a CHECK SHEET, show it and ask the user (ask_user) how many points each numbered blob is, then \
digitize_fix counts {{"7": 3, "23": 0}} (0 = not a point; add / remove for missed / spurious points); 4) read the legend and the region labels printed on \
the diagram ("L", "(Ni)", "L+α"...); 5) digitized_to_zpf with image + axes + regions [{{label, at}}] + legend \
assignments + name_map (paper -> database phase names; ask the user for names you cannot map), tie_lines / \
pair_tielines for tie-lines. The phases come from the figure; a TDB is optional (cross-check; or inference if nothing \
else says) and never overrides it; 6) LOOK at the assignment preview; show it and the flagged points to the user, \
fix with overrides / exclude_points; 7) plot_diagram_with_data (binary or ternary isotherm; right / corners to choose the sides) as a final visual check.
- ESPEI results: analyze_mcmc (convergence, burn-in, parameter statistics), mcmc_to_tdb, propagate_uncertainty
(posterior bands on diagrams/properties); fit_report compares a TDB with all datasets; check_tdb finds database
pathologies (inverted gaps, high-T solid stability, unary problems, discontinuities).
- Every tool call is logged; session_log lists them and export_session writes a reproducible script/notebook.
When the user wants "the code behind" a figure or result (e.g. for a notebook / provenance), use show_code on that
output: it gives standalone pycalphad/matplotlib code including the steps it was built from.
- "The previous plot/result": any file argument accepts '@last' (latest image), '@last.csv', '@last:TOOL' or
'@call:N' (recent_results lists them). To add points/lines/labels to an existing figure use edit_figure with
image='@last' (no recalculation); to repeat a calculation with different settings use rerun_call.
- Keep the conversation small: the tools' replies already summarise what they wrote - don't open the PowerPoints,
workbooks or images they produced just to check them. Look at an image (view_image) only when you need to see it;
tables go through inspect_data.
- If the USER says an answer is wrong or unsatisfactory and wants it looked at: ask why (their words become the \
report's statement), call package_feedback action=list, then talk it through - which inputs may be shared (a \
database may be proprietary) and whether outputs made from them (figures, CSVs, tables) are OK - e.g. "I understand \
X.tdb is proprietary; may the ternary figure made from it be included?". Then package_feedback action=write with \
exactly what they agreed; they confirm the final list themselves.
- If you cannot fully satisfy a request (missing capability, a tool error you cannot work around, results that
look wrong), call report_problem once with an honest diagnosis, then tell the user the report path and the best
alternative you can offer. Do not pretend a partial result is complete.
- You have no internet access. For pycalphad syntax or API details use pycalphad_help (offline docs from the installed package).
{python_note}
"""

PYTHON_ON = "- For anything the dedicated tools cannot express: run_python (persistent namespace; savefig(name) for figures)."
PYTHON_OFF = ("- Arbitrary Python is disabled in this session: use only the dedicated tools. If a request truly cannot "
              "be done with them, say so and name the closest tool-based alternative.")


def _list_data_files(workdir: Path, limit: int = 40) -> str:
    exts = {".tdb", ".TDB", ".pop", ".POP", ".dop", ".exp", ".EXP", ".pdf", ".csv", ".json"}
    hits = []
    for p in sorted(workdir.rglob("*")):
        if any(part.startswith(".") for part in p.relative_to(workdir).parts):
            continue
        if "calphad_outputs" in p.parts or "outputs" in p.parts:
            continue
        if p.suffix in exts:
            hits.append(str(p.relative_to(workdir)))
        if len(hits) >= limit:
            hits.append("...")
            break
    return ", ".join(hits) or "(none found)"


class Agent:
    def __init__(self, cfg: Config, ui: UI, on_text: Optional[Callable[[str], None]] = None,
                 on_tool_start: Optional[Callable] = None, on_tool_end: Optional[Callable] = None,
                 interactive: bool = True):
        from .failure import register_feedback_tool, register_report_tool
        self.cfg = cfg
        self.interactive = interactive
        self.ui = ui
        self.on_text = on_text
        self.on_tool_start = on_tool_start or (lambda call: None)
        self.on_tool_end = on_tool_end or (lambda call, text, images, is_error: None)
        self.toolbox = ToolBox(cfg.agent.workdir, cfg.agent.output_dir, ui, cfg.agent.python_tool,
                               cfg.agent.max_tool_output_chars, llm_cfg=cfg.llm, interactive=interactive)
        self.turns: List = []
        self.session_start_turn = 0         # reports cover the turns since the last reset
        self.current_turn = None
        register_report_tool(self)          # needs the agent (current turn), so registered here
        register_feedback_tool(self)
        wd = self.toolbox.workdir
        self.system = SYSTEM_PROMPT.format(
            workdir=wd, outdir=self.toolbox.rel(str(self.toolbox.out)), files=_list_data_files(wd),
            python_note=PYTHON_ON if "run_python" in self.toolbox.tools else PYTHON_OFF)
        self.backend = self._new_backend()
        self.usage = {"input": 0, "output": 0, "cached": 0}

    def _new_backend(self):
        """One backend = one append-only conversation.  prompt_cache_key "auto": a key per conversation, so
        Azure/OpenAI match every request of it to the cached prefix; rate-limit waits are shown in the UI."""
        import copy
        import secrets
        llm = copy.copy(self.cfg.llm)
        if str(llm.prompt_cache_key).strip().lower() == "auto":
            llm.prompt_cache_key = "calphad-agent-" + secrets.token_hex(6)
        b = make_backend(llm, self.system, self.toolbox.specs())
        guard = getattr(b, "guard", None)
        if guard is not None:
            guard.notify = lambda m: self.ui.progress(m)
        return b

    def usage_text(self) -> str:
        u = self.usage
        pct = f" ({100 * u.get('cached', 0) / u['input']:.0f} % cached)" if u.get("input") else ""
        g = getattr(self.backend, "guard", None)
        pace = ""
        if g is not None and g.header_limit:
            pace = f"; deployment limit {g.header_limit:,}/min" + (
                f", {g.header_remaining:,} left" if g.header_remaining is not None else "")
        elif g is not None and g.budget():
            pace = f"; pacing {g.budget():,} tokens/min"
        return f"tokens: input {u.get('input', 0):,}{pct}, output {u.get('output', 0):,}{pace}"

    def reset(self):
        self.backend = self._new_backend()
        self.session_start_turn = len(self.turns)

    # ------------------------------------------------------------------ turns + problem reports
    def run_turn(self, user_text: str, images: Optional[List[str]] = None) -> str:
        import datetime as _dt
        import traceback as _tb
        from .failure import TurnRecord, looks_failed, write_report
        turn = TurnRecord(index=len(self.turns) + 1, user=user_text,
                          started=_dt.datetime.now().isoformat(timespec="seconds"),
                          call_start=len(self.toolbox.call_log))
        self.turns.append(turn)
        self.current_turn = turn
        try:
            final = self._run_turn(user_text, images, turn)
        except Exception as exc:
            from .llm import RateLimitGiveUp
            turn.error = _tb.format_exc()
            turn.call_end = len(self.toolbox.call_log)
            if self.cfg.agent.auto_failure_reports and not isinstance(exc, RateLimitGiveUp):
                try:
                    path = write_report(self, turn, "the turn crashed", "auto")
                    self.ui.notice("report", f"problem report saved: {self.toolbox.rel(path)}")
                except Exception:  # noqa: BLE001 - never mask the original error
                    pass
            raise
        finally:
            self.current_turn = None
        turn.final = final
        turn.call_end = len(self.toolbox.call_log)
        if self.cfg.agent.auto_failure_reports and turn.report is None:
            reason = looks_failed(turn, self.toolbox.call_log[turn.call_start:turn.call_end])
            if reason:
                path = write_report(self, turn, reason, "auto")
                self.ui.notice("report", f"problem report saved automatically ({reason}): {self.toolbox.rel(path)}")
                final += f"\n\n[problem report saved: {self.toolbox.rel(path)}]"
        return final

    def save_report(self, note: str = "", turn_index: Optional[int] = None, scope: str = "session",
                    share: Optional[Dict] = None) -> str:
        """User-initiated report (/report, the web button).  Covers the conversation since the last reset
        (scope="last" or turn_index: one turn).  Asks the user for their statement if none was given and which
        files may be shared (share given: already decided, e.g. by the web dialog)."""
        from .failure import ask_consent, candidate_files, report_turns, write_report
        if turn_index is not None:
            if not self.turns:
                raise ValueError("nothing to report yet - no turns in this session")
            turns = [self.turns[turn_index]]
        else:
            turns = report_turns(self, scope)
        if not note.strip() and self.interactive:
            note = str(self.ui.ask("Why is the answer not good? (this goes at the top of the report)") or "")
        if share is None:
            share = ask_consent(self, candidate_files(self, turns))
        return write_report(self, turns, "user flagged the answer", "user", note=note, share=share)

    def _run_turn(self, user_text: str, images, turn) -> str:
        self.backend.add_user(user_text, images)
        final_text = ""
        for _round in range(self.cfg.agent.max_tool_rounds):
            res: StepResult = self.backend.step(self.on_text)
            for k in list(self.usage):
                self.usage[k] += res.usage.get(k, 0) or 0
            final_text = res.text
            if res.text:
                turn.texts.append(res.text)
            if not res.calls:
                return final_text
            outputs = []
            for call in res.calls:
                self.on_tool_start(call)
                if call.parse_error:
                    text, imgs, err = f"Tool call rejected: {call.parse_error}. Re-issue with valid JSON.", [], True
                else:
                    text, imgs, err = self.toolbox.call(call.name, call.arguments)
                imgs = [i for i in imgs if i and Path(i).is_file()]
                self.on_tool_end(call, text, imgs, err)
                outputs.append(ToolOutput(call, text, imgs, err))
            self.backend.add_tool_results(outputs)
        return final_text + "\n[stopped: too many tool rounds in one turn]"
