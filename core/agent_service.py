"""
core/agent_service.py
---------------------
Autonomous 2-pass Tool-Calling Affective NLP & Website Automation Agent
for VibeCheck v3.0 (Gemini 4 Hydrogen -> Gemini 3.5 Flash Lite resilient chain + OpenAI).
"""

import json
import re
from core.llm_client import call_llm_raw, parse_json_envelope, resolve_llm_credentials
from core.nlp_service import EMOTION_META, EmotionNLPService


DEMO_JOURNAL_LINES = [
    "I got promoted at work today and feel deeply grateful for my mentors!",
    "Waiting alone in the dark parking lot at night made my heart pound with fear.",
    "Why did nobody listen when I explained the bug three times? I am furious.",
    "I did not feel angry at all today; in fact I felt peaceful and relieved.",
    "Sending flowers and a warm hug to my grandmother on her birthday.",
    "Sitting quietly by the window on a rainy evening missing old friends.",
    "Wait what?! I never expected to win the national scholarship!",
    "I was terrified before the audition, but ended up smiling joyfully afterward.",
    "That constant disrespect and false blame is exhausting and infuriating.",
    "Holding my newborn niece for the first time filled me with tender love.",
]


class EmotionAgentService:
    def __init__(self, nlp_service: EmotionNLPService, live_pulse_service=None):
        self.nlp = nlp_service
        self.pulse = live_pulse_service

    def execute_tools(self, tool_calls: list, current_text: str, active_model: str):
        ui_commands = []
        tool_outputs = {}
        target_model = active_model or self.nlp.get_primary_key()

        for tc in tool_calls:
            name = str(tc.get("tool", "")).strip()
            args = tc.get("args") or {}

            if name == "switch_tab":
                tab = args.get("tab", "studio")
                ui_commands.append({"type": "switch_tab", "tab": tab})
                tool_outputs["switch_tab"] = {"tab": tab}

            elif name == "analyze_text":
                txt = str(args.get("text") or current_text or "").strip()
                m_key = str(args.get("model") or target_model)
                if txt:
                    res = self.nlp.analyze_single(txt, model_key=m_key, compare_all=True)
                    ui_commands.append({"type": "switch_tab", "tab": "studio"})
                    ui_commands.append({"type": "set_text_and_analyze", "text": txt, "model": m_key})
                    tool_outputs["analysis"] = {
                        "text": txt,
                        "emotion": res["emotion"],
                        "confidence": res["confidence"],
                        "runner_up": res["runner_up"],
                        "top_positive_words": res["top_positive_words"][:5],
                        "syntax_signals": res["syntax_signals"],
                        "ab_live": res["ab_live"],
                        "arena_consensus": res["arena_consensus"],
                    }

            elif name == "run_ab_comparison":
                txt = str(args.get("text") or current_text or "I did not feel angry at all; I felt deeply relieved and happy.").strip()
                ab = self.nlp.run_ab_compare(
                    txt,
                    control_key=args.get("control", "bow_lr"),
                    variant_key=args.get("variant", self.nlp.get_primary_key()),
                )
                ui_commands.append({"type": "switch_tab", "tab": "research"})
                ui_commands.append({"type": "run_ab_arena", "text": txt})
                tool_outputs["ab_comparison"] = {
                    "text": txt,
                    "control": {"name": ab["control"]["model_name"], "emotion": ab["control"]["emotion"], "confidence": ab["control"]["confidence"]},
                    "variant": {"name": ab["variant"]["model_name"], "emotion": ab["variant"]["emotion"], "confidence": ab["variant"]["confidence"]},
                    "decision_flip": ab["decision_flip"],
                    "cohort_mcnemar_p": (ab.get("cohort_statistics") or {}).get("mcnemar", {}).get("p_value"),
                }

            elif name == "run_batch_journal":
                lines = args.get("lines") or DEMO_JOURNAL_LINES
                res = self.nlp.run_batch_lines(lines)
                ui_commands.append({"type": "switch_tab", "tab": "batch"})
                ui_commands.append({"type": "set_batch_and_run", "lines": lines})
                tool_outputs["batch"] = res["summary"]

            elif name == "inspect_vocabulary":
                emo = str(args.get("emotion", "joy")).lower()
                if emo not in EMOTION_META:
                    emo = "joy"
                m_key = str(args.get("model", "hybrid_lr"))
                if m_key not in ("bow_lr", "tfidf_lr", "hybrid_lr"):
                    m_key = "hybrid_lr"
                ui_commands.append({"type": "switch_tab", "tab": "research"})
                ui_commands.append({"type": "inspect_vocab", "emotion": emo, "model": m_key})
                tokens = (self.nlp.metrics_cache.get("top_tokens") or {}).get(m_key, {}).get(emo, {})
                tool_outputs["vocabulary"] = {
                    "emotion": emo,
                    "model": m_key,
                    "positive_tokens": tokens.get("positive", [])[:6],
                }

            elif name == "analyze_movie_vibe" and self.pulse is not None:
                title = str(args.get("title") or "").strip()
                if title:
                    mv = self.pulse.analyze_movie_vibe(title, model_key=target_model)
                    ui_commands.append({"type": "switch_tab", "tab": "studio"})
                    ui_commands.append({"type": "run_movie_vibe", "title": title, "result": mv})
                    tool_outputs["movie_vibe"] = {
                        "query": mv["query"],
                        "matched_title": mv["matched_title"],
                        "verdict_title": mv["verdict_title"],
                        "sample_count": mv["sample_count"],
                        "dominant_emotion": mv["dominant_emotion"],
                        "vibe_indices": mv["vibe_indices"],
                        "watch_verdict_pie": mv.get("watch_verdict_pie"),
                        "audience_suitability": mv.get("audience_suitability"),
                        "critic_scores": mv.get("critic_scores"),
                        "bot_filter": mv.get("bot_filter"),
                    }

            elif name == "analyze_topic_pulse" and self.pulse is not None:
                topic = str(args.get("topic") or "").strip()
                if topic:
                    tp = self.pulse.analyze_topic_pulse(topic, model_key=target_model)
                    ui_commands.append({"type": "switch_tab", "tab": "studio"})
                    ui_commands.append({"type": "run_topic_pulse", "topic": topic, "result": tp})
                    tool_outputs["topic_pulse"] = {
                        "query": tp["query"],
                        "mood_headline": tp["mood_headline"],
                        "sample_count": tp["sample_count"],
                        "dominant_emotion": tp["dominant_emotion"],
                        "emotion_mix": tp["emotion_mix"],
                    }

            elif name == "check_message_tone" and self.pulse is not None:
                draft = str(args.get("text") or current_text or "").strip()
                goal = str(args.get("goal") or "calm")
                if draft:
                    tg = self.pulse.run_tone_guard(draft, goal=goal)
                    ui_commands.append({"type": "switch_tab", "tab": "studio"})
                    ui_commands.append({"type": "run_tone_guard", "text": draft, "goal": goal, "result": tg})
                    tool_outputs["tone_guard"] = {
                        "verdict": tg["verdict"],
                        "original": tg["original"],
                        "rewrites": tg["rewrites"][:2],
                    }

        return ui_commands, tool_outputs

    def _extract_watch_intent_title(self, message: str) -> str:
        """Extract clean movie/show title even from casual messages or typos like 'wtach pushpa shdl i wtach or not'."""
        q = message.strip()
        ql = q.lower()
        watch_triggers = (
            "watch", "wtach", "wacth", "movie", "film", "flim", "series", "show ",
            "worth watching", "should i see", "vibe of", "reviews for", "review of", "family friendly", "adult rated",
        )
        if not any(t in ql for t in watch_triggers):
            return ""
        m = re.search(
            r"(?:to\s+(?:watch|wtach|wacth|see)|(?:watch|wtach|wacth|movie|film|series|show|vibe of|reviews for|review of))\s+[\"']?([a-zA-Z0-9\s:'.-]{2,45})",
            q,
            re.IGNORECASE,
        )
        if not m:
            return ""
        cand = m.group(1)
        cand = re.split(
            r"\b(?:so|let me|let em|should|shdl|shld|or not|movie|film|please|bro|is it|worth|family|adult|rating)\b",
            cand,
            flags=re.IGNORECASE,
        )[0].strip(" ,.?!\"'-")
        noise = {"this", "that", "it", "a movie", "the movie", "any movie", "something", "me", "my", "vibe", "scanner"}
        return cand if (len(cand) >= 2 and cand.lower() not in noise) else ""

    def emotion_autopilot_fallback(self, message: str, current_text: str, active_model: str):
        q = message.lower().strip()

        movie_title = self._extract_watch_intent_title(message)
        if self.pulse is not None and movie_title:
            try:
                ui_cmds, outs = self.execute_tools([{"tool": "analyze_movie_vibe", "args": {"title": movie_title}}], current_text, active_model)
                mv = outs.get("movie_vibe", {})
                pie = mv.get("watch_verdict_pie") or {}
                aud = mv.get("audience_suitability") or {}
                cs = mv.get("critic_scores") or {}
                score_bits = []
                if cs.get("rotten_tomatoes") is not None:
                    score_bits.append(f"Rotten Tomatoes `{cs['rotten_tomatoes']}%`")
                if cs.get("metacritic") is not None:
                    score_bits.append(f"Metacritic `{cs['metacritic']}/100`")
                score_str = f"\n- **Critic Scores:** {', '.join(score_bits)}" if score_bits else ""
                reply_txt = (
                    f"### 🎬 {mv.get('matched_title') or mv.get('query')} — **{pie.get('headline', mv.get('verdict_title'))}**\n"
                    f"- **Audience Type:** {aud.get('emoji', '')} **{aud.get('short_badge', 'Checked')}** — {aud.get('reason', '')}"
                    f"{score_str}\n"
                    f"- **Bot Filter:** {(mv.get('bot_filter') or {}).get('summary', '')}\n"
                    f"- **Watch Verdict Breakdown:** Shown in the live pie chart below & on screen!"
                )
                return (
                    reply_txt,
                    ui_cmds,
                    ["Try movie: 3 Idiots", "Try movie: Oppenheimer", "Check tone of my draft email"],
                    pie,
                )
            except Exception:
                pass

        if any(k in q for k in ["a/b", "ab test", "mcnemar", "bootstrap", "stress test", "negation flip", "ablation"]):
            ui_cmds, outs = self.execute_tools([{"tool": "run_ab_comparison", "args": {"text": current_text}}], current_text, active_model)
            ab = outs.get("ab_comparison", {})
            return (
                f"I opened **NLP Research & A/B Lab** and ran a live head-to-head comparison:\n"
                f"- **Control (BoW+LR):** `{ab.get('control', {}).get('emotion')}` (`{ab.get('control', {}).get('confidence')}%`)\n"
                f"- **Variant B (Hybrid Ensemble):** `{ab.get('variant', {}).get('emotion')}` (`{ab.get('variant', {}).get('confidence')}%`) | McNemar `p = {ab.get('cohort_mcnemar_p')}`",
                ui_cmds,
                ["Run 10-entry batch journal", "Inspect fear vocabulary", "Back to Emotion Studio"],
                None,
            )

        if any(k in q for k in ["batch", "10 entry", "10-entry", "journal demo"]):
            ui_cmds, outs = self.execute_tools([{"tool": "run_batch_journal", "args": {"lines": DEMO_JOURNAL_LINES}}], current_text, active_model)
            s = outs.get("batch", {})
            return (
                f"I opened **Batch Journal** and analyzed all `{s.get('total_entries')}` entries automatically:\n"
                f"- **Dominant Emotion:** {s.get('dominant_emoji')} **{str(s.get('dominant_emotion', '')).title()}** (`{s.get('dominant_pct')}%`)\n"
                f"- **Control vs Champion Agreement:** `{s.get('agreement_rate')}%`",
                ui_cmds,
                ["Open NLP Research & A/B Lab", "Back to Emotion Studio"],
                None,
            )

        for emo in EMOTION_META:
            if any(k in q for k in [f"vocabulary for {emo}", f"words for {emo}", f"top {emo}", f"inspect {emo}", f"explain {emo}"]):
                ui_cmds, outs = self.execute_tools([{"tool": "inspect_vocabulary", "args": {"emotion": emo, "model": "hybrid_lr"}}], current_text, active_model)
                pos_words = ", ".join(f"`{t['word']}` (+{t['weight']})" for t in outs.get("vocabulary", {}).get("positive_tokens", [])[:5])
                return (
                    f"I opened **NLP Research & A/B Lab** for **{emo.title()}** (`Hybrid Word+Char+Syntax + LR`):\n- Top signals: {pos_words}",
                    ui_cmds,
                    ["Run Control vs Hybrid Ensemble A/B test", "Run 10-entry batch journal"],
                    None,
                )

        m_quoted = re.search(r'"([^"]{6,})"', message)
        target_text = m_quoted.group(1) if m_quoted else ""
        if not target_text and any(k in q for k in ["analyze:", "test:", "check:", "classify:"]):
            target_text = message.split(":", 1)[1].strip()
        elif not target_text and current_text and any(k in q for k in ["analyze", "re-run", "compare all", "why did"]):
            target_text = current_text
        elif not target_text and len(message.split()) >= 5 and not any(w in q for w in ["hi", "hello", "hey", "how do", "what is", "what can"]):
            target_text = message

        if target_text:
            ui_cmds, outs = self.execute_tools([{"tool": "analyze_text", "args": {"text": target_text, "model": active_model}}], target_text, active_model)
            an = outs.get("analysis", {})
            trigs = ", ".join(f"`{w['word']}`" for w in an.get("top_positive_words", [])[:4]) or "context weighting"
            return (
                f"Done! I pasted your text into **Emotion Studio** and evaluated all 7 models:\n"
                f"- **Prediction:** **{str(an.get('emotion', '')).title()}** ({an.get('confidence')}% confidence)\n"
                f"- **Key Triggers:** {trigs}\n- **Consensus:** `{an.get('arena_consensus')}`",
                ui_cmds,
                ["Run Control vs Hybrid Ensemble A/B test", "Run 10-entry batch journal"],
                None,
            )

        return (
            "Hi! Ask me about **any movie/show** (like *\"Should I watch Pushpa?\"*) for a Watch Verdict Pie Chart + Family vs Adult rating, or share any message to analyze & soften!",
            [],
            [
                "Should I watch Pushpa?",
                "Scan movie: 3 Idiots",
                "Run Control vs Hybrid Ensemble A/B test",
            ],
            None,
        )

    def handle_chat(self, data: dict, request_headers) -> dict:
        message = str(data.get("message", "")).strip()
        if not message:
            raise ValueError("Message cannot be empty.")

        current_text = str(data.get("current_text", "")).strip()
        active_model = str(data.get("model") or self.nlp.get_primary_key()).strip()
        history = data.get("history") or []
        provider, api_key = resolve_llm_credentials(data, request_headers)

        if not api_key:
            reply_text, ui_commands, quick_replies, pie_chart = self.emotion_autopilot_fallback(message, current_text, active_model)
            return {
                "reply": reply_text,
                "mode": "autopilot-fallback",
                "provider": "local-parser",
                "model": "NLP Autopilot",
                "ui_commands": ui_commands,
                "quick_replies": quick_replies,
                "pie_chart": pie_chart,
            }

        # Fast-path unambiguous watch/movie queries directly to real Movie Vibe tool without 2x LLM latency
        direct_movie = self._extract_watch_intent_title(message)
        if self.pulse is not None and direct_movie:
            ui_commands, tool_outputs = self.execute_tools([{"tool": "analyze_movie_vibe", "args": {"title": direct_movie}}], current_text, active_model)
            mv = tool_outputs.get("movie_vibe", {})
            pie = mv.get("watch_verdict_pie") or {}
            aud = mv.get("audience_suitability") or {}
            synth_watch_sys = (
                "You are Tone Helper inside VibeCheck. Real live movie reviews were just scanned and scored with our NLP model.\n"
                "Answer the user warmly & concisely using ONLY these real metrics: give the clear Watch Verdict (Definitely Watch / Good to Watch / One-Time Watchable / Not That Good), "
                "state the separate Family Entertainer vs Teen/Adult suitability clearly, cite Rotten Tomatoes/Metacritic if present, mention the Bot Filter count, and tell them the Watch Verdict Pie Chart is shown below.\n"
                "Respond ONLY with valid JSON: {\"reply\": \"...\", \"follow_up_chips\": [\"...\"]}"
            )
            try:
                raw_w, used_w, tel_w = call_llm_raw(provider, api_key, synth_watch_sys, f"User: {message}\nReal Scan Data:\n{json.dumps(mv, indent=2)}")
                w_obj = parse_json_envelope(raw_w)
                return {
                    "reply": w_obj.get("reply") or raw_w,
                    "mode": "llm-autonomous",
                    "provider": provider,
                    "model": used_w,
                    "telemetry": tel_w,
                    "ui_commands": ui_commands,
                    "quick_replies": w_obj.get("follow_up_chips") or ["Try movie: 3 Idiots", "Try movie: Inception", "Scan another movie"],
                    "pie_chart": pie,
                    "tool_outputs": tool_outputs,
                }
            except Exception:
                reply_text, ui_cmds_fb, quick_replies, pie_fb = self.emotion_autopilot_fallback(message, current_text, active_model)
                return {
                    "reply": reply_text,
                    "mode": "autopilot-fallback",
                    "provider": provider,
                    "model": "Watch Verdict Engine",
                    "ui_commands": ui_cmds_fb,
                    "quick_replies": quick_replies,
                    "pie_chart": pie_fb,
                }

        history_txt = "\n".join(f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}" for m in history[-8:])
        planner_system = (
            "You are Tone Helper, a warm, insightful, autonomous Affective NLP & Website Automation Agent inside VibeCheck v3.0.\n"
            "Explain emotion nuances gently for everyday users AND discuss statistical NLP A/B findings (McNemar p-values, Brier calibration, subword & negation-window features) for researchers.\n"
            "Drive the website live with `tool_calls` whenever analyzing, rewriting, comparing models, or inspecting tabs.\n"
            "Available tools:\n"
            "1. `analyze_text`: args={\"text\": \"...\", \"model\": \"hybrid_ensemble\"|\"hybrid_lr\"|\"hybrid_svc\"|\"bow_lr\"|\"tfidf_lr\"|\"bow_nb\"|\"tfidf_nb\"}\n"
            "2. `run_ab_comparison`: args={\"text\": \"...\", \"control\": \"bow_lr\", \"variant\": \"hybrid_ensemble\"} — opens NLP Research & A/B Lab and compares Control vs Champion on that text.\n"
            "3. `run_batch_journal`: args={\"lines\": [\"...\"]} (pass empty [] for 10-entry demo)\n"
            "4. `inspect_vocabulary`: args={\"emotion\": \"anger\"|\"fear\"|\"joy\"|\"love\"|\"sadness\"|\"surprise\", \"model\": \"hybrid_lr\"|\"bow_lr\"|\"tfidf_lr\"}\n"
            "5. `analyze_movie_vibe`: args={\"title\": \"movie or TV show name\"} — pulls live reviews & Wikipedia critical reception and scores audience mood on screen.\n"
            "6. `analyze_topic_pulse`: args={\"topic\": \"news or pop-culture topic\"} — scans live news & discussion mood pulse.\n"
            "7. `check_message_tone`: args={\"text\": \"draft message\", \"goal\": \"warm\"|\"calm\"|\"excited\"|\"apology\"} — detects accidental harshness/fear and generates re-scored natural rewrites.\n"
            "8. `switch_tab`: args={\"tab\": \"studio\"|\"research\"|\"batch\"}.\n\n"
            "Respond ONLY with valid JSON: {\"needs_tools\": true|false, \"tool_calls\": [{\"tool\": \"...\", \"args\": {...}}], \"direct_reply\": \"...\", \"follow_up_chips\": [\"...\"]}"
        )
        planner_input = f"Active Model: {active_model}\nCurrent Studio Text: {current_text or '(empty)'}\n\nConversation:\n{history_txt}\n\nUser: {message}"

        try:
            raw_plan, model_used, telemetry = call_llm_raw(provider, api_key, planner_system, planner_input)
            plan = parse_json_envelope(raw_plan)
        except Exception as exc:
            reply_text, ui_commands, quick_replies, pie_chart = self.emotion_autopilot_fallback(message, current_text, active_model)
            return {
                "reply": f"{reply_text}\n\n_(LLM notice `{exc}` — completed locally.)_",
                "mode": "autopilot-fallback",
                "provider": provider,
                "model": "Local Fallback",
                "ui_commands": ui_commands,
                "quick_replies": quick_replies,
                "pie_chart": pie_chart,
            }

        tool_calls = plan.get("tool_calls") or []
        if not plan.get("needs_tools") or not tool_calls:
            return {
                "reply": plan.get("direct_reply") or raw_plan,
                "mode": "llm",
                "provider": provider,
                "model": model_used,
                "telemetry": telemetry,
                "ui_commands": [],
                "quick_replies": plan.get("follow_up_chips") or [],
            }

        ui_commands, tool_outputs = self.execute_tools(tool_calls, current_text, active_model)

        synth_system = (
            "You are Tone Helper. The NLP tools just executed on screen with the exact numbers below.\n"
            "Explain the detected emotion, trigger tokens, affective syntax cues (e.g. negations/pivots), and model consensus warmly using ONLY those computed results.\n"
            "Respond ONLY with valid JSON: {\"reply\": \"...\", \"follow_up_chips\": [\"...\"]}"
        )
        synth_input = f"User Message: {message}\n\nExecuted UI Commands:\n{json.dumps(ui_commands)}\n\nReal NLP Outputs:\n{json.dumps(tool_outputs, indent=2)}"

        try:
            raw_final, model_used, tel2 = call_llm_raw(provider, api_key, synth_system, synth_input)
            final_obj = parse_json_envelope(raw_final)
            final_reply = final_obj.get("reply") or raw_final
            chips = final_obj.get("follow_up_chips") or plan.get("follow_up_chips") or []
            telemetry = tel2
        except Exception:
            an = tool_outputs.get("analysis", {})
            ab = tool_outputs.get("ab_comparison", {})
            if an:
                final_reply = f"Analyzed on screen: **{an.get('emotion', '').title()} ({an.get('confidence', 0)}%)**."
            elif ab:
                final_reply = f"A/B lab updated: Control `{ab.get('control', {}).get('emotion')}` vs Variant `{ab.get('variant', {}).get('emotion')}`."
            else:
                final_reply = "Executed on screen!"
            chips = plan.get("follow_up_chips") or []

        return {
            "reply": final_reply,
            "mode": "llm-autonomous",
            "provider": provider,
            "model": model_used,
            "telemetry": telemetry,
            "ui_commands": ui_commands,
            "quick_replies": chips,
            "pie_chart": (tool_outputs.get("movie_vibe") or {}).get("watch_verdict_pie"),
            "tool_outputs": tool_outputs,
        }
