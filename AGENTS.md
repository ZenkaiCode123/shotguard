# AGENTS.md
ShotGuard removes private data from screenshots, fully local. Rules find candidates,
Gemma 4 (Ollama, default gemma4:e4b) judges, code redraws. Spec: SPEC.md.
Rules:
- All model calls go through core/llm.py. Never send user data to a hosted service.
  Hosted mode needs SHOTGUARD_ALLOW_HOSTED=1 and is for fake data only.
- Plain functions, runnable from the CLI: detect, judge, persona, redact, attack, verify.
- Box: {id,x,y,w,h,type,action,text,persona_id}. action: dummy|edit|blackout|keep.
- Judge: {id,type,is_real,keep_prefix}. Persona: {clusters:[{persona_id,member_ids,fake}]}.
  Attacker: {leaks:[{evidence_text,inference,box_id}]}. Ignore claims with no visible evidence.
- Use Pillow only (no OpenCV). Validate every model reply in code.
- Originals live in memory only. Never write secrets to disk or logs.
- Patterns go in rules.yaml, prompts in prompts/.
- Keep changes small. Do not rewrite working code or add dependencies.
- Tests: pytest. Do not install packages. Tell me what to install.
