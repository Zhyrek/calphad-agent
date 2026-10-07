"""Minimal browser chat UI (stdlib only).  `calphad-agent --web` then open http://127.0.0.1:8765

Single-user, local only (binds 127.0.0.1).  The page polls /api/events; images and
generated files (pptx, tdb, json) are served from the working directory only.
"""
from __future__ import annotations

import json
import os
import mimetypes
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

from .tools import UI


class WebUI(UI):
    def __init__(self, hub: "Hub"):
        self.hub = hub

    def _wait(self, ev: Dict) -> str:
        qid = self.hub.new_question()
        ev["qid"] = qid
        self.hub.emit(ev)
        return self.hub.wait_answer(qid)

    def ask(self, question, options=None):
        return self._wait({"type": "question", "text": question, "options": options or []})

    def confirm(self, title, body):
        return self._wait({"type": "confirm", "text": title, "code": body}) == "yes"

    def progress(self, msg, key=None):
        self.hub.emit({"type": "progress", "text": msg, **({"key": key} if key else {})})

    def notice(self, kind, text):
        if kind == "report":
            self.hub.emit({"type": "report", "text": text})
            return
        self.hub.emit({"type": "review", "text": text, "ok": kind == "review_ok"})


class Hub:
    def __init__(self):
        self.events: List[Dict] = []
        self.lock = threading.Lock()
        self.busy = False
        self.answers: Dict[str, str] = {}
        self.cv = threading.Condition()
        self._q = 0

    def emit(self, ev: Dict):
        with self.lock:
            ev["n"] = len(self.events)
            ev["t"] = time.time()
            self.events.append(ev)

    def since(self, n: int) -> List[Dict]:
        with self.lock:
            return self.events[n:]

    def new_question(self) -> str:
        self._q += 1
        return f"q{self._q}"

    def wait_answer(self, qid: str) -> str:
        with self.cv:
            while qid not in self.answers:
                self.cv.wait(0.5)
            return self.answers.pop(qid)

    def answer(self, qid: str, ans: str):
        with self.cv:
            self.answers[qid] = ans
            self.cv.notify_all()


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>CALPHAD agent</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#f7f8fa;--fg:#1f2a37;--muted:#6b7280;--card:#fff;--line:#e5e7eb;--acc:#2c5364;--btnfg:#fff;--tool:#eef4f6;--err:#9b1c1c}
@media (prefers-color-scheme: dark){:root{--bg:#111418;--fg:#e5e7eb;--muted:#9ca3af;--card:#1a1f25;--line:#2b323b;--acc:#7fb3c7;--btnfg:#0b1a20;--tool:#1d262c;--err:#f19999}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:baseline}
header b{font-size:16px}header span{color:var(--muted);font-size:13px;flex:1}header label{font-size:13px;white-space:nowrap}
.rev{border-left:4px solid #d97706;font-size:13px}.rev.ok{border-left-color:#16a34a}
#log{max-width:980px;margin:0 auto;padding:16px 16px 140px}
.msg{margin:12px 0;padding:10px 14px;border-radius:10px;background:var(--card);border:1px solid var(--line);white-space:pre-wrap;overflow-wrap:anywhere}
.user{background:transparent;border-color:var(--acc)}
.tool{background:var(--tool);font:12.5px/1.45 ui-monospace,Menlo,Consolas,monospace;color:var(--muted)}
.tool summary{cursor:pointer;color:var(--fg)}
.err{color:var(--err)}
.prog{color:var(--muted);font-size:12.5px;margin:2px 0 2px 8px}
img.plot{max-width:100%;border-radius:8px;border:1px solid var(--line);margin-top:8px;background:#fff}
pre{background:var(--tool);padding:8px;border-radius:6px;overflow:auto}
.q{border-color:#d97706}.q button{margin:6px 6px 0 0}
button{background:var(--acc);color:var(--btnfg);border:0;border-radius:7px;padding:7px 12px;font:inherit;cursor:pointer}
button.sec{background:transparent;color:var(--fg);border:1px solid var(--line)}
button:disabled{opacity:.55;cursor:default}
footer{position:fixed;bottom:0;left:0;right:0;background:var(--bg);border-top:1px solid var(--line);padding:10px 16px}
footer form{max-width:980px;margin:0 auto;display:flex;gap:8px}
textarea{flex:1;resize:vertical;min-height:48px;max-height:240px;padding:10px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);font:inherit}
a{color:var(--acc)}
</style></head><body>
<header><b>CALPHAD agent</b><span id="meta"></span>
<button id="rep" class="sec" title="Save a problem report of the last answer (to give to a developer / Claude Code)">Report problem</button>
<label id="avl" title="Generated Python is sent to a fresh copy of the model (no conversation context) for a safety review; clean 'safe' verdicts run automatically, anything else asks you."><input type="checkbox" id="av"> auto-run Python after independent safety review</label></header>
<div id="log"></div>
<footer><form id="f"><textarea id="t" placeholder="e.g. plot the ternary isotherm at 1000 K for Co-Cr-Ni in CoCrFeNiV_Choi2019.tdb" autofocus></textarea><button id="b">Send</button></form></footer>
<script>
const log=document.getElementById('log');let n=0,cur=null,busy=false;
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function linkify(s){return esc(s).replace(/```([\s\S]*?)```/g,'<pre>$1</pre>').replace(/\*\*(.+?)\*\*/g,'<b>$1</b>')
 .replace(/(^|[\s(`'">])((?:[\w.\-]+\/)*[\w.\-]+\.(?:pptx|tdb|TDB|json|csv|png|yaml|flux|sh|pdf|zip|md|py|ipynb))/g,(m,p,f)=>p+'<a href="/file?path='+encodeURIComponent(f)+'" target="_blank">'+f+'</a>')}
function nearBottom(){return window.innerHeight+window.scrollY>=document.body.scrollHeight-200}
function bottom(){window.scrollTo(0,document.body.scrollHeight)}
function add(cls,html){const stick=nearBottom();const d=document.createElement('div');d.className='msg '+cls;d.innerHTML=html;log.appendChild(d);
 d.querySelectorAll('img').forEach(i=>i.onload=()=>{if(stick)bottom()});if(stick)bottom();return d}
function answer(qid,a,el){fetch('/api/answer',{method:'POST',body:JSON.stringify({qid,answer:a})});el.querySelectorAll('button,input').forEach(x=>x.disabled=true);el.insertAdjacentHTML('beforeend','<div class="prog">→ '+esc(a)+'</div>')}
function handle(e){
 if(e.type==='user'){add('user',esc(e.text));cur=null}
 else if(e.type==='delta'){if(!cur){cur=add('',"");cur._raw=''}cur._raw+=e.text;cur.innerHTML=linkify(cur._raw)}
 else if(e.type==='assistant'){if(!cur&&e.text){cur=add('',linkify(e.text))}else if(cur&&e.text&&!cur._raw){cur.innerHTML=linkify(e.text)}cur=null}
 else if(e.type==='tool_call'){cur=null;add('tool','<details><summary>⚙ '+esc(e.name)+'</summary>'+esc(JSON.stringify(e.args,null,1))+'</details>')}
 else if(e.type==='tool_result'){const d=add('tool'+(e.error?' err':''),'<details><summary>'+(e.error?'✖ ':'✓ ')+esc(e.name)+'</summary>'+linkify(e.text)+'</details>');
   const stick=nearBottom();(e.images||[]).forEach(p=>{d.insertAdjacentHTML('beforeend','<a href="/file?path='+encodeURIComponent(p)+'" target="_blank"><img class="plot" src="/file?path='+encodeURIComponent(p)+'"></a>')});
   d.querySelectorAll('img').forEach(i=>i.onload=()=>{if(stick)bottom()})}
 else if(e.type==='progress'){const last=log.lastElementChild;
   if(e.key&&last&&last.className==='prog'&&last.dataset.key===e.key){last.firstChild.textContent='… '+e.text}
   else{const d=document.createElement('div');d.className='prog';if(e.key)d.dataset.key=e.key;const sp=document.createElement('span');sp.textContent='… '+e.text;d.appendChild(sp);
     if(e.key&&e.key.startsWith('wait:')){const b=document.createElement('button');b.className='sec';b.textContent='stop waiting';b.title='End the wait (the job keeps running); the agent then reports the job state';
       b.style.cssText='margin-left:8px;padding:1px 8px;font-size:12px';b.onclick=()=>{fetch('/api/stop_wait',{method:'POST',body:'{}'});b.disabled=true};d.appendChild(b)}
     log.appendChild(d)}}
 else if(e.type==='question'){const d=add('q',esc(e.text)+'<br>');(e.options||[]).forEach(o=>{const b=document.createElement('button');b.className='sec';b.textContent=o;b.onclick=()=>answer(e.qid,o,d);d.appendChild(b)});
   const i=document.createElement('input');i.placeholder='or type an answer + Enter';i.style.cssText='width:100%;margin-top:8px;padding:6px';i.onkeydown=ev=>{if(ev.key==='Enter')answer(e.qid,i.value,d)};d.appendChild(i)}
 else if(e.type==='confirm'){const d=add('q',esc(e.text)+'<pre>'+esc(e.code)+'</pre>');const y=document.createElement('button');y.textContent='Run';y.onclick=()=>answer(e.qid,'yes',d);
   const no=document.createElement('button');no.className='sec';no.textContent='Decline';no.onclick=()=>answer(e.qid,'no',d);d.append(y,no)}
 else if(e.type==='report'){add('rev','📋 '+linkify(e.text))}
 else if(e.type==='review'){add('rev'+(e.ok?' ok':''),'🛡 '+esc(e.text))}
 else if(e.type==='error'){add('err',esc(e.text))}
}
async function poll(){try{const r=await fetch('/api/events?since='+n);const j=await r.json();j.events.forEach(e=>{handle(e);n=e.n+1});busy=j.busy;
 document.getElementById('b').disabled=busy;document.getElementById('b').textContent=busy?'Working…':'Send';document.getElementById('meta').textContent=j.meta;
 const av=document.getElementById('av');if(j.python_tool==='off'||j.python_tool==='auto'){av.disabled=true;av.checked=j.python_tool==='auto'}else if(document.activeElement!==av){av.checked=j.python_tool==='auto_verify'}}catch(e){}setTimeout(poll,600)}
document.getElementById('rep').onclick=async()=>{
 const r0=await fetch('/api/report_files?scope=session');const f=await r0.json();
 if(!f.ok){add('err','could not prepare a report: '+esc(f.error||''));return}
 const d=add('q','<b>Report an unsatisfactory answer</b> ('+f.requests+' request(s) since the last reset)<br>'+
  'Why is the answer not good?<textarea class="why" style="width:100%;min-height:60px;margin:6px 0"></textarea>'+
  '<div>Files that may be included - tick only what may be shared (a database may be proprietary; figures and tables made from it can reveal its content):</div>'+
  (f.files.length?f.files.map((x,i)=>'<label style="display:block"><input type="checkbox" data-p="'+esc(x.path)+'"> ['+x.kind+'] '+esc(x.path)+' ('+Math.round(x.size/1024)+' kB'+(x.derived_from.length?'; made from '+esc(x.derived_from.join(', ')):'')+')</label>').join(''):'<div class="prog">(no files involved)</div>')+
  '<label style="display:block;margin-top:6px"><input type="checkbox" class="txt" checked> include tool-result text (can quote database contents)</label>'+
  '<label style="display:block"><input type="checkbox" class="last"> only the last request</label>');
 const b=document.createElement('button');b.textContent='Write report';d.appendChild(b);
 const c=document.createElement('button');c.className='sec';c.textContent='Cancel';c.onclick=()=>d.remove();d.appendChild(c);
 b.onclick=async()=>{const note=d.querySelector('.why').value.trim();if(!note){alert('Please say why the answer is not good.');return}
  const include=[...d.querySelectorAll('input[data-p]')].filter(x=>x.checked).map(x=>x.dataset.p);
  d.querySelectorAll('button,input,textarea').forEach(x=>x.disabled=true);
  const r=await fetch('/api/report',{method:'POST',body:JSON.stringify({note,include,results_text:d.querySelector('.txt').checked,scope:d.querySelector('.last').checked?'last':'session'})});const j=await r.json();
  add('rev',j.ok?'📋 report saved: '+linkify(j.path)+' ('+j.included+' file(s) included) - you can send this zip to the developers':'could not save report: '+esc(j.error||''))}};
document.getElementById('av').onchange=ev=>fetch('/api/settings',{method:'POST',body:JSON.stringify({auto_verify:ev.target.checked})});
document.getElementById('f').onsubmit=ev=>{ev.preventDefault();const t=document.getElementById('t');if(!t.value.trim()||busy)return;
 fetch('/api/send',{method:'POST',body:JSON.stringify({text:t.value})});t.value=''};
document.getElementById('t').addEventListener('keydown',ev=>{if(ev.key==='Enter'&&!ev.shiftKey){ev.preventDefault();document.getElementById('f').requestSubmit()}});
poll();
</script></body></html>"""


def should_open_browser() -> bool:
    """Open the default browser automatically?

    * CALPHAD_NO_BROWSER=1 -> never; CALPHAD_BROWSER=1 -> always (same as --browser).
    * Linux: whenever there is a display - a desktop, a VNC session (DISPLAY=":1") or X forwarded over ssh -X/-Y
      (DISPLAY="localhost:10.0"; e.g. a RealVNC desktop whose terminal ssh-es into the cluster: the browser then
      runs on the cluster next to the agent and shows up in the VNC window).  No display -> no browser (an
      ssh -L port-forward hint is printed instead).
    * macOS / Windows: yes, unless the process runs inside an SSH login."""
    import sys
    env = os.environ
    if env.get("CALPHAD_NO_BROWSER"):
        return False
    if env.get("CALPHAD_BROWSER"):
        return True
    if sys.platform.startswith("linux"):
        return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))
    return not (env.get("SSH_CONNECTION") or env.get("SSH_TTY"))


def _state_file(port: int) -> Path:
    """Private note of a running agent's token (~/.calphad_agent, or CALPHAD_STATE_DIR)."""
    base = os.environ.get("CALPHAD_STATE_DIR") or str(Path.home() / ".calphad_agent")
    return Path(base) / f"web-{port}.json"


def cookie_name(port: int) -> str:
    return f"calphad_token_{port}"          # cookies ignore the port: one name per port


def existing_agent(port: int) -> Optional[str]:
    """Link (with its token) of OUR agent already running on this port, else None.  On a shared login node another
    user's agent may listen on the same port number: it is never reused (we would not have its token)."""
    import urllib.request
    f = _state_file(port)
    try:
        st = json.loads(f.read_text())
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/events?since=0",
                                     headers={"Cookie": f"{cookie_name(port)}={st['token']}"})
        with urllib.request.urlopen(req, timeout=2) as r:
            if "events" in json.loads(r.read().decode() or "{}"):
                return f"http://127.0.0.1:{port}/?token={st['token']}"
    except Exception:  # noqa: BLE001 - no state file, stale file, not ours, not running
        pass
    return None


def running_links() -> list:
    """Links (with token) of this user's agents that are running now, newest first."""
    base = Path(os.environ.get("CALPHAD_STATE_DIR") or str(Path.home() / ".calphad_agent"))
    out = []
    for f in sorted(base.glob("web-*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            port = int(json.loads(f.read_text())["port"])
        except (OSError, ValueError, KeyError):
            continue
        link = existing_agent(port)
        if link:
            out.append(link)
    return out


def ssh_hint(port: int, url: str) -> str:
    """How to reach this server from the machine with the browser (VNC desktop / laptop)."""
    import getpass
    import socket
    host = socket.gethostname()
    return (f"Started over SSH without a display, so no browser was opened here (ssh -X / -Y would let it open "
            f"one). On the machine with your browser (e.g. the VNC "
            f"desktop), connect with a port forward:\n"
            f"    ssh -L {port}:127.0.0.1:{port} {getpass.getuser()}@{host}\n"
            f"  (a compute node behind a login node: add  -J {getpass.getuser()}@<login-node>)\n"
            f"  then open this link there:  {url}")


def open_browser(url: str, delay: float = 0.6) -> None:
    """Open url in the default browser (the BROWSER environment variable, e.g. BROWSER=firefox, picks one)."""
    import webbrowser

    def go():
        try:
            ok = webbrowser.open(url, new=2)
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            print(f"(could not start a browser - open {url} yourself, or set BROWSER=firefox)")
    t = threading.Timer(delay, go)
    t.daemon = True
    t.start()


def serve(cfg, port: int = 8765, browser: Optional[bool] = None) -> int:
    """browser: open the default browser once the server is listening (None = automatically on a desktop)."""
    from .session import Agent
    want_browser = should_open_browser() if browser is None else browser
    running = existing_agent(port)
    if running:                                   # ours, already running (e.g. the launcher was started again)
        print(f"The CALPHAD agent is already running: {running}" + (" - opening it" if want_browser else ""))
        if want_browser:
            import webbrowser
            webbrowser.open(running, new=2)
        return 0
    import secrets
    token = secrets.token_urlsafe(18)             # only whoever has the printed link can use this agent
    hub = Hub()
    ui = WebUI(hub)
    agent = Agent(cfg, ui,
                  on_text=lambda t: hub.emit({"type": "delta", "text": t}),
                  on_tool_start=lambda c: hub.emit({"type": "tool_call", "name": c.name, "args": c.arguments}),
                  on_tool_end=lambda c, text, imgs, err: hub.emit(
                      {"type": "tool_result", "name": c.name, "text": text[:6000], "images": imgs, "error": err}))
    root = agent.toolbox.workdir
    meta = f"{cfg.llm.model} · {cfg.llm.base_url or cfg.llm.provider} · workdir {root}"

    def run(text):
        hub.busy = True
        try:
            out = agent.run_turn(text)
            hub.emit({"type": "assistant", "text": out})
        except Exception as e:  # noqa: BLE001
            from .llm import RateLimitGiveUp
            hub.emit({"type": "error", "text": str(e) if isinstance(e, RateLimitGiveUp) else
                      f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"})
        finally:
            hub.busy = False

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            data = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authed(self, q=None) -> bool:
            """The link's token OR this session's cookie.  Both are checked: after a restart the browser still
            holds the previous session's cookie, and a fresh link must win over it."""
            import hmac
            cands = []
            for part in self.headers.get("Cookie", "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == cookie_name(self.server.server_port):
                    cands.append(v)
            if q is not None:
                cands += q.get("token") or []
            return any(c and hmac.compare_digest(c, token) for c in cands)

        def _forbidden(self):
            data = ("<html><body style='font-family:sans-serif;max-width:680px;margin:40px auto;line-height:1.5'>"
                    "<h3>This CALPHAD agent needs its access link</h3>"
                    "<p>Each agent session has its own secret link, so other users on a shared machine cannot use "
                    "yours. You opened the plain address (for example from a bookmark, an old tab, or "
                    "<code>firefox http://127.0.0.1:8765</code>).</p>"
                    "<p>To get the link, on the machine where the agent runs:</p>"
                    "<pre style='background:#f3f4f6;padding:10px'>pycalphad-agent-venv/bin/calphad-agent --link</pre>"
                    "<p>or copy the line <code>CALPHAD agent web UI: http://127.0.0.1:...?token=...</code> from the "
                    "terminal where it was started. Open that link once; after that this browser remembers it and "
                    "the plain address works too.</p></body></html>").encode()
            self.send_response(403)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            if not self._authed(q):
                return self._forbidden()
            if "token" in q:                       # swap the link's token for a cookie, then a clean address
                self.send_response(303)
                self.send_header("Set-Cookie", f"{cookie_name(self.server.server_port)}={token}; Path=/; HttpOnly; "
                                               "SameSite=Strict")
                self.send_header("Location", u.path or "/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if u.path == "/":
                data = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif u.path == "/api/events":
                self._json({"events": hub.since(int(q.get("since", ["0"])[0])), "busy": hub.busy,
                            "meta": meta + " · " + agent.usage_text(),
                            "python_tool": agent.toolbox.python_tool if "run_python" in agent.toolbox.tools else "off"})
            elif u.path == "/api/report_files":
                try:
                    from .failure import candidate_files, report_turns
                    turns = report_turns(agent, q.get("scope", ["session"])[0])
                    tb_ = agent.toolbox
                    files = [{"path": tb_.rel(p), "kind": i["kind"], "size": i["size"],
                              "derived_from": [Path(x).name for x in i.get("derived_from") or []]}
                             for p, i in candidate_files(agent, turns).items()]
                    self._json({"ok": True, "requests": len(turns), "files": files})
                except Exception as e:  # noqa: BLE001
                    self._json({"ok": False, "error": str(e)})
            elif u.path == "/file":
                p = Path(q.get("path", [""])[0])
                p = (p if p.is_absolute() else root / p).resolve()
                allowed = [root.resolve(), agent.toolbox.out.resolve()]
                if not any(str(p).startswith(str(a)) for a in allowed) or not p.is_file():
                    if not p.is_file():
                        hits = list(agent.toolbox.out.rglob(p.name))
                        if hits:
                            p = hits[0]
                    if not p.is_file() or not any(str(p).startswith(str(a)) for a in allowed):
                        self.send_error(404)
                        return
                data = p.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(str(p))[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)

        def do_POST(self):
            if not self._authed():
                return self._forbidden()
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/send":
                if hub.busy:
                    return self._json({"ok": False, "error": "busy"}, 409)
                hub.emit({"type": "user", "text": body.get("text", "")})
                threading.Thread(target=run, args=(body.get("text", ""),), daemon=True).start()
                self._json({"ok": True})
            elif self.path == "/api/report":
                try:
                    inc = [str(agent.toolbox.p(x)) for x in body.get("include") or []]
                    share = {"files": inc, "results_text": bool(body.get("results_text", True)),
                             "consent": "ticked by the user in the web dialog" if inc else
                             "no files ticked by the user in the web dialog"}
                    path = agent.save_report(note=body.get("note", ""), scope=body.get("scope") or "session",
                                             share=share)
                    self._json({"ok": True, "path": agent.toolbox.rel(path), "included": len(inc)})
                except Exception as e:  # noqa: BLE001
                    self._json({"ok": False, "error": str(e)})
            elif self.path == "/api/settings":
                if "run_python" in agent.toolbox.tools and agent.toolbox.python_tool in ("ask", "auto_verify"):
                    agent.toolbox.python_tool = "auto_verify" if body.get("auto_verify") else "ask"
                    hub.emit({"type": "progress", "text": f"python_tool = {agent.toolbox.python_tool}"})
                self._json({"ok": True, "python_tool": agent.toolbox.python_tool})
            elif self.path == "/api/stop_wait":         # end a wait_job until='done' (the job keeps running)
                agent.toolbox.stop_waiting.set()
                self._json({"ok": True})
            elif self.path == "/api/answer":
                hub.answer(body["qid"], body.get("answer", ""))
                self._json({"ok": True})
            else:
                self.send_error(404)

    srv = None
    for p in range(port, port + 20):              # the port is taken by something else: the next free one
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), H)
            port = p
            break
        except OSError:
            continue
    if srv is None:
        print(f"ports {port}-{port + 19} are all in use; start with --port N")
        return 1
    url = f"http://127.0.0.1:{port}/?token={token}"
    sf = _state_file(port)
    try:                                          # private note of our token, so a second launch can reuse us
        sf.parent.mkdir(mode=0o700, exist_ok=True)
        sf.write_text(json.dumps({"port": port, "token": token, "pid": os.getpid()}))
        os.chmod(sf, 0o600)
    except OSError:
        pass
    print(f"CALPHAD agent web UI: {url}   (Ctrl-C to stop)\n"
          f"  open this full link once (it contains this session's secret); `calphad-agent --link` shows it again")
    if want_browser:
        open_browser(url)
    elif os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        print(ssh_hint(port, url))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if json.loads(sf.read_text()).get("token") == token:
                sf.unlink()
        except (OSError, ValueError):
            pass
    return 0
