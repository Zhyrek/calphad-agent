"""Web UI start-up: opens the default browser once listening (desktop only), reuses a running agent, skips busy
ports.  webbrowser.open is replaced by a recorder, so no real browser window opens."""
import socket
import threading
import time
import webbrowser

import pytest

from calphad_tools.agent import web


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(autouse=True)
def state(tmp_path, monkeypatch):
    """The agent's private token notes live in a temporary folder for these tests (not ~/.calphad_agent)."""
    d = tmp_path / "state"
    monkeypatch.setenv("CALPHAD_STATE_DIR", str(d))
    return d


@pytest.fixture
def opened(monkeypatch):
    urls = []
    monkeypatch.setattr(webbrowser, "open", lambda url, new=0: urls.append(url) or True)
    return urls


def _cfg(tmp_path):
    from calphad_tools.agent.config import load_config
    cfg = load_config(str(tmp_path / "none.toml"))
    cfg.llm.base_url, cfg.llm.api_key = "http://127.0.0.1:9/v1", "sk-test"
    cfg.agent.workdir, cfg.agent.output_dir = str(tmp_path), str(tmp_path / "out")
    return cfg


def _wait(cond, t=10.0):
    end = time.time() + t
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


def test_should_open_browser(monkeypatch):
    import sys
    for v in ("CALPHAD_NO_BROWSER", "CALPHAD_BROWSER", "SSH_CONNECTION", "SSH_TTY", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert web.should_open_browser()                     # Mac desktop
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 5 6.7.8.9 22")
    assert not web.should_open_browser()                 # Mac, logged in over ssh
    monkeypatch.setattr(sys, "platform", "linux")
    assert not web.should_open_browser()                 # cluster over ssh, no display
    monkeypatch.setenv("DISPLAY", ":1")
    assert web.should_open_browser()                     # VNC desktop started from an ssh login
    monkeypatch.setenv("DISPLAY", "localhost:10.0")
    assert web.should_open_browser()                     # ssh -X/-Y: firefox on the cluster shows in the VNC window
    monkeypatch.delenv("DISPLAY")
    assert not web.should_open_browser()                 # plain ssh, no display
    monkeypatch.setenv("CALPHAD_BROWSER", "1")
    assert web.should_open_browser()                     # forced (--browser)
    monkeypatch.setenv("CALPHAD_NO_BROWSER", "1")
    monkeypatch.delenv("CALPHAD_BROWSER")
    monkeypatch.setenv("DISPLAY", ":0")
    assert not web.should_open_browser()                 # switched off


def test_opens_after_listening_then_reuses_running_agent(tmp_path, opened):
    port = _free_port()
    threading.Thread(target=web.serve, args=(_cfg(tmp_path), port, True), daemon=True).start()
    base = f"http://127.0.0.1:{port}/?token="
    assert _wait(lambda: any(u.startswith(base) for u in opened)), opened
    url = opened[0]
    assert web.existing_agent(port) == url
    # the launcher started again: no second server, just a new tab on the running one (same secret link)
    assert web.serve(_cfg(tmp_path), port, True) == 0
    assert opened.count(url) == 2


def test_busy_port_moves_on(tmp_path, opened):
    port = _free_port()
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", port))
    blocker.listen(1)                                    # something that is not the agent
    try:
        assert not web.existing_agent(port)
        threading.Thread(target=web.serve, args=(_cfg(tmp_path), port, True), daemon=True).start()
        assert _wait(lambda: bool(opened))
        other = int(opened[-1].split("127.0.0.1:", 1)[1].split("/", 1)[0])
        assert port < other < port + 20 and web.existing_agent(other)
    finally:
        blocker.close()


def _get(url, cookie=None, method="GET", data=None):
    import urllib.error
    import urllib.request

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    req = urllib.request.Request(url, method=method, data=data, headers={"Cookie": cookie} if cookie else {})
    try:
        r = urllib.request.build_opener(NoRedirect).open(req, timeout=5)
        return r.status, dict(r.headers), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode()


def test_only_the_secret_link_works(tmp_path, opened):
    port = _free_port()
    threading.Thread(target=web.serve, args=(_cfg(tmp_path), port, True), daemon=True).start()
    assert _wait(lambda: bool(opened))
    link = opened[0]
    token = link.split("token=", 1)[1]
    base = f"http://127.0.0.1:{port}"
    # someone else on the same machine, without the link
    for path in ("/", "/api/events?since=0", "/file?path=calphad_agent.toml", "/api/report_files"):
        assert _get(base + path)[0] == 403, path
    assert _get(base + "/api/send", method="POST", data=b'{"text": "hi"}')[0] == 403
    assert _get(base + "/?token=wrong")[0] == 403
    assert _get(base + "/api/events", cookie=f"{web.cookie_name(port)}=wrong")[0] == 403
    # the link: token -> HttpOnly cookie + a clean address
    code, headers, _ = _get(link)
    assert code == 303 and headers["Location"] == "/"
    assert f"{web.cookie_name(port)}={token}" in headers["Set-Cookie"] and "HttpOnly" in headers["Set-Cookie"]
    cookie = f"{web.cookie_name(port)}={token}"
    assert _get(base + "/", cookie=cookie)[0] == 200
    assert '"events"' in _get(base + "/api/events?since=0", cookie=cookie)[2]


def test_another_users_agent_is_not_reused(tmp_path, opened, monkeypatch):
    port = _free_port()
    threading.Thread(target=web.serve, args=(_cfg(tmp_path), port, True), daemon=True).start()
    assert _wait(lambda: bool(opened))
    monkeypatch.setenv("CALPHAD_STATE_DIR", str(tmp_path / "other_user"))   # a different user: no token note
    assert web.existing_agent(port) is None
    threading.Thread(target=web.serve, args=(_cfg(tmp_path), port, True), daemon=True).start()
    assert _wait(lambda: len(opened) >= 2)
    assert f"127.0.0.1:{port}/" not in opened[1]           # got its own server on another port


def test_ssh_without_display_prints_forward_hint(tmp_path, opened, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, "platform", "linux")
    for v in ("DISPLAY", "WAYLAND_DISPLAY", "CALPHAD_BROWSER", "CALPHAD_NO_BROWSER"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 5 6.7.8.9 22")
    port = _free_port()
    threading.Thread(target=web.serve, args=(_cfg(tmp_path), port, None), daemon=True).start()
    time.sleep(3)
    assert not opened                                    # no display: nothing opened
    hint = web.ssh_hint(port, "http://127.0.0.1:1/?token=x")
    assert f"ssh -L {port}:127.0.0.1:{port}" in hint and "token=x" in hint


def test_link_command_finds_the_running_agent(tmp_path, opened, capsys):
    from calphad_tools.agent import cli
    port = _free_port()
    threading.Thread(target=web.serve, args=(_cfg(tmp_path), port, True), daemon=True).start()
    assert _wait(lambda: bool(opened))
    assert opened[0] in web.running_links()
    assert cli.main(["--link"]) == 0 and opened[0] in capsys.readouterr().out
    code, _, body = _get(f"http://127.0.0.1:{port}/")           # the plain address: says how to get the link
    assert code == 403 and "calphad-agent --link" in body


def test_fresh_link_beats_a_stale_cookie_from_a_previous_session(tmp_path, opened):
    """Regression: the agent was restarted; the browser still sends the old session's cookie with the new link."""
    port = _free_port()
    threading.Thread(target=web.serve, args=(_cfg(tmp_path), port, True), daemon=True).start()
    assert _wait(lambda: bool(opened))
    link = opened[0]
    token = link.split("token=", 1)[1]
    stale = f"{web.cookie_name(port)}=token-of-the-previous-session"
    code, headers, _ = _get(link, cookie=stale)
    assert code == 303 and f"{web.cookie_name(port)}={token}" in headers["Set-Cookie"]   # replaced
    assert _get(f"http://127.0.0.1:{port}/", cookie=stale)[0] == 403                        # stale alone: no
    assert _get(f"http://127.0.0.1:{port}/", cookie=f"{web.cookie_name(port)}={token}")[0] == 200
